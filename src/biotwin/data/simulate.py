"""Generate a synthetic T1D cohort with the open UVA/Padova implementation in `simglucose`.

Why a simulator: real CGM datasets (OhioT1DM, etc.) need data-use agreements. The simulator
gives us ground truth (true plasma glucose, known inputs) and lets us run counterfactuals.

Key honesty point: the *data-generating* model (UVA/Padova, 13 states) is NOT the model we
fit in Phase 1 (a 7-state Bergman-style minimal model). If they were the same, the
mechanistic model would win trivially.

Output: one parquet-free npz per cohort with arrays on a 5-min grid:
    cgm    [P, T]  mg/dL, noisy sensor reading (GuardianRT noise model, 5-min sampling)
    bg     [P, T]  mg/dL, true glucose (for analysis only; never a model input)
    insulin[P, T]  U delivered during the 5-min step (basal + bolus)
    carbs  [P, T]  g eaten during the 5-min step
    basal  [P]     U per 5-min step (known pump setting)
"""
from __future__ import annotations

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("simglucose").setLevel(logging.ERROR)

STEP_MIN = 5  # minutes per sample on the output grid


def patient_names() -> list[str]:
    groups = ["adolescent", "adult", "child"]
    return [f"{g}#{i:03d}" for g in groups for i in range(1, 11)]


def _daily_meals(rng: np.random.Generator) -> list[tuple[int, float]]:
    """(minute_of_day, grams). Same structure as simglucose RandomScenario, own RNG."""
    prob = [0.95, 0.3, 0.95, 0.3, 0.95, 0.3]
    mu_t = np.array([7, 9.5, 12, 15, 18, 21.5]) * 60
    sd_t = [60, 30, 60, 30, 60, 30]
    lb = np.array([5, 9, 10, 14, 16, 20]) * 60
    ub = np.array([9, 10, 14, 16, 20, 23]) * 60
    mu_g = [45, 10, 70, 10, 80, 10]
    sd_g = [10, 5, 10, 5, 10, 5]
    meals = []
    for k in range(6):
        if rng.random() < prob[k]:
            t = int(np.clip(rng.normal(mu_t[k], sd_t[k]), lb[k], ub[k]))
            g = max(round(rng.normal(mu_g[k], sd_g[k])), 0)
            if g > 0:
                meals.append((t, float(g)))
    return meals


def simulate_patient(name: str, days: int, seed: int) -> dict:
    """Minute-level closed-loop simulation with an imperfect basal-bolus 'patient behaviour'.

    Imperfections are deliberate so that insulin and carbs are not perfectly collinear
    (otherwise their effects are not separately identifiable from data):
      - carb counting error ~ N(0, 20%)
      - 10% of meals get no bolus, bolus timing jitter in {-10, 0, 0, 15, 30} min
      - rescue carbs (15 g) when CGM < 70 and no rescue in the last 30 min
    """
    from simglucose.patient.t1dpatient import Action, T1DPatient
    from simglucose.sensor.cgm import CGMSensor

    rng = np.random.default_rng(seed)
    patient = T1DPatient.withName(name, seed=seed)
    sensor = CGMSensor.withName("GuardianRT", seed=seed)
    params = pd.read_csv(_param_file("vpatient_params.csv"))
    quest = pd.read_csv(_param_file("Quest.csv"))
    p = params[params.Name == name].iloc[0]
    q = quest[quest.Name == name].iloc[0]
    basal_rate = p.u2ss * p.BW / 6000.0  # U/min, steady-state basal

    n_min = days * 24 * 60
    meal_at = np.zeros(n_min)  # announced grams at minute
    bolus_at = np.zeros(n_min)  # U at minute
    for d in range(days):
        for t, g in _daily_meals(rng):
            m = d * 1440 + t
            meal_at[m] += g
            if rng.random() < 0.9:
                est = g * max(0.3, 1 + 0.2 * rng.standard_normal())
                delay = int(rng.choice([-10, 0, 0, 15, 30]))
                mb = int(np.clip(m + delay, 0, n_min - 1))
                bolus_at[mb] += est / q.CR  # CR: grams covered by 1 U

    T = n_min // STEP_MIN
    cgm = np.zeros(T)
    bg = np.zeros(T)
    ins = np.zeros(T)
    cho = np.zeros(T)
    last_rescue = -10_000
    cur_cgm = float(p.Gb)
    for m in range(n_min):
        meal = meal_at[m]
        if cur_cgm < 70 and m - last_rescue > 30:
            meal += 15.0
            last_rescue = m
        # correction bolus on top of meal bolus, as in simglucose's BBController
        bolus = bolus_at[m]
        if bolus > 0 and cur_cgm > 150:
            bolus += (cur_cgm - 140) / q.CF  # CF: mg/dL drop per 1 U
        u = basal_rate + bolus  # bolus delivered within one minute -> U/min
        patient.step(Action(CHO=meal, insulin=u))
        eaten = patient._last_action.CHO  # grams actually eaten this minute (5 g/min max)
        cur_cgm = sensor.measure(patient)
        k = m // STEP_MIN
        ins[k] += u
        cho[k] += eaten
        if m % STEP_MIN == STEP_MIN - 1:
            cgm[k] = cur_cgm
            bg[k] = patient.observation.Gsub
    return dict(name=name, cgm=cgm, bg=bg, insulin=ins, carbs=cho, basal=basal_rate * STEP_MIN)


def _param_file(fname: str) -> str:
    import os

    import simglucose

    return os.path.join(os.path.dirname(simglucose.__file__), "params", fname)


def _job(args):
    return simulate_patient(*args)


def simulate_cohort(days: int, seed: int, out_path: str, workers: int = 8) -> None:
    names = patient_names()
    jobs = [(n, days, seed * 1000 + i) for i, n in enumerate(names)]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_job, jobs))
    np.savez_compressed(
        out_path,
        names=np.array([r["name"] for r in res]),
        cgm=np.stack([r["cgm"] for r in res]).astype(np.float32),
        bg=np.stack([r["bg"] for r in res]).astype(np.float32),
        insulin=np.stack([r["insulin"] for r in res]).astype(np.float32),
        carbs=np.stack([r["carbs"] for r in res]).astype(np.float32),
        basal=np.array([r["basal"] for r in res], dtype=np.float32),
        step_min=STEP_MIN,
        days=days,
        seed=seed,
        created=str(datetime.now()),
    )
