"""The twin exposed as tools. Everything numeric an agent says must come from here.

simulate_scenario runs paired counterfactuals: the same posterior parameter samples drive the
baseline (basal insulin only, no meal) and the scenario, so the *difference* is less noisy than
either trajectory (common random numbers).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from biotwin.data.dataset import ADAPT_DAYS, LH, STEPS_PER_DAY, Cohort, subject_split
from biotwin.models.mechanistic import CorrectionNet, MinimalModel
from biotwin.uncertainty import _full_theta

VALIDATED_HORIZON_MIN = 60
TOOLS_SPEC = [
    {
        "name": "list_patients",
        "description": "List the virtual (simulated) patients that have a personalised twin, and the days/hours "
                       "of recorded history that scenarios can start from.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "simulate_scenario",
        "description": (
            "Run a what-if simulation on a personalised glucose-insulin digital twin of a SIMULATED patient. "
            "Starts from the patient's real recorded history at (day, hour), then compares a baseline "
            "(basal insulin only) with a scenario (optional meal and/or insulin bolus). Returns median and "
            "80% interval of predicted glucose (mg/dL) for both, peak/nadir, and risk probabilities. "
            "Only the first 60 minutes are validated against held-out data."),
        "input_schema": {
            "type": "object",
            "properties": {
                "patient": {"type": "string", "description": "e.g. 'adult#003'"},
                "day": {"type": "integer", "description": "day index of recorded history (evaluation days)"},
                "hour": {"type": "integer", "description": "hour of day 0-23 at which the scenario starts"},
                "meal_g": {"type": "number", "description": "carbohydrates in grams (0 for none)"},
                "meal_offset_min": {"type": "integer", "description": "minutes after start when the meal is eaten"},
                "bolus_U": {"type": "number", "description": "insulin bolus in units (0 for none)"},
                "bolus_offset_min": {"type": "integer", "description": "minutes after start of the bolus"},
                "horizon_min": {"type": "integer", "description": "simulation length in minutes (5-240)"},
            },
            "required": ["patient", "day", "hour", "meal_g", "meal_offset_min", "bolus_U", "bolus_offset_min",
                         "horizon_min"],
            "additionalProperties": False,
        },
    },
]


class TwinService:
    def __init__(self, data="data/cohort.npz", ckpt_dir="results/checkpoints", n_samples=100, seed=0):
        self.cohort = Cohort.load(data)
        self.test = subject_split(self.cohort.names, seed)["test"]
        self.model = MinimalModel(CorrectionNet(32))
        self.model.load_state_dict(torch.load(Path(ckpt_dir) / "hybrid.pt"))
        self.model.eval()
        self.post = {}
        for n in self.test:
            f = Path(ckpt_dir) / f"personal_{n.replace('#', '_')}.pt"
            if f.exists():
                self.post[n] = torch.load(f)
        self.n_samples = n_samples
        self.gen = torch.Generator().manual_seed(seed)
        self.T_days = self.cohort.cgm.shape[1] // STEPS_PER_DAY

    def list_patients(self) -> dict:
        return {"patients": sorted(self.post), "days": list(range(ADAPT_DAYS, self.T_days)), "hours": "0-23",
                "note": "Virtual patients from the UVA/Padova simulator (simglucose). Not real people."}

    def simulate_scenario(self, patient, day, hour, meal_g=0.0, meal_offset_min=0, bolus_U=0.0,
                          bolus_offset_min=0, horizon_min=120) -> dict:
        if patient not in self.post:
            return {"error": f"unknown patient '{patient}'. Available: {sorted(self.post)}"}
        if not (ADAPT_DAYS <= day < self.T_days) or not (0 <= hour <= 23):
            return {"error": f"day must be in {ADAPT_DAYS}..{self.T_days - 1} and hour in 0..23"}
        if not (5 <= horizon_min <= 240):
            return {"error": "horizon_min must be within 5..240"}
        if meal_g < 0 or bolus_U < 0:
            return {"error": "meal_g and bolus_U must be non-negative"}
        H = horizon_min // 5
        p = self.cohort.idx(patient)
        t0 = day * STEPS_PER_DAY + hour * 12
        c = self.cohort
        if t0 + 1 > c.cgm.shape[1] or t0 < LH:
            return {"error": "not enough recorded history at that time"}
        basal = float(c.basal[p])
        fut_ins = np.full(H, basal, np.float32)
        fut_cho = np.zeros(H, np.float32)
        base = {"cgm_hist": torch.tensor(c.cgm[p, t0 - LH:t0])[None],
                "ins": torch.tensor(np.concatenate([c.insulin[p, t0 - LH:t0], fut_ins]))[None],
                "carbs": torch.tensor(np.concatenate([c.carbs[p, t0 - LH:t0], fut_cho]))[None],
                "basal": torch.tensor([basal]), "y": torch.zeros(1, H)}
        scen = {k: v.clone() for k, v in base.items()}
        km, kb = meal_offset_min // 5, bolus_offset_min // 5
        if meal_g > 0 and km < H:
            scen["carbs"][0, LH + km] += meal_g
        if bolus_U > 0 and kb < H:
            scen["ins"][0, LH + kb] += bolus_U
        post = self.post[patient]
        L = torch.linalg.cholesky(post["cov"])
        pop = self.model.log_theta.detach()
        S = self.n_samples
        phis = post["phi"] + (L @ torch.randn(len(post["phi"]), S, generator=self.gen)).T
        rep = lambda w: {k: v.expand(S, *v.shape[1:]).contiguous() for k, v in w.items()}  # noqa: E731
        lt = torch.stack([_full_theta(pop, ph) for ph in phis])
        with torch.no_grad():
            yb = self.model(rep(base), log_theta=lt)
            ys = self.model(rep(scen), log_theta=lt)
        return {
            "patient": patient, "start": f"day {day} {hour:02d}:00", "current_cgm": round(float(c.cgm[p, t0 - 1]), 1),
            "scenario": {"meal_g": meal_g, "meal_offset_min": meal_offset_min, "bolus_U": bolus_U,
                         "bolus_offset_min": bolus_offset_min, "horizon_min": horizon_min},
            "baseline": _summ(yb), "with_scenario": _summ(ys),
            "difference_median_peak": round(float((ys.max(1).values - yb.max(1).values).median()), 1),
            "difference_median_nadir": round(float((ys.min(1).values - yb.min(1).values).median()), 1),
            "difference_median_mean_glucose": round(float((ys - yb).mean(1).median()), 1),
            "caveats": _caveats(horizon_min, meal_g, bolus_U),
        }


def _summ(y):
    q = lambda x: [round(float(v), 1) for v in torch.quantile(x, torch.tensor([0.1, 0.5, 0.9]))]  # noqa: E731
    out = {}
    for m in (30, 60, 120, 180, 240):
        k = m // 5 - 1
        if k < y.shape[1]:
            out[f"t{m}min_p10_p50_p90"] = q(y[:, k])
    peak, nadir = y.max(1).values, y.min(1).values
    out["peak_median"] = round(float(peak.median()), 1)
    out["nadir_median"] = round(float(nadir.median()), 1)
    out["prob_below_70"] = round(float((nadir < 70).float().mean()), 2)
    out["prob_above_180"] = round(float((peak > 180).float().mean()), 2)
    return out


def _caveats(horizon_min, meal_g, bolus_U):
    c = ["Research prototype on simulated patients; not medical advice.",
         "Intervals reflect posterior parameter uncertainty only (no sensor noise, no model-structure error)."]
    if horizon_min > VALIDATED_HORIZON_MIN:
        c.append(f"Beyond {VALIDATED_HORIZON_MIN} min the forecast is unvalidated extrapolation of the mechanistic model.")
    if meal_g > 100 or bolus_U > 15:
        c.append("Input is larger than most training examples; expect extrapolation error.")
    return c
