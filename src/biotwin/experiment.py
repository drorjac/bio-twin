"""Shared plumbing for the phase scripts: data sets, evaluation, logging, checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from biotwin.data.dataset import Cohort, standard_sets
from biotwin.evaluate import point_metrics
from biotwin.utils import log_run

CKPT = Path("results/checkpoints")
PRED = Path("results/predictions")


def load(seed: int = 0, path: str = "data/cohort.npz"):
    cohort = Cohort.load(path)
    sets = standard_sets(cohort, seed)
    return cohort, sets


def evaluate_and_log(phase, name, pred, sets, config, extra=None):
    w = sets["test_eval"]
    m = point_metrics(pred, w["y"])
    m["rmse_vs_trueBG_60min"] = (pred[:, -1] - w["bg_y"][:, -1]).pow(2).mean().sqrt().item()
    m.update(extra or {})
    log_run(phase, name, config, m)
    PRED.mkdir(parents=True, exist_ok=True)
    torch.save(pred, PRED / f"{name}.pt")
    print(f"[{phase}] {name:22s} RMSE30 {m['rmse_30min']:6.2f}  RMSE60 {m['rmse_60min']:6.2f}  "
          f"MAE60 {m['mae_60min']:6.2f}", flush=True)
    return m


def save_ckpt(model, name):
    CKPT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT / f"{name}.pt")


def results_table(root="results", phases=("phase1", "phase2", "phase3")):
    rows = []
    for ph in phases:
        for f in sorted(Path(root, ph).glob("*.json")):
            r = json.loads(f.read_text())
            if "metrics" not in r:
                continue
            rows.append((ph, r["name"], r["metrics"]))
    return rows
