"""Splits and forecasting windows. This file is where leakage is prevented, so it is tested.

Task (same for every model):
    given  - CGM over the last LH steps (6 h)
           - insulin and carbs over the last LH steps AND the next H steps (known/planned inputs)
    predict- CGM at the next H steps (5..60 min)
Conditioning on future inputs is what makes this a "twin" question: "what if I eat X at t?".

Splits:
    - subject level: train / val / test patients are disjoint (stratified by age group)
    - within each test patient (Phase 2): days [0, ADAPT_DAYS) are used for personalisation
      (last of those days = calibration day), days [ADAPT_DAYS, end) are the only evaluation days.
    - Phase 1/3 population models are evaluated on the SAME test-patient evaluation days so
      every number in the results table is comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

LH = 72  # history length, steps (6 h)
H = 12  # horizon, steps (60 min)
STEPS_PER_DAY = 288
ADAPT_DAYS = 6  # days 0..4 fit, day 5 calibration, days 6.. evaluation


@dataclass
class Cohort:
    names: np.ndarray
    cgm: np.ndarray
    bg: np.ndarray
    insulin: np.ndarray
    carbs: np.ndarray
    basal: np.ndarray

    @classmethod
    def load(cls, path: str = "data/cohort.npz") -> "Cohort":
        z = np.load(path)
        return cls(z["names"], z["cgm"], z["bg"], z["insulin"], z["carbs"], z["basal"])

    def idx(self, name: str) -> int:
        return int(np.where(self.names == name)[0][0])


def subject_split(names, seed: int = 0, n_val: int = 2, n_test: int = 2) -> dict[str, list[str]]:
    """Stratified by group prefix (adolescent/adult/child). Deterministic given seed."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    groups = sorted({str(n).split("#")[0] for n in names})
    for g in groups:
        members = sorted(str(n) for n in names if str(n).startswith(g))
        perm = list(rng.permutation(members))
        out["test"] += perm[:n_test]
        out["val"] += perm[n_test : n_test + n_val]
        out["train"] += perm[n_test + n_val :]
    return out


def day_range(day_lo: int, day_hi: int) -> tuple[int, int]:
    return day_lo * STEPS_PER_DAY, day_hi * STEPS_PER_DAY


def make_windows(
    cohort: Cohort,
    names: list[str],
    t_range: tuple[int, int] | None = None,
    stride: int = 1,
) -> dict[str, torch.Tensor]:
    """Windows whose *forecast targets* all lie in t_range=[lo, hi). History may precede lo.

    Returned tensors (N windows):
        cgm_hist [N, LH], ins [N, LH+H], carbs [N, LH+H], y [N, H], bg_y [N, H],
        basal [N], pid [N] (index into cohort), t0 [N] (index of first target step)
    """
    keys = {k: [] for k in ["cgm_hist", "ins", "carbs", "y", "bg_y", "basal", "pid", "t0"]}
    T = cohort.cgm.shape[1]
    for n in names:
        p = cohort.idx(n)
        lo, hi = (0, T) if t_range is None else t_range
        first = max(lo, LH)  # first target index needs LH steps of history
        for t0 in range(first, hi - H + 1, stride):
            keys["cgm_hist"].append(cohort.cgm[p, t0 - LH : t0])
            keys["ins"].append(cohort.insulin[p, t0 - LH : t0 + H])
            keys["carbs"].append(cohort.carbs[p, t0 - LH : t0 + H])
            keys["y"].append(cohort.cgm[p, t0 : t0 + H])
            keys["bg_y"].append(cohort.bg[p, t0 : t0 + H])
            keys["basal"].append(cohort.basal[p])
            keys["pid"].append(p)
            keys["t0"].append(t0)
    out = {}
    for k, v in keys.items():
        arr = np.array(v)
        out[k] = torch.as_tensor(arr, dtype=torch.long if k in ("pid", "t0") else torch.float32)
    return out


def subset(w: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[mask] for k, v in w.items()}


def batches(w: dict[str, torch.Tensor], bs: int, shuffle: bool, gen: torch.Generator | None = None):
    n = len(w["y"])
    order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        sl = order[i : i + bs]
        yield {k: v[sl] for k, v in w.items()}


def standard_sets(cohort: Cohort, seed: int = 0, train_stride: int = 2) -> dict:
    """The canonical evaluation protocol used by every script."""
    sp = subject_split(cohort.names, seed)
    T = cohort.cgm.shape[1]
    eval_lo, _ = day_range(ADAPT_DAYS, 0)
    return {
        "split": sp,
        "train": make_windows(cohort, sp["train"], stride=train_stride),
        "val": make_windows(cohort, sp["val"], stride=3),
        "test_fit": make_windows(cohort, sp["test"], day_range(0, ADAPT_DAYS - 1)),
        "test_cal": make_windows(cohort, sp["test"], day_range(ADAPT_DAYS - 1, ADAPT_DAYS)),
        "test_eval": make_windows(cohort, sp["test"], (eval_lo, T)),
    }
