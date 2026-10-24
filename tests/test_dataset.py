import numpy as np
import torch

from biotwin.data.dataset import H, LH, day_range, make_windows, subject_split


def test_subject_split_disjoint_stratified_deterministic(toy_cohort):
    sp = subject_split(toy_cohort.names, seed=0)
    sets = [set(v) for v in sp.values()]
    assert sum(len(s) for s in sets) == 30
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    for g in ["adolescent", "adult", "child"]:
        assert sum(n.startswith(g) for n in sp["test"]) == 2
    assert sp == subject_split(toy_cohort.names, seed=0)


def test_windows_alignment_and_no_future_in_history(toy_cohort):
    w = make_windows(toy_cohort, ["adult#001"], stride=7)
    t0 = w["t0"]
    p = toy_cohort.idx("adult#001")
    off = p * 10_000
    # CGM equals the time index (+ offset): history must end exactly at t0-1, targets start at t0
    assert torch.equal(w["cgm_hist"][:, -1] - off, (t0 - 1).float())
    assert torch.equal(w["y"][:, 0] - off, t0.float())
    assert (w["cgm_hist"].max(1).values < w["y"].min(1).values).all()
    assert w["ins"].shape[1] == LH + H


def test_targets_inside_range(toy_cohort):
    lo, hi = day_range(1, 2)
    w = make_windows(toy_cohort, ["child#001"], (lo, hi))
    assert (w["t0"] >= lo).all() and (w["t0"] + H <= hi).all()


def test_future_cgm_cannot_reach_inputs(toy_cohort):
    """Corrupting CGM after t0 must not change anything a model sees for that window."""
    w1 = make_windows(toy_cohort, ["adult#002"], stride=50)
    corrupted = toy_cohort.cgm.copy()
    p = toy_cohort.idx("adult#002")
    t0 = int(w1["t0"][3])
    corrupted[p, t0:] = -1
    toy_cohort.cgm = corrupted
    w2 = make_windows(toy_cohort, ["adult#002"], stride=50)
    for k in ["cgm_hist", "ins", "carbs", "basal"]:
        assert torch.equal(w1[k][3], w2[k][3])
