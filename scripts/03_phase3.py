"""Phase 3: generic sequence models (Transformer, Mamba-style SSM) vs the hybrid twin.

Also a data-efficiency probe: retrain hybrid and Transformer on only 3 training patients
(one per age group). Hypothesis to test, not assume: physics structure helps most when data is scarce.
Compute is CPU-only here, so models are small and training is time-boxed; we report the budget.
"""
import argparse
import time

import torch

from biotwin.data.dataset import make_windows, subset
from biotwin.experiment import CKPT, evaluate_and_log, load, save_ckpt
from biotwin.models.mechanistic import CorrectionNet, MinimalModel
from biotwin.models.neural import build, n_params
from biotwin.train import fit, predict
from biotwin.utils import get_device, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="data/cohort.npz")
ap.add_argument("--minutes", type=float, default=8)
ap.add_argument("--skip-small", action="store_true")
a = ap.parse_args()
dev = get_device()
cohort, S = load(a.seed, a.data)
tr, va, te = S["train"], S["val"], S["test_eval"]
cfg = {"seed": a.seed, "device": str(dev), "minutes_budget": a.minutes}


def latency_ms(model, w, n=256):
    b = {k: v[:n] for k, v in w.items()}
    with torch.no_grad():
        model.eval()
        model(b)
        t = time.time()
        model(b)
    return 1000 * (time.time() - t) / n


for kind in ["transformer", "ssm"]:
    gen = set_seed(a.seed)
    m = build(kind, d=64)
    print(f"fitting {kind} ({n_params(m)} params)")
    t = time.time()
    m, h = fit(m, tr, va, epochs=30, lr=1e-3 if kind == "transformer" else 2e-3, bs=256, gen=gen, device=dev,
               max_minutes=a.minutes)
    save_ckpt(m, kind)
    evaluate_and_log("phase3", kind, predict(m, te, device=dev), S,
                     {**cfg, "n_params": n_params(m), "train_sec": time.time() - t, "hist": h},
                     {"latency_ms_per_window": latency_ms(m, te)})

if not a.skip_small:
    small = [next(n for n in S["split"]["train"] if n.startswith(g)) for g in ["adolescent", "adult", "child"]]
    tr_small = make_windows(cohort, small, stride=2)
    print("small training set:", small, len(tr_small["y"]), "windows")
    gen = set_seed(a.seed)
    ode = MinimalModel()
    ode, _ = fit(ode, subset(tr_small, torch.arange(len(tr_small["y"])) % 3 == 0), va, epochs=25, lr=0.03,
                 bs=512, gen=gen, max_minutes=a.minutes / 2, patience=4, verbose=False)
    hyb = MinimalModel(CorrectionNet(32))
    hyb.log_theta.data.copy_(ode.log_theta.data)
    hyb, _ = fit(hyb, subset(tr_small, torch.arange(len(tr_small["y"])) % 3 == 0), va, epochs=25, lr=3e-3,
                 bs=256, gen=gen, max_minutes=a.minutes / 2, patience=4, verbose=False)
    evaluate_and_log("phase3", "hybrid_3patients", predict(hyb, te), S, {**cfg, "train_patients": small})
    gen = set_seed(a.seed)
    trf = build("transformer", d=64)
    trf, _ = fit(trf, tr_small, va, epochs=30, lr=1e-3, bs=256, gen=gen, max_minutes=a.minutes, verbose=False)
    evaluate_and_log("phase3", "transformer_3patients", predict(trf, te), S, {**cfg, "train_patients": small})
