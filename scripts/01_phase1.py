"""Phase 1: baselines, pure mechanistic ODE, hybrid (ODE + neural correction), pure NN.

All population models are trained on train patients, early-stopped on val patients,
and evaluated on the evaluation days of held-out test patients.
"""
import argparse

import torch

from biotwin.data.dataset import subset
from biotwin.experiment import evaluate_and_log, load, save_ckpt
from biotwin.models.baselines import ARX, Persistence
from biotwin.models.mechanistic import PARAM_NAMES, CorrectionNet, MinimalModel
from biotwin.models.neural import build, n_params
from biotwin.train import fit, predict
from biotwin.utils import get_device, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="data/cohort.npz")
ap.add_argument("--minutes", type=float, default=8, help="time budget per trained model")
a = ap.parse_args()
dev = get_device()
print("device:", dev)
gen = set_seed(a.seed)
cohort, S = load(a.seed, a.data)
tr, va, te = S["train"], S["val"], S["test_eval"]
print({k: len(v["y"]) for k, v in S.items() if k != "split"}, S["split"]["test"])
noise = (te["y"] - te["bg_y"]).pow(2).mean().sqrt().item()
print(f"CGM sensor noise floor (RMSE CGM vs true BG): {noise:.2f} mg/dL")
cfg = {"seed": a.seed, "split": S["split"]}

evaluate_and_log("phase1", "persistence", Persistence().predict(te), S, cfg, {"sensor_noise_rmse": noise})

best = None
for lam in [0.1, 1, 10, 100, 1000]:
    m = ARX(lam=lam).fit(tr)
    r = (m.predict(va) - va["y"]).pow(2).mean().sqrt().item()
    if best is None or r < best[0]:
        best = (r, lam, m)
evaluate_and_log("phase1", "arx", best[2].predict(te), S, {**cfg, "lam": best[1]})

# ---- pure mechanistic: only 8 population parameters are learned
ode_tr = subset(tr, torch.arange(len(tr["y"])) % 3 == 0)
ode = MinimalModel()
print("fitting pure ODE (8 params) ...")
ode, h = fit(ode, ode_tr, va, epochs=25, lr=0.03, bs=512, gen=gen, device=dev, max_minutes=a.minutes, patience=4)
theta = dict(zip(PARAM_NAMES, ode.log_theta.exp().tolist()))
print("fitted population params:", {k: round(v, 4) for k, v in theta.items()})
save_ckpt(ode, "ode")
evaluate_and_log("phase1", "ode", predict(ode, te, device=dev), S, {**cfg, "theta": theta, "hist": h})

# ---- hybrid: ODE initialised from the fit above + neural term in dG/dt
hyb = MinimalModel(CorrectionNet(32))
hyb.log_theta.data.copy_(ode.log_theta.data)
print("fitting hybrid (UDE) ...")
hyb, h = fit(hyb, ode_tr, va, epochs=25, lr=3e-3, bs=256, gen=gen, device=dev, max_minutes=a.minutes, patience=4)
save_ckpt(hyb, "hybrid")
theta_h = dict(zip(PARAM_NAMES, hyb.log_theta.exp().tolist()))
evaluate_and_log("phase1", "hybrid", predict(hyb, te, device=dev), S,
                 {**cfg, "theta": theta_h, "n_params": n_params(hyb), "hist": h})

# ---- pure NN (GRU on the same inputs)
gru = build("gru", d=64)
print(f"fitting GRU ({n_params(gru)} params) ...")
gru, h = fit(gru, tr, va, epochs=30, lr=2e-3, bs=256, gen=gen, device=dev, max_minutes=a.minutes)
save_ckpt(gru, "gru")
evaluate_and_log("phase1", "gru", predict(gru, te, device=dev), S, {**cfg, "n_params": n_params(gru), "hist": h})
