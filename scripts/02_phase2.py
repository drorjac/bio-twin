"""Phase 2: patient-specific adaptation with uncertainty, and calibration checks.

Variants (all evaluated on the same test-patient evaluation days as Phase 1):
  hybrid_pop_gauss      population hybrid + per-horizon Gaussian noise (sigma from val patients)
  hybrid_ensemble       deep ensemble of hybrids (epistemic) + noise
  hybrid_laplace        per-patient MAP of 4 physiological params + Laplace posterior + noise
  hybrid_map_plugin     same MAP, no parameter uncertainty (ablation)
  hybrid_map_conformal  same MAP, split-conformal intervals from the calibration day
  gru_finetuned         black-box comparison: fine-tune the GRU on each patient's fit days
"""
import argparse
import copy
import json
from pathlib import Path

import torch

from biotwin.data.dataset import subset
from biotwin.evaluate import interval_metrics_from_samples, interval_metrics_gaussian, point_metrics
from biotwin.experiment import CKPT, evaluate_and_log, load, save_ckpt
from biotwin.models.mechanistic import CorrectionNet, MinimalModel
from biotwin.models.neural import build
from biotwin.train import fit, predict
from biotwin.uncertainty import (ADAPT, conformal_halfwidth, horizon_sigma, laplace_personalise,
                                 predictive_samples)
from biotwin.utils import get_device, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="data/cohort.npz")
ap.add_argument("--members", type=int, default=4)
ap.add_argument("--minutes", type=float, default=4)
ap.add_argument("--max-patients", type=int, default=None, help="debug: personalise only the first k test patients")
a = ap.parse_args()
dev = get_device()
gen = set_seed(a.seed)
cohort, S = load(a.seed, a.data)
if a.max_patients:
    keep = S["test_eval"]["pid"].unique()[: a.max_patients]
    S["test_eval"] = subset(S["test_eval"], torch.isin(S["test_eval"]["pid"], keep))
tr, va, te = S["train"], S["val"], S["test_eval"]
cfg = {"seed": a.seed}


def hybrid():
    return MinimalModel(CorrectionNet(32))


hyb = hybrid()
hyb.load_state_dict(torch.load(CKPT / "hybrid.pt"))

# ---- A: population + noise
sig_pop = horizon_sigma(predict(hyb, va), va["y"])
mu = predict(hyb, te)
evaluate_and_log("phase2", "hybrid_pop_gauss", mu, S, cfg,
                 interval_metrics_gaussian(mu, sig_pop.expand_as(mu), te["y"]))

# ---- B: deep ensemble (member 0 = phase-1 hybrid; others differ in seed / minibatch order)
members = [hyb]
ode_tr = subset(tr, torch.arange(len(tr["y"])) % 3 == 0)
for m in range(1, a.members):
    path = CKPT / f"hybrid_ens{m}.pt"
    net = hybrid()
    if path.exists():
        net.load_state_dict(torch.load(path))
    else:
        g = set_seed(a.seed + 100 * m)
        net.load_state_dict(torch.load(CKPT / "ode.pt"), strict=False)  # same physics init
        net.correction = CorrectionNet(32)  # fresh random NN init with this seed
        print(f"training ensemble member {m}")
        net, _ = fit(net, ode_tr, va, epochs=25, lr=3e-3, bs=256, gen=g, max_minutes=a.minutes, patience=4,
                     verbose=False)
        save_ckpt(net, f"hybrid_ens{m}")
    members.append(net)
P_va = torch.stack([predict(n, va) for n in members])
P_te = torch.stack([predict(n, te) for n in members])
sig_ens = horizon_sigma(P_va.mean(0), va["y"])
var_te = P_te.var(0) + sig_ens**2
evaluate_and_log("phase2", "hybrid_ensemble", P_te.mean(0), S, {**cfg, "members": a.members},
                 {**interval_metrics_gaussian(P_te.mean(0), var_te.sqrt(), te["y"]),
                  "mean_epistemic_sd_60min": P_te.std(0)[:, -1].mean().item()})

# ---- C/D/E: per-patient personalisation
fit_w, cal_w = S["test_fit"], S["test_cal"]
out = {k: [] for k in ["map", "lap_samples", "plug_samples", "conf_lo", "conf_hi", "gru_ft", "y", "idx"]}
gru = build("gru", d=64)
gru.load_state_dict(torch.load(CKPT / "gru.pt"))
param_report = {}
for p in te["pid"].unique().tolist():
    name = str(cohort.names[p])
    f_mask = (fit_w["pid"] == p) & (fit_w["t0"] % 12 == 0)  # non-overlapping targets
    fw, cw = subset(fit_w, f_mask), subset(cal_w, cal_w["pid"] == p)
    ev_mask = te["pid"] == p
    ew = subset(te, ev_mask)
    phi, cov = laplace_personalise(hyb, fw, sig_pop)
    sd = cov.diag().sqrt()
    param_report[name] = {k: {"pop": float(hyb.log_theta[i].exp()), "map": float(phi[j].exp()),
                              "post_sd_log": float(sd[j])}
                          for j, (k, i) in enumerate(zip(ADAPT, [3, 5, 6, 7]))}
    # observation noise for this patient from the calibration day (MAP predictions)
    zero = torch.zeros_like(cov)
    cal_mean = predictive_samples(hyb, cw, phi, zero, 0.0, n=1, gen=gen)[0]
    sig_p = horizon_sigma(cal_mean, cw["y"])
    lap = predictive_samples(hyb, ew, phi, cov, sig_p, n=50, gen=gen)
    plug = predictive_samples(hyb, ew, phi, zero, sig_p, n=50, gen=gen, param_unc=False)
    mean_map = predictive_samples(hyb, ew, phi, zero, 0.0, n=1, gen=gen)[0]
    hw = conformal_halfwidth(cal_mean, cw["y"], 0.8)
    torch.save({"phi": phi, "cov": cov, "sigma_obs": sig_p}, CKPT / f"personal_{name.replace('#', '_')}.pt")
    # black-box personalisation: fine-tune all GRU weights on the same fit days
    g_ft = copy.deepcopy(gru)
    g_ft, _ = fit(g_ft, subset(fit_w, fit_w["pid"] == p), cw, epochs=10, lr=3e-4, bs=128, gen=gen,
                  patience=3, verbose=False)
    out["gru_ft"].append(predict(g_ft, ew))
    out["map"].append(mean_map)
    out["lap_samples"].append(lap)
    out["plug_samples"].append(plug)
    out["conf_lo"].append(mean_map - hw)
    out["conf_hi"].append(mean_map + hw)
    out["idx"].append(torch.nonzero(ev_mask).squeeze(1))
    print(name, {k: (round(v["pop"], 4), round(v["map"], 4), round(v["post_sd_log"], 3))
                 for k, v in param_report[name].items()}, flush=True)

order = torch.cat(out["idx"])
inv = torch.empty_like(order)
inv[order] = torch.arange(len(order))


def cat(k, dim=0):
    return torch.cat(out[k], dim).index_select(dim, inv)


y = te["y"]
mmap = cat("map")
evaluate_and_log("phase2", "hybrid_laplace", mmap, S, {**cfg, "adapted": ADAPT},
                 interval_metrics_from_samples(cat("lap_samples", 1), y))
evaluate_and_log("phase2", "hybrid_map_plugin", mmap, S, cfg,
                 interval_metrics_from_samples(cat("plug_samples", 1), y))
lo, hi = cat("conf_lo"), cat("conf_hi")
inside = (y >= lo) & (y <= hi)
evaluate_and_log("phase2", "hybrid_map_conformal80", mmap, S, cfg,
                 {"cov80_all": inside.float().mean().item(), "cov80_60min": inside[:, -1].float().mean().item(),
                  "width80_60min": (hi - lo)[:, -1].mean().item()})
evaluate_and_log("phase2", "gru_finetuned", cat("gru_ft"), S, cfg)
Path("results/phase2/personalised_params.json").write_text(json.dumps(param_report, indent=2))
# per-patient RMSE (population vs personalised) for the README
gru_pop = predict(gru, te)
per = {}
for p in te["pid"].unique().tolist():
    m = te["pid"] == p
    per[str(cohort.names[p])] = {
        "hybrid_pop": point_metrics(mu[m], y[m])["rmse_60min"],
        "hybrid_laplace_map": point_metrics(mmap[m], y[m])["rmse_60min"],
        "gru_pop": point_metrics(gru_pop[m], y[m])["rmse_60min"], "gru_finetuned": point_metrics(cat("gru_ft")[m], y[m])["rmse_60min"]}
Path("results/phase2/per_patient_rmse60.json").write_text(json.dumps(per, indent=2))
