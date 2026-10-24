"""Collect results/*.json into a markdown table and draw the figures used in the README."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from biotwin.experiment import load, results_table  # noqa: E402

FIG = Path("results/figures")
FIG.mkdir(parents=True, exist_ok=True)

rows = results_table()
lines = ["| phase | model | RMSE 30 min | RMSE 60 min | MAE 60 min | 80% cov (60 min) | 95% cov (60 min) | 80% width (60 min) |",
         "|---|---|---|---|---|---|---|---|"]
for ph, name, m in rows:
    f = lambda k: f"{m[k]:.2f}" if k in m else "–"  # noqa: E731
    lines.append(f"| {ph} | {name} | {f('rmse_30min')} | {f('rmse_60min')} | {f('mae_60min')} | "
                 f"{f('cov80_60min')} | {f('cov95_60min')} | {f('width80_60min')} |")
table = "\n".join(lines)
Path("results/results_table.md").write_text(table + "\n")
print(table)

# ---- example forecasts on one test patient
cohort, S = load(0)
te = S["test_eval"]
P = Path("results/predictions")
names = [n for n in ["persistence", "arx", "ode", "hybrid", "gru", "transformer", "ssm", "hybrid_laplace"]
         if (P / f"{n}.pt").exists()]
preds = {n: torch.load(P / f"{n}.pt") for n in names}
pid = int(te["pid"][0])
m = te["pid"] == pid
idx = torch.nonzero(m).squeeze(1)
t0s = te["t0"][idx]
fig, ax = plt.subplots(figsize=(11, 4))
lo, hi = int(t0s.min()), int(t0s.min()) + 288
ax.plot(range(lo, hi), cohort.cgm[pid, lo:hi], color="0.6", lw=1, label="CGM (measured)")
ax.plot(range(lo, hi), cohort.bg[pid, lo:hi], color="k", lw=1, ls=":", label="true BG (simulator)")
for j, t in enumerate(range(lo + 24, hi - 12, 36)):
    k = idx[(t0s == t).nonzero().squeeze()]
    for n, c in zip(["arx", "hybrid", "transformer"], ["C0", "C3", "C2"]):
        if n in preds:
            ax.plot(range(t, t + 12), preds[n][k], color=c, lw=2, label=n if j == 0 else None)
ax.set(xlabel="time step (5 min)", ylabel="glucose (mg/dL)", title=f"60-min forecasts, {cohort.names[pid]} (held-out)")
ax.legend(fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig(FIG / "forecasts.png", dpi=130)

# ---- calibration: nominal vs empirical coverage (60 min)
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot([0, 1], [0, 1], "k--", lw=1)
for ph, name, met in rows:
    pts = [(lv / 100, met[f"cov{lv}_60min"]) for lv in (50, 80, 95) if f"cov{lv}_60min" in met]
    if len(pts) > 1:
        ax.plot(*zip(*pts), "o-", label=name)
ax.set(xlabel="nominal coverage", ylabel="empirical coverage (60 min)", xlim=(0.4, 1), ylim=(0.4, 1))
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "calibration.png", dpi=130)

# ---- personalisation per patient
f = Path("results/phase2/per_patient_rmse60.json")
if f.exists():
    per = json.loads(f.read_text())
    fig, ax = plt.subplots(figsize=(8, 3.5))
    keys = ["hybrid_pop", "hybrid_laplace_map", "gru_pop", "gru_finetuned"]
    for i, k in enumerate(keys):
        ax.bar([j + 0.2 * i for j in range(len(per))], [per[p][k] for p in per], 0.2, label=k)
    ax.set_xticks([j + 0.3 for j in range(len(per))], list(per), fontsize=8)
    ax.set(ylabel="RMSE 60 min (mg/dL)", title="Population vs personalised, per test patient")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "personalisation.png", dpi=130)
print("figures in", FIG)
