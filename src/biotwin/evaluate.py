"""Metrics. Point: RMSE/MAE per horizon. Probabilistic: coverage, width, Gaussian NLL, CRPS."""
from __future__ import annotations

import math

import torch

HORIZONS = {"30min": 5, "60min": 11}  # index into the H=12 axis (step k -> (k+1)*5 min)


def point_metrics(pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    err = pred - y
    out = {}
    for name, k in HORIZONS.items():
        out[f"rmse_{name}"] = err[:, k].pow(2).mean().sqrt().item()
        out[f"mae_{name}"] = err[:, k].abs().mean().item()
    out["rmse_all"] = err.pow(2).mean().sqrt().item()
    return out


def gaussian_crps(mu, sigma, y):
    z = (y - mu) / sigma
    pdf = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def interval_metrics_from_samples(samples: torch.Tensor, y: torch.Tensor, levels=(0.5, 0.8, 0.95)):
    """samples [S, N, H] -> empirical coverage of central intervals, at the 60-min horizon and overall."""
    out = {}
    for lv in levels:
        lo = torch.quantile(samples, (1 - lv) / 2, dim=0)
        hi = torch.quantile(samples, 1 - (1 - lv) / 2, dim=0)
        inside = (y >= lo) & (y <= hi)
        out[f"cov{int(lv * 100)}_all"] = inside.float().mean().item()
        out[f"cov{int(lv * 100)}_60min"] = inside[:, -1].float().mean().item()
        out[f"width{int(lv * 100)}_60min"] = (hi - lo)[:, -1].mean().item()
    return out


def interval_metrics_gaussian(mu, sigma, y, levels=(0.5, 0.8, 0.95)):
    out = {}
    for lv in levels:
        z = torch.special.ndtri(torch.tensor(0.5 + lv / 2)).item()
        inside = (y - mu).abs() <= z * sigma
        out[f"cov{int(lv * 100)}_all"] = inside.float().mean().item()
        out[f"cov{int(lv * 100)}_60min"] = inside[:, -1].float().mean().item()
        out[f"width{int(lv * 100)}_60min"] = (2 * z * sigma[:, -1]).mean().item()
    out["crps_60min"] = gaussian_crps(mu[:, -1], sigma[:, -1], y[:, -1]).mean().item()
    out["nll_60min"] = (
        0.5 * math.log(2 * math.pi) + sigma[:, -1].log() + 0.5 * ((y[:, -1] - mu[:, -1]) / sigma[:, -1]) ** 2
    ).mean().item()
    return out
