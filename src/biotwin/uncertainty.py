"""Patient-specific adaptation with uncertainty.

1. Laplace-approximated Bayesian personalisation of a few *physiological* parameters of the
   (hybrid) twin:  p(phi | D_patient) ∝ N(D | twin(phi), sigma^2) N(phi | phi_pop, s^2)
   MAP by gradient descent, posterior covariance = inverse Hessian of the negative log posterior.
2. Predictive distribution = parameter uncertainty (sample phi) + observation noise (per-horizon
   sigma estimated on a held-out calibration day, not on the fit days).
3. Split-conformal intervals (distribution-free) as a sanity reference.

Caveat (say it in interviews): overlapping forecast windows are not independent observations.
We fit on windows with stride = H (non-overlapping targets) to reduce double counting, but the
likelihood is still an approximation, so posterior widths should be checked via coverage.
"""
from __future__ import annotations

import math

import torch

from biotwin.models.mechanistic import PARAM_NAMES

ADAPT = ["SG", "SI", "cg", "Gb"]
ADAPT_IDX = [PARAM_NAMES.index(k) for k in ADAPT]
PRIOR_SD = torch.tensor([0.5, 0.5, 0.5, 0.15])  # log-space prior sd (Gb tighter: ~15%)


def horizon_sigma(pred, y):
    return (pred - y).std(0).clamp_min(1.0)  # [H]


def _full_theta(pop_log_theta, phi):
    lt = pop_log_theta.detach().clone()
    lt[ADAPT_IDX] = phi
    return lt


def laplace_personalise(model, w_fit, sigma, iters=40, fd_eps=1e-3):
    """Returns (phi_map [4], cov [4,4]). Model weights are frozen; only phi moves.

    MAP: L-BFGS on the negative log posterior.
    Covariance: generalised Gauss-Newton Hessian  J^T Sigma^-1 J + prior precision,
    with J = d pred / d phi from central finite differences (4 params -> 8 forward passes).
    GGN is PSD by construction, unlike the exact Hessian away from the optimum.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    pop = model.log_theta.detach().clone()
    mu = pop[ADAPT_IDX]

    def nlp(phi):
        pred = model(w_fit, log_theta=_full_theta(pop, phi))
        nll = 0.5 * (((pred - w_fit["y"]) / sigma) ** 2).sum()
        prior = 0.5 * (((phi - mu) / PRIOR_SD) ** 2).sum()
        return nll + prior

    phi = mu.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([phi], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nlp(phi)
        loss.backward()
        return loss

    opt.step(closure)
    phi_map = phi.detach()
    with torch.no_grad():
        cols = []
        for i in range(len(phi_map)):
            e = torch.zeros_like(phi_map)
            e[i] = fd_eps
            fp = model(w_fit, log_theta=_full_theta(pop, phi_map + e))
            fm = model(w_fit, log_theta=_full_theta(pop, phi_map - e))
            cols.append(((fp - fm) / (2 * fd_eps) / sigma).reshape(-1))
        J = torch.stack(cols, 1)  # [N*H, 4], whitened by sigma
        Hs = J.T @ J + torch.diag(1.0 / PRIOR_SD**2)
    return phi_map, torch.linalg.inv(Hs)


@torch.no_grad()
def predictive_samples(model, w, phi_map, cov, sigma_obs, n=50, gen=None, param_unc=True):
    pop = model.log_theta.detach()
    L = torch.linalg.cholesky(cov + 1e-9 * torch.eye(len(phi_map)))
    out = []
    for _ in range(n):
        phi = phi_map + (L @ torch.randn(len(phi_map), generator=gen) if param_unc else 0)
        mean = model(w, log_theta=_full_theta(pop, phi))
        out.append(mean + sigma_obs * torch.randn(mean.shape, generator=gen))
    return torch.stack(out)  # [S, N, H]


def conformal_halfwidth(pred_cal, y_cal, level):
    """Per-horizon split-conformal quantile of |residual|."""
    r = (pred_cal - y_cal).abs()
    n = len(r)
    q = min(1.0, math.ceil((n + 1) * level) / n)
    return torch.quantile(r, q, dim=0)  # [H]
