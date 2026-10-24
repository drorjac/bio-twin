"""Bergman-style minimal model with insulin and meal absorption compartments, in PyTorch.

State-space view (continuous time, inputs u = insulin deviation from basal [U/min],
r = carbohydrate intake [g/min]):

    Isc1' = u - ka*Isc1                 subcutaneous insulin depot (2nd-order low-pass)
    Isc2' = ka*(Isc1 - Isc2)
    Ip'   = ka*Isc2 - ke*Ip             plasma insulin (deviation)
    X'    = p2*(SI*Ip - X)              remote insulin action  [1/min]
    D1'   = r - kd*D1                   gut (2nd-order low-pass)
    D2'   = kd*(D1 - D2)
    G'    = -SG*(G - Gb) - X*G + cg*kd*D2 (+ f_NN(.) in the hybrid)

Everything except the bilinear X*G term is an LTI filter of known inputs, so the hidden
states at forecast time are reconstructed by running the model over the 6-h input history
(G is pinned to the measured CGM during that warm-up). G0 = last CGM.

Parameters are stored as log-values (positivity) and can be overridden per sample
(`theta` [B, P]) for patient-specific twins.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from biotwin.data.dataset import LH

STEP_MIN = 5.0
PARAM_NAMES = ["ka", "ke", "kd", "SG", "p2", "SI", "cg", "Gb"]
# Literature-scale initial values (orders of magnitude, NOT fitted to the simulator):
INIT = {"ka": 1 / 55, "ke": 0.138, "kd": 1 / 40, "SG": 0.01, "p2": 0.02, "SI": 0.05, "cg": 5.0, "Gb": 140.0}


class MinimalModel(nn.Module):
    def __init__(self, correction: nn.Module | None = None, substeps: int = 2):
        super().__init__()
        self.log_theta = nn.Parameter(torch.tensor([math.log(INIT[k]) for k in PARAM_NAMES]))
        self.correction = correction
        self.substeps = substeps

    # ---- dynamics -------------------------------------------------------
    def f(self, s, u, r, th, ctx):
        ka, ke, kd, SG, p2, SI, cg, Gb = th.unbind(-1)
        Isc1, Isc2, Ip, X, D1, D2, G = s.unbind(-1)
        dG = -SG * (G - Gb) - X * G + cg * kd * D2
        if self.correction is not None:
            dG = dG + self.correction(s, ctx)
        return torch.stack(
            [u - ka * Isc1, ka * (Isc1 - Isc2), ka * Isc2 - ke * Ip, p2 * (SI * Ip - X),
             r - kd * D1, kd * (D1 - D2), dG], -1)

    def step(self, s, u, r, th, ctx):
        """RK4 over one 5-min sample with zero-order-hold inputs."""
        h = STEP_MIN / self.substeps
        for _ in range(self.substeps):
            k1 = self.f(s, u, r, th, ctx)
            k2 = self.f(s + 0.5 * h * k1, u, r, th, ctx)
            k3 = self.f(s + 0.5 * h * k2, u, r, th, ctx)
            k4 = self.f(s + h * k3, u, r, th, ctx)
            s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return s

    # ---- forecasting ----------------------------------------------------
    def theta(self, w, log_theta=None):
        lt = self.log_theta if log_theta is None else log_theta
        th = lt.exp()
        if th.dim() == 1:
            th = th.expand(len(w["y"]), -1)
        return th

    def forward(self, w, log_theta=None, return_states=False):
        th = self.theta(w, log_theta)
        B = len(w["y"])
        H = w["y"].shape[1]
        u_all = (w["ins"] - w["basal"][:, None]) / STEP_MIN  # U/min deviation
        r_all = w["carbs"] / STEP_MIN  # g/min
        g = w["cgm_hist"]
        ctx = (g[:, -1] - g[:, -4]) / 15.0  # recent slope (mg/dL/min): context for the NN only
        s = torch.zeros(B, 7, dtype=g.dtype, device=g.device)
        for t in range(LH):  # warm-up: reconstruct hidden states, pin G to measurements
            s = torch.cat([s[:, :6], g[:, t : t + 1]], 1)
            s = self.step(s, u_all[:, t], r_all[:, t], th, ctx)
        s = torch.cat([s[:, :6], g[:, -1:]], 1)
        out, states = [], []
        for k in range(H):
            s = self.step(s, u_all[:, LH + k], r_all[:, LH + k], th, ctx)
            s = torch.cat([s[:, :6], s[:, 6:].clamp(20, 600)], 1)  # physiological guard
            out.append(s[:, 6])
            states.append(s)
        pred = torch.stack(out, 1)
        return (pred, torch.stack(states, 1)) if return_states else pred


class CorrectionNet(nn.Module):
    """Neural term added to dG/dt (a 'universal differential equation').

    Inputs are normalised mechanistic states + recent CGM slope. Output layer starts at zero,
    so training starts from the pure mechanistic model.
    """

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale = torch.tensor([1.0, 1.0, 1.0, 100.0, 0.05, 0.05, 1 / 100, 1.0])

    def forward(self, s, ctx):
        feats = torch.cat([s, ctx[:, None]], 1)
        feats = feats * self.scale.to(s.device)
        feats = torch.cat([feats[:, :6], (feats[:, 6:7] - 1.4), feats[:, 7:]], 1)
        return self.net(feats).squeeze(-1)  # mg/dL/min
