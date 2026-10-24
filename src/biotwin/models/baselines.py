"""Baselines: persistence and ARX (linear system identification, the SP-native baseline)."""
from __future__ import annotations

import torch

from biotwin.data.dataset import LH


class Persistence:
    """y_hat(t+k) = y(t). Surprisingly hard to beat at 5-15 min."""

    def predict(self, w):
        H = w["y"].shape[1]
        return w["cgm_hist"][:, -1:].expand(-1, H)


class ARX:
    """Direct multi-horizon ARX with exogenous future inputs, fit by ridge regression.

    y(t+k) - y(t) = a_k^T [y(t-j) - y(t)]_{j=1..na} + b_k^T u_dev(past+future) + c_k^T carbs(past+future) + d_k
    One closed-form least-squares problem, all horizons at once (multi-output).
    """

    def __init__(self, na: int = 12, n_in: int = 48, lam: float = 1.0):
        self.na, self.n_in, self.lam = na, n_in, lam
        self.W = None

    def features(self, w):
        g = w["cgm_hist"]
        last = g[:, -1:]
        ar = g[:, -self.na - 1 : -1] - last
        lo = LH - self.n_in
        ins = w["ins"][:, lo:] - w["basal"][:, None]
        cho = w["carbs"][:, lo:]
        one = torch.ones(len(g), 1)
        return torch.cat([ar / 50, ins, cho / 10, one], 1).double()

    def fit(self, w):
        X = self.features(w)
        Y = (w["y"] - w["cgm_hist"][:, -1:]).double()
        A = X.T @ X + self.lam * torch.eye(X.shape[1], dtype=X.dtype)
        self.W = torch.linalg.solve(A, X.T @ Y)
        return self

    def predict(self, w):
        return (self.features(w) @ self.W).float() + w["cgm_hist"][:, -1:]
