"""Generic sequence forecasters (no physics): GRU, Transformer, and a Mamba-style selective SSM.

All three share one token format so the comparison is about the backbone only.
Token at position t in [0, LH+H):
    [cgm_norm (0 if future), observed_flag, insulin_dev, carbs/10]
Future CGM is masked, future inputs are visible (same information as the mechanistic twin).
Output at future positions = (CGM(t) - CGM_last)/50.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from biotwin.data.dataset import LH

G_MU, G_SD = 140.0, 50.0


def tokens(w):
    B, L = w["ins"].shape
    H = L - LH
    g = (w["cgm_hist"] - G_MU) / G_SD
    g = torch.cat([g, torch.zeros(B, H, device=g.device)], 1)
    obs = torch.cat([torch.ones(B, LH, device=g.device), torch.zeros(B, H, device=g.device)], 1)
    u = w["ins"] - w["basal"][:, None]
    c = w["carbs"] / 10
    return torch.stack([g, obs, u, c], -1)


class SeqForecaster(nn.Module):
    def __init__(self, backbone: nn.Module, d_model: int):
        super().__init__()
        self.inp = nn.Linear(4, d_model)
        self.backbone = backbone
        self.head = nn.Linear(d_model, 1)

    def forward(self, w):
        x = self.inp(tokens(w))
        h = self.backbone(x)
        out = self.head(h[:, LH:]).squeeze(-1)
        return w["cgm_hist"][:, -1:] + G_SD * out


# ---------------------------------------------------------------- GRU
class GRUBackbone(nn.Module):
    def __init__(self, d, layers=2):
        super().__init__()
        self.rnn = nn.GRU(d, d, layers, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


# ---------------------------------------------------------------- Transformer
class TransformerBackbone(nn.Module):
    """Bidirectional encoder: safe because future CGM is masked in the tokens."""

    def __init__(self, d, layers=2, heads=4, max_len=128, dropout=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x):
        return self.enc(x + self.pos[:, : x.shape[1]])


# ---------------------------------------------------------------- Mamba-style SSM
class SelectiveSSM(nn.Module):
    """One Mamba-like block, pure PyTorch, sequential scan (fine for L=84 on CPU).

    Continuous diagonal SSM  h' = A h + B(x) x,  y = C(x) h + D x, with input-dependent
    step size Delta(x) ("selection"). Zero-order-hold discretisation:
        Abar = exp(Delta*A),  Bbar ~= Delta*B   (the simplification used in the Mamba paper)
    The official CUDA kernel (mamba-ssm) does a parallel scan; this is the readable reference.
    """

    def __init__(self, d, d_state=16, expand=2, d_conv=4):
        super().__init__()
        di = expand * d
        self.di, self.ds = di, d_state
        self.in_proj = nn.Linear(d, 2 * di)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.x_proj = nn.Linear(di, 2 * d_state + 1)  # B, C, Delta (rank-1 Delta for simplicity)
        self.dt_proj = nn.Linear(1, di)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(di, 1)  # S4D-real init
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(di))
        self.out_proj = nn.Linear(di, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        B_, L, _ = x.shape
        res = x
        x = self.norm(x)
        xz = self.in_proj(x)
        xi, z = xz.chunk(2, -1)
        xi = self.conv(xi.transpose(1, 2))[..., :L].transpose(1, 2)  # causal depthwise conv
        xi = F.silu(xi)
        bcd = self.x_proj(xi)
        Bm, Cm, dt = bcd[..., : self.ds], bcd[..., self.ds : 2 * self.ds], bcd[..., -1:]
        delta = F.softplus(self.dt_proj(dt))  # [B, L, di]
        A = -torch.exp(self.A_log)  # [di, ds]
        h = torch.zeros(B_, self.di, self.ds, device=x.device)
        ys = []
        for t in range(L):
            dA = torch.exp(delta[:, t, :, None] * A)
            dB = delta[:, t, :, None] * Bm[:, t, None, :]
            h = dA * h + dB * xi[:, t, :, None]
            ys.append((h * Cm[:, t, None, :]).sum(-1))
        y = torch.stack(ys, 1) + xi * self.D
        y = y * F.silu(z)
        return res + self.out_proj(y)


class SSMBackbone(nn.Module):
    def __init__(self, d, layers=2, d_state=16):
        super().__init__()
        self.blocks = nn.Sequential(*[SelectiveSSM(d, d_state) for _ in range(layers)])

    def forward(self, x):
        return self.blocks(x)


def build(kind: str, d: int = 64, layers: int = 2) -> SeqForecaster:
    bb = {"gru": GRUBackbone, "transformer": TransformerBackbone, "ssm": SSMBackbone}[kind](d, layers)
    return SeqForecaster(bb, d)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
