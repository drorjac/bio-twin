"""Reference solutions for the tutorial exercises. Try the exercise before reading these."""
from __future__ import annotations

import re

import numpy as np
import torch

# ---------------------------------------------------------------- 00 data


def time_in_range(cgm, lo=70.0, hi=180.0):
    cgm = np.asarray(cgm)
    return float(((cgm >= lo) & (cgm <= hi)).mean())


def noise_floor(cgm, bg):
    return float(np.sqrt(np.mean((np.asarray(cgm) - np.asarray(bg)) ** 2)))


def event_triggered_average(x, events, pre, post):
    x = np.asarray(x)
    segs = [x[e - pre:e + post] for e in events if e - pre >= 0 and e + post <= len(x)]
    return np.mean(segs, axis=0)


# ---------------------------------------------------------------- 01 windows


def windows_for_patient(cgm, ins, carbs, t_lo, t_hi, LH, H):
    out = {k: [] for k in ["cgm_hist", "ins", "carbs", "y", "t0"]}
    for t0 in range(max(t_lo, LH), t_hi - H + 1):
        out["cgm_hist"].append(cgm[t0 - LH:t0])
        out["ins"].append(ins[t0 - LH:t0 + H])
        out["carbs"].append(carbs[t0 - LH:t0 + H])
        out["y"].append(cgm[t0:t0 + H])
        out["t0"].append(t0)
    return {k: torch.as_tensor(np.array(v)) for k, v in out.items()}


def stratified_subject_split(names, n_val, n_test, seed):
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for g in sorted({n.split("#")[0] for n in names}):
        members = list(rng.permutation(sorted(n for n in names if n.startswith(g))))
        out["test"] += members[:n_test]
        out["val"] += members[n_test:n_test + n_val]
        out["train"] += members[n_test + n_val:]
    return out


def nearest_neighbour_forecast(train_x, train_y, test_x):
    idx = torch.cdist(test_x, train_x).argmin(1)
    return train_y[idx]


# ---------------------------------------------------------------- 02 baselines


def persistence(cgm_hist, H):
    return cgm_hist[:, -1:].expand(-1, H)


def arx_features(w, na, n_in, LH):
    g = w["cgm_hist"]
    last = g[:, -1:]
    ar = g[:, -na - 1:-1] - last
    ins = w["ins"][:, LH - n_in:] - w["basal"][:, None]
    cho = w["carbs"][:, LH - n_in:]
    one = torch.ones(len(g), 1)
    return torch.cat([ar / 50, ins, cho / 10, one], 1).double()


def ridge(X, Y, lam):
    A = X.T @ X + lam * torch.eye(X.shape[1], dtype=X.dtype)
    return torch.linalg.solve(A, X.T @ Y)


# ---------------------------------------------------------------- 03 ODE


def rk4_step(f, x, h):
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def minimal_model_rhs(s, u, r, th):
    ka, ke, kd, SG, p2, SI, cg, Gb = th.unbind(-1)
    Isc1, Isc2, Ip, X, D1, D2, G = s.unbind(-1)
    return torch.stack([
        u - ka * Isc1,
        ka * (Isc1 - Isc2),
        ka * Isc2 - ke * Ip,
        p2 * (SI * Ip - X),
        r - kd * D1,
        kd * (D1 - D2),
        -SG * (G - Gb) - X * G + cg * kd * D2,
    ], -1)


def forecast_ode(th, w, LH, substeps=2, step_min=5.0):
    """th [B, 8] (natural units). Returns predicted CGM [B, H]."""
    g = w["cgm_hist"]
    B, H = len(g), w["y"].shape[1]
    u_all = (w["ins"] - w["basal"][:, None]) / step_min
    r_all = w["carbs"] / step_min
    h = step_min / substeps

    def advance(s, t):
        f = lambda x: minimal_model_rhs(x, u_all[:, t], r_all[:, t], th)  # noqa: E731
        for _ in range(substeps):
            s = rk4_step(f, s, h)
        return s

    s = torch.zeros(B, 7, dtype=g.dtype)
    for t in range(LH):
        s = torch.cat([s[:, :6], g[:, t:t + 1]], 1)
        s = advance(s, t)
    s = torch.cat([s[:, :6], g[:, -1:]], 1)
    out = []
    for k in range(H):
        s = advance(s, LH + k)
        s = torch.cat([s[:, :6], s[:, 6:].clamp(20, 600)], 1)
        out.append(s[:, 6])
    return torch.stack(out, 1)


# ---------------------------------------------------------------- 04 hybrid


class ZeroInitMLP(torch.nn.Module):
    def __init__(self, n_in, hidden):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, 1))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def grad_wrt_state(fn, s, ctx, i):
    s = s.detach().clone().requires_grad_(True)
    out = fn(s, ctx)
    (g,) = torch.autograd.grad(out.sum(), s)
    return g[:, i]


# ---------------------------------------------------------------- 05 sequence models


def tokens(w, LH, g_mu=140.0, g_sd=50.0):
    B, L = w["ins"].shape
    H = L - LH
    g = torch.cat([(w["cgm_hist"] - g_mu) / g_sd, torch.zeros(B, H)], 1)
    obs = torch.cat([torch.ones(B, LH), torch.zeros(B, H)], 1)
    u = w["ins"] - w["basal"][:, None]
    c = w["carbs"] / 10
    return torch.stack([g, obs, u, c], -1)


def attention(q, k, v):
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    return torch.softmax(scores, -1) @ v


def selective_scan(x, delta, A, B, C, D):
    """x, delta [b, L, d]; A [d, n]; B, C [b, L, n]; D [d]  ->  y [b, L, d]."""
    b, L, d = x.shape
    h = torch.zeros(b, d, A.shape[1], dtype=x.dtype)
    ys = []
    for t in range(L):
        dA = torch.exp(delta[:, t, :, None] * A)
        dB = delta[:, t, :, None] * B[:, t, None, :]
        h = dA * h + dB * x[:, t, :, None]
        ys.append((h * C[:, t, None, :]).sum(-1))
    return torch.stack(ys, 1) + x * D


# ---------------------------------------------------------------- 06 Bayes


def neg_log_posterior(pred, y, sigma, phi, mu, prior_sd):
    return 0.5 * (((pred - y) / sigma) ** 2).sum() + 0.5 * (((phi - mu) / prior_sd) ** 2).sum()


def jacobian_fd(fn, phi, eps=1e-4):
    cols = []
    for i in range(len(phi)):
        e = torch.zeros_like(phi)
        e[i] = eps
        cols.append((fn(phi + e) - fn(phi - e)).reshape(-1) / (2 * eps))
    return torch.stack(cols, 1)


def ggn_hessian(J, sigma, prior_sd):
    Jw = J / torch.as_tensor(sigma, dtype=J.dtype).reshape(-1, 1)
    return Jw.T @ Jw + torch.diag(1.0 / torch.as_tensor(prior_sd, dtype=J.dtype) ** 2)


def coverage(lo, hi, y):
    return ((y >= lo) & (y <= hi)).float().mean().item()


def conformal_halfwidth(abs_resid, level):
    n = len(abs_resid)
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return torch.quantile(abs_resid, q, dim=0)


# ---------------------------------------------------------------- 07 agent


def grounded_fraction(answer, tool_results):
    def collect(o, acc):
        if isinstance(o, dict):
            for v in o.values():
                collect(v, acc)
        elif isinstance(o, list):
            for v in o:
                collect(v, acc)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            acc.add(round(float(o), 1))
        return acc

    known = collect(tool_results, set())
    found = [round(float(x), 1) for x in re.findall(r"(?<![\d.])-?\d+\.\d+", answer)]
    if not found:
        return 1.0
    return sum(x in known for x in found) / len(found)


def run_agent(policy, tools, question, max_turns=5):
    messages = [{"role": "user", "content": question}]
    calls = []
    for _ in range(max_turns):
        act = policy(messages)
        if act["type"] == "final":
            return {"answer": act["text"], "calls": calls}
        try:
            result = tools[act["name"]](**act["args"])
        except Exception as e:  # noqa: BLE001 - tool errors go back to the model, they don't crash the loop
            result = {"error": f"{type(e).__name__}: {e}"}
        calls.append({"name": act["name"], "args": act["args"], "result": result})
        messages.append({"role": "assistant", "tool_call": act})
        messages.append({"role": "tool", "name": act["name"], "content": result})
    return {"answer": "[max turns reached]", "calls": calls}


def insulin_effect(svc, patient, day, hour, doses=(2, 4)):
    return [svc.simulate_scenario(patient, day, hour, bolus_U=d, horizon_min=120)["difference_median_mean_glucose"]
            for d in doses]
