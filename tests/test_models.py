import torch

from biotwin.data.dataset import H, LH
from biotwin.models.baselines import ARX, Persistence
from biotwin.models.mechanistic import INIT, CorrectionNet, MinimalModel
from biotwin.models.neural import build


def _flat_window(B=4, g=None, basal=0.5):
    g = INIT["Gb"] if g is None else g
    return {"cgm_hist": torch.full((B, LH), g), "ins": torch.full((B, LH + H), basal),
            "carbs": torch.zeros(B, LH + H), "y": torch.zeros(B, H), "basal": torch.full((B,), basal)}


def test_persistence():
    w = _flat_window(g=123.0)
    assert torch.allclose(Persistence().predict(w), torch.full((4, H), 123.0))


def test_arx_recovers_linear_system():
    torch.manual_seed(0)
    T = 4000
    u = (torch.rand(T) < 0.05).float() * 3
    y = torch.zeros(T)
    for t in range(1, T):
        y[t] = 0.9 * y[t - 1] - 4.0 * u[t - 1] + 0.5 * torch.randn(())
    y = y + 140
    idx = torch.arange(LH, T - H)
    w = {"cgm_hist": torch.stack([y[i - LH:i] for i in idx]),
         "ins": torch.stack([u[i - LH:i + H] for i in idx]),
         "carbs": torch.zeros(len(idx), LH + H), "basal": torch.zeros(len(idx)),
         "y": torch.stack([y[i:i + H] for i in idx])}
    m = ARX(lam=1e-3).fit(w)
    rmse = (m.predict(w) - w["y"]).pow(2).mean().sqrt()
    assert rmse < 2.0  # noise-limited, an AR(1) with sd 0.5 accumulates to ~1.2 over 12 steps


def test_mechanistic_steady_state_and_signs():
    m = MinimalModel()
    with torch.no_grad():
        flat = m(_flat_window())
        assert torch.allclose(flat, torch.full_like(flat, INIT["Gb"]), atol=1e-3)
        w = _flat_window()
        w["ins"][:, LH] += 5.0  # 5 U bolus at forecast start
        assert (m(w)[:, -1] < INIT["Gb"] - 1).all()
        w = _flat_window()
        w["carbs"][:, LH] += 25.0
        assert (m(w)[:, -1] > INIT["Gb"] + 1).all()


def test_hybrid_starts_as_mechanistic():
    w = _flat_window(g=180.0)
    with torch.no_grad():
        assert torch.allclose(MinimalModel()(w), MinimalModel(CorrectionNet())(w))


def test_neural_models_ignore_targets():
    w = _flat_window(g=150.0)
    for kind in ["gru", "transformer", "ssm"]:
        m = build(kind, d=16).eval()
        with torch.no_grad():
            a = m(w)
            w2 = dict(w, y=torch.randn(4, H) * 1000)
            assert torch.equal(a, m(w2)), kind
            assert a.shape == (4, H)
