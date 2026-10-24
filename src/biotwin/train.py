"""One generic training loop with early stopping on validation RMSE (val = held-out patients)."""
from __future__ import annotations

import copy
import time

import torch

from biotwin.data.dataset import batches


def to(w, device):
    return {k: v.to(device) for k, v in w.items()}


@torch.no_grad()
def predict(model, w, bs=2048, device="cpu"):
    model.eval()
    outs = [model(to(b, device)).cpu() for b in batches(w, bs, shuffle=False)]
    return torch.cat(outs)


def fit(model, train_w, val_w, *, epochs=30, lr=3e-3, bs=256, wd=0.0, patience=5, gen=None,
        device="cpu", reg_fn=None, max_minutes=None, verbose=True):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad, hist = float("inf"), None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for b in batches(train_w, bs, shuffle=True, gen=gen):
            b = to(b, device)
            pred = model(b)
            loss = ((pred - b["y"]) / 100).pow(2).mean()
            if reg_fn is not None:
                loss = loss + reg_fn(model)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b["y"])
            n += len(b["y"])
        sched.step()
        vp = predict(model, val_w, device=device)
        vrmse = (vp - val_w["y"]).pow(2).mean().sqrt().item()
        hist.append({"epoch": ep, "train_loss": tot / n, "val_rmse": vrmse, "sec": time.time() - t0})
        if verbose:
            print(f"  ep {ep:02d} train {tot / n:.4f} val_rmse {vrmse:.2f} ({time.time() - t0:.0f}s)", flush=True)
        if vrmse < best - 1e-3:
            best, best_state, bad = vrmse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
        if max_minutes and (time.time() - t0) / 60 > max_minutes:
            print("  time budget reached")
            break
    model.load_state_dict(best_state)
    return model, hist
