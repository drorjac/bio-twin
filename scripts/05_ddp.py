"""Phase 5 (optional): DistributedDataParallel + profiler, CPU-only (gloo backend).

What this shows: the DDP mechanics (process group, rank-sharded data, gradient all-reduce, rank-0
logging/checkpointing) and that 2-process DDP with per-rank batch B/2 matches single-process
training with batch B (same global batch -> same gradient up to float reduction order).
What it does NOT show: real multi-GPU speedups. On one CPU, two processes share the same cores, so
expect no speedup. On a CUDA box: backend="nccl", one process per GPU, same code otherwise.

python scripts/05_ddp.py --world 2 --steps 60
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from biotwin.experiment import load
from biotwin.models.neural import build

GLOBAL_BS = 256


def batch_indices(n, step, seed):
    g = torch.Generator().manual_seed(seed + step)
    return torch.randint(0, n, (GLOBAL_BS,), generator=g)


def worker(rank, world, steps, data, out, backend):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29531")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group(backend, rank=rank, world_size=world)
    torch.set_num_threads(max(1, torch.get_num_threads() // world))
    torch.manual_seed(0)  # identical init on all ranks (DDP also broadcasts rank-0 weights)
    _, S = load(0, data)
    tr = S["train"]
    model = DDP(build("transformer", d=64))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses, t0 = [], time.time()
    for step in range(steps):
        idx = batch_indices(len(tr["y"]), step, 0)
        idx = idx[rank::world]  # shard the global batch across ranks
        b = {k: v[idx] for k, v in tr.items()}
        model.eval()  # dropout off: otherwise ranks draw different masks and the runs are not comparable
        loss = ((model(b) - b["y"]) / 100).pow(2).mean()
        opt.zero_grad()
        loss.backward()  # DDP all-reduces (averages) gradients here
        opt.step()
        lt = loss.detach().clone()
        dist.all_reduce(lt)
        losses.append(lt.item() / world)
    if rank == 0:
        Path(out).write_text(json.dumps({"world": world, "losses": losses, "sec": time.time() - t0}))
    dist.destroy_process_group()


def profile(data, steps=5):
    from torch.profiler import ProfilerActivity, profile as prof, record_function

    _, S = load(0, data)
    tr = S["train"]
    rows = {}
    for kind in ["transformer", "ssm", "gru"]:
        torch.manual_seed(0)
        m = build(kind, d=64)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        b = {k: v[:GLOBAL_BS] for k, v in tr.items()}
        with prof(activities=[ProfilerActivity.CPU], record_shapes=False) as p:
            for _ in range(steps):
                with record_function("forward"):
                    loss = ((m(b) - b["y"]) / 100).pow(2).mean()
                with record_function("backward"):
                    opt.zero_grad()
                    loss.backward()
                with record_function("optimizer"):
                    opt.step()
        tab = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=12)
        Path(f"results/phase5/profile_{kind}.txt").write_text(tab)
        ev = {e.key: e.cpu_time_total / 1000 / steps for e in p.key_averages()
              if e.key in ("forward", "backward", "optimizer")}
        rows[kind] = {k: round(v, 1) for k, v in ev.items()}
        print(kind, "ms/step:", rows[kind])
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--data", default="data/cohort.npz")
    a = ap.parse_args()
    Path("results/phase5").mkdir(parents=True, exist_ok=True)
    res = {}
    for w in [1, a.world]:
        f = f"results/phase5/ddp_world{w}.json"
        mp.spawn(worker, args=(w, a.steps, a.data, f, "gloo"), nprocs=w, join=True)
        res[w] = json.loads(Path(f).read_text())
        print(f"world={w}: final loss {res[w]['losses'][-1]:.5f}, {res[w]['sec']:.1f}s")
    l1, l2 = torch.tensor(res[1]["losses"]), torch.tensor(res[a.world]["losses"])
    summary = {"max_abs_loss_diff": (l1 - l2).abs().max().item(), "sec_world1": res[1]["sec"],
               f"sec_world{a.world}": res[a.world]["sec"], "profile_ms_per_step": profile(a.data)}
    print(json.dumps(summary, indent=2))
    Path("results/phase5/summary.json").write_text(json.dumps(summary, indent=2))
