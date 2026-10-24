"""Phase 0: generate the synthetic cohort (30 virtual T1D patients)."""
import argparse
import time

from biotwin.data.simulate import simulate_cohort

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/cohort.npz")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t = time.time()
    simulate_cohort(a.days, a.seed, a.out, a.workers)
    print(f"wrote {a.out} in {time.time() - t:.0f}s")
