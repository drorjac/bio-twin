"""Seeds, device selection, minimal experiment tracking (JSON files, no server)."""
from __future__ import annotations

import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device(pref: str = "auto") -> torch.device:
    """auto: cuda > mps > cpu. Override with BIOTWIN_DEVICE=cpu|mps|cuda."""
    pref = os.environ.get("BIOTWIN_DEVICE", pref)
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def env_info() -> dict:
    return {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "threads": torch.get_num_threads(),
    }


def log_run(phase: str, name: str, config: dict, metrics: dict, root: str = "results") -> Path:
    d = Path(root) / phase
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    rec = {"name": name, "phase": phase, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
           "config": config, "metrics": metrics, "env": env_info()}
    path.write_text(json.dumps(rec, indent=2, default=float))
    return path
