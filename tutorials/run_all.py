"""Execute every tutorial notebook top to bottom (without saving outputs) and report failures.

With TUTOR_VERIFY=1 (set here), every exercise falls back to the reference solution and that solution
must pass its own check, so this doubles as a test of the tutorials.

    .venv/bin/python tutorials/run_all.py            # all notebooks
    .venv/bin/python tutorials/run_all.py 03 04      # just these
"""
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

os.environ["TUTOR_VERIFY"] = "1"
here = Path(__file__).parent
picks = sys.argv[1:]
failed = []
for path in sorted(here.glob("[0-9][0-9]_*.ipynb")):
    if picks and not any(path.name.startswith(p) for p in picks):
        continue
    nb = nbformat.read(path, as_version=4)
    t = time.time()
    try:
        NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(here)}}).execute()
        print(f"ok    {path.name}  ({time.time() - t:.0f}s)", flush=True)
    except Exception as e:  # noqa: BLE001
        failed.append(path.name)
        print(f"FAIL  {path.name}  ({time.time() - t:.0f}s)\n{str(e)[-3000:]}", flush=True)
sys.exit(1 if failed else 0)
