# Tutorials: building a glucose digital twin, from scratch to advanced

Eight workbooks that rebuild the project piece by piece. In each one you write the core
functions yourself. The surrounding code, plots and experiments are already there, so you spend
your time on the ideas and not on plumbing.

| # | Notebook | You build | Time |
|---|---|---|---|
| 00 | [Glucose 101 and the data](00_glucose_and_data.ipynb) | time in range, noise floor, event-triggered averages | 45 min |
| 01 | [Framing the task, and how not to fool yourself](01_task_and_leakage.ipynb) | forecasting windows, patient split, a 1-NN "memoriser" that exposes leakage | 45 min |
| 02 | [Baselines: persistence and ARX](02_baselines_arx.ipynb) | persistence, ARX features, ridge regression; reading learned impulse responses | 45 min |
| 03 | [A mechanistic model from scratch](03_mechanistic_ode.ipynb) | RK4, the minimal-model ODE, forecasting with hidden-state reconstruction; identifiability | 1.5 h |
| 04 | [The hybrid (UDE)](04_hybrid_ude.ipynb) | a zero-initialised correction network; probing what it learned with autograd | 1 h |
| 05 | [Sequence models](05_sequence_models.ipynb) | tokens, attention, a Mamba-style selective scan (and why it's a convolution when frozen) | 1.5 h |
| 06 | [Personalisation with uncertainty](06_personalisation_uncertainty.ipynb) | MAP objective, finite-difference Jacobian, Gauss–Newton Laplace, coverage, split conformal | 1.5 h |
| 07 | [Testing, the agent, and the capstone](07_testing_agent_capstone.ipynb) | metamorphic tests, an agent loop, a grounding check; then an open problem | 1 h + |

Do them in order. Each one ends with a 5-question quiz, with the answers folded.

## How the exercises work

```python
def rk4_step(f, x, h):
    # TODO
    raise NotImplementedError

rk4_step = use(rk4_step, sol.rk4_step, check)
```

`use` runs `check` against an independent reference: the project's own implementation, autograd,
or a closed-form answer.
- `[ok]`: your version is used for the rest of the notebook.
- `[todo]` or `[fail]`: the notebook falls back to the reference so later cells still run.
  Come back to it later.

Hints are folded under each exercise. Full solutions are in [`solutions.py`](solutions.py).
Look only after you've tried.

## Setup

From the repo root:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,notebook]"
```

Then open the notebooks in Jupyter or PyCharm with the `.venv` interpreter. Everything runs on
a laptop CPU. The longest cells train for about a minute.

`python tutorials/run_all.py` executes every notebook using the reference solutions. It's how
the tutorials themselves are tested.

## The capstone

Notebook 07 ends with a problem that isn't solved yet. The hybrid model is the most accurate
one, but in 11 of 72 test situations a 4 U insulin bolus fails to lower its predicted glucose.
The physics-only model never does that. The challenge is to build a twin that gets both right,
using `twin_report` to score it.
