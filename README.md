# bio_twin

A digital twin of glucose–insulin dynamics for type-1 diabetes, built on simulated patients.

The model is a small physiological ODE with a neural network inside it. It's personalised per
patient with Bayesian error bars, and a tool-using agent on top answers "what if" questions
("what happens if adult#003 eats 60 g at noon?").

> Research prototype on simulated data. Not a medical device and not medical advice.

**Learn it step by step:** [`tutorials/`](tutorials/) has 8 workbooks that rebuild the project from scratch, with exercises, hints and quizzes.

**See it end to end:** [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) covers
the whole project in four parts:
- **Problem**: the data and how it's generated.
- **Method**: the theory behind every model, training, uncertainty and the agent.
- **Experiments**: seven experiments, run live.
- **Results**: the table, plus what it does and doesn't show.

It takes about 4 minutes on a laptop CPU, using the committed data and checkpoints.

## What's in here

- **Data**: 30 virtual patients × 10 days from the UVA/Padova simulator
  ([simglucose](https://github.com/jxx123/simglucose), MIT license). The patients' meal and
  insulin habits are deliberately imperfect so that the effects of insulin and carbs can be
  told apart.
- **Models**:
  - persistence and ARX baselines
  - a 7-state Bergman-style minimal model in PyTorch (RK4)
  - a hybrid with an MLP added to dG/dt (a UDE)
  - GRU, Transformer and a Mamba-style SSM
- **Personalisation**: MAP + Laplace posterior over 4 physiological parameters per patient,
  plus split-conformal intervals.
- **Agent**: Claude tool-use loop over the twin, plus an offline regex baseline, with a
  16-question golden set and metamorphic tests.
- **Extras**: CPU DDP equivalence check and a profiler run (`scripts/05_ddp.py`).

## Results

60-minute-ahead RMSE on 6 held-out patients (mg/dL). The CGM sensor noise alone is about 11.5.

| Model | Params | RMSE 60 min |
|---|---|---|
| Persistence | 0 | 52.6 |
| ARX | ~1.6 k | 36.3 |
| Pure ODE | 8 | 36.8 |
| Hybrid (ODE + NN) | 8 + 1.4 k | 32.5 |
| GRU | 50 k | 32.1 |
| Transformer / SSM (6 min CPU budget) | 109 k / 65 k | 36.4 / 39.9 |
| **Hybrid, personalised** | 4 per patient | **27.0** |
| GRU, fine-tuned per patient | 50 k per patient | 25.7 |

What came out of it:

- **Personalisation was the biggest win.** The personalised intervals are calibrated: 80 %
  intervals cover 80 % of held-out points.
- **Fine-tuning a black box was slightly more accurate.** The hybrid earns its place through
  interpretable parameters and uncertainty, not raw RMSE.
- **The NN term can break causality.** In 11 of 72 test situations (5 of 6 patients), the
  population hybrid predicts that a 4 U bolus doesn't lower glucose. The pure ODE: 0 of 72. RMSE
  didn't catch this; metamorphic tests did. See section E5 of the notebook, and
  `tutorials/04` for why (the network learned to cancel the insulin term).

Full tables are in `results/results_table.md` and the raw run logs are in `results/phase*/`.
`docs/claims.md` lists what these results do and don't support.

## Running it

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,notebook]"
.venv/bin/python -m pytest -q              # leakage and model tests
```

Data and the trained checkpoints are committed, so the notebook works straight away. To
regenerate everything from scratch:

```bash
.venv/bin/python scripts/00_simulate.py            # ~15 min on 8 cores
.venv/bin/python scripts/01_phase1.py --minutes 6  # baselines, ODE, hybrid, GRU
.venv/bin/python scripts/02_phase2.py              # personalisation + uncertainty
.venv/bin/python scripts/03_phase3.py --minutes 6  # Transformer, SSM
.venv/bin/python scripts/04_agent_eval.py --backend rules   # or --backend claude (needs an API key)
.venv/bin/python scripts/05_ddp.py --world 2 --steps 40
.venv/bin/python scripts/06_report.py
```

Everything was run on a CPU (M2 Mac). The code picks CUDA, then MPS, then CPU; set
`BIOTWIN_DEVICE` to override.

## Limitations

- It's a simulator. These results are about learning UVA/Padova, not human physiology.
- One seed and one split, so differences of 1–2 mg/dL between models mean nothing.
- The neural models are small and were time-boxed on a CPU.
- Only the first 60 minutes of a forecast are validated. The agent allows up to 4 hours.
- The Claude agent backend hasn't been evaluated yet.

## Layout

```
src/biotwin/   data/ (simulator, windows, splits)  models/  uncertainty.py  agent/
scripts/       one script per phase, 00–06
notebooks/     tutorial.ipynb (end-to-end walkthrough)
tutorials/     8 hands-on workbooks, 00–07
tests/         leakage + model tests
docs/          biology primer, talk outline, claims
results/       metrics JSON, figures, checkpoints
```
