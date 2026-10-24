# Claims I can make vs claims I cannot make

## Can make (backed by files in `results/`)

- On 6 held-out virtual patients from the UVA/Padova simulator, a 7-state minimal model with
  8 population parameters forecasts CGM at 60 min about as well as a ridge ARX model (36.8 vs 36.3
  mg/dL RMSE).
- Adding a neural term to dG/dt (a UDE) reduces 60-min RMSE to 32.5, about the same as a GRU (32.1).
- Bayesian personalisation of 4 physiological parameters, using 5 days of data, reduces the hybrid's
  60-min RMSE from 32.5 to 27.0. Its 80 % intervals cover 80 % of held-out points.
- Fine-tuning a GRU on the same 5 days reaches 25.7. Black-box personalisation was at least as
  accurate here.
- Under a CPU budget of 6 min per model, a small Transformer and a pure-PyTorch Mamba-style SSM did
  not beat the GRU or the hybrid.
- The neural correction can flip the sign of the insulin effect for some patients and states.
  A metamorphic test detected this; disabling the NN term restores the correct sign.
- DDP with 2 processes reproduces single-process training (max loss diff 2.8e-6).
- The evaluation pipeline has automated leakage tests.

## Cannot make

- Anything about real patients, clinical accuracy, safety, or treatment decisions.
- "Hybrid models are more data-efficient": not shown (one seed; 3-patient results 38.2 vs 39.8).
- "Transformers/SSMs are worse for glucose forecasting": only shown for tiny, time-boxed models on CPU.
- Calibration beyond 60 min, or for interventions outside the training distribution.
- That the fitted parameters equal the simulator's true physiological parameters: different model
  structure, and identifiability is weak (SG/Gb).
- That the LLM agent is accurate: the Claude backend has not been evaluated yet. Only the regex
  baseline was scored (14/16).
- Any speedup from distributed training.
