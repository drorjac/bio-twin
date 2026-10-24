# 10-minute talk: "A hybrid digital twin for glucose–insulin dynamics"

Numbers come from `results/results_table.md`. Quote them from there, never from memory.

**0:00–1:00 · Problem.** A digital twin = a mechanistic model + a patient's data + learning, used to
answer *counterfactual* questions ("what if they eat 60 g now?"). Type-1 diabetes is the cleanest
testbed: the input (insulin) is logged, the output (CGM) is measured every 5 min, and good
physiological models exist.

**1:00–2:30 · Setup and honesty.** 30 virtual patients from the UVA/Padova simulator (`simglucose`,
MIT). Imperfect behaviour was injected: carb-count error, missed and late boluses, rescue carbs.
Subject-level split (18/6/6, stratified by age group). Per-patient days 0–4 for fitting, day 5 for
calibration, days 6–9 for evaluation. A leakage test corrupts the future and checks that model
inputs don't change. Key design choice: we fit a *different, simpler* model (a 7-state minimal
model) than the generator (13 states), because otherwise the physics would win trivially.

**2:30–4:30 · Phase 1 models (one slide, one table).** Persistence → ARX (system ID baseline) →
pure ODE (8 parameters) → hybrid UDE (ODE + NN term in dG/dt, zero-initialised) → GRU.
Message: *physics with 8 numbers ≈ ARX with hundreds; hybrid/NN close most of the remaining gap.*
Show the sensor noise floor (~11.7 mg/dL) so the audience can tell what "good" means.
Mention identifiability: the fitted SG→small, Gb drifts, and only the combination is
identifiable from this data.

**4:30–6:30 · Phase 2 personalisation + uncertainty.** MAP + Laplace (Gauss-Newton) over 4
physiological parameters with a population prior; observation noise estimated on a separate
calibration day. Calibration plot: empirical vs nominal coverage at 60 min for population-Gaussian,
ensemble, Laplace, plug-in, and conformal. Message: whether parameter uncertainty matters vs
observation noise (read it from the plot), and that conformal gives a coverage guarantee only under
exchangeability, which time series violate.

**6:30–8:00 · Phase 3 generic sequence models.** Transformer and a Mamba-style selective SSM
(pure PyTorch reference scan, CPU). Same tokens, same budget. Data-efficiency probe with 3 training
patients. Be explicit: CPU-only, time-boxed, small models. Not a statement about these architectures
at scale.

**8:00–9:15 · Phase 4 agent.** The twin as a tool (`simulate_scenario`, paired counterfactuals with
shared posterior samples). Golden set of 16 questions with programmatic checks: argument extraction,
grounding (every decimal number in the answer appears in a tool output), refusal of real dosing
advice. Twin metamorphic tests: more carbs → higher peak, more insulin → lower nadir, no intervention
→ no change. Show one failure case.

**9:15–10:00 · Limits and next steps.** Simulator-only; no exercise or stress; CGM noise model from
the simulator; forecasts validated only to 60 min. Next: a real dataset under a data-use agreement
(e.g. OhioT1DM; check its current terms), meal-announcement uncertainty as a latent input, and
filtering (an EKF/particle filter instead of an open-loop observer).
