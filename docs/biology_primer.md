# Glucose–insulin physiology for a signal-processing person

Written as a translation table. Where I am unsure of a number I say so; check the numbers against
the cited model papers before quoting them in an interview.

## 1. The plant

Think of blood glucose `G(t)` (mg/dL) as the output of a nonlinear, time-varying, MIMO plant.

| Physiology | System-ID view |
|---|---|
| Blood glucose G | measured output (through a sensor with lag + noise) |
| Meal carbohydrates (g) | disturbance input; you know roughly when and how much |
| Insulin (units, U) | control input (in type-1 diabetes it comes from a pump or pen, not the pancreas) |
| Liver glucose production, uptake by muscle and brain | internal feedback loops |
| Insulin sensitivity | a gain that varies between people and over the day (stress, exercise, illness) |
| Gut absorption, subcutaneous insulin absorption | low-pass filters with different time constants |

**Type-1 diabetes (T1D):** the pancreas makes (almost) no insulin, so the body's own feedback
controller is gone. The patient, or an "artificial pancreas" algorithm, closes the loop by hand.
This is why T1D is the standard testbed for glucose modelling: the insulin input is known and
logged.

## 2. Time scales (orders of magnitude; verify before quoting)

- Carbohydrates appear in the blood over roughly 1–3 h; it is a 2nd-order low-pass with a peak
  around 30–90 min.
- Insulin injected under the skin peaks in the blood around 1 h and acts for about 3–5 h. That is
  slower than carbs, which is why people bolus *before* eating.
- A CGM (continuous glucose monitor) measures *interstitial* glucose, which lags blood glucose by
  roughly 5–15 min, plus noise. In this repo the simulated sensor noise is about 11.7 mg/dL RMSE
  (measured, see `results/phase1/persistence.json`).
- Targets: 70–180 mg/dL is "time in range". Below 70 is hypoglycaemia, which is acutely
  dangerous. Above 180 is hyperglycaemia, which does long-term damage.

## 3. The models in this repo

**Data generator: UVA/Padova simulator** (via `simglucose`, MIT license). 13 states, 30 virtual
patients with different parameters. Some versions of this simulator have been accepted by the
FDA as a substitute for preclinical animal trials of control algorithms. That is a statement
about the simulator's use in the literature, not about this repo; check the exact wording before
quoting it.

**Our mechanistic model: a Bergman-style minimal model** with absorption compartments (7 states,
`src/biotwin/models/mechanistic.py`):

```
insulin depot → plasma insulin → "remote insulin action" X   (3 cascaded 1st-order filters)
gut D1 → D2 → appearance of glucose Ra                        (2 cascaded 1st-order filters)
dG/dt = -SG (G - Gb) - X·G + Ra
```

- `SG`, glucose effectiveness: how fast G relaxes to its basal level `Gb` without extra insulin.
  This is the pole of the glucose subsystem.
- `SI`, insulin sensitivity: how strongly insulin action acts on G.
- The `X·G` term is **bilinear**, and it is the only nonlinearity. Everything else is LTI driven by
  known inputs. That is why we can reconstruct the hidden states by running the filters over the
  input history (an "open-loop observer"), then start the forecast from the measured G.

**Why use a different model from the generator?** If we fitted the UVA/Padova equations to
UVA/Padova data, the "mechanistic" baseline would be the true model and would win trivially. The
minimal model is *misspecified on purpose*. That mirrors reality, where every physiological model
is wrong, and it gives the neural correction a real job to do.

## 4. Hybrid model (universal differential equation)

```
dG/dt = mechanistic(G, X, D, θ) + NN(states, recent CGM slope)
```

The NN is initialised to output 0, so training starts from the pure physics model. Analogy: grey-box
system ID. You keep the known structure and learn a residual dynamics term, much like adding a
nonparametric disturbance model to a Kalman filter.

## 5. Personalisation = Bayesian parameter estimation

Each patient differs mostly in a few physiological parameters (here `SI`, `SG`, `cg`, `Gb`). With
a population prior `N(θ_pop, s²)` (in log space), a few days of a patient's data give a MAP
estimate plus a Laplace posterior. This is exactly "MAP estimation with a Gaussian prior, then the
Fisher information / Gauss-Newton Hessian as inverse covariance". You already know this from
estimation theory. Here it is applied to a physiological ODE.

## 6. What a simulator cannot give you (say this before an interviewer does)

- Real-life disturbances: exercise, stress, illness, the menstrual cycle, sensor faults, missed
  data, wrong carb counts beyond the noise we injected.
- Validation: agreement with UVA/Padova only shows we can learn *that simulator*.
- Some of these models (UVA/Padova in particular) were themselves fitted to clinical data,
  so they are physiologically informed. They are still a model.
