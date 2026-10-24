| phase | model | RMSE 30 min | RMSE 60 min | MAE 60 min | 80% cov (60 min) | 95% cov (60 min) | 80% width (60 min) |
|---|---|---|---|---|---|---|---|
| phase1 | arx | 20.54 | 36.27 | 26.11 | – | – | – |
| phase1 | gru | 20.22 | 32.13 | 21.96 | – | – | – |
| phase1 | hybrid | 20.38 | 32.48 | 23.32 | – | – | – |
| phase1 | ode | 23.46 | 36.78 | 25.99 | – | – | – |
| phase1 | persistence | 30.67 | 52.60 | 36.00 | – | – | – |
| phase2 | gru_finetuned | 17.90 | 25.67 | 17.79 | – | – | – |
| phase2 | hybrid_ensemble | 20.83 | 33.19 | 23.63 | 0.86 | 0.95 | 87.05 |
| phase2 | hybrid_laplace | 17.96 | 26.95 | 20.37 | 0.80 | 0.92 | 66.90 |
| phase2 | hybrid_map_conformal80 | 17.96 | 26.95 | 20.37 | 0.80 | – | 64.16 |
| phase2 | hybrid_map_plugin | 17.96 | 26.95 | 20.37 | 0.79 | 0.91 | 66.41 |
| phase2 | hybrid_pop_gauss | 20.38 | 32.48 | 23.32 | 0.85 | 0.94 | 83.27 |
| phase3 | hybrid_3patients | 25.00 | 38.15 | 26.39 | – | – | – |
| phase3 | ssm | 23.27 | 39.87 | 27.40 | – | – | – |
| phase3 | transformer | 23.87 | 36.38 | 24.82 | – | – | – |
| phase3 | transformer_3patients | 25.09 | 39.76 | 26.48 | – | – | – |
