"""Helpers for the tutorial workbooks.

Every exercise follows the same pattern:

    def my_function(...):
        # TODO: your code
        raise NotImplementedError

    my_function = use(my_function, sol.my_function, check_my_function)

`use` runs the check on your version. If it passes, your version is used for the rest of the
notebook. If it isn't written yet or fails, you get a message and the reference solution is used
instead, so later cells still run. Set TUTOR_VERIFY=1 to make a broken reference raise
(that is how the notebooks themselves are tested).
"""
from __future__ import annotations

import os

import torch


def use(mine, ref, check):
    name = getattr(mine, "__name__", "your function")
    try:
        check(mine)
    except NotImplementedError:
        print(f"[todo] {name}: not written yet -> using the reference solution for now")
        _verify(ref, check, name)
        return ref
    except Exception as e:  # noqa: BLE001 - any failure of the learner's code should fall back
        print(f"[fail] {name}: {type(e).__name__}: {e}\n       -> using the reference solution for now")
        _verify(ref, check, name)
        return ref
    print(f"[ok]   {name}: passed the check, your version is used from here on")
    return mine


def _verify(ref, check, name):
    try:
        check(ref)
    except Exception as e:  # noqa: BLE001
        msg = f"reference solution for {name} failed its own check: {e} (bug in the tutorial)"
        if os.environ.get("TUTOR_VERIFY"):
            raise AssertionError(msg) from e
        print("       warning:", msg)


def close(got, expected, tol=1e-4, what="result"):
    """Assert two arrays/tensors match to a relative tolerance."""
    g = torch.as_tensor(got, dtype=torch.float64)
    e = torch.as_tensor(expected, dtype=torch.float64)
    assert g.shape == e.shape, f"{what}: shape {tuple(g.shape)}, expected {tuple(e.shape)}"
    err = (g - e).abs().max().item() if g.numel() else 0.0
    scale = 1.0 + e.abs().max().item() if e.numel() else 1.0
    assert err <= tol * scale, f"{what}: max abs error {err:.3g} (tolerance {tol * scale:.3g})"


def twin_report(model, cohort, sets, bolus_U=4.0, hours=(0, 4, 8, 12, 16, 20), days=(6, 7), horizon=24):
    """Score a (population) twin on accuracy AND on whether insulin lowers glucose.

    accuracy : RMSE at 60 min on the test patients' evaluation days.
    physiology: for every test patient and start time, simulate 2 h with basal only vs a bolus at t=0;
                a violation is a bolus that does NOT lower mean glucose.
    """
    from biotwin.data.dataset import LH, STEPS_PER_DAY
    from biotwin.train import predict

    te = sets["test_eval"]
    rmse60 = (predict(model, te)[:, -1] - te["y"][:, -1]).pow(2).mean().sqrt().item()
    rows = []
    for name in sets["split"]["test"]:
        p = cohort.idx(name)
        for d in days:
            for hr in hours:
                t0 = d * STEPS_PER_DAY + hr * 12
                ins = torch.tensor(cohort.insulin[p, t0 - LH:t0])
                fut = torch.full((horizon,), float(cohort.basal[p]))
                base = {"cgm_hist": torch.tensor(cohort.cgm[p, t0 - LH:t0])[None],
                        "ins": torch.cat([ins, fut])[None],
                        "carbs": torch.cat([torch.tensor(cohort.carbs[p, t0 - LH:t0]), torch.zeros(horizon)])[None],
                        "basal": torch.tensor([float(cohort.basal[p])]), "y": torch.zeros(1, horizon)}
                scen = {k: v.clone() for k, v in base.items()}
                scen["ins"][0, LH] += bolus_U
                with torch.no_grad():
                    model.eval()
                    diff = (model(scen) - model(base)).mean().item()
                rows.append({"patient": name, "day": d, "hour": hr, "effect": diff})
    viol = [r for r in rows if r["effect"] >= 0]
    print(f"RMSE 60 min: {rmse60:.1f} mg/dL | bolus raises or doesn't lower glucose in "
          f"{len(viol)}/{len(rows)} cases")
    return {"rmse60": rmse60, "violations": len(viol), "cases": len(rows), "rows": rows}
