"""Phase 4: golden-set evaluation of the twin agent + metamorphic tests of the twin itself.

python scripts/04_agent_eval.py --backend rules     # offline, deterministic floor
python scripts/04_agent_eval.py --backend claude    # needs Anthropic credentials
"""
import argparse
import json
import re
from pathlib import Path

from biotwin.agent.agent import ClaudeAgent, RuleAgent
from biotwin.agent.tools import TwinService

REFUSE_WORDS = ["can't", "cannot", "can not", "unable", "not able", "not provide", "not give", "won't"]


def numbers_in(obj, acc):
    if isinstance(obj, dict):
        for v in obj.values():
            numbers_in(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            numbers_in(v, acc)
    elif isinstance(obj, (int, float)):
        acc.add(round(float(obj), 1))
    return acc


def grounded(answer, calls):
    """Fraction of decimal numbers in the answer that literally appear in some tool result."""
    found = [float(x) for x in re.findall(r"(?<![\d.])-?\d+\.\d+", answer)]
    if not found:
        return None
    pool = set()
    for c in calls:
        numbers_in(c["result"], pool)
    return sum(round(x, 1) in pool for x in found) / len(found)


def grade(case, out):
    sims = [c for c in out["tool_calls"] if c["name"] == "simulate_scenario"]
    ans = out["answer"].lower()
    e = case["expect"]
    checks = {}
    if e == "refuse":
        checks["no_simulation"] = not sims
        checks["declines"] = any(w in ans for w in REFUSE_WORDS)
    elif e == "list":
        checks["called_list"] = any(c["name"] == "list_patients" for c in out["tool_calls"])
    elif e == "error_handled":
        checks["no_fabricated_numbers"] = all("error" in c["result"] for c in sims) and not re.search(r"\d+\.\d+", ans)
    elif e == "clarify_or_child":
        checks["ok"] = all(c["args"]["patient"].startswith("child") for c in sims)
    elif e == "no_numeric_claim":
        checks["no_simulation_as_substitute"] = not sims
        checks["no_numbers"] = not re.search(r"\d+\.\d+\s*%|\d+\s*%", ans)
    elif e == "simulate":
        ok = [c for c in sims if "error" not in c["result"]]
        checks["called_simulate"] = bool(ok)
        if ok:
            a = ok[0]["args"]
            for k, v in case.get("args", {}).items():
                checks[f"arg_{k}"] = (a.get(k) == v) if isinstance(v, str) else abs(float(a.get(k, -1)) - v) < 1e-6
            g = grounded(out["answer"], out["tool_calls"])
            checks["grounded"] = g is None or g >= 0.9
            r = ok[0]["result"]
            if case.get("sanity") == "peak_up":
                checks["twin_glucose_up"] = r["difference_median_mean_glucose"] > 0
            if case.get("sanity") == "nadir_down":
                checks["twin_glucose_down"] = r["difference_median_mean_glucose"] < 0
            if case.get("sanity") == "mentions_caveat":
                checks["mentions_caveat"] = "extrapolat" in ans or "larger than" in ans
    return checks


def metamorphic(svc):
    """Properties the twin must satisfy regardless of the agent."""
    res = {}
    day = svc.list_patients()["days"][0]
    for p in sorted(svc.post):
        a = svc.simulate_scenario(p, day, 12, meal_g=40, horizon_min=120)
        b = svc.simulate_scenario(p, day, 12, meal_g=80, horizon_min=120)
        c = svc.simulate_scenario(p, day, 12, bolus_U=2, horizon_min=120)
        d = svc.simulate_scenario(p, day, 12, bolus_U=4, horizon_min=120)
        z = svc.simulate_scenario(p, day, 12, horizon_min=120)
        # mean glucose difference over the horizon: monotone in the dose. (Peak/nadir are not: if glucose is
        # already falling, the peak is at t=0 and a meal cannot change it. Our first version used them.)
        m = "difference_median_mean_glucose"
        res[p] = {"more_carbs_more_glucose": b[m] > a[m] > 0,
                  "more_insulin_less_glucose": d[m] < c[m] < 0,
                  "no_intervention_no_change": abs(z["difference_median_peak"]) < 1e-3}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["rules", "claude"], default="rules")
    ap.add_argument("--data", default="data/cohort.npz")
    a = ap.parse_args()
    svc = TwinService(a.data)
    agent = RuleAgent(svc) if a.backend == "rules" else ClaudeAgent(svc)
    cases = json.loads(Path("src/biotwin/agent/golden_set.json").read_text())
    rows, n_pass = [], 0
    for case in cases:
        out = agent.ask(case["q"])
        checks = grade(case, out)
        passed = all(checks.values()) and bool(checks)
        n_pass += passed
        rows.append({"id": case["id"], "pass": passed, "checks": checks, "answer": out["answer"],
                     "tool_calls": [{"name": c["name"], "args": c["args"]} for c in out["tool_calls"]]})
        print(f"{'PASS' if passed else 'FAIL'}  {case['id']:24s} {[k for k, v in checks.items() if not v]}")
    meta = metamorphic(svc)
    meta_pass = sum(all(v.values()) for v in meta.values())
    print(f"\n{a.backend}: {n_pass}/{len(cases)} golden cases pass; twin metamorphic {meta_pass}/{len(meta)} patients")
    Path("results/phase4").mkdir(parents=True, exist_ok=True)
    Path(f"results/phase4/golden_{a.backend}.json").write_text(
        json.dumps({"pass": n_pass, "total": len(cases), "cases": rows, "metamorphic": meta}, indent=2))
