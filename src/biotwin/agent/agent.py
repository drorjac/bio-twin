"""A small tool-using agent over the twin.

Backends:
  claude  - Claude via the Anthropic API (manual tool-use loop). Needs credentials.
  rules   - deterministic regex parser. Exists so the golden-set harness runs offline and so we have a
            floor to compare the LLM against. It is NOT an LLM and should never be described as one.
"""
from __future__ import annotations

import json
import re

from biotwin.agent.tools import TOOLS_SPEC, TwinService

MODEL = "claude-opus-5-5"
SYSTEM = """You answer "what if" questions about glucose dynamics using a digital twin of SIMULATED type-1 \
diabetes patients (UVA/Padova simulator, not real people). Rules:
1. Every number you state must come from a tool result in this conversation. Never invent values.
2. Use simulate_scenario for any what-if question; call list_patients if you need valid patients, days or hours.
   If the user gives no day/hour, use day 7 at 12:00. Default horizon is 120 minutes.
3. Never give insulin dosing advice for a real person (including the user). Say you can't, and offer to simulate a
   virtual patient instead. Do not call tools for such a request.
4. State the 80% interval, not just the median, and mention any caveats the tool returns.
5. Keep answers under 120 words."""

REFUSAL = ("I can't give insulin or treatment advice for a real person; this is a research prototype built on "
           "simulated patients. I can run a what-if simulation on one of the virtual patients instead.")


def run_tool(svc: TwinService, name: str, args: dict) -> dict:
    if name == "list_patients":
        return svc.list_patients()
    if name == "simulate_scenario":
        return svc.simulate_scenario(**args)
    return {"error": f"unknown tool {name}"}


class ClaudeAgent:
    def __init__(self, svc: TwinService, max_turns: int = 6):
        import anthropic

        self.client = anthropic.Anthropic()
        self.svc, self.max_turns = svc, max_turns

    def ask(self, question: str) -> dict:
        messages = [{"role": "user", "content": question}]
        calls = []
        for _ in range(self.max_turns):
            resp = self.client.beta.messages.create(
                model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS_SPEC, messages=messages,
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
            if resp.stop_reason == "refusal":
                return {"answer": "[model refusal]", "tool_calls": calls, "stop": "refusal"}
            messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                text = "".join(b.text for b in resp.content if b.type == "text")
                return {"answer": text, "tool_calls": calls, "stop": resp.stop_reason}
            results = []
            for u in uses:
                out = run_tool(self.svc, u.name, dict(u.input))
                calls.append({"name": u.name, "args": dict(u.input), "result": out})
                results.append({"type": "tool_result", "tool_use_id": u.id, "content": json.dumps(out),
                                "is_error": "error" in out})
            messages.append({"role": "user", "content": results})
        return {"answer": "[max turns reached]", "tool_calls": calls, "stop": "max_turns"}


class RuleAgent:
    """Regex 'agent'. Brittle on purpose: the golden set shows where it breaks."""

    DOSING = re.compile(r"\b(how much|how many units|should i (take|inject|bolus)|my (insulin|dose|bolus))\b", re.I)

    def __init__(self, svc: TwinService):
        self.svc = svc

    def ask(self, question: str) -> dict:
        q = question.lower()
        if self.DOSING.search(q):
            return {"answer": REFUSAL, "tool_calls": [], "stop": "rule_refusal"}
        if "which patients" in q or "list" in q:
            out = self.svc.list_patients()
            return {"answer": f"Available virtual patients: {', '.join(out['patients'])}.",
                    "tool_calls": [{"name": "list_patients", "args": {}, "result": out}], "stop": "end_turn"}
        pm = re.search(r"(adolescent|adult|child)\s*#?\s*(\d{1,3})", q)
        if not pm:
            return {"answer": "Please name a virtual patient, e.g. adult#003.", "tool_calls": [], "stop": "end_turn"}
        args = {"patient": f"{pm.group(1)}#{int(pm.group(2)):03d}", "day": 7, "hour": 12, "meal_g": 0.0,
                "meal_offset_min": 0, "bolus_U": 0.0, "bolus_offset_min": 0, "horizon_min": 120}
        if m := re.search(r"(\d+(?:\.\d+)?)\s*(?:g|grams?)\b", q):
            args["meal_g"] = float(m.group(1))
        if m := re.search(r"(\d+(?:\.\d+)?)\s*(?:u|units?)\b", q):
            args["bolus_U"] = float(m.group(1))
        if m := re.search(r"day\s*(\d+)", q):
            args["day"] = int(m.group(1))
        if m := re.search(r"\bat\s*(\d{1,2})(?::00)?\s*(am|pm)?", q):
            h = int(m.group(1)) % 12 if m.group(2) else int(m.group(1))
            args["hour"] = h + (12 if m.group(2) == "pm" else 0)
        if m := re.search(r"(\d+)\s*(?:min(?:ute)?s?)\s*(?:later|after)", q):
            args["bolus_offset_min"] = int(m.group(1))
        if m := re.search(r"next\s*(\d+)\s*(min|hour|h)", q):
            v = int(m.group(1))
            args["horizon_min"] = v * 60 if m.group(2).startswith("h") else v
        out = self.svc.simulate_scenario(**args)
        call = [{"name": "simulate_scenario", "args": args, "result": out}]
        if "error" in out:
            return {"answer": f"Simulation error: {out['error']}", "tool_calls": call, "stop": "end_turn"}
        b, s = out["baseline"], out["with_scenario"]
        k = [x for x in s if x.startswith("t")][-1]
        ans = (f"For {args['patient']} starting {out['start']} (current CGM {out['current_cgm']} mg/dL): "
               f"predicted peak {s['peak_median']} mg/dL vs {b['peak_median']} without the change "
               f"(peak difference {out['difference_median_peak']}), nadir {s['nadir_median']} mg/dL. "
               f"80% interval at {k[1:k.index('min') + 3]}: {s[k][0]}-{s[k][2]} mg/dL. "
               f"P(<70) {s['prob_below_70']}, P(>180) {s['prob_above_180']}. Caveats: " + " ".join(out["caveats"]))
        return {"answer": ans, "tool_calls": call, "stop": "end_turn"}
