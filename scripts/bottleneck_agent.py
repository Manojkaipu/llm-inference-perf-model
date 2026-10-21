"""Agent that runs parameter sweeps on the analytical model and writes up where
the inference bottleneck shifts.

    python scripts/bottleneck_agent.py [--task "..."] [--effort high] [--max-iterations 40]

Needs ANTHROPIC_API_KEY (environment or .env). Writes reports/bottleneck_<time>.md
and a tool-call log to results/agent_runs/<time>.json. Every number in the report
is checked against the tool outputs; unmatched numbers are listed in the report.
"""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import anthropic
from anthropic import beta_tool
from dotenv import load_dotenv

from perfmodel import agent_tools

MODEL = "claude-opus-5-5"
PRICE_PER_MTOK = {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00}

SYSTEM = """You are a GPU performance analyst. Your tools wrap an analytical roofline \
model of Llama-style LLM inference: per-op FLOPs and bytes, timed against each GPU's \
compute and memory roofs with efficiency profiles. Use them to map where the \
bottleneck shifts (weight bandwidth, KV-cache bandwidth, compute, kernel launch \
overhead, memory capacity) across batch size, context length, phase, model size and \
GPU, then write a report in Markdown.

Ground rules:
- Every number in the report must come from a tool result; rounding is fine. If you \
state a ratio or a change, give the two tool-reported numbers it is based on. Never \
estimate a number yourself.
- Locate transitions with find_crossover rather than reading them off a coarse grid.
- Only the T4 "calibrated" profile is validated against measurements (TinyLlama-1.1B \
on llama.cpp). Say which profile each finding uses. Present H100 results, and other \
models on the T4, as model predictions rather than measurements, and point out where \
a conclusion depends on an uncalibrated assumption (for example, the H100 \
kernel-overhead prior).
- The report is for engineers deciding how to batch and size inference. Lead with a \
short summary of the most decision-relevant findings, use compact tables where they \
help, and keep it under about 1500 words. End with a short caveats section.
- Your final message is saved verbatim as the report, so it must contain only the \
report."""

DEFAULT_TASK = """Map the bottleneck transitions for every model on both GPUs. Cover \
decode across batch size and context length, prefill across prompt length, the batch \
and context limits set by memory capacity, and how the picture differs between the \
T4 and H100 and between model sizes. Explain the mechanism behind each shift."""


def logged(fn, log: list):
    """Wrap a tool so every call and its output are recorded; return input errors
    to the model as tool output instead of crashing the run."""
    @functools.wraps(fn)
    def inner(**kwargs):
        t0 = time.time()
        try:
            out = fn(**kwargs)
        except (ValueError, KeyError) as e:
            out = json.dumps({"error": str(e)})
        log.append({"tool": fn.__name__, "input": kwargs, "output": out,
                    "seconds": round(time.time() - t0, 3)})
        args = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"  [{len(log):>2}] {fn.__name__}({args[:110]})", flush=True)
        return out
    return inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--max-iterations", type=int, default=40)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()
    log: list[dict] = []
    tools = [beta_tool(logged(f, log)) for f in (agent_tools.describe_setup, agent_tools.sweep,
                                                agent_tools.find_crossover, agent_tools.op_breakdown)]

    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": args.task}],
        tools=tools,
        thinking={"type": "adaptive"},
        output_config={"effort": args.effort},
        cache_control={"type": "ephemeral"},       # cache the growing conversation prefix
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",                       # re-run on a fallback model if declined
        max_iterations=args.max_iterations,
    )

    usage = dict.fromkeys(PRICE_PER_MTOK, 0)
    last = None
    print(f"Running {MODEL} (effort {args.effort})...", flush=True)
    for message in runner:
        last = message
        u = message.usage
        usage["input"] += u.input_tokens or 0
        usage["output"] += u.output_tokens or 0
        usage["cache_read"] += u.cache_read_input_tokens or 0
        usage["cache_write"] += u.cache_creation_input_tokens or 0
    cost = sum(usage[k] * PRICE_PER_MTOK[k] / 1e6 for k in usage)

    if last is None:
        sys.exit("No response from the API.")
    if last.stop_reason == "refusal":
        sys.exit(f"Request declined: {last.stop_details}")
    report = "\n".join(b.text for b in last.content if b.type == "text").strip()
    if last.stop_reason == "tool_use" or not report:
        sys.exit(f"Run ended before a report was written (stop_reason={last.stop_reason}, "
                 f"{len(log)} tool calls). Raise --max-iterations and retry.")
    if last.stop_reason == "max_tokens":
        print("Warning: report hit max_tokens and may be truncated.")

    sources = [json.dumps(c["input"]) + c["output"] for c in log]
    audit = agent_tools.audit_numbers(report, sources)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "results" / "agent_runs").mkdir(parents=True, exist_ok=True)
    unverified = ", ".join(audit["unverified"]) or "none"
    provenance = f"""

---

## Provenance

Generated by `scripts/bottleneck_agent.py` with `{last.model}` (effort `{args.effort}`) on
{time.strftime('%Y-%m-%d')}, from {len(log)} tool calls to the analytical model
(log: `results/agent_runs/{stamp}.json`).

Numeric check: {audit['checked']} numbers in this report were compared with the tool
outputs; {len(audit['unverified'])} could not be matched: {unverified}.

Tokens: {usage['input']:,} input, {usage['cache_read']:,} cache read,
{usage['cache_write']:,} cache write, {usage['output']:,} output (~${cost:.2f}).
"""
    out = ROOT / "reports" / f"bottleneck_{stamp}.md"
    out.write_text(report + provenance, encoding="utf-8")
    (ROOT / "results" / "agent_runs" / f"{stamp}.json").write_text(json.dumps(
        {"model": last.model, "effort": args.effort, "task": args.task, "usage": usage,
         "cost_usd": cost, "audit": audit, "tool_calls": log}, indent=1))

    print(f"\n{len(log)} tool calls, ~${cost:.2f}. "
          f"Numbers checked: {audit['checked']}, unmatched: {unverified}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
