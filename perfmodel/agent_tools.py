"""Tool functions exposed to the bottleneck agent, plus a numeric-claim audit.

Each tool returns a JSON string with numbers rounded to 4 significant figures,
so the report's numbers can be traced back to tool output (see audit_numbers).
"""
import json
import re
from pathlib import Path

import pandas as pd

from .calibration import RESULTS, load_profiles
from .config import MODELS
from .hardware import GPUS
from .ops import decode_ops, prefill_ops
from .predict import op_table
from .sweep import EVENTS, find_transition, point

MAX_POINTS = 80
PROFILES = load_profiles()


def _round(x):
    if isinstance(x, float):
        return float(f"{x:.4g}")
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def _dump(obj) -> str:
    return json.dumps(_round(obj), separators=(",", ":"))


def _resolve(model: str, gpu: str, profile: str):
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {list(MODELS)}")
    if gpu not in GPUS:
        raise ValueError(f"unknown gpu {gpu!r}; choose from {list(GPUS)}")
    if profile not in PROFILES[gpu]:
        raise ValueError(f"profile {profile!r} not available for {gpu}; "
                         f"choose from {list(PROFILES[gpu])}")
    return MODELS[model], GPUS[gpu], PROFILES[gpu][profile][0]


def describe_setup() -> str:
    """Describe the available models, GPUs, efficiency profiles and validation accuracy.

    Call this first. Profiles: "peak" (datasheet upper bound), "prior" (hand-set
    estimate) and, where measurements exist, "calibrated" (fitted to measured runs).
    """
    models = {n: {"params_b": m.params() / 1e9, "layers": m.n_layers, "hidden": m.hidden,
                  "q_heads": m.n_heads, "kv_heads": m.n_kv_heads,
                  "weights_gb_fp16": m.weight_bytes() / 1e9,
                  "kv_cache_kb_per_token": m.kv_bytes_per_token() / 1e3}
              for n, m in MODELS.items()}
    gpus = {n: {"fp16_tensor_tflops": g.matmul_flops / 1e12, "mem_bw_gb_s": g.mem_bw / 1e9,
                "memory_gb": g.mem_capacity / 1e9, "ridge_flop_per_byte": g.ridge_point,
                "profiles": {p: {"provenance": prov, "compute_eff": e.compute,
                                 "memory_eff": e.memory,
                                 "kernel_overhead_us": e.kernel_overhead_s * 1e6,
                                 "attention_half_work": e.attention_half_work,
                                 "overlap": None if e.overlap == float("inf") else e.overlap}
                             for p, (e, prov) in PROFILES[n].items()}}
            for n, g in GPUS.items()}
    validation = []
    for path in sorted(RESULTS.glob("validation_summary_*.csv")):
        df = pd.read_csv(path)
        held = df[(df.split == "held-out") & (df.profile.isin(["roofline", "refined"]))]
        validation.append({"run": path.stem.removeprefix("validation_summary_"),
                           "held_out_mape_pct": held[["phase", "profile", "mape_pct"]]
                           .to_dict("records")})
    return _dump({"models": models, "gpus": gpus, "validation": validation,
                  "notes": ["decode tokens_per_s = batch / time of one decode step at that context",
                            "prefill tokens_per_s = batch * prompt / time of one prefill pass",
                            "memory fit assumes weights + KV cache <= 90% of GPU memory"]})


def sweep(model: str, gpu: str, profile: str, phase: str, batches: list[int],
          contexts: list[int]) -> str:
    """Predict one forward pass at every (batch, context) grid point.

    Each point reports tokens/s, pass time, the share of time in weight matmuls
    ("linear"), attention / KV cache, elementwise ops and kernel launch overhead,
    the dominant component, whether linear and attention ops are compute- or
    memory-bound, GB of weights and KV cache read, and whether it fits in memory.

    Args:
        model: Model name from describe_setup.
        gpu: GPU name from describe_setup.
        profile: Efficiency profile available for that GPU.
        phase: "decode" (one new token per sequence, `context` tokens of history) or "prefill" (prompts of `context` tokens).
        batches: Batch sizes (number of sequences).
        contexts: Context lengths in tokens. At most 80 grid points in total.
    """
    cfg, g, eff = _resolve(model, gpu, profile)
    if phase not in ("decode", "prefill"):
        raise ValueError("phase must be 'decode' or 'prefill'")
    if len(batches) * len(contexts) > MAX_POINTS:
        raise ValueError(f"grid too large ({len(batches) * len(contexts)} points, max {MAX_POINTS})")
    rows = [point(cfg, g, eff, phase, b, c) for c in contexts for b in batches]
    return _dump({"model": model, "gpu": gpu, "profile": profile, "phase": phase,
                  "points": rows})


def find_crossover(model: str, gpu: str, profile: str, phase: str, event: str,
                   axis: str, fixed: int, search_max: int = 1 << 20) -> str:
    """Find the exact batch size or context length where a bottleneck transition happens.

    Returns the smallest value on `axis` at which the event becomes true, plus
    the full point just before and at the transition.

    Args:
        model: Model name from describe_setup.
        gpu: GPU name from describe_setup.
        profile: Efficiency profile available for that GPU.
        phase: "decode" or "prefill".
        event: One of "attention_exceeds_linear" (KV-cache/attention time overtakes weight-matmul time), "linear_compute_bound" (weight matmuls cross the ridge point), "overhead_below_10pct" (launch overhead drops under 10% of pass time), "exceeds_memory" (weights + KV cache stop fitting).
        axis: "batch" or "context": the dimension to search along.
        fixed: Value of the other dimension, held constant.
        search_max: Largest value to search up to.
    """
    cfg, g, eff = _resolve(model, gpu, profile)
    if event not in EVENTS:
        raise ValueError(f"unknown event {event!r}; choose from {list(EVENTS)}")
    if axis not in ("batch", "context"):
        raise ValueError("axis must be 'batch' or 'context'")
    res = find_transition(cfg, g, eff, phase, event, axis, fixed, hi=search_max)
    return _dump({"model": model, "gpu": gpu, "profile": profile, "phase": phase, **res})


def op_breakdown(model: str, gpu: str, profile: str, phase: str, batch: int,
                 context: int) -> str:
    """Per-operation detail for one forward pass: arithmetic intensity, which roof
    limits it, and its share of total time. Use to explain why a component dominates.

    Args:
        model: Model name from describe_setup.
        gpu: GPU name from describe_setup.
        profile: Efficiency profile available for that GPU.
        phase: "decode" or "prefill".
        batch: Batch size (sequences).
        context: Context length (decode) or prompt length (prefill), in tokens.
    """
    cfg, g, eff = _resolve(model, gpu, profile)
    ops = decode_ops(cfg, batch, context) if phase == "decode" else prefill_ops(cfg, batch, context)
    df = op_table(ops, g, eff)
    cols = ["op", "kind", "count", "intensity", "bound", "time_share", "total_time_s"]
    df = df[cols].sort_values("time_share", ascending=False)
    df["total_time_ms"] = df.pop("total_time_s") * 1e3
    return _dump({"model": model, "gpu": gpu, "profile": profile, "phase": phase,
                  "batch": batch, "context": context, "ridge_flop_per_byte": g.ridge_point,
                  "ops": df.to_dict("records")})


# --- numeric-claim audit --------------------------------------------------------

_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
_IDENTIFIERS = [*MODELS, *GPUS, "H100", "T4", "fp16", "FP16", "Llama-3", "SwiGLU"]


def extract_numbers(text: str) -> list[tuple[str, float]]:
    for ident in sorted(_IDENTIFIERS, key=len, reverse=True):
        text = text.replace(ident, " ")
    text = re.sub(r"(?m)^\s*#+.*$", " ", text)            # headings
    text = re.sub(r"(?m)^\s*\d+\.\s", " ", text)          # list markers
    out = []
    for m in _NUM.finditer(text):
        s = m.group().replace(",", "")
        try:
            out.append((m.group(), float(s)))
        except ValueError:
            pass
    return out


def audit_numbers(report: str, sources: list[str], skip_small_ints: int = 16) -> dict:
    """Check that every number in the report matches some number in the tool
    inputs/outputs, allowing for the rounding the report itself shows (3.1
    matches 3.142) or 1%. Whole numbers <= skip_small_ints (counts, list items)
    are not checked."""
    pool = [v for s in sources for _, v in extract_numbers(s)]
    checked, unverified = 0, []
    for raw, v in extract_numbers(report):
        if "." not in raw and abs(v) <= skip_small_ints:
            continue
        checked += 1
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tol_round = 0.5 * 10 ** -decimals
        # also accept values a tool reported as a fraction and the report shows as a percent
        ok = any(abs(v - s) <= max(tol_round, 0.01 * abs(s)) or
                 abs(v - 100 * s) <= max(tol_round, 0.01 * abs(100 * s)) for s in pool)
        if not ok:
            unverified.append(raw)
    return {"checked": checked, "unverified": sorted(set(unverified), key=unverified.index)}
