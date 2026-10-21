"""Parameter sweeps and bottleneck-transition search (no LLM involved).

A "point" is one forward pass:
  decode   batch sequences each generate one token at `context` tokens of history
  prefill  batch prompts of `context` tokens are processed in one pass
"""
from .config import ModelConfig
from .hardware import GPU, Efficiency
from .ops import ATTENTION, ELEMENTWISE, LINEAR, decode_ops, prefill_ops
from .predict import fits, memory_footprint, op_time

COMPONENTS = (LINEAR, ATTENTION, ELEMENTWISE, "overhead")


def point(cfg: ModelConfig, gpu: GPU, eff: Efficiency, phase: str, batch: int,
          context: int) -> dict:
    ops = decode_ops(cfg, batch, context) if phase == "decode" else prefill_ops(cfg, batch, context)
    time_by = dict.fromkeys(COMPONENTS, 0.0)
    t_c = {LINEAR: 0.0, ATTENTION: 0.0}
    t_m = {LINEAR: 0.0, ATTENTION: 0.0}
    weight_b = kv_b = 0.0
    for o in ops:
        c, m, t = op_time(o, gpu, eff)
        time_by[o.kind] += (t - eff.kernel_overhead_s) * o.count
        time_by["overhead"] += eff.kernel_overhead_s * o.count
        if o.kind in t_c:
            t_c[o.kind] += c * o.count
            t_m[o.kind] += m * o.count
        weight_b += o.weight_bytes * o.count
        kv_b += o.kv_bytes * o.count
    total = sum(time_by.values())
    tokens = batch if phase == "decode" else batch * context
    shares = {k: v / total for k, v in time_by.items()}
    max_ctx = context + (1 if phase == "decode" else 0)
    return {
        "phase": phase, "batch": batch, "context": context,
        "fits_in_memory": fits(cfg, gpu, batch, max_ctx, eff),
        "memory_needed_gb": memory_footprint(cfg, batch, max_ctx) / 1e9,
        "pass_time_ms": total * 1e3,
        "tokens_per_s": tokens / total,
        "share": shares,
        "dominant": max(shares, key=shares.get),
        # Which roof limits each op family: compute if its compute time exceeds its memory time.
        "linear_bound": "compute" if t_c[LINEAR] > t_m[LINEAR] else "memory",
        "attention_bound": "compute" if t_c[ATTENTION] > t_m[ATTENTION] else "memory",
        "weight_gb_read": weight_b / 1e9,
        "kv_gb_read": kv_b / 1e9,
    }


# --- transition search ---------------------------------------------------------

EVENTS = {
    # name: (description, predicate on a point; transition = first value where it is True)
    "attention_exceeds_linear": (
        "attention (KV-cache) time exceeds weight-matmul time",
        lambda p: p["share"][ATTENTION] > p["share"][LINEAR]),
    "linear_compute_bound": (
        "weight matmuls become compute-bound (cross the ridge point)",
        lambda p: p["linear_bound"] == "compute"),
    "overhead_below_10pct": (
        "kernel launch overhead falls below 10% of pass time",
        lambda p: p["share"]["overhead"] < 0.10),
    "exceeds_memory": (
        "weights + KV cache no longer fit in GPU memory",
        lambda p: not p["fits_in_memory"]),
}


def find_transition(cfg, gpu, eff, phase: str, event: str, axis: str, fixed: int,
                    lo: int = 1, hi: int = 1 << 20) -> dict:
    """Smallest value on `axis` ("batch" or "context") at which `event` becomes true,
    holding the other axis at `fixed`. Doubles from `lo` to bracket, then bisects.
    Assumes the event, once true, stays true along the axis (checked at the bracket)."""
    desc, pred = EVENTS[event]
    at = lambda v: point(cfg, gpu, eff, phase, v, fixed) if axis == "batch" \
        else point(cfg, gpu, eff, phase, fixed, v)
    if pred(at(lo)):
        return {"event": event, "description": desc, "axis": axis, "fixed": fixed,
                "result": "already true at the smallest value", "value": lo, "at": at(lo)}
    prev, v = lo, lo * 2
    while v <= hi and not pred(at(v)):
        prev, v = v, v * 2
    if v > hi:
        return {"event": event, "description": desc, "axis": axis, "fixed": fixed,
                "result": f"never true up to {hi}", "value": None, "at": at(prev)}
    a, b = prev, v                      # pred(a) False, pred(b) True
    while b - a > 1:
        m = (a + b) // 2
        a, b = (m, b) if not pred(at(m)) else (a, m)
    return {"event": event, "description": desc, "axis": axis, "fixed": fixed,
            "result": "found", "value": b, "before": at(a), "at": at(b)}
