"""Roofline timing and throughput prediction.

Each op takes max(compute time, memory time) + a fixed launch overhead, i.e.
compute and memory traffic overlap perfectly within an op and ops run serially.
"""
from dataclasses import dataclass

import pandas as pd

from .config import ModelConfig
from .hardware import GPU, PEAK, Efficiency
from .ops import ELEMENTWISE, Op, decode_ops, prefill_ops


def op_time(op: Op, gpu: GPU, eff: Efficiency = PEAK) -> tuple[float, float, float]:
    """Return (compute_s, memory_s, total_s) for one instance of `op`."""
    peak = gpu.vector_flops if op.kind == ELEMENTWISE else gpu.matmul_flops
    t_c = op.flops / (peak * eff.compute)
    t_m = op.bytes / (gpu.mem_bw * eff.memory)
    return t_c, t_m, max(t_c, t_m) + eff.kernel_overhead_s


def pass_time(ops: list[Op], gpu: GPU, eff: Efficiency = PEAK) -> float:
    return sum(op_time(o, gpu, eff)[2] * o.count for o in ops)


def op_table(ops: list[Op], gpu: GPU, eff: Efficiency = PEAK) -> pd.DataFrame:
    rows = []
    for o in ops:
        t_c, t_m, t = op_time(o, gpu, eff)
        rows.append({
            "op": o.name, "kind": o.kind, "count": o.count,
            "flops": o.flops, "bytes": o.bytes,
            "weight_bytes": o.weight_bytes, "act_bytes": o.act_bytes, "kv_bytes": o.kv_bytes,
            "intensity": o.intensity,
            "compute_s": t_c, "memory_s": t_m, "time_s": t,
            "total_time_s": t * o.count,
            "overhead_s": eff.kernel_overhead_s,
            "bound": ("overhead" if eff.kernel_overhead_s > max(t_c, t_m)
                      else "compute" if t_c > t_m else "memory"),
            "achieved_flops": o.flops / t if t else 0.0,
        })
    df = pd.DataFrame(rows)
    df["time_share"] = df["total_time_s"] / df["total_time_s"].sum()
    return df


def memory_footprint(cfg: ModelConfig, batch: int, max_context: int) -> float:
    """Weights + full KV cache, bytes (ignores activation workspace)."""
    return cfg.weight_bytes() + batch * max_context * cfg.kv_bytes_per_token()


def fits(cfg: ModelConfig, gpu: GPU, batch: int, max_context: int,
         eff: Efficiency = PEAK) -> bool:
    return memory_footprint(cfg, batch, max_context) <= gpu.mem_capacity * eff.mem_utilization


@dataclass
class Prediction:
    ttft_s: float              # prefill latency for the whole batch
    decode_s: float            # time to generate all gen_len tokens after prefill
    batch: int
    prompt_len: int
    gen_len: int
    fits: bool

    @property
    def prefill_tok_s(self) -> float:
        return self.batch * self.prompt_len / self.ttft_s

    @property
    def decode_tok_s(self) -> float:
        """Generated tokens/s during the decode phase (all sequences)."""
        return self.batch * self.gen_len / self.decode_s if self.gen_len else float("nan")

    @property
    def tpot_s(self) -> float:
        """Mean time per output token per sequence (inter-token latency)."""
        return self.decode_s / self.gen_len if self.gen_len else float("nan")

    @property
    def e2e_tok_s(self) -> float:
        """Generated tokens/s including prefill."""
        return self.batch * self.gen_len / (self.ttft_s + self.decode_s)


def decode_step_time(cfg, gpu, batch, context_len, eff=PEAK) -> float:
    return pass_time(decode_ops(cfg, batch, context_len), gpu, eff)


def predict(cfg: ModelConfig, gpu: GPU, batch: int, prompt_len: int, gen_len: int,
            eff: Efficiency = PEAK) -> Prediction:
    """Static-batch generation: prefill all prompts, then gen_len decode steps.

    The prefill pass emits the first token, so decoding runs gen_len - 1 steps
    over contexts prompt_len .. prompt_len + gen_len - 2.
    """
    ttft = pass_time(prefill_ops(cfg, batch, prompt_len), gpu, eff)
    decode = sum(decode_step_time(cfg, gpu, batch, prompt_len + i, eff)
                 for i in range(gen_len - 1))
    # Report decode throughput over the gen_len - 1 steps actually run.
    return Prediction(ttft, decode, batch, prompt_len, max(gen_len - 1, 0),
                      fits(cfg, gpu, batch, prompt_len + gen_len, eff))
