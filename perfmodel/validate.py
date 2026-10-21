"""Compare predictions with measured llama.cpp runs and calibrate efficiencies.

Expected measurement directory (written by notebooks/t4_llamacpp_benchmark.ipynb):
  llama_bench.csv     llama-bench -o csv  (batch 1: pp rows and tg rows at depth)
  batched_bench.csv   parsed llama-batched-bench table (multi-sequence)
  microbench.csv      metric,value  (torch-measured bandwidth and matmul FLOP/s)
  env.json            GPU / driver / llama.cpp commit metadata
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ModelConfig
from .hardware import GPU, Efficiency
from .ops import ELEMENTWISE, Op, decode_ops, forward_ops


# --- workloads as llama.cpp executes them ------------------------------------

def prefill_pass_ops(cfg: ModelConfig, batch: int, prompt: int, past: int = 0,
                     ubatch: int = 512) -> list[Op]:
    """Prompt processing split into micro-batches of `ubatch` tokens.

    Long prompts are processed one sequence-chunk at a time; short prompts are
    packed several sequences per micro-batch.
    """
    ops: list[Op] = []
    if batch == 1 or prompt >= ubatch:
        for start in range(0, prompt, ubatch):
            n = min(ubatch, prompt - start)
            last = start + n == prompt
            chunk = forward_ops(cfg, 1, n, past + start, logits=last)
            ops += [replace(o, count=o.count * batch) for o in chunk]
        return ops
    per_pass = ubatch // prompt
    full, rem = divmod(batch, per_pass)
    ops += [replace(o, count=o.count * full) for o in forward_ops(cfg, per_pass, prompt, past)]
    if rem:
        ops += forward_ops(cfg, rem, prompt, past)
    return ops


def decode_run_ops(cfg: ModelConfig, batch: int, start_ctx: int, steps: int) -> list[Op]:
    ops: list[Op] = []
    for i in range(steps):
        ops += decode_ops(cfg, batch, start_ctx + i)
    return ops


# --- vectorized evaluation (same formula as predict.op_time) ------------------

class CompiledCases:
    """Flatten every case's op list into arrays so one efficiency setting can be
    evaluated over all cases in a few numpy operations (needed for fitting)."""

    def __init__(self, op_lists: list[list[Op]]):
        rows, mm, vec, byt, cnt = [], [], [], [], []
        for i, ops in enumerate(op_lists):
            for o in ops:
                rows.append(i)
                mm.append(0.0 if o.kind == ELEMENTWISE else o.flops)
                vec.append(o.flops if o.kind == ELEMENTWISE else 0.0)
                byt.append(o.bytes)
                cnt.append(o.count)
        self.n = len(op_lists)
        self.row = np.array(rows)
        self.mm, self.vec = np.array(mm), np.array(vec)
        self.bytes, self.count = np.array(byt), np.array(cnt, dtype=float)

    def times(self, gpu: GPU, eff: Efficiency) -> np.ndarray:
        t_c = (self.mm / gpu.matmul_flops + self.vec / gpu.vector_flops) / eff.compute
        t_m = self.bytes / (gpu.mem_bw * eff.memory)
        t = (np.maximum(t_c, t_m) + eff.kernel_overhead_s) * self.count
        return np.bincount(self.row, weights=t, minlength=self.n)


# --- loading measurements -----------------------------------------------------

def _flash_attn_on(v) -> bool:
    if isinstance(v, (int, float, np.number)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "on", "yes", "enabled"}


def load_cases(meas_dir: Path) -> pd.DataFrame:
    """One row per measured throughput number, with the workload it describes."""
    meas_dir = Path(meas_dir)
    cases = []

    lb_path = meas_dir / "llama_bench.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        for _, r in lb.iterrows():
            depth = int(r.get("n_depth", 0) or 0)
            ub = int(r.get("n_ubatch", 512))
            fa = _flash_attn_on(r.get("flash_attn", 1))
            if r.n_prompt > 0 and r.n_gen == 0:
                cases.append(dict(source="llama-bench", phase="prefill", batch=1,
                                  prompt=int(r.n_prompt), depth=depth, gen=0, ubatch=ub,
                                  flash_attn=fa, measured_tok_s=float(r.avg_ts),
                                  measured_std=float(r.get("stddev_ts", np.nan))))
            elif r.n_gen > 0 and r.n_prompt == 0:
                cases.append(dict(source="llama-bench", phase="decode", batch=1,
                                  prompt=0, depth=depth, gen=int(r.n_gen), ubatch=ub,
                                  flash_attn=fa, measured_tok_s=float(r.avg_ts),
                                  measured_std=float(r.get("stddev_ts", np.nan))))

    bb_path = meas_dir / "batched_bench.csv"
    if bb_path.exists():
        bb = pd.read_csv(bb_path)
        for _, r in bb.iterrows():
            ub = int(r.get("n_ubatch", 512))
            fa = _flash_attn_on(r.get("flash_attn", 1))
            base = dict(source="batched-bench", batch=int(r.b), ubatch=ub, flash_attn=fa,
                        measured_std=np.nan)
            cases.append({**base, "phase": "prefill", "prompt": int(r.pp), "depth": 0,
                          "gen": 0, "measured_tok_s": float(r.s_pp)})
            cases.append({**base, "phase": "decode", "prompt": 0, "depth": int(r.pp),
                          "gen": int(r.tg), "measured_tok_s": float(r.s_tg)})
    if not cases:
        raise FileNotFoundError(f"no llama_bench.csv or batched_bench.csv in {meas_dir}")
    return pd.DataFrame(cases)


def case_ops(cfg: ModelConfig, case) -> list[Op]:
    if case.phase == "prefill":
        return prefill_pass_ops(cfg, case.batch, case.prompt, case.depth, case.ubatch)
    return decode_run_ops(cfg, case.batch, case.depth, case.gen)


def case_tokens(case) -> int:
    return case.batch * (case.prompt if case.phase == "prefill" else case.gen)


def microbench_efficiency(meas_dir: Path, gpu: GPU, kernel_overhead_s: float) -> Efficiency | None:
    """Efficiency from torch-measured bandwidth and matmul throughput (no fitting)."""
    path = Path(meas_dir) / "microbench.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path).set_index("metric")["value"]
    bw = max(m.get("read_bw", 0), m.get("copy_bw", 0))
    flops = m.get("matmul_flops_sustained", m.get("matmul_flops_burst", 0))
    if not bw or not flops:
        return None
    return Efficiency(compute=min(flops / gpu.matmul_flops, 1.0),
                      memory=min(bw / gpu.mem_bw, 1.0), kernel_overhead_s=kernel_overhead_s)


def load_env(meas_dir: Path) -> dict:
    path = Path(meas_dir) / "env.json"
    return json.loads(path.read_text()) if path.exists() else {}


# --- prediction, error, calibration -------------------------------------------

def predict_cases(cases: pd.DataFrame, compiled: CompiledCases, gpu: GPU,
                  eff: Efficiency) -> np.ndarray:
    tokens = np.array([case_tokens(c) for c in cases.itertuples()], dtype=float)
    return tokens / compiled.times(gpu, eff)


def log_rmse(pred: np.ndarray, meas: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.log(pred / meas) ** 2)))


def fit_efficiency(cases: pd.DataFrame, compiled: CompiledCases, gpu: GPU,
                   mask: np.ndarray | None = None) -> Efficiency:
    """Coarse-to-fine grid search for (compute, memory, overhead) minimizing the
    RMS log error of tokens/s over the cases selected by `mask`."""
    meas = cases.measured_tok_s.to_numpy()
    mask = np.ones(len(cases), bool) if mask is None else mask
    tokens = np.array([case_tokens(c) for c in cases.itertuples()], dtype=float)

    def loss(ec, em, oh):
        t = compiled.times(gpu, Efficiency(ec, em, oh))
        return log_rmse(tokens[mask] / t[mask], meas[mask])

    lo = np.array([0.05, 0.30, 0.0])
    hi = np.array([1.00, 1.00, 30e-6])
    best = None
    for _ in range(4):
        grids = [np.linspace(a, b, 11) for a, b in zip(lo, hi)]
        for ec in grids[0]:
            for em in grids[1]:
                for oh in grids[2]:
                    val = loss(ec, em, oh)
                    if best is None or val < best[0]:
                        best = (val, ec, em, oh)
        step = (hi - lo) / 10
        center = np.array(best[1:])
        lo = np.maximum(center - step, [0.01, 0.05, 0.0])
        hi = np.minimum(center + step, [1.0, 1.0, 100e-6])
    _, ec, em, oh = best
    return Efficiency(compute=float(ec), memory=float(em), kernel_overhead_s=float(oh))


def error_summary(df: pd.DataFrame, profiles: list[str]) -> pd.DataFrame:
    """Mean absolute % error and mean signed % error by source x phase x profile."""
    out = []
    for (src, phase), g in df.groupby(["source", "phase"]):
        for p in profiles:
            err = g[f"err_{p}"]
            out.append({"source": src, "phase": phase, "profile": p, "n": len(g),
                        "mape_pct": 100 * err.abs().mean(),
                        "bias_pct": 100 * err.mean(),
                        "worst_pct": 100 * err.loc[err.abs().idxmax()]})
    return pd.DataFrame(out)
