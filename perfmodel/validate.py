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
from scipy.optimize import minimize

from .config import ModelConfig
from .hardware import GPU, Efficiency
from .ops import ATTENTION, ELEMENTWISE, Op, decode_ops, forward_ops
from .predict import attention_mem_eff, combine


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
        rows, mm, vec, byt, cnt, attn, par = [], [], [], [], [], [], []
        for i, ops in enumerate(op_lists):
            for o in ops:
                rows.append(i)
                mm.append(0.0 if o.kind == ELEMENTWISE else o.flops)
                vec.append(o.flops if o.kind == ELEMENTWISE else 0.0)
                byt.append(o.bytes)
                cnt.append(o.count)
                attn.append(o.kind == ATTENTION)
                par.append(o.parallelism)
        self.n = len(op_lists)
        self.row = np.array(rows)
        self.mm, self.vec = np.array(mm), np.array(vec)
        self.bytes, self.count = np.array(byt), np.array(cnt, dtype=float)
        self.attn, self.par = np.array(attn), np.array(par, dtype=float)

    def times(self, gpu: GPU, eff: Efficiency) -> np.ndarray:
        t_c = (self.mm / gpu.matmul_flops + self.vec / gpu.vector_flops) / eff.compute
        mem_eff = np.where(self.attn, attention_mem_eff(eff, self.par), eff.memory)
        t_m = self.bytes / (gpu.mem_bw * mem_eff)
        t = (combine(t_c, t_m, eff.overlap) + eff.kernel_overhead_s) * self.count
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


FIT_BOUNDS = {
    "compute": (0.02, 1.0),
    "memory": (0.3, 1.0),
    "attention_memory": (0.02, 1.0),
    "attention_half_work": (0.1, 2000.0),
    "kernel_overhead_s": (0.0, 50e-6),
    "overlap": (1.0, 16.0),
}


def fit_efficiency(cases: pd.DataFrame, compiled: CompiledCases, gpu: GPU,
                   mask: np.ndarray | None = None,
                   free=("compute", "memory", "kernel_overhead_s"),
                   base: Efficiency = Efficiency()) -> Efficiency:
    """Fit the `free` Efficiency fields (others taken from `base`) to minimize the
    RMS log error of tokens/s over the cases selected by `mask`.

    Bounded Powell search in [0, 1]-normalized coordinates, from several starts.
    """
    meas = cases.measured_tok_s.to_numpy()
    mask = np.ones(len(cases), bool) if mask is None else mask
    tokens = np.array([case_tokens(c) for c in cases.itertuples()], dtype=float)
    lo = np.array([FIT_BOUNDS[f][0] for f in free])
    hi = np.array([FIT_BOUNDS[f][1] for f in free])

    def to_eff(u):
        return replace(base, **dict(zip(free, (lo + np.clip(u, 0, 1) * (hi - lo)).tolist())))

    def loss(u):
        t = compiled.times(gpu, to_eff(u))
        return log_rmse(tokens[mask] / t[mask], meas[mask])

    starts = [np.full(len(free), s) for s in (0.2, 0.5, 0.8)]
    best = min((minimize(loss, u0, method="Powell", bounds=[(0, 1)] * len(free),
                         options={"xtol": 1e-4, "ftol": 1e-7}) for u0 in starts),
               key=lambda r: r.fun)
    return to_eff(best.x)


def flag_anomalies(cases: pd.DataFrame) -> pd.Series:
    """Rule-based flags for measurements no roofline-style model can reproduce.

    - cold start: a batched-bench prefill at least 2x slower than llama-bench on
      the identical workload (batch 1, same prompt, depth 0).
    - non-monotonic decode: a batched decode step slower than the step at the
      next larger batch for the same prompt. More work cannot take less time
      under any additive cost model, so this points to a kernel-dispatch change.
    """
    flags = pd.Series("", index=cases.index)
    ref = cases[(cases.source == "llama-bench") & (cases.phase == "prefill") & (cases.depth == 0)]
    ref = ref.groupby("prompt").measured_tok_s.mean()
    bb = cases.source == "batched-bench"
    for i, c in cases[bb & (cases.phase == "prefill") & (cases.batch == 1)].iterrows():
        if c.prompt in ref.index and c.measured_tok_s < 0.5 * ref[c.prompt]:
            flags[i] = "cold start"
    dec = cases[bb & (cases.phase == "decode")].copy()
    dec["step_s"] = dec.batch / dec.measured_tok_s
    for _, g in dec.groupby("depth"):
        g = g.sort_values("batch")
        slower = g.step_s.to_numpy()[:-1] > g.step_s.to_numpy()[1:]
        for i in g.index[:-1][slower]:
            flags[i] = "non-monotonic decode"
    return flags


def gpu_log_summary(meas_dir: Path, busy_util: float = 50) -> dict:
    """Clock / power behaviour while the GPU was busy (nvidia-smi log)."""
    path = Path(meas_dir) / "gpu_log.csv"
    if not path.exists():
        return {}
    g = pd.read_csv(path, skipinitialspace=True)
    num = lambda col: g[col].astype(str).str.extract(r"([\d.]+)")[0].astype(float)
    cols = {c.split(" ")[0]: c for c in g.columns}
    sm, util = num(cols["clocks.current.sm"]), num(cols["utilization.gpu"])
    busy = util >= busy_util
    out = {"busy_samples": int(busy.sum()),
           "sm_clock_median_mhz": float(sm[busy].median()),
           "power_median_w": float(num(cols["power.draw"])[busy].median()),
           "temp_max_c": float(num(cols["temperature.gpu"]).max())}
    reason_col = next((c for c in g.columns if "reasons" in c), None)
    if reason_col:
        # bit 0x4 = software power cap
        bits = g[reason_col][busy].astype(str).apply(lambda s: int(s, 16) if s.startswith("0x") else 0)
        out["power_capped_frac"] = float(((bits & 0x4) > 0).mean())
    return out


def error_summary(df: pd.DataFrame, profiles: list[str], by=("split", "phase")) -> pd.DataFrame:
    """Mean absolute % error and mean signed % error per group and profile."""
    out = []
    for keys, g in df.groupby(list(by)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        for p in profiles:
            err = g[f"err_{p}"]
            out.append({**dict(zip(by, keys)), "profile": p, "n": len(g),
                        "mape_pct": 100 * err.abs().mean(),
                        "bias_pct": 100 * err.mean(),
                        "worst_pct": 100 * err.loc[err.abs().idxmax()]})
    return pd.DataFrame(out)
