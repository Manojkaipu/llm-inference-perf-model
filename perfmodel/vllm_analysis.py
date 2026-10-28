"""Turn vLLM serving measurements (notebooks/vllm_serving_benchmark.ipynb) into
summaries, a fitted vLLM efficiency profile, and model-vs-measured comparisons.

Calibration uses only the continuous-batching sweep. Prefix-caching TTFT and
speculative-decoding speedup are then predicted out-of-sample.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DRAFT_MODELS, ModelConfig
from .hardware import GPU, PEAK, Efficiency
from .serving import (alpha_from_acceptance_rate, breakeven_acceptance, continuous_step_ops,
                      expected_tokens_per_step, predict_continuous, prefill_time,
                      simulate_closed_loop, spec_speedup, spec_step_times)
from .validate import CompiledCases, fit_efficiency


def _tpot(df: pd.DataFrame) -> pd.Series:
    """Per-request time per output token after the first."""
    return (df.e2e_s - df.ttft_s) / (df.out_tokens - 1)


def load(meas_dir: Path) -> dict:
    meas_dir = Path(meas_dir)
    read = lambda name: pd.read_csv(meas_dir / name) if (meas_dir / name).exists() else None
    data = {"env": json.loads((meas_dir / "env.json").read_text()),
            "micro": read("microbench.csv"), "cb": read("cb_requests.csv"),
            "cb_levels": read("cb_levels.csv")}
    pcs = [read(f"prefix_{s}.csv") for s in ("on", "off")]
    data["prefix"] = pd.concat([p for p in pcs if p is not None], ignore_index=True) if any(
        p is not None for p in pcs) else None
    reqs, levels = [], []
    for path in sorted(meas_dir.glob("spec_*.csv")):
        (levels if path.stem.endswith("_levels") else reqs).append(pd.read_csv(path))
    data["spec"] = pd.concat(reqs, ignore_index=True) if reqs else None
    data["spec_levels"] = pd.concat(levels, ignore_index=True) if levels else None
    return data


def memory_efficiency(micro: pd.DataFrame | None, gpu: GPU, default: float = 0.85) -> float:
    if micro is None:
        return default
    m = micro.set_index("metric")["value"]
    bw = max(m.get("read_bw", 0), m.get("copy_bw", 0))
    return min(bw / gpu.mem_bw, 1.0) if bw else default


# --- 1. continuous batching -----------------------------------------------------

def summarize_cb(cb: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    cb = cb.assign(tpot_s=_tpot(cb))
    g = cb.groupby("concurrency")
    s = pd.DataFrame({
        "ttft_p50_s": g.ttft_s.median(), "ttft_p90_s": g.ttft_s.quantile(0.9),
        "tpot_p50_s": g.tpot_s.median(), "tpot_p90_s": g.tpot_s.quantile(0.9),
        "e2e_p50_s": g.e2e_s.median(), "e2e_p90_s": g.e2e_s.quantile(0.9),
    }).reset_index()
    lv = levels.set_index("concurrency")
    s["output_tok_s"] = (lv.out_tokens / lv.wall_s).reindex(s.concurrency).to_numpy()
    s["requests_per_s"] = (lv.n_requests / lv.wall_s).reindex(s.concurrency).to_numpy()
    # The counter only: vLLM also exports a per-request preemption histogram.
    s["preemptions"] = [
        sum(v for k, v in json.loads(m).items() if k.endswith("num_preemptions_total"))
        for m in lv.metrics.reindex(s.concurrency)]
    return s


def assign_waves(cb: pd.DataFrame, gap_s: float = 0.02) -> pd.DataFrame:
    """Group requests that started within `gap_s` of each other (per concurrency
    level) into arrival waves, with each request's rank inside its wave.

    With a closed-loop client and fixed output lengths, requests that start
    together finish together, so their replacements also arrive together."""
    out = []
    for _, g in cb.sort_values(["concurrency", "t_start"]).groupby("concurrency"):
        wave = np.concatenate([[0], np.cumsum(np.diff(g.t_start.to_numpy()) > gap_s)])
        g = g.assign(wave=wave)
        g["rank"] = g.groupby("wave").cumcount()
        g["wave_size"] = g.groupby("wave").t_start.transform("size")
        out.append(g)
    return pd.concat(out)


def infer_prefill_budget(cb: pd.DataFrame, input_len: int, default: int = 2048) -> int:
    """Prefill tokens per engine step, read from the data: in the largest arrival
    wave, prompts prefilled in the same step get their first token together."""
    w = assign_waves(cb)
    big = w[w.wave_size == w.wave_size.max()]
    big = big[big.concurrency == big.concurrency.max()]
    t = np.sort((big.t_start + big.ttft_s).to_numpy())
    if len(t) < 8:
        return default
    sizes = [len(x) for x in np.split(t, np.where(np.diff(t) > 0.004)[0] + 1)][1:]
    return int(pd.Series(sizes).mode().iloc[0] * input_len) if sizes else default


def simulate_cb(cfg, gpu, eff, summary: pd.DataFrame, cb: pd.DataFrame, input_len: int,
                output_len: int, budget: int, max_seqs: int) -> pd.DataFrame:
    """Add scheduler-simulation predictions (same concurrency and request count as
    measured) to the per-concurrency summary."""
    out = summary.copy()
    rows = []
    for c in out.concurrency:
        n = int((cb.concurrency == c).sum())
        sim = pd.DataFrame(simulate_closed_loop(cfg, gpu, eff, int(c), n, input_len,
                                                output_len, budget, max_seqs))
        rows.append({"sim_ttft_p50_s": sim.ttft_s.median(), "sim_ttft_p90_s": sim.ttft_s.quantile(0.9),
                     "sim_tpot_p50_s": sim.tpot_s.median(), "sim_e2e_p50_s": sim.e2e_s.median(),
                     "sim_output_tok_s": n * output_len / sim.t_end.max()})
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def fit_vllm_profile(cfg: ModelConfig, gpu: GPU, summary: pd.DataFrame, input_len: int,
                     output_len: int, memory_eff: float) -> Efficiency:
    """Fit compute efficiency, per-kernel overhead and compute/memory overlap to the
    measured median TPOT at each concurrency (memory efficiency pinned to the
    measured bandwidth)."""
    cases = pd.DataFrame({"tokens": summary.concurrency.astype(float),
                          "measured_tok_s": summary.concurrency / summary.tpot_p50_s})
    compiled = CompiledCases([continuous_step_ops(cfg, c, input_len, output_len)
                              for c in summary.concurrency])
    return fit_efficiency(cases, compiled, gpu,
                          free=("compute", "kernel_overhead_s", "overlap"),
                          base=replace(PEAK, memory=memory_eff))


def predict_cb(cfg, gpu, eff, summary: pd.DataFrame, input_len: int, output_len: int,
               prefix: str = "pred") -> pd.DataFrame:
    out = summary.copy()
    preds = [predict_continuous(cfg, gpu, eff, c, input_len, output_len)
             for c in out.concurrency]
    out[f"{prefix}_tpot_s"] = [p["tpot_s"] for p in preds]
    out[f"{prefix}_ttft_s"] = [p["ttft_s"] for p in preds]
    out[f"{prefix}_output_tok_s"] = [p["output_tok_s"] for p in preds]
    return out


# --- 2. prefix caching ------------------------------------------------------------

def summarize_prefix(pc: pd.DataFrame) -> pd.DataFrame:
    m = pc[~pc.warm.astype(bool)]
    g = m.groupby(["config", "prefix_len"])
    return pd.DataFrame({"ttft_p50_s": g.ttft_s.median(), "ttft_min_s": g.ttft_s.min(),
                         "cached_tokens_p50": g.cached_tokens.median(),
                         "prompt_len": g.prompt_len.first(), "n": g.size()}).reset_index()


def predict_prefix(cfg, gpu, eff, summary: pd.DataFrame) -> pd.DataFrame:
    """Predicted prefill compute for the uncached tokens. The residual (measured -
    predicted) is the per-request overhead the model does not include (HTTP,
    tokenization-free scheduling, first sampling); if the model captures the
    prefix effect it is roughly constant across prefix lengths."""
    out = summary.copy()
    cached = np.where(out.config == "caching_on", out.prefix_len, 0)
    out["pred_compute_s"] = [prefill_time(cfg, gpu, eff, int(L), int(c))
                             for L, c in zip(out.prompt_len, cached)]
    out["residual_s"] = out.ttft_p50_s - out.pred_compute_s
    return out


# --- 3. speculative decoding -------------------------------------------------------

def acceptance_from_metrics(metrics_json: str) -> dict:
    """Draft counts from vLLM's spec-decode counters (names vary slightly by version)."""
    m = json.loads(metrics_json) if isinstance(metrics_json, str) else {}
    pick = lambda pred: sum(v for k, v in m.items() if pred(k))
    drafts = pick(lambda k: "spec_decode" in k and "num_drafts" in k)
    draft_tokens = pick(lambda k: "spec_decode" in k and "num_draft_tokens" in k)
    accepted = pick(lambda k: "spec_decode" in k and "num_accepted_tokens" in k
                    and "per_pos" not in k)
    return {"drafts": drafts, "draft_tokens": draft_tokens, "accepted": accepted,
            "acceptance_rate": accepted / draft_tokens if draft_tokens else np.nan,
            "tokens_per_step": 1 + accepted / drafts if drafts else np.nan}


def spec_config_info(env: dict) -> dict:
    """name -> (draft repo or None, k, method) from the notebook's recorded configs."""
    defs = {**(env.get("sd", {}).get("configs") or {}), **(env.get("sd", {}).get("ngram") or {})}
    info = {}
    for name, sc in defs.items():
        if sc is None:
            continue
        info[name] = (sc.get("model"), int(sc["num_speculative_tokens"]), sc.get("method", "draft_model"))
    return info


def summarize_spec(spec: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    spec = spec.assign(tpot_s=_tpot(spec))
    keys = ["config", "prompt_type", "concurrency"]
    s = spec.groupby(keys).agg(tpot_p50_s=("tpot_s", "median"), n=("tpot_s", "size")).reset_index()
    acc = pd.DataFrame([{**{k: r[k] for k in keys}, **acceptance_from_metrics(r["metrics"]),
                         "output_tok_s": r["out_tokens"] / r["wall_s"]}
                        for _, r in levels.iterrows()])
    s = s.merge(acc, on=keys, how="left")
    base = s[s.config == "baseline"].set_index(["prompt_type", "concurrency"])
    idx = pd.MultiIndex.from_frame(s[["prompt_type", "concurrency"]])
    s["baseline_tpot_s"] = base.tpot_p50_s.reindex(idx).to_numpy()
    s["baseline_output_tok_s"] = base.output_tok_s.reindex(idx).to_numpy()
    s["speedup"] = s.baseline_tpot_s / s.tpot_p50_s
    s["throughput_speedup"] = s.output_tok_s / s.baseline_output_tok_s
    # Cost of one engine step: speculative steps emit tokens_per_step tokens each.
    s["step_s"] = np.where(s.config == "baseline", s.tpot_p50_s, s.tpot_p50_s * s.tokens_per_step)
    return s


def predict_spec_from_step_cost(summary: pd.DataFrame) -> pd.DataFrame:
    """Speedup predicted from measured tokens per step and the step costs measured
    on the *other* prompt types (leave one prompt type out). This isolates how
    speedup depends on acceptance from how well the draft cost is modeled."""
    out = summary.copy()
    preds = []
    for r in out.itertuples():
        same = out[(out.concurrency == r.concurrency) & (out.prompt_type != r.prompt_type)]
        spec_step = same[same.config == r.config].step_s.median()
        base_step = same[same.config == "baseline"].step_s.median()
        preds.append(np.nan if r.config == "baseline" else r.tokens_per_step * base_step / spec_step)
    out["pred_speedup_step_cost"] = preds
    return out


def predict_spec(target: ModelConfig, gpu: GPU, eff: Efficiency, summary: pd.DataFrame,
                 info: dict, ctx: int) -> pd.DataFrame:
    """Predicted speedup from the measured tokens per step and the model's draft,
    verify and target step costs."""
    out = summary.copy()
    preds = []
    for r in out.itertuples():
        if r.config not in info or pd.isna(r.tokens_per_step):
            preds.append(np.nan)
            continue
        draft_repo, k, method = info[r.config]
        t = _spec_times(target, draft_repo, gpu, eff, k, int(r.concurrency), ctx)
        preds.append(spec_speedup(t, k, r.tokens_per_step))
    out["pred_speedup"] = preds
    return out


def _spec_times(target, draft_repo, gpu, eff, k, batch, ctx) -> dict:
    if draft_repo in DRAFT_MODELS:
        return spec_step_times(target, DRAFT_MODELS[draft_repo], gpu, eff, k, batch, ctx)
    # n-gram (prompt lookup) drafting runs on the CPU: no draft model cost
    t = spec_step_times(target, target, gpu, eff, k, batch, ctx)
    return {**t, "draft_step_s": 0.0, "spec_step_s": t["verify_step_s"]}


def spec_curve(target, gpu, eff, info: dict, name: str, batch: int, ctx: int,
               rates=np.linspace(0.0, 0.98, 50), times: dict | None = None) -> pd.DataFrame:
    """Speedup across draft acceptance rates (i.i.d. per-token acceptance), from the
    model's step costs or from measured ones passed as `times`."""
    draft_repo, k, _ = info[name]
    t = times or _spec_times(target, draft_repo, gpu, eff, k, batch, ctx)
    rows = []
    for rate in rates:
        alpha = alpha_from_acceptance_rate(rate, k)
        rows.append({"acceptance_rate": rate,
                     "speedup": spec_speedup(t, k, expected_tokens_per_step(alpha, k))})
    return pd.DataFrame(rows)


def breakeven_rate(target, gpu, eff, info: dict, name: str, batch: int, ctx: int) -> float:
    """Draft acceptance rate below which speculation slows decoding (model costs)."""
    draft_repo, k, _ = info[name]
    t = _spec_times(target, draft_repo, gpu, eff, k, batch, ctx)
    return breakeven_acceptance(t["spec_step_s"], t["target_step_s"], k)


def measured_step_times(summary: pd.DataFrame, name: str, batch: int) -> dict:
    """Median measured step costs for a config, in the shape spec_curve expects."""
    at = summary[summary.concurrency == batch]
    return {"target_step_s": at[at.config == "baseline"].step_s.median(),
            "spec_step_s": at[at.config == name].step_s.median()}
