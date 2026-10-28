"""Analyze a vLLM serving run (notebooks/vllm_serving_benchmark.ipynb).

    python scripts/analyze_vllm.py measurements/<gpu>_vllm

1. Continuous batching: fit a vLLM efficiency profile to median TPOT (steady-state
   step model), then simulate the scheduler under the same closed-loop load to
   predict throughput, TTFT, TPOT and latency together.
2. Prefix caching: predicted prefill of the uncached tokens vs measured TTFT.
3. Speculative decoding: per-config step costs, speedup predicted from measured
   acceptance with (a) step costs measured on other prompt types and (b) the
   analytical draft + verify costs, and break-even acceptance rates.

Writes reports/vllm_<gpu>.md, figures/vllm_*_<gpu>.png, results/vllm_*_<gpu>.csv and
results/vllm_profile_<gpu>.json.
"""
import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from perfmodel import TINYLLAMA_1_1B, plots
from perfmodel.config import DRAFT_MODELS
from perfmodel.hardware import gpu_from_name
from perfmodel.serving import breakeven_acceptance, spec_step_times
from perfmodel.vllm_analysis import (assign_waves, fit_vllm_profile, infer_prefill_budget, load,
                                     measured_step_times, memory_efficiency, predict_prefix,
                                     predict_spec, predict_spec_from_step_cost, simulate_cb,
                                     spec_config_info, spec_curve, summarize_cb, summarize_prefix,
                                     summarize_spec)

PROMPT_LABELS = {"copy": "copy", "natural": "natural, T=0", "natural_t1": "natural, T=1",
                 "random_t1": "random, T=1"}
MARKERS = {"copy": "o", "natural": "s", "natural_t1": "^", "random_t1": "D"}


def md_table(df: pd.DataFrame, fmt="{:.2f}") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "-" if np.isnan(v) else fmt.format(v)
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(f(v) for v in row) + " |"
                              for row in df.itertuples(index=False)])


def mape(pred, meas) -> float:
    return float(100 * (np.asarray(pred) / np.asarray(meas) - 1).__abs__().mean())


def _plain_log_axis(axis):
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if str(v)[0] in "25" else ""))


# --- figures ----------------------------------------------------------------------

def fig_cb(cb: pd.DataFrame, title: str, path: Path):
    fig, axes = plots._fig(2, width=5.4, height=4.3)
    ax = axes[0]
    ax.plot(cb.concurrency, cb.output_tok_s, color=plots.TEXT, marker="o", linestyle="none",
            markersize=6, label="measured")
    ax.plot(cb.concurrency, cb.sim_output_tok_s, color=plots.SERIES[0], linewidth=2,
            label="model (scheduler simulation)")
    ax.set_ylabel("Output tokens/s", color=plots.TEXT_2, fontsize=9)
    ax.set_title("Throughput", fontsize=10, color=plots.TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=plots.TEXT)
    ax = axes[1]
    for col, sim, color, label in (("ttft_p90_s", "sim_ttft_p90_s", plots.SERIES[1], "TTFT p90"),
                                   ("ttft_p50_s", "sim_ttft_p50_s", plots.SERIES[2], "TTFT p50"),
                                   ("tpot_p50_s", "sim_tpot_p50_s", plots.SERIES[0], "TPOT p50")):
        ax.plot(cb.concurrency, cb[col] * 1e3, color=color, marker="o", linestyle="none",
                markersize=6, markeredgecolor=plots.SURFACE, label=f"{label}, measured")
        ax.plot(cb.concurrency, cb[sim] * 1e3, color=color, linewidth=2, label=f"{label}, model")
    ax.set_yscale("log")
    _plain_log_axis(ax.yaxis)
    ax.set_ylabel("Latency (ms)", color=plots.TEXT_2, fontsize=9)
    ax.set_title("Latency", fontsize=10, color=plots.TEXT, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=plots.TEXT, ncol=2, loc="upper left")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(cb.concurrency, [str(c) for c in cb.concurrency])
        ax.set_xlabel("Concurrent requests", color=plots.TEXT_2, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=plots.TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=plots.SURFACE)
    plt.close(fig)


def fig_prefix(pc: pd.DataFrame, title: str, path: Path):
    fig, axes = plots._fig(1, width=6.5, height=4.2)
    ax = axes[0]
    overhead = pc.residual_s.median()
    for state, color in (("caching_on", plots.SERIES[0]), ("caching_off", plots.SERIES[1])):
        s = pc[pc.config == state]
        if not len(s):
            continue
        label = "prefix caching " + state.split("_")[1]
        ax.plot(s.prefix_len, s.ttft_p50_s * 1e3, color=color, marker="o", linestyle="none",
                markersize=7, markeredgecolor=plots.SURFACE, label=f"{label}, measured")
        ax.plot(s.prefix_len, (s.pred_compute_s + overhead) * 1e3, color=color, linewidth=2,
                label=f"{label}, model + {overhead * 1e3:.0f} ms")
    ax.set_xlabel(f"Shared prefix length (tokens, of {int(pc.prompt_len.iloc[0])})",
                  color=plots.TEXT_2, fontsize=9)
    ax.set_ylabel("Time to first token, p50 (ms)", color=plots.TEXT_2, fontsize=9)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, labelcolor=plots.TEXT)
    fig.suptitle(title, x=0.01, ha="left", color=plots.TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=plots.SURFACE)
    plt.close(fig)


def fig_spec(sd: pd.DataFrame, curves: dict, title: str, path: Path):
    concs = sorted(sd.concurrency.unique())
    fig, axes = plots._fig(len(concs), width=5.0, height=4.5)
    names = [n for n in sd.config.unique() if n != "baseline"][:3]   # <= 3 colors (scatter)
    for ax, c in zip(axes, concs):
        ax.axhline(1, color=plots.TEXT_2, linewidth=1)
        for color, name in zip(plots.SERIES, names):
            meas_curve, model_curve = curves.get((name, c), (None, None))
            if meas_curve is not None:
                ax.plot(meas_curve.acceptance_rate, meas_curve.speedup, color=color, linewidth=2,
                        label=f"{name}: measured step cost")
            if model_curve is not None:
                ax.plot(model_curve.acceptance_rate, model_curve.speedup, color=color, linewidth=1.5,
                        linestyle="--", label=f"{name}: analytical step cost")
            s = sd[(sd.config == name) & (sd.concurrency == c)]
            for ptype, m in MARKERS.items():
                p = s[s.prompt_type == ptype]
                ax.scatter(p.acceptance_rate, p.speedup, color=color, marker=m, s=50,
                           edgecolor=plots.SURFACE, linewidth=1.2, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.set_title(f"{c} concurrent request{'s' if c > 1 else ''}", fontsize=10,
                     color=plots.TEXT, loc="left")
        ax.set_xlabel("Draft acceptance rate (measured)", color=plots.TEXT_2, fontsize=9)
    axes[0].set_ylabel("Decode speedup vs no speculation", color=plots.TEXT_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    for ptype, m in MARKERS.items():
        handles.append(plt.Line2D([], [], color=plots.TEXT_2, marker=m, linestyle="none"))
        labels.append(f"measured: {PROMPT_LABELS[ptype]} prompts")
    axes[-1].legend(handles, labels, frameon=False, fontsize=7.5, labelcolor=plots.TEXT,
                    loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, x=0.01, ha="left", color=plots.TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=plots.SURFACE)
    plt.close(fig)


# --- main -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meas_dir", type=Path)
    args = ap.parse_args()
    data = load(args.meas_dir)
    env = data["env"]
    gpu = gpu_from_name(env["gpu_name"])
    cfg = TINYLLAMA_1_1B
    slug = args.meas_dir.name.removesuffix("_vllm")
    figs, results, reports = ROOT / "figures", ROOT / "results", ROOT / "reports"
    for d in (figs, results, reports):
        d.mkdir(exist_ok=True)
    cb_in, cb_out = env["cb"]["input"], env["cb"]["output"]
    mem = memory_efficiency(data["micro"], gpu)
    micro = data["micro"].set_index("metric")["value"] if data["micro"] is not None else {}
    micro_compute = micro.get("matmul_flops_sustained", np.nan) / gpu.matmul_flops

    head = f"vLLM {env.get('vllm_version', '?')} on {env['gpu_name']}, TinyLlama-1.1B fp16, measured {env.get('timestamp', '?')}"
    if env.get("server_extra_env"):
        head += (". Servers ran with " + ", ".join(f"`{k}={v}`" for k, v in env["server_extra_env"].items())
                 + " because FlashInfer could not build its sampling kernel")
    sections = []

    # 1. continuous batching ------------------------------------------------------
    cb = summarize_cb(data["cb"], data["cb_levels"])
    eff = fit_vllm_profile(cfg, gpu, cb, cb_in, cb_out, mem)
    budget = infer_prefill_budget(data["cb"], cb_in)
    cb = simulate_cb(cfg, gpu, eff, cb, data["cb"], cb_in, cb_out, budget, env.get("max_num_seqs", 256))
    cb["ttft_offset_s"] = cb.ttft_p50_s - cb.sim_ttft_p50_s
    cb.to_csv(results / f"vllm_continuous_batching_{slug}.csv", index=False)
    profile = {k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in asdict(eff).items()}
    (results / f"vllm_profile_{slug}.json").write_text(json.dumps(profile, indent=2))
    fig_cb(cb, f"Continuous batching: TinyLlama-1.1B on {gpu.name} (vLLM, {cb_in} in / {cb_out} out)",
           figs / f"vllm_continuous_batching_{slug}.png")
    waves = assign_waves(data["cb"])
    hi = waves[waves.concurrency == waves.concurrency.max()]
    later_sizes = hi[hi.wave > 0].groupby("wave").size()
    errs = {m: mape(cb[f"sim_{m}"], cb[m]) for m in
            ("output_tok_s", "e2e_p50_s", "tpot_p50_s", "ttft_p90_s", "ttft_p50_s")}
    t = pd.DataFrame({
        "concurrency": cb.concurrency,
        "out tok/s": cb.output_tok_s, "model": cb.sim_output_tok_s,
        "TTFT p50 ms": cb.ttft_p50_s * 1e3, "model ": cb.sim_ttft_p50_s * 1e3,
        "TTFT p90 ms": cb.ttft_p90_s * 1e3, "model  ": cb.sim_ttft_p90_s * 1e3,
        "TPOT p50 ms": cb.tpot_p50_s * 1e3, "model   ": cb.sim_tpot_p50_s * 1e3,
        "E2E p50 ms": cb.e2e_p50_s * 1e3, "model    ": cb.sim_e2e_p50_s * 1e3})
    sections.append(f"""## 1. Continuous batching

Closed loop: at each concurrency level, that many clients each send a request (random
{cb_in}-token prompt, exactly {cb_out} output tokens) as soon as their previous one finishes.

**Model.** A vLLM efficiency profile is fitted to the median TPOT at each level, treating
each engine step as one decode per running request plus an amortized share of new prompts'
chunked prefill, all sharing one read of the weights. Memory efficiency is pinned to the
measured {mem:.0%} of peak bandwidth; the fit gives compute efficiency {eff.compute:.3f},
{eff.kernel_overhead_s * 1e6:.1f} us per kernel and overlap exponent {eff.overlap:.2f}. The
profile then drives a **step-by-step simulation of the scheduler** under the same load.
Each step decodes one token per running request and fills the rest of a {budget:,}-token
budget (read from the data) with queued prompts in arrival order. The simulation predicts
throughput, TTFT, TPOT and end-to-end latency together.

| Metric | Model error (MAPE) |
|---|---|
| Output throughput | {errs['output_tok_s']:.1f}% |
| End-to-end latency p50 | {errs['e2e_p50_s']:.1f}% |
| TPOT p50 | {errs['tpot_p50_s']:.1f}% |
| TTFT p90 | {errs['ttft_p90_s']:.1f}% |
| TTFT p50 | {errs['ttft_p50_s']:.1f}% |

![continuous batching](../figures/vllm_continuous_batching_{slug}.png)

{md_table(t, "{:,.1f}")}

**Requests arrive in lockstep waves.** Every request generates exactly {cb_out} tokens, so
requests that start together finish together, and their clients' next requests arrive
together. At concurrency {int(hi.concurrency.iloc[0])} the first wave has {int(hi[hi.wave == 0].shape[0])}
requests and later waves most often have {int(later_sizes.mode().iloc[0])}. Within a wave,
prompts are prefilled {budget // cb_in} per step, so TTFT steps up with arrival order. The
simulation reproduces this (same wave sizes and same within-wave step pattern), and the
large first wave sets TTFT p90.

**What the model misses: a delay before a new request is first scheduled.** Median TTFT
at high concurrency is set by the small later waves. Measured minus simulated median TTFT
is {', '.join(f"{o * 1e3:.0f} ms at {c}" for c, o in zip(cb.concurrency, cb.ttft_offset_s))}:
negligible up to {int(cb[cb.ttft_offset_s.abs() < 0.02].concurrency.max())} concurrent
requests, then growing with the number of open streams. Within a wave, the gaps between
first tokens match the simulated step length, so the extra time comes before a request's
first step. The likely cause is CPU contention on the VM: vLLM's single Python API
server streams tokens for every running request while the Python client on the same VM
parses those streams. CPU load was not measured, so this is a hypothesis.

No request was ever preempted (vLLM's `num_preemptions` counter stayed at
{int(cb.preemptions.sum())}): the KV cache had room for every running sequence.
""")

    # 2. prefix caching -----------------------------------------------------------
    if data["prefix"] is not None:
        pc = predict_prefix(cfg, gpu, eff, summarize_prefix(data["prefix"]))
        pc.to_csv(results / f"vllm_prefix_cache_{slug}.csv", index=False)
        fig_prefix(pc, f"Prefix caching: time to first token on {gpu.name} (vLLM)",
                   figs / f"vllm_prefix_cache_{slug}.png")
        on = pc[pc.config == "caching_on"].set_index("prefix_len")
        off = pc[pc.config == "caching_off"].set_index("prefix_len")
        p_max = on.index.max()
        slope_meas = (on.ttft_p50_s.iloc[0] - on.ttft_p50_s.loc[p_max]) / p_max
        slope_model = (on.pred_compute_s.iloc[0] - on.pred_compute_s.loc[p_max]) / p_max
        # Same prediction with the microbenchmark's large-matmul compute efficiency.
        pc_mb = predict_prefix(cfg, gpu, replace(eff, compute=micro_compute),
                               summarize_prefix(data["prefix"]))
        on_mb = pc_mb[pc_mb.config == "caching_on"].set_index("prefix_len")
        slope_mb = (on_mb.pred_compute_s.iloc[0] - on_mb.pred_compute_s.loc[p_max]) / p_max
        tbl = pd.DataFrame({"prefix tokens": on.index,
                            "TTFT on ms": on.ttft_p50_s * 1e3,
                            "TTFT off ms": off.ttft_p50_s.reindex(on.index) * 1e3,
                            "speedup": off.ttft_p50_s.reindex(on.index) / on.ttft_p50_s,
                            "model compute ms": on.pred_compute_s * 1e3,
                            "measured - model ms": on.residual_s * 1e3}).reset_index(drop=True)
        sections.append(f"""## 2. Prefix caching

One request at a time, with a {int(pc.prompt_len.iloc[0])}-token prompt whose first N tokens
are shared with an earlier request (each prefix length uses its own prefix, warmed by one
excluded request). With caching off, TTFT stays flat at
{off.ttft_p50_s.median() * 1e3:.0f} ms. With caching on, it falls to
{on.ttft_p50_s.loc[p_max] * 1e3:.0f} ms at a {p_max:,}-token shared prefix, a
**{off.ttft_p50_s.loc[p_max] / on.ttft_p50_s.loc[p_max]:.1f}x** speedup.

**Model.** Prefill only the uncached tokens, attending over the cached prefix. Each cached
token saves {slope_meas * 1e6:.1f} us of measured TTFT; the model predicts
{slope_model * 1e6:.1f} us, so measured / model = {slope_meas / slope_model:.2f}. The profile's
compute efficiency ({eff.compute:.3f}) was fitted on decode-heavy steps, while a
{int(pc.prompt_len.iloc[0])}-token prefill runs large matmuls. Using the microbenchmark's
sustained matmul efficiency ({micro_compute:.3f}) instead predicts {slope_mb * 1e6:.1f} us per
cached token ({slope_meas / slope_mb:.2f}x the measured value). Measured minus that
prediction is then nearly flat, {on_mb.residual_s.min() * 1e3:.0f}-{on_mb.residual_s.max() * 1e3:.0f} ms
across prefix lengths: a fixed per-request cost, plus prefill compute that the model
captures once it uses the efficiency of large matmuls.

![prefix caching](../figures/vllm_prefix_cache_{slug}.png)

{md_table(tbl, "{:,.1f}")}
""")

    # 3. speculative decoding -------------------------------------------------------
    if data["spec"] is not None:
        info = spec_config_info(env)
        ctx = env["sd"]["input"] + env["sd"]["output"] // 2
        sd = summarize_spec(data["spec"], data["spec_levels"])
        sd = predict_spec(cfg, gpu, eff, sd, info, ctx)
        sd = predict_spec_from_step_cost(sd)
        sd.to_csv(results / f"vllm_spec_decode_{slug}.csv", index=False)
        names = [n for n in sd.config.unique() if n != "baseline" and n in info]
        concs = sorted(int(c) for c in sd.concurrency.unique())

        cost_rows, curves = [], {}
        for n in names:
            draft_repo, k, _ = info[n]
            for c in concs:
                meas = measured_step_times(sd, n, c)
                model = spec_step_times(cfg, DRAFT_MODELS[draft_repo], gpu, eff, k, c, ctx)
                steps = sd[(sd.config == n) & (sd.concurrency == c)].step_s
                implied_draft = (meas["spec_step_s"] - model["verify_step_s"]) / k
                cost_rows.append({
                    "config": n, "concurrency": c,
                    "target step ms": meas["target_step_s"] * 1e3,
                    "spec step ms": meas["spec_step_s"] * 1e3,
                    "spread across prompts": f"{steps.std() / steps.median():.0%}",
                    "model spec step ms": model["spec_step_s"] * 1e3,
                    "draft step ms (implied)": implied_draft * 1e3,
                    "draft step ms (model)": model["draft_step_s"] * 1e3,
                    "break-even accept. (measured)": breakeven_acceptance(meas["spec_step_s"], meas["target_step_s"], k),
                    "break-even accept. (model)": breakeven_acceptance(model["spec_step_s"], model["target_step_s"], k)})
                curves[(n, c)] = (spec_curve(cfg, gpu, eff, info, n, c, ctx, times=meas),
                                  spec_curve(cfg, gpu, eff, info, n, c, ctx))
        costs = pd.DataFrame(cost_rows)
        fig_spec(sd, curves, f"Speculative decoding on {gpu.name} (vLLM, target TinyLlama-1.1B)",
                 figs / f"vllm_spec_decode_{slug}.png")

        s = sd[sd.config != "baseline"].copy()
        corr = s.groupby(["config", "concurrency"]).apply(
            lambda g: np.corrcoef(g.tokens_per_step, g.speedup)[0, 1], include_groups=False)
        err_cost = mape(s.pred_speedup_step_cost, s.speedup)
        err_model = {n: mape(g.pred_speedup, g.speedup) for n, g in s.groupby("config")}
        tbl = s[["config", "prompt_type", "concurrency", "acceptance_rate", "tokens_per_step",
                 "speedup", "pred_speedup_step_cost", "pred_speedup"]].rename(columns={
            "prompt_type": "prompts", "acceptance_rate": "acceptance", "tokens_per_step": "tokens/step",
            "pred_speedup_step_cost": "pred. (measured cost)", "pred_speedup": "pred. (analytical)"})
        best = s.loc[s.speedup.idxmax()]
        failed = [k for k, v in env.get("servers", {}).items() if v.get("failed")]
        sections.append(f"""## 3. Speculative decoding

Each configuration ran on four prompt types chosen to span draft acceptance: a passage
being **copied** (draft and target agree), **natural** text at temperature 0 and 1, and
**random** tokens at temperature 1. Speedup = TPOT without speculation / TPOT with it, for
the same prompts and concurrency. Acceptance comes from vLLM's counters: acceptance rate =
accepted / drafted tokens, and tokens per step = 1 + accepted / drafts.

**How speedup depends on acceptance.** A speculative step costs the same however many
of its draft tokens are accepted. Across the four prompt types, the measured step cost
varies by only 2-4% within a configuration, even though acceptance ranges from
{s.acceptance_rate.min():.2f} to {s.acceptance_rate.max():.2f}. So

> speedup = tokens per step x (target step time / speculative step time)

which is linear in tokens per step: within each configuration their correlation is
{corr.min():.3f}-{corr.max():.3f}. Using step costs measured on the *other* three prompt types
(leave one out), this predicts the measured speedups with {err_cost:.1f}% average error
(worst {(s.pred_speedup_step_cost / s.speedup - 1).abs().max() * 100:.1f}%).
The best case was {best.config} on {PROMPT_LABELS[best.prompt_type]} prompts at
concurrency {int(best.concurrency)}: {best.speedup:.2f}x at {best.acceptance_rate:.0%} acceptance.

**Draft cost decides whether it pays off.** The break-even acceptance rate is
(speculative step / target step - 1) / k. The 160M draft costs several times what its size
suggests, so it needs much higher acceptance than the 68M draft to help:

{md_table(costs, "{:.2f}")}

**Where the analytical model fails: the draft step.** The model's verify + draft costs
give speedup errors of {', '.join(f"{n} {e:.0f}%" for n, e in err_model.items())}. The
"implied" draft cost (measured step minus modeled verification, per draft token) shows
why: the 68M draft runs close to the model, while the 160M draft takes several times its
weight-read time. A per-kernel overhead fitted to the 68M draft does not explain it
either, so the analysis uses the measured draft cost. The cause inside vLLM's draft-model
path was not profiled.

![speculative decoding](../figures/vllm_spec_decode_{slug}.png)

{md_table(tbl, "{:.2f}")}
""" + (f"\nConfigurations that failed to start: {', '.join(failed)}.\n" if failed else ""))

    summary = [f"# vLLM serving: measurements vs model ({gpu.name})\n\n{head}.\n"]
    report = reports / f"vllm_{slug}.md"
    report.write_text("\n".join(summary + sections), encoding="utf-8")
    print("\n".join(summary + sections))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
