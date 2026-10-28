"""Analyze a vLLM serving run (notebooks/vllm_serving_benchmark.ipynb).

    python scripts/analyze_vllm.py measurements/<gpu>_vllm

Fits a vLLM efficiency profile on the continuous-batching sweep, predicts prefix
caching and speculative decoding out-of-sample, and writes reports/vllm_<gpu>.md,
figures/vllm_*_<gpu>.png, results/vllm_*_<gpu>.csv and results/vllm_profile_<gpu>.json.
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perfmodel import TINYLLAMA_1_1B, plots
from perfmodel.calibration import load_profiles
from perfmodel.hardware import gpu_from_name
from perfmodel.vllm_analysis import (breakeven_rate, fit_vllm_profile, load, memory_efficiency,
                                     predict_cb, predict_prefix, predict_spec, spec_config_info,
                                     spec_curve, summarize_cb, summarize_prefix, summarize_spec)

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PROMPT_LABELS = {"copy": "copy", "natural": "natural, T=0", "natural_t1": "natural, T=1",
                 "random_t1": "random, T=1"}
MARKERS = {"copy": "o", "natural": "s", "natural_t1": "^", "random_t1": "D"}


def md_table(df: pd.DataFrame, fmt="{:.2f}") -> str:
    f = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else (
        "-" if isinstance(v, (float, np.floating)) else str(v))
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(f(v) for v in row) + " |"
                              for row in df.itertuples(index=False)])


def pct_err(pred, meas):
    return 100 * (pred / meas - 1)


def fig_cb(cb: pd.DataFrame, title: str, path: Path):
    fig, axes = plots._fig(2, width=5.2, height=4.2)
    ax = axes[0]
    ax.plot(cb.concurrency, cb.output_tok_s, color=plots.TEXT, marker="o", linestyle="none",
            markersize=6, label="measured")
    ax.plot(cb.concurrency, cb.pred_output_tok_s, color=plots.SERIES[0], linewidth=2,
            label="model (vLLM fit)")
    if "llamacpp_output_tok_s" in cb:
        ax.plot(cb.concurrency, cb.llamacpp_output_tok_s, color=plots.SERIES[1], linewidth=2,
                linestyle="--", label="model (llama.cpp T4 profile)")
    ax.set_ylabel("Output tokens/s", color=plots.TEXT_2, fontsize=9)
    ax.set_title("Throughput", fontsize=10, color=plots.TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=plots.TEXT)
    ax = axes[1]
    for col, pcol, color, label in (("ttft_p50_s", "pred_ttft_s", plots.SERIES[1], "TTFT"),
                                    ("tpot_p50_s", "pred_tpot_s", plots.SERIES[0], "TPOT")):
        ax.plot(cb.concurrency, cb[col] * 1e3, color=color, marker="o", linestyle="none",
                markersize=6, markeredgecolor=plots.SURFACE, label=f"{label} p50, measured")
        ax.plot(cb.concurrency, cb[pcol] * 1e3, color=color, linewidth=2, label=f"{label}, model")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if str(v)[0] in "25" else ""))
    ax.set_ylabel("Latency (ms)", color=plots.TEXT_2, fontsize=9)
    ax.set_title("Latency", fontsize=10, color=plots.TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=plots.TEXT)
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
                label=f"{label}, model + {overhead * 1e3:.0f} ms overhead")
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
    fig, axes = plots._fig(len(concs), width=5.0, height=4.4)
    names = [n for n in sd.config.unique() if n != "baseline"][:3]   # <= 3 colors (scatter)
    for ax, c in zip(axes, concs):
        ax.axhline(1, color=plots.TEXT_2, linewidth=1)
        for color, name in zip(plots.SERIES, names):
            if (name, c) in curves:
                cv = curves[(name, c)]
                ax.plot(cv.acceptance_rate, cv.speedup, color=color, linewidth=2, label=f"{name}, model")
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
    axes[0].set_ylabel("Decode speedup vs no speculation (TPOT)", color=plots.TEXT_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    for ptype, m in MARKERS.items():
        handles.append(plt.Line2D([], [], color=plots.TEXT_2, marker=m, linestyle="none"))
        labels.append(f"measured: {PROMPT_LABELS[ptype]}")
    axes[-1].legend(handles, labels, frameon=False, fontsize=8, labelcolor=plots.TEXT,
                    loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, x=0.01, ha="left", color=plots.TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=plots.SURFACE)
    plt.close(fig)


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
    head = (f"vLLM {env.get('vllm_version', '?')} on {env['gpu_name']}, "
            f"TinyLlama-1.1B fp16, measured {env.get('timestamp', '?')}")
    if env.get("server_extra_env"):
        head += (". Servers ran with " + ", ".join(f"`{k}={v}`" for k, v in env["server_extra_env"].items())
                 + " because FlashInfer could not build its sampling kernel, so token sampling used "
                   "vLLM's PyTorch implementation")
    sections = [f"# vLLM serving: measurements vs model ({gpu.name})\n\n{head}.\n"]

    # 1. continuous batching -> fit
    cb = summarize_cb(data["cb"], data["cb_levels"])
    eff = fit_vllm_profile(cfg, gpu, cb, cb_in, cb_out, mem)
    cb = predict_cb(cfg, gpu, eff, cb, cb_in, cb_out)
    llamacpp = load_profiles()[gpu.name].get("calibrated")
    if llamacpp:
        cb = predict_cb(cfg, gpu, llamacpp[0], cb, cb_in, cb_out, prefix="llamacpp")
    cb.to_csv(results / f"vllm_continuous_batching_{slug}.csv", index=False)
    profile = {k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in asdict(eff).items()}
    (results / f"vllm_profile_{slug}.json").write_text(json.dumps(profile, indent=2))
    fig_cb(cb, f"Continuous batching: TinyLlama-1.1B on {gpu.name} (vLLM, {cb_in} in / {cb_out} out)",
           figs / f"vllm_continuous_batching_{slug}.png")
    t = cb[["concurrency", "output_tok_s", "pred_output_tok_s", "ttft_p50_s", "pred_ttft_s",
            "tpot_p50_s", "pred_tpot_s", "e2e_p50_s", "preemptions"]].copy()
    for col in ("ttft_p50_s", "pred_ttft_s", "tpot_p50_s", "pred_tpot_s", "e2e_p50_s"):
        t[col] = t[col] * 1e3
    t.columns = ["concurrency", "out tok/s", "model tok/s", "TTFT p50 ms", "model TTFT ms",
                 "TPOT p50 ms", "model TPOT ms", "E2E p50 ms", "preemptions"]
    tpot_mape = (cb.pred_tpot_s / cb.tpot_p50_s - 1).abs().mean() * 100
    ttft_mape = (cb.pred_ttft_s / cb.ttft_p50_s - 1).abs().mean() * 100
    llama_note = ""
    if llamacpp:
        e = (cb.llamacpp_tpot_s / cb.tpot_p50_s - 1).abs().mean() * 100
        llama_note = (f" For comparison, the llama.cpp-calibrated {gpu.name} profile (no refit) "
                      f"predicts vLLM's TPOT with {e:.1f}% MAPE.")
    sections.append(f"""## 1. Continuous batching (calibration set)

Closed loop: each concurrency level keeps that many requests in flight with random
{cb_in}-token prompts and {cb_out} output tokens. The model treats each engine step as one
decode token per running request plus, amortized, the chunked prefill of the
requests that replace finished ones, all sharing one read of the weights.

Fitted vLLM profile (memory efficiency pinned to the measured {mem:.0%} of peak bandwidth):
compute {eff.compute:.3f}, per-kernel overhead {eff.kernel_overhead_s * 1e6:.1f} us,
overlap exponent {eff.overlap:.2f}. Fit error: TPOT {tpot_mape:.1f}% MAPE (in-sample);
TTFT {ttft_mape:.1f}% MAPE (TTFT was not fitted).{llama_note}

![continuous batching](../figures/vllm_continuous_batching_{slug}.png)

{md_table(t, "{:,.1f}")}
""")

    # 2. prefix caching
    if data["prefix"] is not None:
        pc = predict_prefix(cfg, gpu, eff, summarize_prefix(data["prefix"]))
        pc.to_csv(results / f"vllm_prefix_cache_{slug}.csv", index=False)
        fig_prefix(pc, f"Prefix caching: time to first token on {gpu.name} (vLLM)",
                   figs / f"vllm_prefix_cache_{slug}.png")
        on = pc[pc.config == "caching_on"].set_index("prefix_len")
        off = pc[pc.config == "caching_off"].set_index("prefix_len")
        tbl = pd.DataFrame({"prefix_len": on.index,
                            "TTFT on ms": on.ttft_p50_s * 1e3,
                            "TTFT off ms": off.ttft_p50_s.reindex(on.index) * 1e3,
                            "speedup": (off.ttft_p50_s.reindex(on.index) / on.ttft_p50_s),
                            "cached tokens": on.cached_tokens_p50,
                            "model compute ms": on.pred_compute_s * 1e3,
                            "residual ms": on.residual_s * 1e3}).reset_index(drop=True)
        sections.append(f"""## 2. Prefix caching (out-of-sample)

One request at a time; each prefix length uses its own shared prefix, warmed by one
request that is excluded. The model predicts only the prefill compute for the uncached
tokens (attending over the cached prefix), so measured minus predicted is the
per-request overhead it leaves out. If the model captures the caching effect, that
residual stays flat as the prefix grows: here it ranges
{pc.residual_s.min() * 1e3:.1f}-{pc.residual_s.max() * 1e3:.1f} ms (median {pc.residual_s.median() * 1e3:.1f} ms).

![prefix caching](../figures/vllm_prefix_cache_{slug}.png)

{md_table(tbl, "{:,.1f}")}
""")

    # 3. speculative decoding
    if data["spec"] is not None:
        info = spec_config_info(env)
        ctx = env["sd"]["input"] + env["sd"]["output"] // 2
        sd = predict_spec(cfg, gpu, eff, summarize_spec(data["spec"], data["spec_levels"]), info, ctx)
        sd.to_csv(results / f"vllm_spec_decode_{slug}.csv", index=False)
        spec_names = [n for n in sd.config.unique() if n != "baseline" and n in info]
        curves = {(n, c): spec_curve(cfg, gpu, eff, info, n, int(c), ctx)
                  for n in spec_names for c in sd.concurrency.unique()}
        fig_spec(sd, curves, f"Speculative decoding on {gpu.name} (vLLM, target TinyLlama-1.1B)",
                 figs / f"vllm_spec_decode_{slug}.png")
        s = sd[sd.config != "baseline"]
        tbl = s[["config", "prompt_type", "concurrency", "acceptance_rate", "tokens_per_step",
                 "speedup", "pred_speedup"]].copy()
        tbl["error %"] = pct_err(tbl.pred_speedup, tbl.speedup)
        be = pd.DataFrame([{"config": n, "concurrency": int(c),
                            "break-even acceptance": breakeven_rate(cfg, gpu, eff, info, n, int(c), ctx)}
                           for n in spec_names for c in sorted(sd.concurrency.unique())])
        failed = [k for k, v in env.get("servers", {}).items() if v.get("failed")]
        corr = s[["acceptance_rate", "speedup"]].corr().iloc[0, 1] if len(s) > 2 else float("nan")
        sections.append(f"""## 3. Speculative decoding (out-of-sample)

Speedup = baseline TPOT / speculative TPOT for the same prompt type and concurrency.
Acceptance rate = accepted / drafted tokens from vLLM's counters; tokens per step =
1 + accepted / drafts. The model predicts speedup from the measured tokens per step
and its own costs for k draft steps plus one (k+1)-token verification pass, with the
profile fitted in part 1. Across all configurations, the correlation between
acceptance rate and speedup is {corr:.2f}.

![speculative decoding](../figures/vllm_spec_decode_{slug}.png)

{md_table(tbl, "{:.2f}")}

Predicted break-even acceptance rate (below it, speculation slows decoding):

{md_table(be, "{:.2f}")}
""" + (f"\nConfigurations that failed to start: {', '.join(failed)}.\n" if failed else ""))

    report = reports / f"vllm_{slug}.md"
    report.write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
