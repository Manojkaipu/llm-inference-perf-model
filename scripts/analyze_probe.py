"""Analyze a llama.cpp kernel probe (notebooks/llamacpp_kernel_probe.ipynb).

    python scripts/analyze_probe.py measurements/t4_kernel_probe

Writes reports/kernel_probe_<gpu>.md, figures/kernel_probe_<gpu>.png and
results/kernel_probe_<gpu>_kernels.csv.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perfmodel import TINYLLAMA_1_1B, pass_time
from perfmodel.calibration import load_profiles
from perfmodel.hardware import gpu_from_name
from perfmodel.validate import decode_run_ops

# (category, pattern) in priority order; matched against lower-cased kernel names
KERNEL_CATEGORIES = [
    ("flash attention", r"flash_attn|fattn"),
    ("custom matvec (mul_mat_vec)", r"mul_mat_vec|mmv"),
    ("custom quantized matmul (mmq)", r"mul_mat_q|mmq"),
    ("cuBLAS / CUTLASS GEMM", r"gemm|cutlass|xmma|cublas|s1688|s16816|h1688|h16816|turing|ampere|sm\d\d_"),
    ("type convert / copy", r"convert|cpy|copy|dequantize|to_fp16|to_fp32"),
    ("norm", r"rms_norm|norm"),
    ("rope", r"rope"),
    ("softmax", r"soft_max|softmax"),
    ("elementwise", r"add|mul|silu|swiglu|glu|scale|unary|binary"),
]


def categorize(name: str) -> str:
    n = name.lower()
    return next((cat for cat, pat in KERNEL_CATEGORIES if re.search(pat, n)), "other")


def load_kernels(meas: Path, steps: int) -> pd.DataFrame:
    rows = []
    for path in sorted(meas.glob("kern_*_cuda_gpu_kern_sum.csv")):
        batch = int(re.search(r"kern_(\d+)_", path.name).group(1))
        df = pd.read_csv(path)
        col = lambda key: next(c for c in df.columns if key.lower() in c.lower())
        tot, inst, name = col("Total Time"), col("Instances"), col("Name")
        for _, r in df.iterrows():
            rows.append({"batch": batch, "kernel": r[name], "category": categorize(r[name]),
                         "ms_per_step": r[tot] / 1e6 / steps, "launches_per_step": r[inst] / steps})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, fmt="{:.2f}") -> str:
    f = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(f(v) for v in row) + " |"
                              for row in df.itertuples(index=False)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meas_dir", type=Path)
    args = ap.parse_args()
    meas = args.meas_dir
    env = json.loads((meas / "env.json").read_text())
    gpu = gpu_from_name(env["gpu_name"])
    slug = meas.name.removesuffix("_kernel_probe")
    npp, ntg = env.get("probe_npp", 512), env.get("probe_ntg", 64)

    fine = pd.read_csv(meas / "probe_fine.csv").sort_values("b")
    off_path = meas / "probe_graphs_off.csv"
    off = pd.read_csv(off_path).sort_values("b") if off_path.exists() else None

    # Model prediction for the same decode steps, using this GPU's calibrated profile if any.
    profiles = load_profiles()[gpu.name]
    prof_name = "calibrated" if "calibrated" in profiles else "prior"
    eff = profiles[prof_name][0]
    fine["predicted_ms"] = [pass_time(decode_run_ops(TINYLLAMA_1_1B, b, npp, ntg), gpu, eff)
                            / ntg * 1e3 for b in fine.b]
    fine["error_pct"] = 100 * (fine.predicted_ms / fine.step_ms - 1)

    # A larger batch should never step faster than a smaller one.
    step = fine.step_ms.to_numpy()
    nonmono = [(int(b), float(s)) for b, s, later in
               zip(fine.b, step, [step[i + 1:].min() if i + 1 < len(step) else np.inf
                                  for i in range(len(step))]) if s > later * 1.05]

    kernels = pd.DataFrame()
    nsys_ntg = env.get("nsys_ntg")
    if nsys_ntg:
        kernels = load_kernels(meas, nsys_ntg)
    if len(kernels):
        (ROOT / "results").mkdir(exist_ok=True)
        kernels.to_csv(ROOT / "results" / f"kernel_probe_{slug}_kernels.csv", index=False)

    # --- figure ---------------------------------------------------------------------
    from perfmodel import plots
    import matplotlib.pyplot as plt
    fig, axes = plots._fig(1, width=7.0, height=4.4)
    ax = axes[0]
    ax.plot(fine.b, fine.step_ms, color=plots.SERIES[0], marker="o", markersize=5, linewidth=2,
            markeredgecolor=plots.SURFACE, label="measured, CUDA graphs default")
    if off is not None:
        ax.plot(off.b, off.step_ms, color=plots.SERIES[1], marker="s", markersize=5, linewidth=2,
                markeredgecolor=plots.SURFACE, label="measured, CUDA graphs off")
    ax.plot(fine.b, fine.predicted_ms, color=plots.SERIES[2], linestyle="--", linewidth=2,
            label=f"model ({prof_name} profile)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(fine.b, [str(b) for b in fine.b], fontsize=7)
    ax.set_xlabel("Batch size (sequences)", color=plots.TEXT_2, fontsize=9)
    ax.set_ylabel("Decode step time (ms)", color=plots.TEXT_2, fontsize=9)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, labelcolor=plots.TEXT)
    fig.suptitle(f"TinyLlama-1.1B decode step on {env['gpu_name']} (llama.cpp, prompt {npp})",
                 x=0.01, ha="left", color=plots.TEXT, fontsize=11)
    fig.tight_layout()
    (ROOT / "figures").mkdir(exist_ok=True)
    fig_path = ROOT / "figures" / f"kernel_probe_{slug}.png"
    fig.savefig(fig_path, dpi=150, facecolor=plots.SURFACE)
    plt.close(fig)

    # --- report ---------------------------------------------------------------------
    sections = [f"""# Kernel probe: TinyLlama-1.1B decode on {env['gpu_name']} (llama.cpp)

llama.cpp `{env.get('llama_cpp_commit', '?')[:10]}`, measured {env.get('timestamp', '?')}.
Decode step = `t_tg / tg` from llama-batched-bench at prompt {npp}, {ntg} generated tokens.
The model column uses the **{prof_name}** profile for {gpu.name}.

![step time](../figures/kernel_probe_{slug}.png)

## A. Step time by batch size

{md_table(fine[['b', 'step_ms', 'predicted_ms', 'error_pct']].rename(columns={
    'b': 'batch', 'step_ms': 'measured ms', 'predicted_ms': 'model ms', 'error_pct': 'model error %'}))}

Batch sizes that step >5% slower than some larger batch (impossible for any additive
cost model, so a sign of a different code path):
{', '.join(f'{b} ({s:.1f} ms)' for b, s in nonmono) or 'none'}.
"""]
    if off is not None:
        ab = off[["b", "step_ms"]].merge(fine[["b", "step_ms"]], on="b",
                                         suffixes=("_off", "_default"))
        ab["off_vs_default_pct"] = 100 * (ab.step_ms_off / ab.step_ms_default - 1)
        sections.append(f"""## B. CUDA graphs off vs default

{md_table(ab.rename(columns={'b': 'batch', 'step_ms_off': 'graphs off ms',
                             'step_ms_default': 'default ms',
                             'off_vs_default_pct': 'change %'}))}

If the batch-2 penalty disappears or the batch-1 time rises to match batch 2 with graphs
off, the gap is CUDA-graph capture. If batch 2 is unchanged, it is the kernel choice.
""")
    if len(kernels):
        by_cat = (kernels.groupby(["batch", "category"])
                  .agg(ms_per_step=("ms_per_step", "sum"),
                       launches_per_step=("launches_per_step", "sum")).reset_index())
        pivot = by_cat.pivot(index="category", columns="batch", values="ms_per_step").fillna(0)
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        launches = by_cat.groupby("batch").launches_per_step.sum()
        matmul = kernels[kernels.category.str.contains("matvec|GEMM|mmq")]
        top_matmul = (matmul.sort_values("ms_per_step", ascending=False)
                      .groupby("batch").head(2)[["batch", "kernel", "ms_per_step"]]
                      .sort_values(["batch", "ms_per_step"], ascending=[True, False]))
        top_matmul["kernel"] = top_matmul.kernel.str.slice(0, 90)
        sections.append(f"""## C. Kernel time per decode step (Nsight Systems, ms)

Each batch size was profiled separately (prompt {env.get('nsys_npp')}, {nsys_ntg} decode steps);
per-step values divide totals by {nsys_ntg}, so they include a small share of prefill.

{md_table(pivot.reset_index())}

Kernel launches per step: {', '.join(f'batch {b}: {v:.0f}' for b, v in launches.items())}.

Top matrix-multiply kernels per batch:

{md_table(top_matmul, "{:.3f}")}
""")
    else:
        sections.append("## C. Kernel summary\n\nNo Nsight Systems data in this probe.\n")

    (ROOT / "reports").mkdir(exist_ok=True)
    out = ROOT / "reports" / f"kernel_probe_{slug}.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
