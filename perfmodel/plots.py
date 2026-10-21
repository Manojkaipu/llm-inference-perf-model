"""Static matplotlib figures (light theme, reference palette)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .hardware import GPU, PEAK, Efficiency
from .ops import ATTENTION, ELEMENTWISE, LINEAR
from .predict import fits, op_table, predict

SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # blue, orange, aqua: all-pairs CVD safe
SERIES_4 = "#eda100"  # yellow: adjacent-pair safe only, so used in stacked bars, not scatter
KIND_STYLE = {
    LINEAR: (SERIES[0], "o", "Linear (weights)"),
    ATTENTION: (SERIES[1], "s", "Attention (KV cache)"),
    ELEMENTWISE: (SERIES[2], "^", "Elementwise / norms"),
}
LABELLED_OPS = {"qkv_proj", "gate_up_proj", "attention", "lm_head"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(ncols, width=5.2, height=4.2):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes[0]:
        _style(ax)
    return fig, axes[0]


def _label_clusters(df, tol=0.25):
    """Merge labels of points within `tol` decades of each other on both axes."""
    clusters = []
    for _, r in df.sort_values("intensity").iterrows():
        lx, ly = np.log10(r.intensity), np.log10(r.achieved_flops)
        for c in clusters:
            if abs(c[1] - lx) < tol and abs(c[2] - ly) < tol:
                c[0].append(r.op)
                break
        else:
            clusters.append([[r.op], lx, ly])
    return [(names, 10 ** lx, 10 ** ly) for names, lx, ly in clusters]


def _no_fit(ax, gpu):
    ax.text(0.5, 0.5, f"Model does not fit in {gpu.mem_capacity / 1e9:.0f} GB",
            transform=ax.transAxes, ha="center", va="center", color=TEXT_2, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def roofline(gpu: GPU, scenarios: list[tuple[str, list]], eff: Efficiency = PEAK,
             path=None, title=None):
    """One panel per scenario: ops placed at (intensity, achieved FLOP/s)."""
    fig, axes = _fig(len(scenarios))
    xs = np.logspace(-1, 4.5, 200)
    roof = np.minimum(gpu.matmul_flops, gpu.mem_bw * xs)
    real_roof = np.minimum(gpu.matmul_flops * eff.compute, gpu.mem_bw * eff.memory * xs)
    for ax, (label, ops) in zip(axes, scenarios):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(xs, roof, color=TEXT_2, linewidth=2, zorder=1)
        if eff is not PEAK:
            ax.plot(xs, real_roof, color=TEXT_2, linewidth=1.2, linestyle="--", zorder=1)
        ax.axvline(gpu.ridge_point, color=GRID, linewidth=1.2, zorder=0)
        ax.text(gpu.ridge_point * 1.15, gpu.matmul_flops * 1.4,
                f"ridge {gpu.ridge_point:.0f} FLOP/B", color=TEXT_2, fontsize=8)
        df = op_table(ops, gpu, eff)
        df = df[df.flops > 0]
        for kind, (color, marker, name) in KIND_STYLE.items():
            sub = df[df.kind == kind]
            ax.scatter(sub.intensity, sub.achieved_flops, s=56, color=color, marker=marker,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3, label=name)
        for names, x, y in _label_clusters(df[df.op.isin(LABELLED_OPS)]):
            ax.annotate(" / ".join(names), (x, y), xytext=(7, -12),
                        textcoords="offset points", fontsize=8, color=TEXT)
        ax.set_title(label, fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("Arithmetic intensity (FLOP/byte)", color=TEXT_2, fontsize=9)
        ax.set_ylim(gpu.matmul_flops * 1e-6, gpu.matmul_flops * 3)
    axes[0].set_ylabel("Attained FLOP/s", color=TEXT_2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", labelcolor=TEXT)
    fig.suptitle(title or f"{gpu.name} roofline", x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def decode_throughput(cfg, gpus, effs, contexts, batches, gen_len=64, path=None):
    """Decode tokens/s vs batch, one panel per GPU, one line per context length.
    Lines stop where weights + KV cache no longer fit in GPU memory."""
    fig, axes = _fig(len(gpus))
    for ax, gpu in zip(axes, gpus):
        eff = effs[gpu.name]
        for color, ctx in zip(SERIES, contexts):
            pts = [(b, predict(cfg, gpu, b, ctx, gen_len, eff).decode_tok_s)
                   for b in batches if fits(cfg, gpu, b, ctx + gen_len, eff)]
            if not pts:
                continue
            bx, ty = zip(*pts)
            ax.plot(bx, ty, color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, label=f"context {ctx:,}")
            ax.annotate(f"{ty[-1]:,.0f}", (bx[-1], ty[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=TEXT, va="center")
        ax.set_title(gpu.name, fontsize=10, color=TEXT, loc="left")
        if not ax.lines:
            _no_fit(ax, gpu)
            continue
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(batches, [str(b) for b in batches])
        ax.set_xlabel("Batch size (sequences)", color=TEXT_2, fontsize=9)
    axes[0].set_ylabel("Decode throughput (tokens/s)", color=TEXT_2, fontsize=9)
    next(ax for ax in axes if ax.lines).legend(frameon=False, fontsize=8, labelcolor=TEXT)
    fig.suptitle(f"{cfg.name}: predicted decode throughput", x=0.01, ha="left",
                 color=TEXT, fontsize=12)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def decode_breakdown(cfg, gpus, effs, context, batches, path=None):
    """Share of decode-step time by op kind vs batch (100% stacked bars).
    Kernel launch overhead is split out as its own segment."""
    from .ops import decode_ops
    styles = {**KIND_STYLE, "overhead": (SERIES_4, None, "Launch overhead")}
    fig, axes = _fig(len(gpus))
    for ax, gpu in zip(axes, gpus):
        eff = effs[gpu.name]
        shares = {k: [] for k in styles}
        labels = []
        for b in batches:
            if not fits(cfg, gpu, b, context, eff):
                continue
            df = op_table(decode_ops(cfg, b, context), gpu, eff)
            df["work_s"] = (df.time_s - df.overhead_s) * df["count"]
            by_kind = df.groupby("kind").work_s.sum().to_dict()
            by_kind["overhead"] = (df.overhead_s * df["count"]).sum()
            total = df.total_time_s.sum()
            for k in styles:
                shares[k].append(by_kind.get(k, 0.0) / total * 100)
            labels.append(str(b))
        ax.set_title(gpu.name, fontsize=10, color=TEXT, loc="left")
        if not labels:
            _no_fit(ax, gpu)
            continue
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        for kind, (color, _, name) in styles.items():
            vals = np.array(shares[kind])
            ax.bar(x, vals, bottom=bottom, color=color, width=0.7, label=name,
                   edgecolor=SURFACE, linewidth=2)
            bottom += vals
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Batch size (sequences)", color=TEXT_2, fontsize=9)
    axes[0].set_ylabel("Share of decode step time (%)", color=TEXT_2, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left",
                    bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(f"{cfg.name}: where decode time goes (context {context:,})", x=0.01,
                 ha="left", color=TEXT, fontsize=12)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
