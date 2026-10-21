"""Validate predictions against measured llama.cpp runs and write an error report.

    python scripts/validate.py measurements/t4_llamacpp [--model TinyLlama-1.1B] [--gpu T4]

Profiles compared:
  peak        datasheet peaks, no overhead
  prior       hand-set REALISTIC efficiencies (hardware.py)
  microbench  bandwidth / FLOP/s measured with torch on the same GPU (no fitting)
  fitted      efficiencies fitted on batch-1 llama-bench rows only; batched-bench
              rows are held out, so their error is out-of-sample
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perfmodel import GPUS, MODELS, PEAK, REALISTIC, plots
from perfmodel.validate import (CompiledCases, case_ops, error_summary, fit_efficiency,
                                load_cases, load_env, microbench_efficiency, predict_cases)

LABELS = {"peak": "peak", "prior": "prior", "microbench": "microbench", "fitted": "fitted"}


def md_table(df: pd.DataFrame, floatfmt="{:.1f}") -> str:
    cols = list(df.columns)
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meas_dir", type=Path)
    ap.add_argument("--model", default="TinyLlama-1.1B", choices=MODELS)
    ap.add_argument("--gpu", default="T4", choices=GPUS)
    args = ap.parse_args()

    cfg, gpu = MODELS[args.model], GPUS[args.gpu]
    env = load_env(args.meas_dir)
    all_cases = load_cases(args.meas_dir)

    # The model assumes fused (flash) attention; FA-off rows are reported separately.
    fa_on = all_cases[all_cases.flash_attn].reset_index(drop=True)
    fa_off = all_cases[~all_cases.flash_attn].reset_index(drop=True)
    cases = fa_on if len(fa_on) else all_cases
    compiled = CompiledCases([case_ops(cfg, c) for c in cases.itertuples()])

    prior = REALISTIC[gpu.name]
    effs = {"peak": PEAK, "prior": prior}
    micro = microbench_efficiency(args.meas_dir, gpu, prior.kernel_overhead_s)
    if micro:
        effs["microbench"] = micro
    train = (cases.source == "llama-bench").to_numpy()
    if not train.any():
        train = np.ones(len(cases), bool)
    effs["fitted"] = fit_efficiency(cases, compiled, gpu, mask=train)

    for name, eff in effs.items():
        cases[f"pred_{name}"] = predict_cases(cases, compiled, gpu, eff)
        cases[f"err_{name}"] = cases[f"pred_{name}"] / cases.measured_tok_s - 1
    cases["fit_split"] = np.where(train, "train", "held-out")

    out, figs, reports = ROOT / "results", ROOT / "figures", ROOT / "reports"
    for d in (out, figs, reports):
        d.mkdir(exist_ok=True)
    tag = f"{cfg.name}_{gpu.name}_llamacpp"
    cases.to_csv(out / f"validation_cases_{tag}.csv", index=False)
    summary = error_summary(cases, list(effs))
    summary.to_csv(out / f"validation_summary_{tag}.csv", index=False)

    shown = [p for p in ("prior", "microbench", "fitted") if p in effs]
    plots.validation_parity(cases, shown, LABELS, path=figs / f"validation_parity_{tag}.png",
                            title=f"{cfg.name} on {gpu.name} (llama.cpp): predicted vs measured")
    if ((cases.source == "batched-bench") & (cases.phase == "decode")).any():
        plots.validation_batched(cases, shown, LABELS, path=figs / f"validation_batched_{tag}.png",
                                 title=f"{cfg.name} on {gpu.name}: batched decode, measured vs predicted")

    fa_off_note = ""
    if len(fa_off) and len(fa_on):
        c_off = CompiledCases([case_ops(cfg, c) for c in fa_off.itertuples()])
        e = predict_cases(fa_off, c_off, gpu, effs["fitted"]) / fa_off.measured_tok_s - 1
        fa_off_note = (f"\nFlash attention **off** ({len(fa_off)} rows, fitted profile): "
                       f"MAPE {100 * e.abs().mean():.1f}%, bias {100 * e.mean():+.1f}%. "
                       "The model assumes fused attention, so a positive bias here means "
                       "unfused attention adds traffic the model does not count.\n")

    eff_rows = pd.DataFrame([{"profile": k, "compute_eff": v.compute, "memory_eff": v.memory,
                              "overhead_us": v.kernel_overhead_s * 1e6} for k, v in effs.items()])
    held = cases[~train]
    fitted_heldout = (100 * held.err_fitted.abs().mean()) if len(held) else float("nan")
    report = f"""# Validation: {cfg.name} on {gpu.name} (llama.cpp)

GPU: {env.get('gpu_name', '?')} | driver {env.get('driver', '?')} |
llama.cpp {env.get('llama_cpp_commit', '?')[:10]} | measured {env.get('timestamp', '?')}

Error is `predicted / measured - 1` in tokens/s. The **fitted** profile is fit on
batch-1 `llama-bench` rows only ({int(train.sum())} rows). Its error on the
{len(held)} held-out `batched-bench` rows is out-of-sample: **{fitted_heldout:.1f}% MAPE**.

## Efficiency profiles

{md_table(eff_rows, "{:.3f}")}

## Error by source and phase (%)

{md_table(summary)}
{fa_off_note}
## Figures

![parity](../figures/validation_parity_{tag}.png)

![batched](../figures/validation_batched_{tag}.png)
"""
    (reports / f"validation_{tag}.md").write_text(report, encoding="utf-8")
    print(eff_rows.to_string(index=False))
    print()
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    print(f"\nReport: {reports / f'validation_{tag}.md'}")


if __name__ == "__main__":
    main()
