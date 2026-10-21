"""Validate predictions against measured llama.cpp runs and write an error report.

    python scripts/validate.py measurements/t4_llamacpp [--model TinyLlama-1.1B] [--gpu T4]

Profiles compared:
  peak        datasheet peaks, no overhead
  prior       hand-set REALISTIC efficiencies (hardware.py)
  microbench  bandwidth / FLOP/s measured with torch on the same GPU (no fitting)
  roofline    classic max(compute, memory) roofline; compute, memory and
              per-kernel overhead fitted
  refined     memory efficiency pinned to the measured bandwidth; fits compute,
              per-kernel overhead, a separate attention bandwidth efficiency and
              the compute/memory overlap exponent

Both fitted profiles use the same split: all llama-bench rows plus batched-bench
rows at --train-prompt are training data; batched-bench rows at other prompt
lengths are held out. Rows flagged as anomalies are excluded from fitting and
reported separately.
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

from perfmodel import GPUS, MODELS, PEAK, REALISTIC, plots
from perfmodel.validate import (CompiledCases, case_ops, error_summary, fit_efficiency,
                                flag_anomalies, gpu_log_summary, load_cases, load_env,
                                microbench_efficiency, predict_cases)

LABELS = {"peak": "peak", "prior": "prior", "microbench": "microbench",
          "roofline": "fitted roofline", "refined": "refined"}


def md_table(df: pd.DataFrame, floatfmt="{:.1f}") -> str:
    cols = list(df.columns)
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def headline(summary: pd.DataFrame, split: str) -> pd.DataFrame:
    s = summary[summary.split == split]
    return s.pivot(index="profile", columns="phase", values="mape_pct").reindex(
        [p for p in LABELS if p in set(s.profile)]).reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meas_dir", type=Path)
    ap.add_argument("--model", default="TinyLlama-1.1B", choices=MODELS)
    ap.add_argument("--gpu", default="T4", choices=GPUS)
    ap.add_argument("--train-prompt", type=int, default=512,
                    help="batched-bench prompt length used for fitting; others are held out")
    args = ap.parse_args()

    cfg, gpu = MODELS[args.model], GPUS[args.gpu]
    env = load_env(args.meas_dir)
    gpulog = gpu_log_summary(args.meas_dir)
    all_cases = load_cases(args.meas_dir)

    # The model assumes fused (flash) attention; FA-off rows are reported separately.
    fa_on = all_cases[all_cases.flash_attn].reset_index(drop=True)
    fa_off = all_cases[~all_cases.flash_attn].reset_index(drop=True)
    cases = fa_on if len(fa_on) else all_cases
    cases["anomaly"] = flag_anomalies(cases)
    is_bb = cases.source == "batched-bench"
    cases["split"] = np.where(is_bb & (cases.prompt.where(cases.phase == "prefill", cases.depth)
                                       != args.train_prompt), "held-out", "train")
    cases.loc[cases.anomaly != "", "split"] = "anomaly"
    train = (cases.split == "train").to_numpy()
    compiled = CompiledCases([case_ops(cfg, c) for c in cases.itertuples()])

    prior = REALISTIC[gpu.name]
    effs = {"peak": PEAK, "prior": prior}
    micro = microbench_efficiency(args.meas_dir, gpu, prior.kernel_overhead_s)
    if micro:
        effs["microbench"] = micro
    effs["roofline"] = fit_efficiency(cases, compiled, gpu, mask=train)
    effs["refined"] = fit_efficiency(
        cases, compiled, gpu, mask=train,
        free=("compute", "kernel_overhead_s", "attention_half_work", "overlap"),
        base=replace(PEAK, memory=(micro or prior).memory))

    for name, eff in effs.items():
        cases[f"pred_{name}"] = predict_cases(cases, compiled, gpu, eff)
        cases[f"err_{name}"] = cases[f"pred_{name}"] / cases.measured_tok_s - 1

    out, figs, reports = ROOT / "results", ROOT / "figures", ROOT / "reports"
    for d in (out, figs, reports):
        d.mkdir(exist_ok=True)
    tag = f"{cfg.name}_{gpu.name}_llamacpp"
    cases.to_csv(out / f"validation_cases_{tag}.csv", index=False)
    summary = error_summary(cases, list(effs), by=("split", "phase"))
    summary.to_csv(out / f"validation_summary_{tag}.csv", index=False)
    calib = {k: {f: (None if v is None or (isinstance(v, float) and np.isinf(v)) else v)
                 for f, v in asdict(e).items()} for k, e in effs.items()}
    (out / f"calibrated_{tag}.json").write_text(json.dumps(calib, indent=2))

    shown = [p for p in ("microbench", "roofline", "refined") if p in effs]
    clean = cases[cases.split != "anomaly"]
    plots.validation_parity(clean, shown, LABELS, path=figs / f"validation_parity_{tag}.png",
                            title=f"{cfg.name} on {gpu.name} (llama.cpp): predicted vs measured")
    if (is_bb & (cases.phase == "decode")).any():
        plots.validation_batched(cases, shown, LABELS, path=figs / f"validation_batched_{tag}.png",
                                 title=f"{cfg.name} on {gpu.name}: batched decode, measured vs predicted")

    # --- report -----------------------------------------------------------------
    eff_rows = pd.DataFrame([{
        "profile": k, "compute_eff": e.compute, "memory_eff": e.memory,
        "attn_memory_eff": e.attention_memory if e.attention_memory else e.memory,
        "attn_half_work": e.attention_half_work or "-",
        "overhead_us": e.kernel_overhead_s * 1e6,
        "overlap_p": "inf (max)" if np.isinf(e.overlap) else f"{e.overlap:.2f}"}
        for k, e in effs.items()])

    anomalies = cases[cases.split == "anomaly"][
        ["anomaly", "source", "phase", "batch", "prompt", "depth", "measured_tok_s",
         "pred_refined", "err_refined"]].copy()
    anomalies["err_refined"] *= 100

    fa_off_note = ""
    if len(fa_off) and len(fa_on):
        c_off = CompiledCases([case_ops(cfg, c) for c in fa_off.itertuples()])
        e = predict_cases(fa_off, c_off, gpu, effs["refined"]) / fa_off.measured_tok_s - 1
        fa_off["err"] = e
        by = fa_off.groupby("phase").err.agg(lambda x: 100 * x.mean())
        fa_off_note = ("Flash attention **off** (llama-bench, refined profile), mean signed error: "
                       + ", ".join(f"{p} {v:+.1f}%" for p, v in by.items())
                       + ". The model assumes fused attention, so positive error means unfused "
                       "attention costs more than the model counts.\n")

    throttle = ""
    if gpulog:
        sm_max = float(str(env.get("max_sm_clock", "0")).split()[0] or 0)
        frac = gpulog.get("power_capped_frac")
        throttle = (f"While busy ({gpulog['busy_samples']} samples at 0.5 s), the median SM clock was "
                    f"**{gpulog['sm_clock_median_mhz']:.0f} MHz**"
                    + (f" (max {sm_max:.0f} MHz, so ~{100 * gpulog['sm_clock_median_mhz'] / sm_max:.0f}% "
                       f"of the boost clock the 65 TFLOPS datasheet figure assumes)" if sm_max else "")
                    + f", median power {gpulog['power_median_w']:.1f} W"
                    + (f", software power cap active in {100 * frac:.0f}% of samples" if frac is not None else "")
                    + f", max temperature {gpulog['temp_max_c']:.0f} C.\n")

    report = f"""# Validation: {cfg.name} on {gpu.name} (llama.cpp)

GPU: {env.get('gpu_name', '?')} | driver {env.get('driver', '?')} |
llama.cpp `{env.get('llama_cpp_commit', '?')[:10]}` | measured {env.get('timestamp', '?')}

Error = `predicted / measured - 1` in tokens/s; MAPE = mean |error|.
Training rows: llama-bench (batch 1, flash attention on) + batched-bench at prompt
{args.train_prompt} ({int(train.sum())} rows). Held out: batched-bench at the other
prompt lengths ({int((cases.split == 'held-out').sum())} rows). Anomalies excluded
from both ({len(anomalies)} rows, listed below).

## Held-out error (MAPE %)

{md_table(headline(summary, 'held-out'))}

## Training error (MAPE %)

{md_table(headline(summary, 'train'))}

## Efficiency profiles

{md_table(eff_rows, "{:.3f}")}

## GPU clocks during the run

{throttle or 'No gpu_log.csv.'}
## Anomalies (excluded from fit and from the error tables)

{md_table(anomalies, "{:.1f}") if len(anomalies) else 'None.'}

{fa_off_note}
## Full error table (%)

{md_table(summary)}

## Figures

![parity](../figures/validation_parity_{tag}.png)

![batched](../figures/validation_batched_{tag}.png)
"""
    (reports / f"validation_{tag}.md").write_text(report, encoding="utf-8")
    print(eff_rows.to_string(index=False))
    print("\nheld-out MAPE %\n", headline(summary, "held-out").round(1).to_string(index=False))
    print("\ntrain MAPE %\n", headline(summary, "train").round(1).to_string(index=False))
    print("\nanomalies\n", anomalies.round(1).to_string(index=False))
    print("\n" + fa_off_note + throttle)
    print(f"Report: {reports / f'validation_{tag}.md'}")


if __name__ == "__main__":
    main()
