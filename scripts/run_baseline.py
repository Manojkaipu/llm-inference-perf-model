"""Generate baseline analytical results: per-op tables, rooflines, throughput sweeps.

    python scripts/run_baseline.py [--model TinyLlama-1.1B] [--peak]
"""
import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perfmodel import GPUS, MODELS, PEAK, REALISTIC, decode_ops, op_table, predict, prefill_ops
from perfmodel import plots

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
CONTEXTS = [512, 2048, 8192]
BASELINE_GPUS = {g: GPUS[g] for g in ("T4", "H100-SXM")}
GEN_LEN = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama-1.1B", choices=MODELS)
    ap.add_argument("--peak", action="store_true", help="use peak instead of realistic efficiency")
    args = ap.parse_args()

    cfg = MODELS[args.model]
    effs = {g: PEAK for g in BASELINE_GPUS} if args.peak else REALISTIC
    tag = f"{cfg.name}_{'peak' if args.peak else 'realistic'}"
    out, figs = ROOT / "results", ROOT / "figures"
    out.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)

    scenarios = [
        ("Prefill, batch 1, prompt 512", prefill_ops(cfg, 1, 512)),
        ("Decode, batch 1, context 512", decode_ops(cfg, 1, 512)),
        ("Decode, batch 64, context 2048", decode_ops(cfg, 64, 2048)),
    ]
    for gname, gpu in BASELINE_GPUS.items():
        tables = []
        for label, ops in scenarios:
            df = op_table(ops, gpu, effs[gname])
            df.insert(0, "scenario", label)
            tables.append(df)
        pd.concat(tables).to_csv(out / f"ops_{tag}_{gname}.csv", index=False)
        plots.roofline(gpu, scenarios, effs[gname], path=figs / f"roofline_{tag}_{gname}.png",
                       title=f"{cfg.name} on {gpu.name} roofline")

    rows = []
    for (gname, gpu), b, ctx in product(BASELINE_GPUS.items(), BATCHES, CONTEXTS):
        p = predict(cfg, gpu, b, ctx, GEN_LEN, effs[gname])
        rows.append({"gpu": gname, "batch": b, "prompt_len": ctx, "gen_len": GEN_LEN,
                     "fits": p.fits, "ttft_s": p.ttft_s, "prefill_tok_s": p.prefill_tok_s,
                     "tpot_s": p.tpot_s, "decode_tok_s": p.decode_tok_s,
                     "e2e_tok_s": p.e2e_tok_s})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / f"sweep_{tag}.csv", index=False)

    gpus = list(BASELINE_GPUS.values())
    plots.decode_throughput(cfg, gpus, effs, CONTEXTS, BATCHES, path=figs / f"decode_tps_{tag}.png")
    plots.decode_breakdown(cfg, gpus, effs, 2048, BATCHES, path=figs / f"decode_breakdown_{tag}.png")

    print(sweep[sweep.prompt_len == 2048].to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
    print(f"\nWrote CSVs to {out} and figures to {figs}")


if __name__ == "__main__":
    main()
