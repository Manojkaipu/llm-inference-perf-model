import numpy as np
import pandas as pd
import pytest

from perfmodel import T4, TINYLLAMA_1_1B, Efficiency, decode_ops, pass_time, prefill_ops, totals
from perfmodel.validate import (CompiledCases, case_ops, case_tokens, decode_run_ops,
                                fit_efficiency, flag_anomalies, load_cases, predict_cases,
                                prefill_pass_ops)

CFG = TINYLLAMA_1_1B
TRUE = Efficiency(compute=0.42, memory=0.78, kernel_overhead_s=5e-6)


REFINED = Efficiency(compute=0.25, memory=0.85, kernel_overhead_s=9e-6,
                     attention_half_work=13.0, overlap=1.1)


@pytest.mark.parametrize("eff", [TRUE, REFINED])
def test_compiled_matches_scalar_pass_time(eff):
    lists = [decode_ops(CFG, 8, 1000), prefill_ops(CFG, 2, 300)]
    got = CompiledCases(lists).times(T4, eff)
    want = [pass_time(ops, T4, eff) for ops in lists]
    assert got == pytest.approx(want, rel=1e-9)


def test_attention_ramp_slows_small_batch_decode_attention_only():
    ramp = Efficiency(memory=0.85, attention_half_work=13.0)
    flat = Efficiency(memory=0.85)
    attn = lambda b, e: next(o for o in decode_ops(CFG, b, 4096) if o.name == "attention")
    from perfmodel import op_time
    slow = op_time(attn(1, ramp), T4, ramp)[1] / op_time(attn(1, flat), T4, flat)[1]
    fast = op_time(attn(128, ramp), T4, ramp)[1] / op_time(attn(128, flat), T4, flat)[1]
    assert slow == pytest.approx((4 + 13) / 4)       # batch 1 x 4 KV heads
    assert fast == pytest.approx((512 + 13) / 512)


def test_flag_anomalies():
    rows = [dict(source="llama-bench", phase="prefill", batch=1, prompt=128, depth=0,
                 measured_tok_s=5000.0)]
    rows += [dict(source="batched-bench", phase="prefill", batch=1, prompt=128, depth=0,
                  measured_tok_s=1500.0)]
    for b, tps in [(1, 100.0), (2, 120.0), (4, 360.0), (8, 700.0)]:
        rows.append(dict(source="batched-bench", phase="decode", batch=b, prompt=0, depth=128,
                         measured_tok_s=tps))
    flags = flag_anomalies(pd.DataFrame(rows))
    assert flags.tolist() == ["", "cold start", "", "non-monotonic decode", "", ""]


@pytest.mark.parametrize("batch, prompt", [(1, 2048), (4, 2048), (16, 128), (5, 128)])
def test_chunked_prefill_conserves_linear_and_attention_work(batch, prompt):
    chunked = totals(prefill_pass_ops(CFG, batch, prompt, ubatch=512))
    whole = totals(prefill_ops(CFG, batch, prompt))
    lm_head = 2 * batch * CFG.hidden * CFG.vocab
    # Same per-token matmul and causal-attention FLOPs; only logits may differ.
    assert chunked["flops"] == pytest.approx(whole["flops"], abs=lm_head * 0.01, rel=1e-6)
    # Chunking re-reads layer weights once per micro-batch (the LM head only on
    # passes that emit logits, hence the 10% slack).
    passes = -(-batch * prompt // 512)
    ratio = chunked["weight_bytes"] / whole["weight_bytes"]
    assert 0.9 * passes <= ratio <= passes


def _synthetic_measurements(tmp_path, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    lb, bb = [], []
    for p in (128, 512, 1024, 2048):
        for d in (0, 2048):
            ops = prefill_pass_ops(CFG, 1, p, d)
            lb.append(dict(n_prompt=p, n_gen=0, n_depth=d, n_ubatch=512, flash_attn=1,
                           avg_ts=p / pass_time(ops, T4, TRUE), stddev_ts=0))
    for d in (0, 512, 2048, 8192):
        ops = decode_run_ops(CFG, 1, d, 128)
        lb.append(dict(n_prompt=0, n_gen=128, n_depth=d, n_ubatch=512, flash_attn=1,
                       avg_ts=128 / pass_time(ops, T4, TRUE), stddev_ts=0))
    for pp in (128, 512, 2048):
        for b in (1, 4, 16, 64):
            t_pp = pass_time(prefill_pass_ops(CFG, b, pp), T4, TRUE)
            t_tg = pass_time(decode_run_ops(CFG, b, pp, 128), T4, TRUE)
            bb.append(dict(pp=pp, tg=128, b=b, s_pp=b * pp / t_pp, s_tg=b * 128 / t_tg))
    lb, bb = pd.DataFrame(lb), pd.DataFrame(bb)
    for df, col in ((lb, "avg_ts"), (bb, "s_pp"), (bb, "s_tg")):
        df[col] *= rng.lognormal(0, noise, len(df))
    lb.to_csv(tmp_path / "llama_bench.csv", index=False)
    bb.to_csv(tmp_path / "batched_bench.csv", index=False)
    return tmp_path


def test_fit_on_batch1_recovers_efficiency_and_generalizes(tmp_path):
    cases = load_cases(_synthetic_measurements(tmp_path))
    assert cases.groupby(["source", "phase"]).size().to_dict() == {
        ("batched-bench", "decode"): 12, ("batched-bench", "prefill"): 12,
        ("llama-bench", "decode"): 4, ("llama-bench", "prefill"): 8}
    assert cases.flash_attn.all()
    compiled = CompiledCases([case_ops(CFG, c) for c in cases.itertuples()])
    train = (cases.source == "llama-bench").to_numpy()
    fit = fit_efficiency(cases, compiled, T4, mask=train)
    assert fit.compute == pytest.approx(TRUE.compute, rel=0.1)
    assert fit.memory == pytest.approx(TRUE.memory, rel=0.1)
    pred = predict_cases(cases, compiled, T4, fit)
    err = np.abs(pred / cases.measured_tok_s - 1)[~train]
    assert err.mean() < 0.06


def test_case_tokens():
    cases = pd.DataFrame([dict(phase="prefill", batch=4, prompt=512, gen=0),
                          dict(phase="decode", batch=4, prompt=0, gen=128)])
    assert [case_tokens(c) for c in cases.itertuples()] == [2048, 512]
