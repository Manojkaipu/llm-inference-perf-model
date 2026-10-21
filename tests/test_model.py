import math

import pytest

from perfmodel import (H100, LLAMA_3_1_8B, PEAK, T4, TINYLLAMA_1_1B, decode_ops, op_table,
                       predict, prefill_ops, totals)


@pytest.mark.parametrize("cfg, expected", [(TINYLLAMA_1_1B, 1.100e9), (LLAMA_3_1_8B, 8.030e9)])
def test_param_counts_match_published(cfg, expected):
    assert cfg.params() == pytest.approx(expected, rel=0.01)


def test_decode_flops_close_to_2x_params_at_short_context():
    cfg = TINYLLAMA_1_1B
    t = totals(decode_ops(cfg, batch=1, context_len=1))
    # Embedding lookup does no matmul; the remainder is ~2 FLOP per parameter.
    matmul_params = cfg.params() - cfg.vocab * cfg.hidden
    assert t["flops"] == pytest.approx(2 * matmul_params, rel=0.02)


def test_decode_b1_reads_every_weight_once():
    cfg = TINYLLAMA_1_1B
    t = totals(decode_ops(cfg, batch=1, context_len=1))
    # Embedding table is gathered (one row), not streamed.
    streamed = cfg.weight_bytes() - cfg.vocab * cfg.hidden * cfg.dtype_bytes
    assert t["weight_bytes"] == pytest.approx(streamed, rel=1e-3)


def test_kv_read_scales_with_batch_and_context():
    cfg = TINYLLAMA_1_1B
    a = totals(decode_ops(cfg, 4, 1000))["kv_bytes"]
    b = totals(decode_ops(cfg, 8, 2000))["kv_bytes"]
    assert b / a == pytest.approx(4, rel=0.01)


def test_causal_attention_halves_prefill_scores():
    cfg = TINYLLAMA_1_1B
    attn = lambda causal: next(o for o in prefill_ops(cfg, 1, 2048, causal=causal)
                               if o.name == "attention").flops
    assert attn(True) / attn(False) == pytest.approx(0.5, rel=0.01)


def test_decode_small_batch_is_memory_bound_and_prefill_compute_bound():
    cfg = TINYLLAMA_1_1B
    for gpu in (T4, H100):
        dec = op_table(decode_ops(cfg, 1, 512), gpu)
        assert (dec.loc[dec.kind == "linear", "bound"] == "memory").all()
        pre = op_table(prefill_ops(cfg, 1, 2048), gpu)
        assert (pre.loc[pre.op == "gate_up_proj", "bound"] == "compute").all()


def test_decode_b1_peak_bound_equals_bandwidth_over_bytes():
    cfg = TINYLLAMA_1_1B
    p = predict(cfg, T4, batch=1, prompt_len=16, gen_len=2, eff=PEAK)
    bytes_ = totals(decode_ops(cfg, 1, 16))["bytes"]
    assert p.decode_tok_s == pytest.approx(T4.mem_bw / bytes_, rel=0.02)


def test_8b_does_not_fit_on_t4():
    assert not predict(LLAMA_3_1_8B, T4, 1, 128, 2).fits
    assert predict(LLAMA_3_1_8B, H100, 1, 128, 2).fits
