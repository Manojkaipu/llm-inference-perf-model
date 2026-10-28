import pytest

from perfmodel import PEAK, T4, TINYLLAMA_1_1B, Efficiency, decode_ops, forward_ops, totals
from perfmodel.config import LLAMA_68M, LLAMA_160M
from perfmodel.ops import mixed_forward_ops
from perfmodel.predict import pass_time
from perfmodel.serving import (acceptance_rate, alpha_from_acceptance_rate,
                               expected_tokens_per_step, predict_continuous, prefill_time,
                               spec_speedup, spec_step_times)

CFG = TINYLLAMA_1_1B
EFF = Efficiency(compute=0.25, memory=0.85, kernel_overhead_s=8e-6)


@pytest.mark.parametrize("cfg, expected", [(LLAMA_68M, 68e6), (LLAMA_160M, 162e6)])
def test_draft_param_counts(cfg, expected):
    assert cfg.params() == pytest.approx(expected, rel=0.02)


def test_mixed_pass_reads_weights_once_and_sums_attention():
    a, b = (8, 1, 1000), (1, 512, 0)
    mixed = totals(mixed_forward_ops(CFG, [a, b]))
    sep = [totals(forward_ops(CFG, *a)), totals(forward_ops(CFG, *b))]
    assert mixed["weight_bytes"] == pytest.approx(sep[0]["weight_bytes"])   # not doubled
    assert mixed["kv_bytes"] == pytest.approx(sep[0]["kv_bytes"] + sep[1]["kv_bytes"])
    assert mixed["flops"] == pytest.approx(sep[0]["flops"] + sep[1]["flops"], rel=1e-9)


def test_mixed_pass_is_cheaper_than_separate_passes_when_memory_bound():
    a, b = (8, 1, 1000), (1, 64, 0)
    t_mixed = pass_time(mixed_forward_ops(CFG, [a, b]), T4, EFF)
    t_sep = pass_time(forward_ops(CFG, *a), T4, EFF) + pass_time(forward_ops(CFG, *b), T4, EFF)
    assert t_mixed < t_sep


def test_empty_segments_are_ignored_with_attention_ramp():
    # Concurrency 1 leaves no other running decodes; with an attention-parallelism
    # ramp a zero-work segment used to divide by zero.
    ramp = Efficiency(compute=0.25, memory=0.85, attention_half_work=13.0, overlap=1.1)
    p = predict_continuous(CFG, T4, ramp, 1, 512, 128)
    assert p["ttft_s"] > 0
    assert len(mixed_forward_ops(CFG, [(0, 1, 100), (1, 8, 0)])) == len(forward_ops(CFG, 1, 8, 0))


def test_continuous_batching_throughput_grows_and_tpot_rises():
    preds = [predict_continuous(CFG, T4, EFF, c, 512, 128) for c in (1, 8, 64)]
    assert preds[0]["output_tok_s"] < preds[1]["output_tok_s"] < preds[2]["output_tok_s"]
    assert preds[0]["tpot_s"] < preds[1]["tpot_s"] < preds[2]["tpot_s"]
    for p in preds:
        assert p["ttft_s"] > p["tpot_s"] * 0.5


def test_prefix_cache_ttft_falls_with_cached_prefix():
    times = [prefill_time(CFG, T4, EFF, 1920, p) for p in (0, 512, 1024, 1792)]
    assert times == sorted(times, reverse=True)
    assert times[-1] < times[0] / 4


def test_acceptance_math_round_trips():
    for k in (1, 3, 5):
        for alpha in (0.1, 0.5, 0.9):
            rate = acceptance_rate(alpha, k)
            assert alpha_from_acceptance_rate(rate, k) == pytest.approx(alpha, abs=1e-6)
    assert expected_tokens_per_step(0.0, 4) == 1
    assert expected_tokens_per_step(1.0, 4) == 5


def test_spec_speedup_behaves():
    t = spec_step_times(CFG, LLAMA_68M, T4, EFF, k=3, batch=1, ctx=256)
    # Verifying 4 positions costs about one decode step when memory-bound.
    assert t["verify_step_s"] < 1.2 * t["target_step_s"]
    assert spec_speedup(t, 3, 1.0) < 1                    # nothing accepted: pure loss
    assert spec_speedup(t, 3, 4.0) > 2                    # everything accepted
    # Larger batches leave less memory-bound slack, so the same acceptance helps less.
    t64 = spec_step_times(CFG, LLAMA_68M, T4, EFF, k=3, batch=64, ctx=256)
    assert spec_speedup(t64, 3, 3.0) < spec_speedup(t, 3, 3.0)


def test_simulation_matches_single_request_and_queues_a_burst():
    from perfmodel.serving import simulate_closed_loop
    eff = Efficiency(compute=0.3, memory=0.85, kernel_overhead_s=3e-6)
    one = simulate_closed_loop(CFG, T4, eff, 1, 3, 512, 32, budget=2048)
    prefill = pass_time(forward_ops(CFG, 1, 512, 0), T4, eff)
    step = pass_time(decode_ops(CFG, 1, 512 + 16), T4, eff)
    assert one[0]["ttft_s"] == pytest.approx(prefill, rel=1e-6)
    assert one[0]["tpot_s"] == pytest.approx(step, rel=0.05)
    assert [r["t_start"] for r in one] == sorted(r["t_start"] for r in one)   # closed loop
    # 8 prompts arriving at once with a 2048-token budget: the first step prefills 4.
    # In the next step those 4 decode first, leaving 2044 tokens, so 3 more prompts
    # finish and the 8th spills into a short third step (its last 4 tokens + 7 decodes),
    # as vLLM schedules it.
    burst = sorted(r["ttft_s"] for r in simulate_closed_loop(CFG, T4, eff, 8, 8, 512, 4, budget=2048))
    assert burst[0] == pytest.approx(burst[3])
    assert burst[4] == pytest.approx(burst[6]) and burst[7] > burst[6]
    assert burst[4] == pytest.approx(2 * burst[0], rel=0.05)
