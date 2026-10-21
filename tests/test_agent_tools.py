import json
import math

import pytest

from perfmodel import T4, TINYLLAMA_1_1B, LLAMA_3_1_8B, H100, PEAK, Efficiency
from perfmodel.agent_tools import (PROFILES, audit_numbers, describe_setup, extract_numbers,
                                   find_crossover, op_breakdown, sweep)
from perfmodel.sweep import find_transition, point


def test_point_shares_sum_to_one_and_time_matches_throughput():
    p = point(TINYLLAMA_1_1B, T4, PEAK, "decode", 8, 1024)
    assert sum(p["share"].values()) == pytest.approx(1.0)
    assert p["tokens_per_s"] == pytest.approx(8 / (p["pass_time_ms"] / 1e3))


@pytest.mark.parametrize("event", ["attention_exceeds_linear", "linear_compute_bound"])
def test_find_transition_brackets_exact_flip(event):
    from perfmodel.sweep import EVENTS
    pred = EVENTS[event][1]
    r = find_transition(TINYLLAMA_1_1B, T4, PEAK, "decode", event, "batch", 2048)
    assert r["result"] == "found"
    b = r["value"]
    assert pred(point(TINYLLAMA_1_1B, T4, PEAK, "decode", b, 2048))
    assert not pred(point(TINYLLAMA_1_1B, T4, PEAK, "decode", b - 1, 2048))


def test_linear_compute_bound_batch_near_ridge_point():
    # Weight GEMM intensity at batch B is ~B FLOP/byte (2 FLOP per 2-byte weight per row),
    # so at peak efficiency the flip should sit near the T4 ridge point (~203).
    r = find_transition(TINYLLAMA_1_1B, T4, PEAK, "decode", "linear_compute_bound", "batch", 128)
    assert 150 < r["value"] < 260


def test_exceeds_memory_on_t4():
    r = find_transition(LLAMA_3_1_8B, T4, PEAK, "decode", "exceeds_memory", "batch", 512)
    assert r["result"] == "already true at the smallest value"
    r = find_transition(TINYLLAMA_1_1B, T4, PEAK, "decode", "exceeds_memory", "batch", 4096)
    assert r["result"] == "found"
    assert r["before"]["fits_in_memory"] and not r["at"]["fits_in_memory"]


def test_tools_return_json_and_validate_inputs():
    setup = json.loads(describe_setup())
    assert "calibrated" in setup["gpus"]["T4"]["profiles"]
    assert "calibrated" not in setup["gpus"]["H100-SXM"]["profiles"]
    s = json.loads(sweep("TinyLlama-1.1B", "T4", "calibrated", "decode", [1, 8], [512]))
    assert len(s["points"]) == 2
    c = json.loads(find_crossover("TinyLlama-1.1B", "T4", "calibrated", "decode",
                                  "attention_exceeds_linear", "batch", 2048))
    assert c["result"] == "found" and c["value"] > 1
    o = json.loads(op_breakdown("TinyLlama-1.1B", "H100-SXM", "prior", "prefill", 1, 512))
    assert abs(sum(r["time_share"] for r in o["ops"]) - 1) < 0.01
    with pytest.raises(ValueError):
        sweep("TinyLlama-1.1B", "H100-SXM", "calibrated", "decode", [1], [512])
    with pytest.raises(ValueError):
        sweep("TinyLlama-1.1B", "T4", "prior", "decode", list(range(1, 100)), [512])


def test_calibrated_profile_loaded_with_overlap():
    eff = PROFILES["T4"]["calibrated"][0]
    assert isinstance(eff, Efficiency)
    assert eff.attention_half_work and not math.isinf(eff.overlap)


def test_extract_numbers_ignores_identifiers_and_list_markers():
    text = "## 3. Results\n1. On the T4, Llama-3.1-8B needs 16.06 GB; TinyLlama-1.1B hits 3,720 tok/s."
    assert [v for _, v in extract_numbers(text)] == [16.06, 3720.0]


def test_audit_numbers():
    sources = ['{"tokens_per_s":3720,"share":{"attention":0.6731},"pass_time_ms":9.712}']
    report = ("Throughput reaches 3,720 tok/s; attention is 67.3% of time and a step takes "
              "9.7 ms. Batch 8 is fine. The made-up figure is 4242 tok/s.")
    r = audit_numbers(report, sources)
    assert r["unverified"] == ["4242"]
    assert r["checked"] == 4
