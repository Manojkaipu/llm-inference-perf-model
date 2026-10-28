import json

import numpy as np
import pandas as pd
import pytest

from perfmodel import T4, TINYLLAMA_1_1B, Efficiency
from perfmodel.serving import (acceptance_rate, expected_tokens_per_step, predict_continuous,
                               prefill_time, spec_step_times)
from perfmodel.config import LLAMA_68M
from perfmodel.vllm_analysis import (acceptance_from_metrics, fit_vllm_profile,
                                     load, memory_efficiency, predict_cb, predict_prefix,
                                     predict_spec, predict_spec_from_step_cost, spec_config_info, summarize_cb,
                                     summarize_prefix, summarize_spec, breakeven_rate)

CFG = TINYLLAMA_1_1B
TRUE = Efficiency(compute=0.30, memory=0.85, kernel_overhead_s=6e-6, overlap=1.5)
OVERHEAD_S = 0.015            # fixed per-request cost the model does not include
CB_IN, CB_OUT = 512, 128
SD_IN, SD_OUT = 256, 128
ALPHAS = {"copy": 0.9, "natural": 0.6, "natural_t1": 0.4, "random_t1": 0.1}


def _write_synthetic(d, seed=0):
    rng = np.random.default_rng(seed)
    noise = lambda n=None: rng.lognormal(0, 0.02, n)
    configs = {"baseline": None,
               "llama68m_k3": {"model": "JackFram/llama-68m", "num_speculative_tokens": 3}}
    env = {"gpu_name": "Tesla T4", "cb": {"input": CB_IN, "output": CB_OUT},
           "pc": {"prompt_len": 1920, "output": 8},
           "sd": {"input": SD_IN, "output": SD_OUT, "configs": configs, "ngram": {}}}
    (d / "env.json").write_text(json.dumps(env))
    pd.DataFrame({"metric": ["read_bw"], "value": [TRUE.memory * T4.mem_bw], "unit": ["B/s"]}
                 ).to_csv(d / "microbench.csv", index=False)

    rows, levels = [], []
    for c in (1, 4, 16, 64):
        p = predict_continuous(CFG, T4, TRUE, c, CB_IN, CB_OUT)
        n = 4 * c
        ttft = p["ttft_s"] * noise(n)
        tpot = p["tpot_s"] * noise(n)
        rows += [{"concurrency": c, "ttft_s": a, "e2e_s": a + (CB_OUT - 1) * b,
                  "out_tokens": CB_OUT, "t_start": i * 0.5} for i, (a, b) in enumerate(zip(ttft, tpot))]
        metrics = {"vllm:num_preemptions_total": 0.0, "vllm:request_num_preemptions_bucket": 9.0 * n}
        levels.append({"concurrency": c, "n_requests": n, "out_tokens": n * CB_OUT,
                       "wall_s": n * CB_OUT / p["output_tok_s"], "metrics": json.dumps(metrics)})
    pd.DataFrame(rows).to_csv(d / "cb_requests.csv", index=False)
    pd.DataFrame(levels).to_csv(d / "cb_levels.csv", index=False)

    for state in ("on", "off"):
        pr = []
        for p in (0, 512, 1024, 1792):
            cached = p if state == "on" else 0
            t = prefill_time(CFG, T4, TRUE, 1920, cached) + OVERHEAD_S
            pr += [{"config": f"caching_{state}", "prefix_len": p, "prompt_len": 1920,
                    "warm": i == 0, "ttft_s": t * noise() * (1.5 if i == 0 else 1),
                    "cached_tokens": cached} for i in range(5)]
        pd.DataFrame(pr).to_csv(d / f"prefix_{state}.csv", index=False)

    ctx = SD_IN + SD_OUT // 2
    for name, sc in configs.items():
        reqs, lv = [], []
        for ptype, alpha in ALPHAS.items():
            for c in (1, 8):
                t = spec_step_times(CFG, LLAMA_68M, T4, TRUE, 3, c, ctx)
                if sc is None:
                    tpot, metrics = t["target_step_s"], {}
                else:
                    tps = expected_tokens_per_step(alpha, 3)
                    tpot = t["spec_step_s"] / tps
                    steps = 100.0
                    metrics = {"vllm:spec_decode_num_drafts_total": steps,
                               "vllm:spec_decode_num_draft_tokens_total": 3 * steps,
                               "vllm:spec_decode_num_accepted_tokens_total": (tps - 1) * steps,
                               "vllm:spec_decode_num_accepted_tokens_per_pos_total": (tps - 1) * steps}
                reqs += [{"config": name, "prompt_type": ptype, "concurrency": c, "ttft_s": 0.05,
                          "e2e_s": 0.05 + (SD_OUT - 1) * tpot * noise(), "out_tokens": SD_OUT}
                         for _ in range(4)]
                lv.append({"config": name, "prompt_type": ptype, "concurrency": c,
                           "out_tokens": 4 * SD_OUT, "wall_s": 4 * SD_OUT * tpot / c,
                           "n_requests": 4, "metrics": json.dumps(metrics)})
        pd.DataFrame(reqs).to_csv(d / f"spec_{name}.csv", index=False)
        pd.DataFrame(lv).to_csv(d / f"spec_{name}_levels.csv", index=False)
    return env


def test_acceptance_from_metrics_ignores_per_position_counter():
    m = json.dumps({"vllm:spec_decode_num_drafts_total": 10,
                    "vllm:spec_decode_num_draft_tokens_total": 30,
                    "vllm:spec_decode_num_accepted_tokens_total": 15,
                    "vllm:spec_decode_num_accepted_tokens_per_pos_total": 15})
    a = acceptance_from_metrics(m)
    assert a["acceptance_rate"] == pytest.approx(0.5)
    assert a["tokens_per_step"] == pytest.approx(2.5)
    assert np.isnan(acceptance_from_metrics("{}")["acceptance_rate"])


def test_full_analysis_recovers_synthetic_ground_truth(tmp_path):
    _write_synthetic(tmp_path)
    data = load(tmp_path)
    mem = memory_efficiency(data["micro"], T4)
    assert mem == pytest.approx(TRUE.memory)

    cb = summarize_cb(data["cb"], data["cb_levels"])
    assert (cb.preemptions == 0).all()          # histogram buckets must not be counted
    fit = fit_vllm_profile(CFG, T4, cb, CB_IN, CB_OUT, mem)
    cb = predict_cb(CFG, T4, fit, cb, CB_IN, CB_OUT)
    err = (cb.pred_tpot_s / cb.tpot_p50_s - 1).abs()
    assert err.max() < 0.04

    pc = predict_prefix(CFG, T4, fit, summarize_prefix(data["prefix"]))
    on = pc[pc.config == "caching_on"]
    assert on.ttft_p50_s.is_monotonic_decreasing
    # Residual = the fixed overhead we injected, nearly constant across prefix lengths.
    assert pc.residual_s.median() == pytest.approx(OVERHEAD_S, abs=0.006)
    assert pc.residual_s.std() < 0.006

    info = spec_config_info(data["env"])
    assert info == {"llama68m_k3": ("JackFram/llama-68m", 3, "draft_model")}
    sd = predict_spec(CFG, T4, fit, summarize_spec(data["spec"], data["spec_levels"]), info,
                      SD_IN + SD_OUT // 2)
    sd = sd[sd.config != "baseline"]
    assert sd.acceptance_rate.tolist() == pytest.approx(
        [acceptance_rate(ALPHAS[p], 3) for p in sd.prompt_type], rel=1e-6)
    assert (sd.pred_speedup / sd.speedup - 1).abs().max() < 0.08
    # Step cost does not depend on acceptance, so other prompt types' steps predict speedup.
    sc = predict_spec_from_step_cost(summarize_spec(data["spec"], data["spec_levels"]))
    sc = sc[sc.config != "baseline"]
    assert (sc.pred_speedup_step_cost / sc.speedup - 1).abs().max() < 0.06
    # High acceptance speeds decoding up; very low acceptance slows it down at batch 1.
    s1 = sd[sd.concurrency == 1].set_index("prompt_type").speedup
    assert s1["copy"] > 1.5 and s1["random_t1"] < 1
    be = breakeven_rate(CFG, T4, fit, info, "llama68m_k3", 1, SD_IN + SD_OUT // 2)
    assert 0 < be < acceptance_rate(ALPHAS["natural"], 3)
