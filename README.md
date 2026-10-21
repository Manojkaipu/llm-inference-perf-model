# Analytical Performance Model for LLM Inference

Predicts per-layer FLOPs, memory traffic, roofline placement and tokens/s for
Llama-style decoders on NVIDIA T4 and H100, to be validated against measured
runs on a Colab T4.

## Status

| Phase | State |
|---|---|
| 1. Per-op FLOP / byte model (prefill + decode, GQA, SwiGLU, KV cache) | done |
| 2. Rooflines for T4 and H100 | done |
| 3. Tokens/s prediction (TTFT, TPOT, decode and end-to-end throughput) | done |
| 4. Measured T4 runs (llama.cpp) and error report | done ([report](reports/validation_TinyLlama-1.1B_T4_llamacpp.md)) |
| 5. Sweep agent that writes up bottleneck shifts | planned |

## Layout

```
perfmodel/config.py    model presets (TinyLlama-1.1B, Llama-3.2-1B, Llama-3.1-8B)
perfmodel/hardware.py  GPU specs and efficiency profiles (peak / realistic priors)
perfmodel/ops.py       per-op FLOPs and bytes for one forward pass
perfmodel/predict.py   roofline timing, memory-fit check, throughput prediction
perfmodel/plots.py     roofline, throughput, time-breakdown and validation figures
perfmodel/validate.py  llama.cpp workloads, vectorized evaluation, efficiency fitting
scripts/run_baseline.py  writes results/*.csv and figures/*.png
scripts/validate.py    measured vs predicted error report -> reports/
notebooks/t4_llamacpp_benchmark.ipynb  Colab T4 measurement run
tests/                 sanity checks, plus fit recovery on synthetic measurements
```

```
pip install -r requirements.txt
python scripts/run_baseline.py                         # TinyLlama, realistic priors
python scripts/run_baseline.py --model Llama-3.1-8B --peak
python -m pytest -q
```

## Validation workflow

1. Open `notebooks/t4_llamacpp_benchmark.ipynb` in Colab and choose Run all (T4 runtime, ~30 min).
2. Unzip the downloaded `t4_llamacpp.zip` into `measurements/t4_llamacpp/`.
3. `python scripts/validate.py measurements/t4_llamacpp` writes `reports/validation_*.md` with the error report.

The report compares four efficiency profiles: datasheet **peak**, the hand-set
**prior**, **microbench** (bandwidth and FLOP/s measured with PyTorch on the
same GPU, no fitting) and **fitted**. The fitted profile is fit on batch-1
`llama-bench` rows only. The `batched-bench` rows (batch 1-128) are held out, so
their error is out-of-sample.

To match what llama.cpp actually executes, the predicted workloads include
prompt splitting into 512-token micro-batches (weights are re-read per chunk) and
decode over the exact context range each benchmark covers.

## Modeling assumptions

- **Ideal traffic**: each weight, activation and KV entry crosses DRAM once
  (perfect tiling reuse, fused FlashAttention-style attention, no S x S
  score matrix in memory).
- **Op time** = `combine(FLOPs / (peak * eff_compute), bytes / (BW * eff_memory)) + launch overhead`.
  `combine` is `max` by default (classic roofline) or the soft maximum
  `(t_c^p + t_m^p)^(1/p)` when calibrated. Ops run serially; there is no overlap between ops.
- Attention can optionally have its bandwidth scaled by available parallelism
  (`Efficiency.attention_half_work`). It is off by default and calibrated in validation.
- Matmuls and attention use the FP16 tensor-core peak. Norms and activations use the FP32 CUDA-core peak.
- Prefill attention exploits causal masking (half the score matrix). Logits
  are computed only for each sequence's last position.
- Static batching: all sequences share prompt and generation length.
- Memory fit = weights + full KV cache <= 90% of device memory (activation
  workspace ignored).
- `REALISTIC` efficiencies (T4: 45% compute, 80% bandwidth, 4 us/kernel;
  H100: 70%, 85%, 3 us) are **priors** that phase 4 calibrates.

## Validation results: TinyLlama-1.1B fp16 on a Colab T4 (llama.cpp)

Training data: 35 rows (batch-1 `llama-bench` and batched rows at a 512-token prompt).
Held out: 29 batched rows at 128- and 2048-token prompts, batch 1-128.

| Held-out MAPE | decode | prefill |
|---|---|---|
| Datasheet peaks | 103% | 313% |
| Hand-set prior | 47% | 96% |
| Measured bandwidth / FLOP/s (no fitting) | 52% | 56% |
| Fitted classic roofline (max) | 13.5% | 3.5% |
| **Refined** | **8.7%** | **4.4%** |

What closed the gap, in order of size:

1. **The T4 is power-capped, not compute-capped.** Under load it sat at its 70 W
   limit 100% of the time, and the SM clock dropped to a median of 690 MHz
   (boost: 1590 MHz). Effective tensor-core throughput is ~1/4 of the 65 TFLOPS
   datasheet figure. Memory clocks did not throttle, so bandwidth held at 85% of peak.
2. **Decode attention needs parallelism to use the bandwidth.** At batch 1 there
   are only 4 KV heads' worth of work for 40 SMs, so attention reached ~20% of bandwidth.
   The refined model scales attention bandwidth by `w / (w + 12.9)`, where
   `w` = sequences x KV heads.
3. **Compute and memory barely overlap within a kernel.** The fitted soft-max
   exponent is 1.1, so op time is close to `t_compute + t_memory`, not `max`.
   This matters for mid-size batches near the ridge point.
4. **Per-kernel overhead is ~10 us,** about 3 ms of the ~10 ms batch-1 decode step.

Not modeled, so flagged and excluded by explicit rules:

- **Batch-2 decode is slower per step than batch 4** (17 ms vs 11 ms), at every prompt
  length. That is likely a llama.cpp kernel-dispatch effect.
- **The first batched-bench measurement was a cold start,** 3.6x slower than the same
  workload in `llama-bench`.

The remaining decode error is a consistent +14-17% over-prediction at batch 16-32.
With flash attention off, prefill runs ~2x slower than predicted, because the model
assumes fused attention.

## Early observations (before validation)

- TinyLlama decode at batch 1 is weight-bandwidth bound. The T4 ceiling is ~150 tok/s
  at peak bandwidth and ~110 tok/s with the realistic prior.
- As batch grows, the weight GEMMs climb the roofline toward the ridge point,
  and **KV-cache reads in attention become the dominant cost**. On the T4 at
  2k context the crossover falls between batch 32 and 64. Attention intensity is stuck near
  `n_heads / n_kv_heads` FLOP/byte no matter the batch size, so batching does not help it.
- On the H100, kernel launch overhead is about half of the batch-1 decode step for a
  1B model. That makes small-model decode there latency-bound, not bandwidth-bound.
