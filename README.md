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
| 4. Measured T4 runs (llama.cpp) and error report | notebook + validation code ready; awaiting Colab run |
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
- **Op time** = `max(FLOPs / (peak * eff_compute), bytes / (BW * eff_memory)) + launch overhead`.
  Ops run serially; there is no overlap between ops.
- Matmuls and attention use the FP16 tensor-core peak. Norms and activations use the FP32 CUDA-core peak.
- Prefill attention exploits causal masking (half the score matrix). Logits
  are computed only for each sequence's last position.
- Static batching: all sequences share prompt and generation length.
- Memory fit = weights + full KV cache <= 90% of device memory (activation
  workspace ignored).
- `REALISTIC` efficiencies (T4: 45% compute, 80% bandwidth, 4 us/kernel;
  H100: 70%, 85%, 3 us) are **priors** that phase 4 calibrates.

## Early observations (before validation)

- TinyLlama decode at batch 1 is weight-bandwidth bound. The T4 ceiling is ~150 tok/s
  at peak bandwidth and ~110 tok/s with the realistic prior.
- As batch grows, the weight GEMMs climb the roofline toward the ridge point,
  and **KV-cache reads in attention become the dominant cost**. On the T4 at
  2k context the crossover falls between batch 32 and 64. Attention intensity is stuck near
  `n_heads / n_kv_heads` FLOP/byte no matter the batch size, so batching does not help it.
- On the H100, kernel launch overhead is about half of the batch-1 decode step for a
  1B model. That makes small-model decode there latency-bound, not bandwidth-bound.
