"""Generate the Colab notebooks (they share setup cells, so they are built here).

    python notebooks/make_notebooks.py

  gpu_llamacpp_benchmark.ipynb  measurements for scripts/validate.py, on any supported GPU
  llamacpp_kernel_probe.ipynb   fine batch sweep, CUDA-graphs A/B and Nsight kernel summary,
                                for scripts/analyze_probe.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n")}


def write(path, cells, gpu_type="T4"):
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i}"
        c["source"] = c["source"].splitlines(keepends=True)
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": gpu_type, "provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}}}
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", path.name)


# --- shared cells -----------------------------------------------------------------

HELPERS = """
import datetime, json, os, re, shutil, subprocess, sys
import pandas as pd

BIN = "/content/llama.cpp/build/bin"
GGUF = "/content/tinyllama-1.1b-f16.gguf"
KV_BYTES_PER_TOKEN = 2 * 22 * 256 * 2      # TinyLlama: K+V x layers x kv_dim x fp16
WEIGHT_BYTES = 2.2e9


def run(cmd, check=False, env=None, **kw):
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env={**os.environ, **(env or {})}, **kw)
    if check and r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\\n")
        raise RuntimeError(f"command failed ({r.returncode})")
    return r


def gpu_slug(name):
    n = name.upper()
    for key, slug in (("T4", "t4"), ("L4", "l4"), ("H100", "h100"), ("A100", "a100")):
        if key in n:
            if key == "H100" and "PCIE" in n:
                return "h100_pcie"
            if key == "A100" and "80GB" in n:
                return "a100_80gb"
            return slug
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def parse_batched_bench(log, **extra):
    rows = []
    for line in log.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 10 and cells[0].isdigit():
            pp, tg, b, n_kv = map(int, cells[:4])
            t_pp, s_pp, t_tg, s_tg, t, s = map(float, cells[4:])
            rows.append(dict(pp=pp, tg=tg, b=b, n_kv=n_kv, t_pp=t_pp, s_pp=s_pp,
                             t_tg=t_tg, s_tg=s_tg, t=t, s=s, **extra))
    return pd.DataFrame(rows)


def batched_bench(npp, ntg, npl, env=None, prefix=()):
    \"\"\"Run llama-batched-bench with flash attention on; returns (CompletedProcess, fa_args).\"\"\"
    ctx = max(npl) * (max(npp) + ntg)
    cmd = [*prefix, f"{BIN}/llama-batched-bench", "-m", GGUF, "-ngl", 99, "-c", ctx,
           "-b", N_BATCH, "-ub", N_UBATCH, "-npp", ",".join(map(str, npp)),
           "-ntg", ntg, "-npl", ",".join(map(str, npl))]
    for fa in (["-fa", "on"], ["-fa"], []):           # newer: -fa on|off|auto, older: bare -fa
        r = run(cmd + fa, env=env)
        if r.returncode == 0:
            return r, fa
        print(r.stderr[-1500:])
    return r, None
"""

GPU_CHECK = """
q = "name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem,compute_cap"
info = run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], check=True).stdout
info = info.strip().splitlines()[0]
print(info)
name, driver, mem, power, sm_max, mem_max, cc = [s.strip() for s in info.split(",")]
SLUG = gpu_slug(name)
MEM_BYTES = float(mem.split()[0]) * 2**20
CUDA_ARCH = cc.replace(".", "")
OUT = f"/content/results_{SLUG}"
os.makedirs(OUT, exist_ok=True)
env = dict(gpu_name=name, driver=driver, memory_total=mem, power_limit=power,
           max_sm_clock=sm_max, max_mem_clock=mem_max, compute_cap=cc, hf_model=HF_MODEL,
           n_batch=N_BATCH, n_ubatch=N_UBATCH,
           timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
print(f"GPU {name} -> results in {OUT}, building for sm_{CUDA_ARCH}")

# Throttle-reason field was renamed in newer drivers; use whichever this one accepts.
fields = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu"
for reason in ("clocks_event_reasons.active", "clocks_throttle_reasons.active", None):
    f = fields + (f",{reason}" if reason else "")
    if run(["nvidia-smi", f"--query-gpu={f}", "--format=csv"]).returncode == 0:
        fields = f
        break
gpu_log = open(f"{OUT}/gpu_log.csv", "w")
gpu_logger = subprocess.Popen(["nvidia-smi", f"--query-gpu={fields}", "--format=csv", "-lms", "500"],
                              stdout=gpu_log)
"""

BUILD = """
shutil.rmtree("/content/llama.cpp", ignore_errors=True)
run(["git", "clone", "--depth", "1", "--branch", LLAMA_CPP_REF,
     "https://github.com/ggml-org/llama.cpp", "/content/llama.cpp"], check=True)
env["llama_cpp_ref"] = LLAMA_CPP_REF
env["llama_cpp_commit"] = run(["git", "-C", "/content/llama.cpp", "rev-parse", "HEAD"],
                              check=True).stdout.strip()
print("commit", env["llama_cpp_commit"])

run(["cmake", "-S", "/content/llama.cpp", "-B", "/content/llama.cpp/build",
     "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}",   # this GPU's arch only
     "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"], check=True)
run(["cmake", "--build", "/content/llama.cpp/build", "--config", "Release", "-j", os.cpu_count(),
     "--target", "llama-bench", "llama-batched-bench"], check=True)
assert os.path.exists(f"{BIN}/llama-bench") and os.path.exists(f"{BIN}/llama-batched-bench")
"""

MODEL = """
from huggingface_hub import snapshot_download

model_dir = snapshot_download(HF_MODEL, allow_patterns=["*.json", "*.safetensors", "tokenizer.model"])
if not any(f.endswith(".safetensors") for f in os.listdir(model_dir)):
    model_dir = snapshot_download(HF_MODEL, allow_patterns=["*.json", "*.bin", "tokenizer.model"])

run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece"], check=True)
convert = [sys.executable, "/content/llama.cpp/convert_hf_to_gguf.py", model_dir,
           "--outtype", "f16", "--outfile", GGUF]
r = run(convert)
if r.returncode != 0:   # converter needs a dependency Colab lacks; install its requirements
    print(r.stderr[-2000:])
    run([sys.executable, "-m", "pip", "install", "-q", "-r",
         "/content/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"], check=True)
    run(convert, check=True)
print(f"{os.path.getsize(GGUF) / 1e9:.2f} GB")
"""


def package(zip_stem):
    return f"""
gpu_logger.terminate()
gpu_log.close()
json.dump(env, open(f"{{OUT}}/env.json", "w"), indent=2)

g = pd.read_csv(f"{{OUT}}/gpu_log.csv", skipinitialspace=True)
print(g.describe().loc[["min", "50%", "max"]].to_string())

archive = f"/content/{{SLUG}}_{zip_stem}"
shutil.make_archive(archive, "zip", OUT)
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    os.makedirs("/content/drive/MyDrive/llm_perf_model", exist_ok=True)
    shutil.copy(archive + ".zip", "/content/drive/MyDrive/llm_perf_model/")
from google.colab import files
files.download(archive + ".zip")
"""


# --- benchmark notebook ------------------------------------------------------------

benchmark = [
    md("""
# llama.cpp benchmark: TinyLlama-1.1B (fp16) on any Colab GPU

Measures the numbers that `scripts/validate.py` compares against the analytical model.
Works on a T4, L4, A100 or H100: the GPU is detected, llama.cpp is built for its
architecture, and the batch grid is sized to its memory.

| File | What |
|---|---|
| `microbench.csv` | achievable DRAM bandwidth and fp16 matmul FLOP/s (PyTorch) |
| `llama_bench.csv` | batch-1 prompt processing and generation at several context depths |
| `batched_bench.csv` | multi-sequence prefill and decode across batch sizes |
| `gpu_log.csv` | SM/memory clocks, power and temperature every 0.5 s |
| `env.json` | GPU, driver, llama.cpp commit, flags used |

**How to run:** Runtime > Change runtime type > pick a GPU, then Runtime > Run all.
The last cell downloads `<gpu>_llamacpp.zip`. Unzip it into `measurements/<gpu>_llamacpp/`
in the repo and run `python scripts/validate.py measurements/<gpu>_llamacpp`.
"""),
    code("""
# --- configuration -----------------------------------------------------------
LLAMA_CPP_REF = "master"   # branch or release tag; the exact commit is recorded
HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SAVE_TO_DRIVE = False      # also copy results to My Drive/llm_perf_model/

N_BATCH, N_UBATCH = 2048, 512

# llama-bench (batch 1): prompt sizes, generated tokens, and context depths
BENCH_PROMPTS = [128, 512, 1024, 2048]
BENCH_GEN = 128
BENCH_DEPTHS = [0, 512, 2048, 8192, 32768]
REPS = 5

# llama-batched-bench: prompt lengths; batch sizes are powers of two up to what fits
BB_NPP = [128, 512, 2048]
BB_NTG = 128
BB_MAX_NPL = 512
""" + HELPERS),
    md("## 1. Detect the GPU and start a clock/power logger"),
    code(GPU_CHECK),
    md("""
## 2. Microbenchmarks: what this GPU actually sustains

Runs in a separate process so PyTorch releases all GPU memory before llama.cpp starts.
"""),
    code('''
MICRO = r"""
import time, torch
ev = lambda: torch.cuda.Event(enable_timing=True)

def timed(fn, iters):
    out = []
    for _ in range(iters):
        s, e = ev(), ev()
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / 1e3)
    return out

med = lambda xs: sorted(xs)[len(xs) // 2]
n = 512 * 2**20                                   # 1 GiB of fp16, well beyond any L2
x = torch.empty(n, dtype=torch.float16, device="cuda").uniform_()
y = torch.empty_like(x)
timed(lambda: y.copy_(x), 3)
copy_bw = 2 * n * 2 / med(timed(lambda: y.copy_(x), 20))      # read + write
timed(lambda: x.sum(dtype=torch.float32), 3)
read_bw = n * 2 / med(timed(lambda: x.sum(dtype=torch.float32), 20))
del x, y

m = 8192
a = torch.randn(m, m, dtype=torch.float16, device="cuda"); b = torch.randn_like(a)
flop = 2 * m**3
timed(lambda: a @ b, 3)
burst = flop / med(timed(lambda: a @ b, 10))
ts, t0 = [], time.time()
while time.time() - t0 < 30:                      # 30 s to reach thermal/power steady state
    ts += timed(lambda: a @ b, 10)
sustained = flop / med(ts[len(ts) // 2:])

print("metric,value,unit")
print(f"copy_bw,{copy_bw:.6g},B/s")
print(f"read_bw,{read_bw:.6g},B/s")
print(f"matmul_flops_burst,{burst:.6g},FLOP/s")
print(f"matmul_flops_sustained,{sustained:.6g},FLOP/s")
"""
r = run([sys.executable, "-c", MICRO], check=True)
open(f"{OUT}/microbench.csv", "w").write(r.stdout)
pd.read_csv(f"{OUT}/microbench.csv")
'''),
    md("## 3. Build llama.cpp with CUDA for this GPU (~10-20 min on Colab's CPUs)"),
    code(BUILD),
    md("## 4. Download TinyLlama and convert to an fp16 GGUF"),
    code(MODEL),
    md("""
## 5. llama-bench: batch 1, prompt processing (pp) and generation (tg) at each context depth

Runs with flash attention both off and on. The model assumes fused attention, so FA-on rows
are the main comparison.
"""),
    code("""
bench = [f"{BIN}/llama-bench", "-m", GGUF, "-ngl", 99, "-b", N_BATCH, "-ub", N_UBATCH, "-r", REPS,
         "-p", ",".join(map(str, BENCH_PROMPTS)), "-n", BENCH_GEN,
         "-d", ",".join(map(str, BENCH_DEPTHS)), "-o", "csv"]
for fa in (["-fa", "0,1"], ["-fa", "off,on"]):     # flag syntax differs across versions
    r = run(bench + fa)
    if r.returncode == 0 and r.stdout.strip():
        break
    print(r.stderr[-1500:])
else:
    raise RuntimeError("llama-bench failed")
env["llama_bench_fa_arg"] = " ".join(fa)
open(f"{OUT}/llama_bench.csv", "w").write(r.stdout)
open(f"{OUT}/llama_bench.log", "w").write(r.stderr)
lb = pd.read_csv(f"{OUT}/llama_bench.csv")
lb[["n_prompt", "n_gen", "n_depth", "flash_attn", "avg_ts", "stddev_ts"]]
"""),
    md("""
## 6. llama-batched-bench: many sequences at once

Batch sizes are powers of two up to the largest whose KV cache fits in ~85% of GPU memory.
"""),
    code("""
per_seq = (max(BB_NPP) + BB_NTG) * KV_BYTES_PER_TOKEN
fit = int((0.85 * MEM_BYTES - WEIGHT_BYTES - 1.5e9) // per_seq)
npl = [b for b in (2**i for i in range(10)) if b <= min(fit, BB_MAX_NPL)]
print("batch sizes:", npl)

r, fa = batched_bench(BB_NPP, BB_NTG, npl)
while fa is None and len(npl) > 1:     # most likely out of memory: drop the largest batch
    npl = npl[:-1]
    print("retrying with npl =", npl)
    r, fa = batched_bench(BB_NPP, BB_NTG, npl)
assert fa is not None, "llama-batched-bench failed"
env["batched_bench_fa_arg"] = " ".join(fa) or "(default)"
env["batched_bench_npl"] = npl
log = r.stdout + "\\n" + r.stderr
open(f"{OUT}/batched_bench.log", "w").write(log)
bb = parse_batched_bench(log, n_ubatch=N_UBATCH, flash_attn=fa != [])
assert len(bb), "could not parse the batched-bench table; see batched_bench.log"
bb.to_csv(f"{OUT}/batched_bench.csv", index=False)
bb
"""),
    md("## 7. Package the results"),
    code(package("llamacpp")),
]

# --- kernel probe notebook ---------------------------------------------------------

probe = [
    md("""
# llama.cpp kernel probe: why is batch-2 decode slow?

On the T4, batch-2 decode was slower per step than batch 4 (17 ms vs 11 ms), and
batch 16-32 ran ~15% slower than the model predicts. This notebook collects evidence
to explain both:

| Experiment | File | Question |
|---|---|---|
| A. Fine batch sweep (1-64) | `probe_fine.csv` | exactly where does step time jump? |
| B. CUDA graphs off | `probe_graphs_off.csv` | is the batch-2 penalty lost graph capture? |
| C. Nsight Systems kernel summary | `kern_<batch>_cuda_gpu_kern_sum.csv` | which kernels run at each batch size? |

Experiment C installs Nsight Systems if needed and is skipped if that fails.

**How to run:** use the same GPU type as the run you want to explain (T4 by default), then
Runtime > Run all (~25 min, mostly the build). Unzip `<gpu>_kernel_probe.zip` into
`measurements/<gpu>_kernel_probe/` and run `python scripts/analyze_probe.py measurements/<gpu>_kernel_probe`.
"""),
    code("""
# --- configuration -----------------------------------------------------------
LLAMA_CPP_REF = "master"
HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SAVE_TO_DRIVE = False
N_BATCH, N_UBATCH = 2048, 512

NPP, NTG = 512, 64                       # same prompt length as the training rows
FINE_NPL = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64]
AB_NPL = [1, 2, 3, 4, 8, 16, 32]         # CUDA graphs on vs off
NSYS_NPL = [1, 2, 4, 16, 32]             # kernel summaries
NSYS_NPP, NSYS_NTG = 32, 128             # short prompt so decode kernels dominate the profile
""" + HELPERS),
    md("## 1. Detect the GPU"),
    code(GPU_CHECK),
    md("## 2. Build llama.cpp"),
    code(BUILD),
    md("## 3. Model"),
    code(MODEL),
    md("## A. Fine batch sweep (default settings)"),
    code("""
r, fa = batched_bench([NPP], NTG, FINE_NPL)
assert fa is not None, "llama-batched-bench failed"
env["fa_arg"] = " ".join(fa) or "(default)"
open(f"{OUT}/probe_fine.log", "w").write(r.stdout + "\\n" + r.stderr)
fine = parse_batched_bench(r.stdout + "\\n" + r.stderr, graphs="default")
fine["step_ms"] = fine.t_tg / fine.tg * 1e3
fine.to_csv(f"{OUT}/probe_fine.csv", index=False)
fine[["b", "step_ms", "s_tg"]]
"""),
    md("## B. Same batches with CUDA graphs disabled"),
    code("""
r, _ = batched_bench([NPP], NTG, AB_NPL, env={"GGML_CUDA_DISABLE_GRAPHS": "1"})
open(f"{OUT}/probe_graphs_off.log", "w").write(r.stdout + "\\n" + r.stderr)
off = parse_batched_bench(r.stdout + "\\n" + r.stderr, graphs="off")
off["step_ms"] = off.t_tg / off.tg * 1e3
off.to_csv(f"{OUT}/probe_graphs_off.csv", index=False)
off.merge(fine[["b", "step_ms"]], on="b", suffixes=("_graphs_off", "_default"))
"""),
    md("""
## C. Nsight Systems: which kernels run at each batch size

Each batch size is profiled in its own short run (prompt 32, 128 decode steps), so per-step
kernel counts are roughly `instances / 128`.
"""),
    code("""
def find_nsys():
    path = shutil.which("nsys")
    if path:
        return path
    for root in ("/usr/local/cuda/bin", "/opt/nvidia/nsight-systems"):
        for dirpath, _, files_ in os.walk(root):
            if "nsys" in files_:
                return os.path.join(dirpath, "nsys")
    # Not preinstalled: try NVIDIA's apt repo (package names carry a version).
    run(["apt-get", "update", "-qq"])
    pkgs = run(["apt-cache", "search", "nsight-systems"]).stdout.split("\\n")
    names = sorted(p.split()[0] for p in pkgs if p.split() and "nsight-systems" in p.split()[0])
    for pkg in reversed(names):
        if run(["apt-get", "install", "-y", "-qq", pkg]).returncode == 0 and shutil.which("nsys"):
            return shutil.which("nsys")
    return None

NSYS = find_nsys()
env.update(nsys=NSYS, probe_npp=NPP, probe_ntg=NTG, nsys_npp=NSYS_NPP, nsys_ntg=NSYS_NTG)
print("nsys:", NSYS)
if NSYS:
    for b in NSYS_NPL:
        rep = f"/content/prof_{b}"
        r, _ = batched_bench([NSYS_NPP], NSYS_NTG, [b],
                             prefix=(NSYS, "profile", "-o", rep, "--force-overwrite", "true",
                                     "--trace", "cuda", "--sample", "none", "--cpuctxsw", "none"))
        s = run([NSYS, "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                 "--output", f"{OUT}/kern_{b}", "--force-export", "true", rep + ".nsys-rep"])
        print(s.stdout[-500:], s.stderr[-500:])
    print(sorted(f for f in os.listdir(OUT) if f.startswith("kern_")))
else:
    print("Nsight Systems unavailable; experiments A and B still answer most of the question.")
"""),
    md("## Package the results"),
    code(package("kernel_probe")),
]

if __name__ == "__main__":
    write(HERE / "gpu_llamacpp_benchmark.ipynb", benchmark, gpu_type="T4")
    write(HERE / "llamacpp_kernel_probe.ipynb", probe, gpu_type="T4")
