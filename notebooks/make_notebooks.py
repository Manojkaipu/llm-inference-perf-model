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
if PERSIST:   # survive disconnects: results and caches live on Google Drive
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/llm_perf_model"
else:
    BASE = "/content"
OUT = f"{BASE}/results_{SLUG}{RESULTS_SUFFIX}"
CACHE = f"{BASE}/cache"
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
ENV_PATH = f"{OUT}/env.json"
env = json.load(open(ENV_PATH)) if os.path.exists(ENV_PATH) else {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
env.update(gpu_name=name, driver=driver, memory_total=mem, power_limit=power,
           max_sm_clock=sm_max, max_mem_clock=mem_max, compute_cap=cc, hf_model=HF_MODEL,
           n_batch=N_BATCH, n_ubatch=N_UBATCH)


def save_env():
    json.dump(env, open(ENV_PATH, "w"), indent=2)


def done(fname):
    \"\"\"True if a previous (disconnected) session already produced this output.\"\"\"
    path = f"{OUT}/{fname}"
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"{fname} already exists from an earlier session; skipping. Delete it to re-measure.")
        return True
    return False


save_env()
print(f"GPU {name} -> results in {OUT}, building for sm_{CUDA_ARCH}")

# Throttle-reason field was renamed in newer drivers; use whichever this one accepts.
fields = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu"
for reason in ("clocks_event_reasons.active", "clocks_throttle_reasons.active", None):
    f = fields + (f",{reason}" if reason else "")
    if run(["nvidia-smi", f"--query-gpu={f}", "--format=csv"]).returncode == 0:
        fields = f
        break
resumed = os.path.exists(f"{OUT}/gpu_log.csv")
gpu_log = open(f"{OUT}/gpu_log.csv", "a")      # append across sessions, header only once
gpu_logger = subprocess.Popen(
    ["nvidia-smi", f"--query-gpu={fields}", "--format=csv" + (",noheader" if resumed else ""),
     "-lms", "500"], stdout=gpu_log)
"""

BUILD = """
BUILD_CACHE = f"{CACHE}/llamacpp_sm{CUDA_ARCH}"
cached = os.path.exists(f"{BUILD_CACHE}/bin/llama-batched-bench") and not REBUILD
shutil.rmtree("/content/llama.cpp", ignore_errors=True)
if cached:
    # Reuse binaries from an earlier session; fetch the matching source for the converter.
    commit = open(f"{BUILD_CACHE}/commit.txt").read().strip()
    os.makedirs("/content/llama.cpp")
    run(["git", "-C", "/content/llama.cpp", "init", "-q"], check=True)
    run(["git", "-C", "/content/llama.cpp", "fetch", "-q", "--depth", "1",
         "https://github.com/ggml-org/llama.cpp", commit], check=True)
    run(["git", "-C", "/content/llama.cpp", "checkout", "-q", "FETCH_HEAD"], check=True)
    shutil.copytree(f"{BUILD_CACHE}/bin", BIN, dirs_exist_ok=True)
    for f in os.listdir(BIN):
        os.chmod(f"{BIN}/{f}", 0o755)            # Drive drops the executable bit
    print("using cached build", commit)
else:
    run(["git", "clone", "--depth", "1", "--branch", LLAMA_CPP_REF,
         "https://github.com/ggml-org/llama.cpp", "/content/llama.cpp"], check=True)
    commit = run(["git", "-C", "/content/llama.cpp", "rev-parse", "HEAD"], check=True).stdout.strip()
    run(["cmake", "-S", "/content/llama.cpp", "-B", "/content/llama.cpp/build",
         "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH}",   # this GPU's arch only
         "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"], check=True)
    run(["cmake", "--build", "/content/llama.cpp/build", "--config", "Release", "-j", os.cpu_count(),
         "--target", "llama-bench", "llama-batched-bench"], check=True)
    shutil.copytree(BIN, f"{BUILD_CACHE}/bin", dirs_exist_ok=True)
    open(f"{BUILD_CACHE}/commit.txt", "w").write(commit)
assert os.path.exists(f"{BIN}/llama-bench") and os.path.exists(f"{BIN}/llama-batched-bench")
if env.get("llama_cpp_commit", commit) != commit:
    print("WARNING: earlier results in this folder used llama.cpp", env["llama_cpp_commit"])
env.update(llama_cpp_ref=LLAMA_CPP_REF, llama_cpp_commit=commit)
save_env()
"""

MODEL = """
GGUF_CACHE = f"{CACHE}/tinyllama-1.1b-f16.gguf"
if os.path.exists(GGUF_CACHE):
    shutil.copy(GGUF_CACHE, GGUF)
    print("using cached GGUF")
else:
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
    shutil.copy(GGUF, GGUF_CACHE)
print(f"{os.path.getsize(GGUF) / 1e9:.2f} GB")
"""


def package(zip_stem):
    return f"""
gpu_logger.terminate()
gpu_log.close()
save_env()

g = pd.read_csv(f"{{OUT}}/gpu_log.csv", skipinitialspace=True)
print(g.describe().loc[["min", "50%", "max"]].to_string())

archive = f"/content/{{SLUG}}_{zip_stem}"
shutil.make_archive(archive, "zip", OUT)
if PERSIST:
    shutil.copy(archive + ".zip", BASE)
    print("zip also saved to", BASE)
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

**How to run:** Runtime > Change runtime type > pick a GPU, then Runtime > Run all, and
approve the Google Drive prompt. Results, the llama.cpp build and the model are saved to
`My Drive/llm_perf_model/` as each step finishes, so **if the runtime disconnects, reconnect
and Run all again**: finished steps are skipped and the build is reused.
The last cell downloads `<gpu>_llamacpp.zip`. Unzip it into `measurements/<gpu>_llamacpp/`
in the repo and run `python scripts/validate.py measurements/<gpu>_llamacpp`.
"""),
    code("""
# --- configuration -----------------------------------------------------------
LLAMA_CPP_REF = "master"   # branch or release tag; the exact commit is recorded
HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PERSIST = True             # results + build/model caches on Google Drive (resumable)
REBUILD = False            # True forces a fresh llama.cpp build even if one is cached
RESULTS_SUFFIX = "_llamacpp"

N_BATCH, N_UBATCH = 2048, 512

# llama-bench (batch 1): prompt sizes, generated tokens, and context depths
BENCH_PROMPTS = [128, 512, 1024, 2048]
BENCH_GEN = 128
BENCH_DEPTHS = [0, 512, 2048, 8192, 32768]
REPS = 5

# llama-batched-bench: prompt lengths; batch sizes are powers of two up to what fits
BB_NPP = [128, 512, 2048]
BB_NTG = 128
BB_MAX_NPL = 256
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
if not done("microbench.csv"):
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
if not done("llama_bench.csv"):
    for fa in (["-fa", "0,1"], ["-fa", "off,on"]):     # flag syntax differs across versions
        r = run(bench + fa)
        if r.returncode == 0 and r.stdout.strip():
            break
        print(r.stderr[-1500:])
    else:
        raise RuntimeError("llama-bench failed")
    env["llama_bench_fa_arg"] = " ".join(fa)
    save_env()
    open(f"{OUT}/llama_bench.log", "w").write(r.stderr)
    open(f"{OUT}/llama_bench.csv", "w").write(r.stdout)
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

frames = []
for npp in BB_NPP:                     # one run per prompt length, saved as it finishes
    fname = f"batched_bench_pp{npp}.csv"
    if not done(fname):
        sizes = list(npl)
        r, fa = batched_bench([npp], BB_NTG, sizes)
        while fa is None and len(sizes) > 1:   # most likely out of memory: drop the largest batch
            sizes = sizes[:-1]
            print("retrying with npl =", sizes)
            r, fa = batched_bench([npp], BB_NTG, sizes)
        assert fa is not None, "llama-batched-bench failed"
        log = r.stdout + "\\n" + r.stderr
        open(f"{OUT}/batched_bench_pp{npp}.log", "w").write(log)
        df = parse_batched_bench(log, n_ubatch=N_UBATCH, flash_attn=fa != [])
        assert len(df), f"could not parse the batched-bench table; see batched_bench_pp{npp}.log"
        env["batched_bench_fa_arg"] = " ".join(fa) or "(default)"
        env.setdefault("batched_bench_npl", {})[str(npp)] = sizes
        save_env()
        df.to_csv(f"{OUT}/{fname}", index=False)
    frames.append(pd.read_csv(f"{OUT}/{fname}"))
bb = pd.concat(frames, ignore_index=True)
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
Runtime > Run all (~25 min, mostly the build) and approve the Google Drive prompt. After a
disconnect, Run all again: finished experiments are skipped and the build is reused. Unzip `<gpu>_kernel_probe.zip` into
`measurements/<gpu>_kernel_probe/` and run `python scripts/analyze_probe.py measurements/<gpu>_kernel_probe`.
"""),
    code("""
# --- configuration -----------------------------------------------------------
LLAMA_CPP_REF = "master"
HF_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PERSIST = True             # results + build/model caches on Google Drive (resumable)
REBUILD = False            # True forces a fresh llama.cpp build even if one is cached
RESULTS_SUFFIX = "_kernel_probe"
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
if not done("probe_fine.csv"):
    r, fa = batched_bench([NPP], NTG, FINE_NPL)
    assert fa is not None, "llama-batched-bench failed"
    env.update(fa_arg=" ".join(fa) or "(default)", probe_npp=NPP, probe_ntg=NTG)
    save_env()
    open(f"{OUT}/probe_fine.log", "w").write(r.stdout + "\\n" + r.stderr)
    fine = parse_batched_bench(r.stdout + "\\n" + r.stderr, graphs="default")
    fine["step_ms"] = fine.t_tg / fine.tg * 1e3
    fine.to_csv(f"{OUT}/probe_fine.csv", index=False)
fine = pd.read_csv(f"{OUT}/probe_fine.csv")
fine[["b", "step_ms", "s_tg"]]
"""),
    md("## B. Same batches with CUDA graphs disabled"),
    code("""
if not done("probe_graphs_off.csv"):
    r, _ = batched_bench([NPP], NTG, AB_NPL, env={"GGML_CUDA_DISABLE_GRAPHS": "1"})
    open(f"{OUT}/probe_graphs_off.log", "w").write(r.stdout + "\\n" + r.stderr)
    off = parse_batched_bench(r.stdout + "\\n" + r.stderr, graphs="off")
    off["step_ms"] = off.t_tg / off.tg * 1e3
    off.to_csv(f"{OUT}/probe_graphs_off.csv", index=False)
off = pd.read_csv(f"{OUT}/probe_graphs_off.csv")
off.merge(fine[["b", "step_ms"]], on="b", suffixes=("_graphs_off", "_default"))
"""),
    md("""
## C. Nsight Systems: which kernels run at each batch size

Each batch size is profiled in its own short run (prompt 32, 128 decode steps), so per-step
kernel counts are roughly `instances / 128`. llama.cpp runs decode as CUDA graphs, and by
default Nsight records a graph as one item instead of listing its kernels, so this
uses `--cuda-graph-trace=node`. If this nsys version lacks that option, graphs are disabled
for the profiled runs instead (experiment B shows that barely changes step time).
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
print("nsys:", NSYS)
if NSYS:
    node_trace = "cuda-graph-trace" in run([NSYS, "profile", "--help"]).stdout
    graph_args = ["--cuda-graph-trace", "node"] if node_trace else []
    graph_env = None if node_trace else {"GGML_CUDA_DISABLE_GRAPHS": "1"}
    env.update(nsys=NSYS, nsys_npp=NSYS_NPP, nsys_ntg=NSYS_NTG,
               nsys_graph_mode="node trace" if node_trace else "graphs disabled")
    save_env()
    for b in NSYS_NPL:
        if done(f"kern_{b}_cuda_gpu_kern_sum.csv"):
            continue
        rep = f"/content/prof_{b}"
        r, _ = batched_bench([NSYS_NPP], NSYS_NTG, [b], env=graph_env,
                             prefix=(NSYS, "profile", "-o", rep, "--force-overwrite", "true",
                                     "--trace", "cuda", "--sample", "none", "--cpuctxsw", "none",
                                     *graph_args))
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
