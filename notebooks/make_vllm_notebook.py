"""Generate notebooks/vllm_serving_benchmark.ipynb.

    python notebooks/make_vllm_notebook.py

Measures continuous batching, prefix caching and speculative decoding with vLLM's
OpenAI-compatible server, for scripts/analyze_vllm.py.
"""
from pathlib import Path

from make_notebooks import code, md, write

HERE = Path(__file__).resolve().parent

# Original text used for natural-language and copy prompts (~600 tokens).
PASSAGE = (
    "Lighthouses were once the most important pieces of infrastructure on any coast. "
    "Before radio and satellite navigation, a ship approaching land at night had little more "
    "than a compass, a chart and the light of a tower to tell it where the rocks were. Each "
    "light had its own pattern of flashes, so a navigator who counted the seconds between "
    "them could work out which headland lay ahead. The keepers who tended these lights lived "
    "on remote islands for months at a time. They trimmed the wicks, polished the lenses, "
    "wound the clockwork that turned the lamp, and wrote the weather into a logbook every few "
    "hours. In storms they stayed awake through the night, because a light that went dark "
    "could cost a crew their lives. The great lenses that made the beams visible from twenty "
    "miles away were invented in France in the early nineteenth century. Instead of one thick "
    "curved piece of glass, the designer used rings of prisms arranged around a central lens, "
    "which bent the light into a narrow beam while keeping the whole assembly light enough to "
    "rotate. The idea spread quickly, and within a few decades most major lighthouses used it. "
    "Over the twentieth century the keepers were replaced, one station at a time, by "
    "electric lamps, automatic switches and remote monitoring. The last keepers left their "
    "towers in the closing decades of the century, and many of the old stations became "
    "museums, guest houses or simply quiet landmarks. Ships today rely on satellite "
    "positioning, radar and electronic charts, but the lights still turn, partly as a backup "
    "and partly because sailors trust a signal they can see with their own eyes. Walking up "
    "the spiral stairs of an old tower, a visitor can still find the brass fittings, the "
    "narrow bunks and the small windows through which the keepers watched the sea. The logbooks "
    "that survive record ordinary days in careful handwriting: the direction of the wind, the "
    "ships that passed, the supplies that arrived by boat, and now and then a rescue. Read "
    "together, they describe a way of life built around a single duty, which was to keep the "
    "light burning from sunset to sunrise, whatever the weather."
)

CONFIG = r'''
# --- configuration -----------------------------------------------------------
TARGET = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DRAFTS = {"llama68m": "JackFram/llama-68m", "llama160m": "JackFram/llama-160m"}
VLLM_SPEC = ""            # e.g. "==0.11.0" to pin a version; empty = latest
PERSIST = True            # save results to Google Drive as each part finishes (resumable)
MAX_MODEL_LEN = 2048      # TinyLlama's trained context
GPU_MEM_UTIL = 0.85
MAX_NUM_SEQS = 128
PORT = 8000
SERVER_TIMEOUT_S = 1200   # first start downloads the model and compiles

# 1. Continuous batching: closed loop, `concurrency` requests always in flight
CB_INPUT, CB_OUTPUT = 512, 128
CB_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128]
CB_MIN_REQUESTS = 32      # per level: max(CB_MIN_REQUESTS, 4 x concurrency)

# 2. Prefix caching: one request at a time, prompts sharing a prefix of each length
PC_PROMPT_LEN, PC_OUTPUT = 1920, 8
PC_PREFIX_LENS = [0, 256, 512, 1024, 1536, 1792]
PC_REPEATS = 8            # plus one warm-up request per prefix length

# 3. Speculative decoding: each config is a separate server
SD_INPUT, SD_OUTPUT = 256, 128
SD_CONCURRENCY = [1, 8]
SD_REQUESTS = {1: 6, 8: 16}
SD_CONFIGS = {            # name -> --speculative-config (None = no speculation)
    "baseline": None,
    "llama68m_k3": {"model": DRAFTS["llama68m"], "num_speculative_tokens": 3},
    "llama160m_k3": {"model": DRAFTS["llama160m"], "num_speculative_tokens": 3},
    "llama160m_k5": {"model": DRAFTS["llama160m"], "num_speculative_tokens": 5},
}
# Used only if no draft-model config can start on this vLLM/GPU: prompt-lookup speculation.
SD_NGRAM_FALLBACK = {
    "ngram_k3": {"method": "ngram", "num_speculative_tokens": 3,
                 "prompt_lookup_max": 4, "prompt_lookup_min": 2},
}

import asyncio, datetime, json, os, re, shutil, signal, subprocess, sys, time, urllib.request
import numpy as np
import pandas as pd

VENV = "/content/vllm-env"
URL = f"http://127.0.0.1:{PORT}"


def run(cmd, check=False, **kw):
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if check and r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
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
'''

GPU_SETUP = r'''
q = "name,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.mem,compute_cap"
info = run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], check=True).stdout
name, driver, mem, power, sm_max, mem_max, cc = [s.strip() for s in info.strip().splitlines()[0].split(",")]
SLUG = gpu_slug(name)

BASE = "/content"
if PERSIST:
    from google.colab import drive
    try:
        drive.mount("/content/drive")
        BASE = "/content/drive/MyDrive/llm_perf_model"
    except Exception as e:   # popup dismissed/timed out, or the account blocks Drive mounts
        print(f"WARNING: Google Drive mount failed ({e}). Continuing with local storage: "
              "results are still downloaded at the end, but a disconnect cannot be resumed.")
OUT = f"{BASE}/results_{SLUG}_vllm"
os.makedirs(f"{OUT}/logs", exist_ok=True)
ENV_PATH = f"{OUT}/env.json"
env = json.load(open(ENV_PATH)) if os.path.exists(ENV_PATH) else {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
env.update(gpu_name=name, driver=driver, memory_total=mem, power_limit=power,
           max_sm_clock=sm_max, max_mem_clock=mem_max, compute_cap=cc, target=TARGET,
           max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_NUM_SEQS,
           cb=dict(input=CB_INPUT, output=CB_OUTPUT),
           pc=dict(prompt_len=PC_PROMPT_LEN, output=PC_OUTPUT),
           sd=dict(input=SD_INPUT, output=SD_OUTPUT, configs=SD_CONFIGS, ngram=SD_NGRAM_FALLBACK))


def save_env():
    json.dump(env, open(ENV_PATH, "w"), indent=2)


def done(fname):
    """True if a previous (disconnected) session already produced this output."""
    path = f"{OUT}/{fname}"
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"{fname} already exists from an earlier session; skipping. Delete it to re-measure.")
        return True
    return False


save_env()
print(f"GPU {name} (sm_{cc.replace('.', '')}) -> results in {OUT}")

fields = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu"
for reason in ("clocks_event_reasons.active", "clocks_throttle_reasons.active", None):
    f = fields + (f",{reason}" if reason else "")
    if run(["nvidia-smi", f"--query-gpu={f}", "--format=csv"]).returncode == 0:
        fields = f
        break
resumed = os.path.exists(f"{OUT}/gpu_log.csv")
gpu_log = open(f"{OUT}/gpu_log.csv", "a")
gpu_logger = subprocess.Popen(
    ["nvidia-smi", f"--query-gpu={fields}", "--format=csv" + (",noheader" if resumed else ""),
     "-lms", "500"], stdout=gpu_log)
'''

INSTALL = r'''
# vLLM goes in its own Python 3.12 environment: Colab's kernel Python may be newer than
# vLLM's wheels support, and this keeps vLLM's PyTorch separate from Colab's.
if not os.path.exists(f"{VENV}/bin/vllm"):
    run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    run([sys.executable, "-m", "uv", "venv", VENV, "--python", "3.12"], check=True)
    pkg = f"vllm{VLLM_SPEC}"
    r = run([sys.executable, "-m", "uv", "pip", "install", "--python", f"{VENV}/bin/python",
             pkg, "--torch-backend=auto"])
    if r.returncode != 0:          # older uv without --torch-backend
        print(r.stderr[-1500:])
        run([sys.executable, "-m", "uv", "pip", "install", "--python", f"{VENV}/bin/python", pkg],
            check=True)
ver = run([f"{VENV}/bin/python", "-c",
           "import vllm, torch; print(vllm.__version__, torch.__version__, torch.version.cuda)"],
          check=True).stdout.split()
env.update(vllm_version=ver[0], vllm_torch=ver[1], vllm_torch_cuda=ver[2])
print("vLLM", ver[0], "| torch", ver[1], "| CUDA", ver[2])

# The server runs from the venv without activating it, so put the venv's tools
# (e.g. ninja, which FlashInfer uses to JIT-compile kernels) on PATH explicitly.
SERVER_ENV = {**os.environ, "PATH": f"{VENV}/bin:" + os.environ.get("PATH", "")}
if os.path.exists("/usr/local/cuda/bin/nvcc"):
    SERVER_ENV.setdefault("CUDA_HOME", "/usr/local/cuda")
EXTRA_ENV = dict(env.get("server_extra_env") or {})   # fallbacks found in an earlier session

# Precompiled FlashInfer kernels, so vLLM does not need to compile any at startup.
fi = run([f"{VENV}/bin/python", "-c", "import flashinfer; print(flashinfer.__version__)"])
fi_ver = fi.stdout.strip() if fi.returncode == 0 else ""
if fi_ver:
    r = run([sys.executable, "-m", "uv", "pip", "install", "--python", f"{VENV}/bin/python",
             f"flashinfer-cubin=={fi_ver}"])
    env["flashinfer_cubin_installed"] = r.returncode == 0
dk = run([f"{VENV}/bin/vllm", "download-kernels"], env=SERVER_ENV)
env.update(flashinfer_version=fi_ver, vllm_download_kernels_ok=dk.returncode == 0)
save_env()
print("FlashInfer", fi_ver or "not installed", "| cubin:", env.get("flashinfer_cubin_installed"),
      "| vllm download-kernels:", "ok" if dk.returncode == 0 else "failed")

run([sys.executable, "-m", "pip", "install", "-q", "aiohttp"], check=True)

# Model configs, so the analysis can check its architecture presets against them.
from huggingface_hub import hf_hub_download
os.makedirs(f"{OUT}/configs", exist_ok=True)
for repo in [TARGET, *DRAFTS.values()]:
    shutil.copy(hf_hub_download(repo, "config.json"), f"{OUT}/configs/{repo.split('/')[-1]}.json")
'''

MICRO = r'''
MICRO = """
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
n = 512 * 2**20
x = torch.empty(n, dtype=torch.float16, device="cuda").uniform_()
y = torch.empty_like(x)
timed(lambda: y.copy_(x), 3)
copy_bw = 2 * n * 2 / med(timed(lambda: y.copy_(x), 20))
timed(lambda: x.sum(dtype=torch.float32), 3)
read_bw = n * 2 / med(timed(lambda: x.sum(dtype=torch.float32), 20))
del x, y
m = 8192
a = torch.randn(m, m, dtype=torch.float16, device="cuda"); b = torch.randn_like(a)
timed(lambda: a @ b, 3)
ts, t0 = [], time.time()
while time.time() - t0 < 15:
    ts += timed(lambda: a @ b, 10)
print("metric,value,unit")
print(f"copy_bw,{copy_bw:.6g},B/s")
print(f"read_bw,{read_bw:.6g},B/s")
print(f"matmul_flops_sustained,{2 * m**3 / med(ts[len(ts) // 2:]):.6g},FLOP/s")
"""
if not done("microbench.csv"):
    r = run([f"{VENV}/bin/python", "-c", MICRO], check=True)
    open(f"{OUT}/microbench.csv", "w").write(r.stdout)
pd.read_csv(f"{OUT}/microbench.csv")
'''

CLIENT = r'''
import aiohttp

# --- server lifecycle ----------------------------------------------------------
def _healthy():
    try:
        return urllib.request.urlopen(f"{URL}/health", timeout=2).status == 200
    except Exception:
        return False


def wait_gpu_free(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        used = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).stdout
        if float(used.split()[0]) < 1500:
            return
        time.sleep(3)
    print("WARNING: GPU memory still in use")


def stop_server(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


# A startup failure while FlashInfer builds a kernel (no ninja/nvcc, or a toolkit that
# does not match vLLM's CUDA) can be avoided with vLLM's PyTorch sampler instead.
FLASHINFER_BUILD_ERROR = re.compile(r"No such file or directory: 'ninja'|flashinfer[./]jit", re.I)


def flashinfer_build_failed(log):
    # Every vLLM log may start with a harmless "FlashInfer kernels are disabled" warning.
    lines = [l for l in log.splitlines() if "FlashInfer kernels are disabled" not in l]
    return bool(FLASHINFER_BUILD_ERROR.search("\n".join(lines)))


def _launch(tag, i, extra, attempt):
    log_path = f"{OUT}/logs/server_{tag}_{i}{'_retry' if attempt else ''}.log"
    cmd = [f"{VENV}/bin/vllm", "serve", TARGET, "--served-model-name", "target",
           "--dtype", "half", "--max-model-len", MAX_MODEL_LEN,
           "--gpu-memory-utilization", GPU_MEM_UTIL, "--max-num-seqs", MAX_NUM_SEQS,
           "--port", PORT, "--seed", 0, *extra]
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), " ".join(f"{k}={v}" for k, v in EXTRA_ENV.items()), flush=True)
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                            start_new_session=True, env={**SERVER_ENV, **EXTRA_ENV})
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < SERVER_TIMEOUT_S:
        if _healthy():
            return proc, log_path, time.time() - t0
        time.sleep(5)
    stop_server(proc)
    return None, log_path, time.time() - t0


def start_server(tag, variants):
    """Start `vllm serve` with the first argument variant that comes up healthy
    (flag spellings differ between vLLM versions)."""
    for i, extra in enumerate(variants):
        for attempt in range(2):
            wait_gpu_free()
            proc, log_path, secs = _launch(tag, i, extra, attempt)
            if proc:
                print(f"server '{tag}' ready after {secs:.0f} s")
                env.setdefault("servers", {})[tag] = {"args": extra, "startup_s": round(secs),
                                                      "extra_env": dict(EXTRA_ENV)}
                save_env()
                return proc
            log = open(log_path).read()
            print(f"server '{tag}' variant {i} did not start; last log lines:")
            print("\n".join(log.splitlines()[-25:]))
            if attempt == 0 and "VLLM_USE_FLASHINFER_SAMPLER" not in EXTRA_ENV \
                    and flashinfer_build_failed(log):
                print("FlashInfer could not build its sampling kernel; retrying with vLLM's "
                      "PyTorch sampler (VLLM_USE_FLASHINFER_SAMPLER=0) for this and later servers.")
                EXTRA_ENV["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
                env["server_extra_env"] = dict(EXTRA_ENV)
                save_env()
                continue
            break
    env.setdefault("servers", {})[tag] = {"failed": True, "variants": variants}
    save_env()
    return None


async def with_server(tag, variants, outputs, body):
    """Run `body` against a fresh server unless all its outputs already exist."""
    if all(done(f) for f in outputs):
        return True
    proc = start_server(tag, variants)
    if proc is None:
        return False
    try:
        await run_load([random_prompt(64) for _ in range(8)], 4, 16, 0.0)   # warm-up
        await body()
    finally:
        stop_server(proc)
    return True


# --- metrics -------------------------------------------------------------------
METRIC_KEYS = ("spec_decode", "prefix_cache", "preemption")


def scrape():
    """Sum of each vLLM Prometheus counter we care about, across label sets."""
    try:
        text = urllib.request.urlopen(f"{URL}/metrics", timeout=5).read().decode()
    except Exception:
        return {}
    vals = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([A-Za-z_:][\w:]*)(?:\{[^}]*\})?\s+(\S+)", line)
        if m and any(k in m.group(1) for k in METRIC_KEYS) and not m.group(1).endswith("_created"):
            try:
                vals[m.group(1)] = vals.get(m.group(1), 0.0) + float(m.group(2))
            except ValueError:
                pass
    return vals


# --- streaming client ----------------------------------------------------------
async def one_request(session, prompt_ids, max_tokens, temperature):
    payload = {"model": "target", "prompt": prompt_ids, "max_tokens": max_tokens,
               "temperature": temperature, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    t_first, usage, chunks = None, None, 0
    async with session.post(f"{URL}/v1/completions", json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("choices") and obj["choices"][0].get("text"):
                chunks += 1
                if t_first is None:
                    t_first = time.perf_counter()
            if obj.get("usage"):
                usage = obj["usage"]
    t_end = time.perf_counter()
    usage = usage or {}
    return {"t_start": t0, "ttft_s": (t_first - t0) if t_first else float("nan"),
            "e2e_s": t_end - t0, "out_tokens": usage.get("completion_tokens", chunks),
            "prompt_tokens": usage.get("prompt_tokens", len(prompt_ids)), "chunks": chunks,
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}


async def run_load(prompts, concurrency, max_tokens, temperature):
    """Closed loop: `concurrency` workers, each sending its next request as soon as
    the previous one finishes, until all prompts are used."""
    results, nxt = [], 0
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=timeout) as s:
        async def worker():
            nonlocal nxt
            while nxt < len(prompts):
                p = prompts[nxt]
                nxt += 1
                results.append(await one_request(s, p, max_tokens, temperature))
        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        wall = time.perf_counter() - t0
    for r in results:
        r["t_start"] -= t0
    return results, wall


async def measure(prompts, concurrency, max_tokens, temperature, **tags):
    before = await asyncio.to_thread(scrape)      # blocking HTTP; keep the event loop free
    results, wall = await run_load(prompts, concurrency, max_tokens, temperature)
    after = await asyncio.to_thread(scrape)
    delta = {k: after[k] - before.get(k, 0.0) for k in after}
    rows = [{**tags, "concurrency": concurrency, "temperature": temperature, **r} for r in results]
    level = {**tags, "concurrency": concurrency, "temperature": temperature,
             "n_requests": len(results), "wall_s": wall,
             "out_tokens": sum(r["out_tokens"] for r in results), "metrics": json.dumps(delta)}
    return rows, level
'''

PROMPTS = r'''
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(TARGET)
BOS = tok.bos_token_id
PASSAGE = """''' + PASSAGE + r'''"""
PASSAGE_IDS = tok(PASSAGE, add_special_tokens=False).input_ids
rng = np.random.default_rng(0)
print(len(PASSAGE_IDS), "passage tokens")


def random_ids(n, gen=rng):
    return gen.integers(100, 31000, n).tolist()       # skip special/byte tokens


def random_prompt(n):
    return [BOS] + random_ids(n - 1)


def natural_prompt(i, n=SD_INPUT):
    start = (7 * i) % max(1, len(PASSAGE_IDS) - n)
    return [BOS] + PASSAGE_IDS[start:start + n - 1]


def copy_prompt(i, n=SD_INPUT):
    # A passage segment shown twice, then its start again: the model (and draft)
    # continue by copying, which gives high draft acceptance.
    seg = PASSAGE_IDS[7 * i: 7 * i + 120]
    return ([BOS] + seg + seg + seg)[:n]


# prompt type -> (prompt builder, temperature); ordered from high to low expected acceptance
PROMPT_TYPES = {
    "copy": (copy_prompt, 0.0),
    "natural": (natural_prompt, 0.0),
    "natural_t1": (natural_prompt, 1.0),
    "random_t1": (lambda i: random_prompt(SD_INPUT), 1.0),
}
'''

BASELINE = r'''
async def continuous_batching():
    rows, levels = [], []
    for c in CB_CONCURRENCY:
        n = max(CB_MIN_REQUESTS, 4 * c)
        r, lv = await measure([random_prompt(CB_INPUT) for _ in range(n)], c, CB_OUTPUT, 0.0,
                              experiment="continuous_batching", config="baseline")
        rows += r
        levels.append(lv)
        d = pd.DataFrame(r)
        tpot = ((d.e2e_s - d.ttft_s) / (d.out_tokens - 1)).median()
        print(f"concurrency {c:3d}: {lv['out_tokens'] / lv['wall_s']:8.0f} output tok/s | "
              f"TTFT p50 {d.ttft_s.median() * 1e3:7.1f} ms | TPOT p50 {tpot * 1e3:6.2f} ms")
    pd.DataFrame(rows).to_csv(f"{OUT}/cb_requests.csv", index=False)
    pd.DataFrame(levels).to_csv(f"{OUT}/cb_levels.csv", index=False)


async def spec_run(name):
    rows, levels = [], []
    for ptype, (make, temp) in PROMPT_TYPES.items():
        for c in SD_CONCURRENCY:
            prompts = [make(i) for i in range(SD_REQUESTS[c])]
            r, lv = await measure(prompts, c, SD_OUTPUT, temp, experiment="spec_decode",
                                  config=name, prompt_type=ptype)
            rows += r
            levels.append(lv)
            d = pd.DataFrame(r)
            tpot = ((d.e2e_s - d.ttft_s) / (d.out_tokens - 1)).median()
            print(f"{name:14s} {ptype:11s} c={c}: TPOT p50 {tpot * 1e3:6.2f} ms")
    pd.DataFrame(levels).to_csv(f"{OUT}/spec_{name}_levels.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{OUT}/spec_{name}.csv", index=False)


async def baseline_body():
    if not done("cb_requests.csv"):
        await continuous_batching()
    if not done("spec_baseline.csv"):
        await spec_run("baseline")

await with_server("baseline", [[]], ["cb_requests.csv", "spec_baseline.csv"], baseline_body)
'''

PREFIX = r'''
async def prefix_run(state):
    rows = []
    gen = np.random.default_rng(1)
    for p in PC_PREFIX_LENS:
        shared = random_ids(p, gen)
        prompts = [[BOS] + shared + random_ids(PC_PROMPT_LEN - 1 - p, gen)
                   for _ in range(PC_REPEATS + 1)]
        r, _ = await measure(prompts, 1, PC_OUTPUT, 0.0, experiment="prefix_cache",
                             config=f"caching_{state}", prefix_len=p, prompt_len=PC_PROMPT_LEN)
        for i, x in enumerate(r):
            x["warm"] = i == 0           # first request fills the cache
        rows += r
        d = pd.DataFrame(r[1:])
        print(f"caching {state:3s} prefix {p:5d}: TTFT p50 {d.ttft_s.median() * 1e3:7.1f} ms, "
              f"cached tokens {d.cached_tokens.median()}")
    pd.DataFrame(rows).to_csv(f"{OUT}/prefix_{state}.csv", index=False)

for state, variants in [("on", [["--enable-prefix-caching"]]),
                        ("off", [["--no-enable-prefix-caching"]])]:
    await with_server(f"prefix_{state}", variants, [f"prefix_{state}.csv"],
                      lambda state=state: prefix_run(state))
'''

SPEC = r'''
def spec_variants(sc):
    base = ["--speculative-config", json.dumps(sc)]
    if "method" in sc:
        return [base]
    return [base, ["--speculative-config", json.dumps({**sc, "method": "draft_model"})]]

measured = []
for name, sc in SD_CONFIGS.items():
    if sc is None:
        continue
    if await with_server(f"spec_{name}", spec_variants(sc), [f"spec_{name}.csv"],
                         lambda name=name: spec_run(name)):
        measured.append(name)
if not measured:
    print("No draft-model configuration started on this vLLM/GPU; measuring n-gram "
          "(prompt-lookup) speculation instead.")
    for name, sc in SD_NGRAM_FALLBACK.items():
        if await with_server(f"spec_{name}", spec_variants(sc), [f"spec_{name}.csv"],
                             lambda name=name: spec_run(name)):
            measured.append(name)
env["spec_configs_measured"] = measured
save_env()
print("measured:", measured)
'''

PACKAGE = r'''
gpu_logger.terminate()
gpu_log.close()
save_env()
archive = f"/content/{SLUG}_vllm"
shutil.make_archive(archive, "zip", OUT)
if BASE != "/content":
    shutil.copy(archive + ".zip", BASE)
    print("zip also saved to", BASE)
from google.colab import files
files.download(archive + ".zip")
'''

cells = [
    md("""
# vLLM serving benchmark: continuous batching, prefix caching, speculative decoding

Runs TinyLlama-1.1B (fp16) behind vLLM's OpenAI-compatible server and measures:

| Part | What | Output |
|---|---|---|
| 1 | **Continuous batching**: throughput, TTFT, TPOT and latency at 1-128 concurrent requests | `cb_requests.csv`, `cb_levels.csv` |
| 2 | **Prefix caching**: TTFT vs shared-prefix length, caching on vs off | `prefix_on.csv`, `prefix_off.csv` |
| 3 | **Speculative decoding**: speedup and draft acceptance for llama-68m / llama-160m drafts, on prompts from easy to predict (copying) to hard (random, temperature 1) | `spec_<config>.csv`, `spec_<config>_levels.csv` |

Acceptance counts come from vLLM's own Prometheus counters (`/metrics`), read before and after each measurement.

**How to run:** Runtime > Change runtime type > GPU (T4 works if this vLLM version supports
it; L4 or A100 are safer), then Runtime > Run all and approve the Google Drive prompt.
Expect ~45-60 min: installing vLLM, then one server start per configuration (each
compiles and captures CUDA graphs). If the runtime disconnects, reconnect and Run all:
finished parts are skipped. Unzip the downloaded `<gpu>_vllm.zip` into
`measurements/<gpu>_vllm/` and run `python scripts/analyze_vllm.py measurements/<gpu>_vllm`.

If a server fails to start, its log tail is printed and saved under `logs/`. The most
likely cause on a T4 is a vLLM release that no longer supports Turing GPUs: switch to an
L4 or A100, or pin an older version with `VLLM_SPEC`.
"""),
    code(CONFIG),
    md("## Setup: GPU, storage, clock logger"),
    code(GPU_SETUP),
    md("## Install vLLM (separate Python 3.12 environment, ~5 min)"),
    code(INSTALL),
    md("## Microbenchmark: achievable bandwidth and matmul throughput"),
    code(MICRO),
    md("## Client, server control and prompts"),
    code(CLIENT),
    code(PROMPTS),
    md("""
## Part 1 + speculative baseline: one server, no speculation

Continuous batching sweep (random prompts, so no prefix-cache hits), then the
no-speculation baseline for part 3 on the same server.
"""),
    code(BASELINE),
    md("## Part 2: prefix caching on vs off"),
    code(PREFIX),
    md("""
## Part 3: speculative decoding

One server per configuration. If a draft-model configuration cannot start on this
vLLM version, it is recorded as failed and the next one is tried; if none start,
n-gram (prompt-lookup) speculation is measured instead so acceptance can still be studied.
"""),
    code(SPEC),
    md("## Package the results"),
    code(PACKAGE),
]

if __name__ == "__main__":
    write(HERE / "vllm_serving_benchmark.ipynb", cells, gpu_type="T4")
