"""Serving-engine effects on top of the per-pass model: continuous batching,
prefix caching and speculative decoding.

Each function returns predicted times in seconds; efficiencies come from the
caller (e.g. a profile fitted to vLLM measurements).
"""
from collections import deque

from .config import ModelConfig
from .hardware import GPU, Efficiency
from .ops import decode_ops, forward_ops, mixed_forward_ops
from .predict import pass_time


# --- continuous batching ------------------------------------------------------------

def continuous_step_ops(cfg: ModelConfig, concurrency: float, input_len: int, output_len: int):
    """Ops for one steady-state engine step with `concurrency` requests in flight
    (closed loop: a finished request is immediately replaced).

    Every step decodes one token for each running request, at the average context
    input_len + output_len / 2. New requests' prompts are prefilled in chunks
    inside the same steps (chunked prefill), so each step also carries, amortized,
    concurrency / output_len prompts' worth of prefill.
    """
    ctx = input_len + output_len / 2
    prefills_per_step = concurrency / output_len
    return mixed_forward_ops(cfg, [(concurrency, 1, ctx), (prefills_per_step, input_len, 0)],
                             logits_rows=concurrency + prefills_per_step)


def predict_continuous(cfg, gpu: GPU, eff: Efficiency, concurrency: float, input_len: int,
                       output_len: int) -> dict:
    step = pass_time(continuous_step_ops(cfg, concurrency, input_len, output_len), gpu, eff)
    # A new request's prompt is prefilled whole in the step it is admitted to,
    # alongside the running decodes.
    first = pass_time(mixed_forward_ops(
        cfg, [(max(concurrency - 1, 0), 1, input_len + output_len / 2), (1, input_len, 0)]),
        gpu, eff)
    return {"tpot_s": step, "ttft_s": first, "e2e_s": first + (output_len - 1) * step,
            "output_tok_s": concurrency / step}


def simulate_closed_loop(cfg, gpu: GPU, eff: Efficiency, concurrency: int, n_requests: int,
                         input_len: int, output_len: int, budget: int = 2048,
                         max_seqs: int = 256) -> list[dict]:
    """Step-by-step simulation of a vLLM-style scheduler under a closed-loop client.

    Each step first gives every running request one decode token, then fills the
    remaining token budget with queued prompts in arrival order, chunking a prompt
    that does not fit. A request's first token comes at the end of the step that
    finishes its prompt; it then decodes one token per step until it has
    output_len tokens, and its client immediately sends the next request. Step
    durations come from the per-pass model for exactly that mix of work.
    Returns one record per request (ttft_s, e2e_s, tpot_s, t_start, t_end).
    """
    waiting: deque = deque()
    running: list[dict] = []
    results: list[dict] = []
    issued, t = 0, 0.0

    def submit(at):
        nonlocal issued
        waiting.append({"arrival": at, "prefilled": 0, "generated": 0, "first": None})
        issued += 1

    for _ in range(min(concurrency, n_requests)):
        submit(0.0)
    while waiting or running:
        segments, room = [], budget - len(running)
        if running:
            ctx = sum(input_len + r["generated"] for r in running) / len(running)
            segments.append((len(running), 1, ctx))
        finished_prefill = []
        while waiting and room > 0 and len(running) + len(finished_prefill) < max_seqs:
            r = waiting[0]
            chunk = min(input_len - r["prefilled"], room)
            segments.append((1, chunk, r["prefilled"]))
            r["prefilled"] += chunk
            room -= chunk
            if r["prefilled"] < input_len:
                break                                    # budget exhausted mid-prompt
            finished_prefill.append(waiting.popleft())
        t += pass_time(mixed_forward_ops(cfg, segments,
                                         logits_rows=len(running) + len(finished_prefill)), gpu, eff)
        for r in running:
            r["generated"] += 1
        for r in finished_prefill:
            r["first"], r["generated"] = t, 1
            running.append(r)
        still = []
        for r in running:
            if r["generated"] < output_len:
                still.append(r)
                continue
            results.append({"t_start": r["arrival"], "t_end": t, "ttft_s": r["first"] - r["arrival"],
                            "e2e_s": t - r["arrival"],
                            "tpot_s": (t - r["first"]) / max(output_len - 1, 1)})
            if issued < n_requests:
                submit(t)
        running = still
    return results


# --- prefix caching -----------------------------------------------------------------

def prefill_time(cfg, gpu: GPU, eff: Efficiency, prompt_len: int, cached_len: int = 0) -> float:
    """Prefill of one prompt whose first `cached_len` tokens are already in the KV
    cache: only the remaining tokens run through the model, attending over the
    cached prefix."""
    return pass_time(forward_ops(cfg, 1, prompt_len - cached_len, cached_len), gpu, eff)


# --- speculative decoding -----------------------------------------------------------

def expected_tokens_per_step(alpha: float, k: int) -> float:
    """Tokens emitted per verification step when each draft token is accepted
    independently with probability alpha: accepted prefix + 1 target token."""
    if alpha >= 1:
        return k + 1
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def acceptance_rate(alpha: float, k: int) -> float:
    """Fraction of drafted tokens accepted (what vLLM reports), given per-token alpha."""
    return (expected_tokens_per_step(alpha, k) - 1) / k


def alpha_from_acceptance_rate(rate: float, k: int) -> float:
    """Invert acceptance_rate (monotonic in alpha) by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if acceptance_rate(mid, k) < rate else (lo, mid)
    return (lo + hi) / 2


def spec_step_times(target: ModelConfig, draft: ModelConfig, gpu: GPU, eff: Efficiency,
                    k: int, batch: int, ctx: int) -> dict:
    """Costs of one speculative step vs one normal decode step.

    draft:  k sequential draft decode steps
    verify: one target pass over k + 1 positions per sequence, with logits at
            every position so each draft token can be checked
    """
    t_target = pass_time(decode_ops(target, batch, ctx), gpu, eff)
    t_draft = pass_time(decode_ops(draft, batch, ctx), gpu, eff)
    t_verify = pass_time(mixed_forward_ops(target, [(batch, k + 1, ctx)],
                                           logits_rows=batch * (k + 1)), gpu, eff)
    return {"target_step_s": t_target, "draft_step_s": t_draft, "verify_step_s": t_verify,
            "spec_step_s": k * t_draft + t_verify}


def breakeven_acceptance(spec_step_s: float, target_step_s: float, k: int) -> float:
    """Draft acceptance rate at which speculation breaks even: the speculative
    step must emit spec_step / target_step tokens to pay for itself."""
    return (spec_step_s / target_step_s - 1) / k


def spec_speedup(times: dict, k: int, tokens_per_step: float) -> float:
    """Decode speedup vs no speculation, given tokens emitted per speculative step."""
    return tokens_per_step * times["target_step_s"] / times["spec_step_s"]
