"""Serving-engine effects on top of the per-pass model: continuous batching,
prefix caching and speculative decoding.

Each function returns predicted times in seconds; efficiencies come from the
caller (e.g. a profile fitted to vLLM measurements).
"""
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


def spec_speedup(times: dict, k: int, tokens_per_step: float) -> float:
    """Decode speedup vs no speculation, given tokens emitted per speculative step."""
    return tokens_per_step * times["target_step_s"] / times["spec_step_s"]
