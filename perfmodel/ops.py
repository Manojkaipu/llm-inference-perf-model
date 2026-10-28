"""Per-op FLOP and memory-traffic accounting for one forward pass.

A forward pass processes `batch` sequences, each adding `new_tokens` tokens on
top of `past_tokens` already in the KV cache:
  prefill     -> new_tokens = prompt length, past_tokens = 0
  decode step -> new_tokens = 1,             past_tokens = current context

Bytes are the ideal DRAM traffic: every weight, input and output is moved
exactly once (perfect on-chip reuse, fused FlashAttention-style attention).
"""
from dataclasses import dataclass

from .config import ModelConfig

LINEAR, ATTENTION, ELEMENTWISE = "linear", "attention", "elementwise"


@dataclass(frozen=True)
class Op:
    name: str
    kind: str
    flops: float
    weight_bytes: float = 0.0
    act_bytes: float = 0.0
    kv_bytes: float = 0.0
    count: int = 1          # repeats per forward pass (n_layers for per-layer ops)
    parallelism: int = 0    # independent work items (attention: sequences x KV heads x queries)

    @property
    def bytes(self) -> float:
        return self.weight_bytes + self.act_bytes + self.kv_bytes

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes if self.bytes else float("inf")


def _linear(name, n_tok, d_in, d_out, b, count):
    return Op(name, LINEAR, flops=2 * n_tok * d_in * d_out,
              weight_bytes=d_in * d_out * b,
              act_bytes=n_tok * (d_in + d_out) * b, count=count)


def _rmsnorm(name, n_tok, d, b, count):
    # square, reduce, rsqrt-scale, weight multiply
    return Op(name, ELEMENTWISE, flops=4 * n_tok * d, weight_bytes=d * b,
              act_bytes=2 * n_tok * d * b, count=count)


def forward_ops(cfg: ModelConfig, batch: int, new_tokens: int, past_tokens: int,
                causal: bool = True, logits: bool = True) -> list[Op]:
    """`logits=False` drops the LM head, e.g. for non-final chunks of a split prompt."""
    return mixed_forward_ops(cfg, [(batch, new_tokens, past_tokens)], causal=causal,
                             logits_rows=batch if logits else 0)


def _attention(cfg: ModelConfig, name: str, batch, new_tokens, past_tokens, causal: bool) -> Op:
    b, bkv, hd = cfg.dtype_bytes, cfg.kv_dtype_bytes, cfg.head_dim
    total_ctx = past_tokens + new_tokens
    # (query, key) pairs scored per head per sequence; causal masking skips the
    # upper triangle of the new-token block.
    if causal:
        pairs = new_tokens * past_tokens + new_tokens * (new_tokens + 1) / 2
    else:
        pairs = new_tokens * total_ctx
    return Op(name, ATTENTION,
              # QK^T and PV matmuls (2 FLOP/MAC each) + ~5 FLOP/score for softmax
              flops=batch * cfg.n_heads * pairs * (4 * hd + 5),
              act_bytes=2 * batch * new_tokens * cfg.q_dim * b,      # read Q, write O
              kv_bytes=batch * total_ctx * 2 * cfg.kv_dim * bkv,     # read K and V once
              count=cfg.n_layers, parallelism=batch * cfg.n_kv_heads * new_tokens)


def mixed_forward_ops(cfg: ModelConfig, segments, causal: bool = True,
                      logits_rows=None) -> list[Op]:
    """One forward pass over sequences in different states, as a serving engine
    batches them (e.g. continuous batching mixes decodes with prefill chunks).

    segments: (batch, new_tokens, past_tokens) groups. All their tokens share one
    pass through the weight matmuls (weights read once); each group gets its own
    attention. Batch may be fractional to represent work amortized across steps.
    logits_rows: rows sent through the LM head (default: one per sequence).
    """
    b, bkv = cfg.dtype_bytes, cfg.kv_dtype_bytes
    d, L = cfg.hidden, cfg.n_layers
    segments = [s for s in segments if s[0] * s[1] > 0]      # drop groups with no work
    n = sum(bs * new for bs, new, _ in segments)
    if logits_rows is None:
        logits_rows = sum(bs for bs, _, _ in segments)
    attention = [_attention(cfg, "attention" if len(segments) == 1 else f"attention[{i}]",
                            bs, new, past, causal)
                 for i, (bs, new, past) in enumerate(segments)]

    ops = [
        Op("embed", ELEMENTWISE, flops=0, act_bytes=2 * n * d * b),
        _rmsnorm("attn_norm", n, d, b, L),
        _linear("qkv_proj", n, d, cfg.q_dim + 2 * cfg.kv_dim, b, L),
        Op("rope", ELEMENTWISE, flops=3 * n * (cfg.q_dim + cfg.kv_dim),
           act_bytes=2 * n * (cfg.q_dim + cfg.kv_dim) * b, count=L),
        Op("kv_write", ELEMENTWISE, flops=0, kv_bytes=2 * n * cfg.kv_dim * bkv, count=L),
        *attention,
        _linear("o_proj", n, cfg.q_dim, d, b, L),
        Op("residual_attn", ELEMENTWISE, flops=n * d, act_bytes=3 * n * d * b, count=L),
        _rmsnorm("mlp_norm", n, d, b, L),
        _linear("gate_up_proj", n, d, 2 * cfg.intermediate, b, L),
        Op("silu_mul", ELEMENTWISE, flops=5 * n * cfg.intermediate,
           act_bytes=3 * n * cfg.intermediate * b, count=L),
        _linear("down_proj", n, cfg.intermediate, d, b, L),
        Op("residual_mlp", ELEMENTWISE, flops=n * d, act_bytes=3 * n * d * b, count=L),
        _rmsnorm("final_norm", n, d, b, 1),
    ]
    if logits_rows:
        # Serving engines compute logits only for positions they sample from.
        ops.append(_linear("lm_head", logits_rows, d, cfg.vocab, b, 1))
    return ops


def prefill_ops(cfg, batch, prompt_len, **kw):
    return forward_ops(cfg, batch, new_tokens=prompt_len, past_tokens=0, **kw)


def decode_ops(cfg, batch, context_len, **kw):
    return forward_ops(cfg, batch, new_tokens=1, past_tokens=context_len, **kw)


def totals(ops: list[Op]) -> dict:
    return {
        "flops": sum(o.flops * o.count for o in ops),
        "bytes": sum(o.bytes * o.count for o in ops),
        "weight_bytes": sum(o.weight_bytes * o.count for o in ops),
        "kv_bytes": sum(o.kv_bytes * o.count for o in ops),
        "act_bytes": sum(o.act_bytes * o.count for o in ops),
    }
