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
    b, bkv = cfg.dtype_bytes, cfg.kv_dtype_bytes
    d, hd, L = cfg.hidden, cfg.head_dim, cfg.n_layers
    n = batch * new_tokens
    total_ctx = past_tokens + new_tokens

    # (query, key) pairs scored per head per sequence; causal masking skips the
    # upper triangle of the new-token block.
    if causal:
        pairs = new_tokens * past_tokens + new_tokens * (new_tokens + 1) / 2
    else:
        pairs = new_tokens * total_ctx

    ops = [
        Op("embed", ELEMENTWISE, flops=0, act_bytes=2 * n * d * b),
        _rmsnorm("attn_norm", n, d, b, L),
        _linear("qkv_proj", n, d, cfg.q_dim + 2 * cfg.kv_dim, b, L),
        Op("rope", ELEMENTWISE, flops=3 * n * (cfg.q_dim + cfg.kv_dim),
           act_bytes=2 * n * (cfg.q_dim + cfg.kv_dim) * b, count=L),
        Op("kv_write", ELEMENTWISE, flops=0, kv_bytes=2 * n * cfg.kv_dim * bkv, count=L),
        Op("attention", ATTENTION,
           # QK^T and PV matmuls (2 FLOP/MAC each) + ~5 FLOP/score for softmax
           flops=batch * cfg.n_heads * pairs * (4 * hd + 5),
           act_bytes=2 * n * cfg.q_dim * b,                       # read Q, write O
           kv_bytes=batch * total_ctx * 2 * cfg.kv_dim * bkv,     # read K and V once
           count=L),
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
    if logits:
        # Serving engines compute logits only for each sequence's last position.
        ops.append(_linear("lm_head", batch, d, cfg.vocab, b, 1))
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
