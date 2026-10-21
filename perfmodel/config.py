"""Llama-style decoder model configurations."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden: int            # d_model
    n_layers: int
    n_heads: int           # query heads
    n_kv_heads: int        # key/value heads (GQA when < n_heads)
    intermediate: int      # SwiGLU hidden size
    vocab: int
    tied_embeddings: bool = False
    dtype_bytes: int = 2   # weights + activations (fp16)
    kv_dtype_bytes: int = 2

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def layer_params(self) -> int:
        attn = self.hidden * (self.q_dim + 2 * self.kv_dim) + self.q_dim * self.hidden
        mlp = 3 * self.hidden * self.intermediate
        norms = 2 * self.hidden
        return attn + mlp + norms

    def params(self) -> int:
        embed = self.vocab * self.hidden
        head = 0 if self.tied_embeddings else self.vocab * self.hidden
        return self.n_layers * self.layer_params() + embed + head + self.hidden

    def weight_bytes(self) -> int:
        return self.params() * self.dtype_bytes

    def kv_bytes_per_token(self) -> int:
        return 2 * self.n_layers * self.kv_dim * self.kv_dtype_bytes


TINYLLAMA_1_1B = ModelConfig(
    name="TinyLlama-1.1B", hidden=2048, n_layers=22, n_heads=32, n_kv_heads=4,
    intermediate=5632, vocab=32000,
)

LLAMA_3_2_1B = ModelConfig(
    name="Llama-3.2-1B", hidden=2048, n_layers=16, n_heads=32, n_kv_heads=8,
    intermediate=8192, vocab=128256, tied_embeddings=True,
)

LLAMA_3_1_8B = ModelConfig(
    name="Llama-3.1-8B", hidden=4096, n_layers=32, n_heads=32, n_kv_heads=8,
    intermediate=14336, vocab=128256,
)

MODELS = {m.name: m for m in (TINYLLAMA_1_1B, LLAMA_3_2_1B, LLAMA_3_1_8B)}
