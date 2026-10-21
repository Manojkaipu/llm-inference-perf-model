"""GPU specs and efficiency profiles.

Peak numbers are vendor datasheet values (dense, no sparsity). The "realistic"
efficiencies are priors, not measurements -- phase 4 calibrates them against
measured T4 runs.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    matmul_flops: float   # FP16 tensor-core peak, FLOP/s (dense)
    vector_flops: float   # FP32 CUDA-core peak, FLOP/s (norms, activations)
    mem_bw: float         # DRAM bandwidth, bytes/s
    mem_capacity: float   # bytes

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) where tensor-core compute = bandwidth."""
        return self.matmul_flops / self.mem_bw


T4 = GPU("T4", matmul_flops=65e12, vector_flops=8.1e12, mem_bw=320e9, mem_capacity=16e9)
H100 = GPU("H100-SXM", matmul_flops=989.4e12, vector_flops=66.9e12, mem_bw=3.35e12,
           mem_capacity=80e9)

GPUS = {g.name: g for g in (T4, H100)}


@dataclass(frozen=True)
class Efficiency:
    compute: float = 1.0          # fraction of peak FLOP/s achieved
    memory: float = 1.0           # fraction of peak bandwidth achieved
    kernel_overhead_s: float = 0.0  # fixed cost per kernel launch
    mem_utilization: float = 0.9  # usable fraction of memory for weights + KV cache
    # Bandwidth fraction achieved by attention kernels (KV-cache reads); None = `memory`.
    attention_memory: float | None = None
    # Attention needs enough independent work (sequences x KV heads x queries) to
    # fill the GPU: its bandwidth is scaled by w / (w + attention_half_work).
    # None = no ramp.
    attention_half_work: float | None = None
    # How compute and memory time combine within an op: (t_c^p + t_m^p)^(1/p).
    # inf = perfect overlap (classic roofline max); 1 = no overlap (sum).
    overlap: float = math.inf


PEAK = Efficiency()

# The T4 is a 70 W card and throttles under sustained tensor-core load, so its
# compute prior is lower than the H100's.
REALISTIC = {
    "T4": Efficiency(compute=0.45, memory=0.80, kernel_overhead_s=4e-6),
    "H100-SXM": Efficiency(compute=0.70, memory=0.85, kernel_overhead_s=3e-6),
}
