"""Engine profiles: static VRAM overhead, memory utilization, flag recommendations.

vLLM allocates a KV-cache block pool up front; gpu_memory_utilization (0.9
default) caps the fraction of VRAM it may use. llama.cpp offloads layers to
the GPU (-ngl) and grows KV allocations on demand, so usable fraction is
higher but context buffers are preallocated.
"""

ENGINES = {
    "vllm": {
        "label": "vLLM",
        "default_utilization": 0.9,
        "static_overhead_gib": 1.5,   # CUDA graphs, kernels, activations slack
    },
    "llama.cpp": {
        "label": "llama.cpp",
        "default_utilization": 0.95,
        "static_overhead_gib": 0.5,
    },
}


def known_engines():
    return sorted(ENGINES)


def engine_profile(name: str) -> dict:
    key = name.strip().lower().replace("-", ".").replace(" ", "")
    if key not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; known: {', '.join(known_engines())}")
    return dict(ENGINES[key])


def recommended_flags(engine: str, ctx: int, concurrency: int,
                      tensor_parallel: int, kv_dtype: str,
                      utilization: float) -> list:
    """Concrete engine flags to pass at launch."""
    engine = engine.strip().lower().replace("-", ".")
    flags = []
    if engine == "vllm":
        flags += [
            f"--max-model-len {ctx}",
            f"--max-num-seqs {concurrency}",
            f"--gpu-memory-utilization {utilization}",
        ]
        if tensor_parallel and tensor_parallel > 1:
            flags.append(f"--tensor-parallel-size {tensor_parallel}")
        if kv_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2", "int4", "nf4"):
            flags.append(f"--kv-cache-dtype {kv_dtype}")
    elif engine == "llama.cpp":
        flags += [
            f"--ctx-size {ctx}",
            "-ngl 999",           # offload all layers to GPU
            f"--parallel {concurrency}",
        ]
    return flags
