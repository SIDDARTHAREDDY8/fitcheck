"""VRAM budget + verdict + ASCII budget bar + max-safe-context math."""

import math

from .kv import kv_cache_bytes, per_token_bytes
from .models import weight_dtype_bytes

GIB = 2 ** 30


def weights_bytes(model: dict, w_dtype: str | None = None) -> float:
    if model.get("weight_bytes"):
        return float(model["weight_bytes"])
    dtype = w_dtype or model.get("weight_dtype", "bf16")
    if not model.get("params"):
        raise ValueError("model has no param count and no weight_bytes; pass --params")
    return float(model["params"]) * weight_dtype_bytes(dtype)


def budget(model, vram_gib, gpus, tensor_parallel, ctx, kv_dtype,
           w_dtype, concurrency, engine_profile, utilization, reserve_pct):
    """Compute the full VRAM budget. Returns a dict of byte-denominated parts."""
    total_vram = vram_gib * GIB * gpus
    tp = tensor_parallel or gpus
    if tp > gpus:
        raise ValueError(f"--tensor-parallel {tp} exceeds --gpus {gpus}")
    # TP shards weights and KV across tp GPUs; static overhead is per-GPU.
    weights_per_gpu = weights_bytes(model, w_dtype) / tp
    kv_per_gpu = kv_cache_bytes(model, ctx, kv_dtype, concurrency) / tp
    static_per_gpu = engine_profile["static_overhead_gib"] * GIB
    reserve_per_gpu = vram_gib * GIB * reserve_pct

    usable_per_gpu = vram_gib * GIB * utilization
    used_per_gpu = weights_per_gpu + kv_per_gpu + static_per_gpu
    # Reserve is NOT part of "used"; it shrinks what we consider available.
    capacity_per_gpu = vram_gib * GIB * (1.0 - reserve_pct)

    ptb = per_token_bytes(model, kv_dtype)
    # KV bytes per context-token per sequence on one GPU
    kv_per_ctx_token_per_seq = kv_per_gpu / (ctx * concurrency) if ctx and concurrency else ptb / tp
    headroom_per_gpu = capacity_per_gpu * utilization - used_per_gpu

    # Max safe context: solve used == usable minus weights/overhead, at same concurrency
    kv_budget_per_gpu = max(0.0, capacity_per_gpu * utilization
                            - weights_per_gpu - static_per_gpu)
    if kv_per_ctx_token_per_seq > 0 and concurrency > 0:
        max_ctx = int(kv_budget_per_gpu / (kv_per_ctx_token_per_seq * concurrency))
    else:
        max_ctx = 0

    ratio = used_per_gpu / (vram_gib * GIB)  # share of raw VRAM used
    # Verdict bands:
    #   pool = what the engine will actually allocate (vLLM fails at startup
    #          only when it cannot fit inside gpu_memory_utilization * VRAM).
    #   safe = the conservative planning budget (pool minus safety reserve).
    # OOM only when the engine physically cannot allocate the config; TIGHT
    # when it fits the pool but eats the safety reserve (vLLM may trim
    # max_model_len at startup profiling, or need --gpu-memory-utilization
    # raised). This keeps headline cases honest: e.g. Llama-3.1-8B at 32k on
    # a 4090 is TIGHT (servable, ~1 GiB pool headroom), not OOM.
    pool_per_gpu = vram_gib * GIB * utilization
    safe_per_gpu = capacity_per_gpu * utilization
    if used_per_gpu > pool_per_gpu:
        verdict = "OOM"
    elif used_per_gpu > safe_per_gpu:
        verdict = "TIGHT"
    else:
        verdict = "FITS"

    return {
        "gpus": gpus,
        "tensor_parallel": tp,
        "vram_per_gpu_gib": vram_gib,
        "total_vram_gib": total_vram / GIB,
        "weights_gib": weights_per_gpu / GIB,
        "kv_gib": kv_per_gpu / GIB,
        "static_overhead_gib": static_per_gpu / GIB,
        "reserve_gib": reserve_per_gpu / GIB,
        "used_gib": used_per_gpu / GIB,
        "headroom_gib": headroom_per_gpu / GIB,
        "pool_headroom_gib": (pool_per_gpu - used_per_gpu) / GIB,
        "utilization": utilization,
        "reserve_pct": reserve_pct,
        "ctx": ctx,
        "concurrency": concurrency,
        "kv_per_token_bytes": ptb,
        "max_safe_ctx": max_ctx,
        "verdict": verdict,
        "used_ratio": ratio,
    }


def ascii_bar(b: dict, width: int = 42) -> str:
    """VRAM budget bar: weights | KV cache | engine overhead | headroom."""
    total = b["vram_per_gpu_gib"]
    parts = [
        ("weights", b["weights_gib"]),
        ("KV cache", b["kv_gib"]),
        ("overhead", b["static_overhead_gib"]),
    ]
    headroom = max(0.0, total - b["used_gib"])
    parts.append(("headroom", headroom))
    overflow = max(0.0, b["used_gib"] - total)
    chars = ["#", "=", "+", "-"]
    bar = ""
    for (_, gib), ch in zip(parts, chars):
        n = int(round(width * gib / total))
        bar += ch * n
    bar = bar[:width].ljust(width)
    lines = [f"VRAM per GPU ({total:g} GiB)  [{bar}]"]
    for (label, gib), ch in zip(parts, chars):
        lines.append(f"  {ch} {label:<10} {gib:8.2f} GiB")
    if overflow:
        lines.append(f"  ! OVER BUDGET by {overflow:.2f} GiB")
    return "\n".join(lines)
