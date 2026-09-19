"""KV-cache math: per-token bytes for MHA/GQA/MQA, MLA, sliding-window attention.

All sizes are in bytes. We report GiB (2**30) everywhere, matching what
vLLM/llama.cpp print at runtime.

Formulae:
  Standard attention (MHA/GQA/MQA):
      per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
      (the factor of 2 accounts for K and V)
  MLA (DeepSeek absorbed-KV formulation, DeepSeek-V3 paper, Fig. 5):
      the model caches only the compressed latent c_kv (kv_lora_rank)
      and the decoupled RoPE key (qk_rope_head_dim) per layer:
      per_token = 2 * n_layers * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes
  Sliding-window attention:
      KV for a windowed layer is bounded by the window, not the context:
      kv_tokens(layer) = min(ctx, sliding_window)
      Global-attention layers (e.g. Gemma 3's 1-in-6 global layers) use ctx.
"""

DTYPE_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,    # fp8_e4m3 / fp8_e5m2
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "int8": 1.0,
    "int4": 0.5,   # nf4 / int4 quantized KV
    "nf4": 0.5,
    # Weight-only dtypes also accepted for weights size; KV is never gguf-stored,
    # so a gguf/awq weight flag never reaches this table.
}


def dtype_bytes(name: str) -> float:
    norm = name.strip().lower().replace("-", "").replace("_", "")
    table = {k.replace("-", "").replace("_", ""): v for k, v in DTYPE_BYTES.items()}
    if norm not in table:
        raise ValueError(f"unknown KV dtype {name!r}; known: {sorted(DTYPE_BYTES)}")
    return table[norm]


def per_token_bytes(model: dict, kv_dtype: str = "fp16") -> float:
    """Bytes of KV cache for ONE token at batch size 1.

    model keys: layers, attention ("standard" or "mla"), kv_heads, head_dim
    (standard), kv_lora_rank, qk_rope_head_dim (mla), sliding_window (optional),
    global_layers (optional count of full-attention layers when sliding window
    is set).
    """
    b = dtype_bytes(kv_dtype)
    layers = model["layers"]
    att = model.get("attention", "standard")
    if att == "mla":
        base = 2.0 * layers * (model["kv_lora_rank"] + model["qk_rope_head_dim"]) * b
        return base
    if att != "standard":
        raise ValueError(f"unknown attention kind {att!r}")
    return 2.0 * layers * model["kv_heads"] * model["head_dim"] * b


def kv_cache_bytes(model: dict, ctx: int, kv_dtype: str = "fp16",
                   concurrency: int = 1) -> float:
    """Total KV cache bytes for `concurrency` sequences of `ctx` tokens each.

    Sliding-window layers only cache up to the window size; global layers
    (Gemma 3 style) cache the full context.
    """
    b = dtype_bytes(kv_dtype)
    layers = model["layers"]
    att = model.get("attention", "standard")

    if att == "mla":
        per_layer_per_token = 2.0 * (model["kv_lora_rank"] + model["qk_rope_head_dim"]) * b
        return per_layer_per_token * layers * ctx * concurrency

    per_layer_per_token = 2.0 * model["kv_heads"] * model["head_dim"] * b
    window = model.get("sliding_window")
    global_layers = model.get("global_layers", 0)
    if window:
        windowed_layers = layers - global_layers
        tokens = (windowed_layers * min(ctx, window)
                  + global_layers * ctx)
        return per_layer_per_token * tokens * concurrency
    return per_layer_per_token * layers * ctx * concurrency
