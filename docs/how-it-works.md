# How fitcheck works

## The VRAM budget

Serving an LLM on GPUs consumes VRAM in three buckets:

1. **Model weights** — fixed per dtype: `params × bytes_per_param`.
2. **KV cache** — grows linearly with context length × concurrency. This is
   the bucket everyone forgets, and the one this tool exists to compute.
3. **Engine overhead** — CUDA graphs, kernels, activation workspace, and a
   safety reserve.

fitcheck sizes each bucket per GPU (sharded by tensor-parallel degree when
`--tensor-parallel > 1`), then compares against usable VRAM.

## The per-token KV formula

For each generated token, the model must store a key and a value vector for
every layer and every KV head:

```
per_token_bytes = 2 × n_layers × n_kv_heads × head_dim × dtype_bytes
```

The factor of 2 is K and V. For GQA (`n_kv_heads < n_query_heads`) the cache
shrinks proportionally — Llama 3's 8 KV heads against 32 query heads is
exactly why its cache is 4× smaller than an MHA equivalent. The formula is
given in the same form by Richard Sakaguchi's vLLM KV-preemption writeup
(dev.to/ji_ai), which prescribes computing it to set `max_num_seqs`.

Worked example — Llama 3.1 8B (32 layers, 8 KV heads, head_dim 128, fp16):

```
2 × 32 × 8 × 128 × 2 = 131,072 bytes/token = 128 KiB/token
131,072 tokens × 131,072 bytes = 16 GiB at 128K context
```

That 16 GiB *exceeds the model's own 15 GiB bf16 weights* — the whole point.

## MLA (DeepSeek V2/V3)

Multi-head Latent Attention compresses the KV cache into a low-rank latent
plus a decoupled RoPE key. Per the DeepSeek-V3 paper, the absorbed
formulation caches only `c_kv` (dimension `kv_lora_rank`) and the RoPE key
(dimension `qk_rope_head_dim`) per layer:

```
per_token_bytes = 2 × n_layers × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes
```

For DeepSeek V3 (61 layers, kv_lora_rank=512, qk_rope_head_dim=64):

```
2 × 61 × 576 × 2 = 140,544 bytes/token ≈ 137 KiB/token (fp16)
```

vs ~2.3 MiB/token a 61-layer MHA model of the same width would need.
Notably, `qk_nope_head_dim` (128) is *not* cached — it is absorbed into the
up-projection at inference time.

## Sliding-window attention

Mistral-style windowed layers only attend to the last `sliding_window`
tokens, so their KV is bounded by the window, not the context:

```
kv_tokens(layer) = min(ctx, sliding_window)
```

Gemma 3 mixes 5 local (1K window) layers with 1 global layer, so fitcheck
models its 10 global layers (out of 62) at full context and the rest at the
1K window. If a config has no `sliding_window`, every layer sees the full
context.

## Weights

`params × weight_dtype_bytes`, using the preset/config dtype (override with
`--w-dtype`). If you point fitcheck at a local checkpoint directory, it
sums the actual `.safetensors` file sizes (via the index) instead of
estimating. Tensor parallelism divides both weights and KV across GPUs;
engine static overhead is per-GPU and is not sharded.

## Conservative defaults (why they're pessimistic)

- **8% VRAM reserve** (`--reserve`): covers CUDA context, fragmentation, and
  driver-visible vs usable VRAM gaps.
- **vLLM `gpu_memory_utilization = 0.9`** (llama.cpp 0.95): matches vLLM's
  default; the KV block pool is preallocated up front, so the tool sizes
  against 90% of raw VRAM, not 100%.
- **Static overhead**: 1.5 GiB (vLLM: graphs/kernels), 0.5 GiB (llama.cpp).

## Verdict rules

Two different "budgets" are in play, and the verdicts distinguish them:

- the **engine pool** = VRAM × `gpu_memory_utilization` — what vLLM will
  actually allocate (it fails at startup beyond this);
- the **safe budget** = pool minus the 8% reserve — the conservative
  planning line. The compliance-org war story is why the reserve exists:
  with ~130GB of weights on 2×A100, vLLM in practice capped context at
  ~92K tokens, well under naive pool math.

- **OOM** — total usage exceeds the engine pool. The config cannot be
  allocated at all. Exit code 1.
- **TIGHT** — fits inside the engine pool but eats the safety reserve
  (e.g. Llama-3.1-8B at 32k on a 4090: servable, ~1 GiB pool headroom).
  vLLM may trim `max_model_len` at startup profiling; raise
  `--gpu-memory-utilization` or lower `--ctx` for margin.
- **FITS** — inside the safe budget with the full reserve intact.

**Max safe context** inverts the safe budget at your stated concurrency:
`(safe − weights − overhead) / (bytes per ctx-token per sequence)`.

## What it does NOT model

- vLLM's paged/blocked KV allocation granularity (16-token blocks) and
  prefix-caching savings.
- Prefill activation spikes — approximated by static overhead; on very long
  contexts with small cards, prefill can transiently exceed the estimate.
- MoE expert-parallel / pipeline-parallel memory layouts; TP is the only
  sharding modeled.
- Custom MLA variants that cache additional projections.

## Citations

- "Context kills VRAM: running LLMs on a local GPU" — Medium, @lyx_62906
  (7.5GB weights + 7.0GB KV at 128K = 15GB > 12GB card)
- "You don't need more VRAM, you need to fix your KV cache" — Medium,
  Coding Nexus (the 15GB/12GB OOM case)
- "Running local AI models for compliance-sensitive organizations" — Medium,
  @leonid_91859 (130GB weights on 2×A100 → ~11GB KV headroom; vLLM capped at
  92,544 tokens on a 196K-context model)
- "KV cache preemption: why vLLM throughput collapses under load" — dev.to,
  ji_ai (the exact per-token formula; setting max_num_seqs from KV pool)
- "A Free Tool to Check VRAM Requirements for Any HuggingFace Model" — dev.to,
  vramio (demand proof; weights-only — the gap fitcheck fills)
- DeepSeek-V3 Technical Report (MLA absorbed-KV formulation)
