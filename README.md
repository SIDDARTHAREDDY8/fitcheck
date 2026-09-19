# fitcheck

**Will this model fit on your GPU(s) at this context length with this engine?**

`fitcheck` is a zero-dependency Python CLI that answers the question every
self-hoster asks before a long-context OOM crash: it computes the real
KV-cache footprint from the model's architecture (MHA/GQA/MQA, DeepSeek MLA,
sliding-window attention) plus weights, engine overhead, and tensor
parallelism — then gives a verdict, an ASCII VRAM budget bar, the max safe
context, and ready-to-paste engine flags.

Weights-only calculators exist. **The KV cache is what actually kills you:**
it grows linearly with context length, and at 128K tokens it can exceed the
model weights themselves.

## Install

```bash
pip install git+https://github.com/SIDDARTHAREDDY8/fitcheck.git
```

(Or clone and run with `python3 -m fitcheck` — no dependencies, stdlib only.
Requires Python 3.9+.)

## Quickstart

```bash
# Will Llama 3.1 8B run at 128K on a single RTX 4090?
fitcheck llama-3.1-8b --gpu 4090 --ctx 128k

# Qwen3 32B on 2x A100 80GB with fp8 KV cache
fitcheck qwen3-32b --gpu a100-80 --gpus 2 --ctx 64k --kv-dtype fp8

# Any HF repo, fetched live
fitcheck --from-hf Qwen/Qwen3-32B --gpu 5090 --ctx 32k

# A local checkpoint dir (reads config.json + safetensors sizes)
fitcheck ./my-model --gpu-vram 48 --ctx 1m --engine llama.cpp

# Custom architecture, no preset needed
fitcheck --layers 32 --kv-heads 4 --head-dim 128 --params 3.75B \
         --gpu 3080 --ctx 8k

# Machine-readable
fitcheck llama-3.1-8b --gpu 4090 --ctx 8k --json
```

Example output:

```
fitcheck 0.1.0  |  llama-3.1-8b
================================================================
engine: vLLM    ctx: 8,192 tokens    concurrency: 1    GPUs: 1x24 GiB
weights: bf16    KV dtype: fp16    KV/token: 131,072 B (128.0 KiB)

VRAM per GPU (24 GiB)  [##########################==+++-----------]
  # weights       14.96 GiB
  = KV cache       1.00 GiB
  + overhead       1.50 GiB
  - headroom       6.54 GiB

verdict: [OK] FITS
  headroom: 2.41 GiB — you could push to ~27,971 tokens at this concurrency

recommended vLLM flags:
  --max-model-len 8192
  --max-num-seqs 1
  --gpu-memory-utilization 0.9
```

Exit code is `1` on OOM, so you can gate scripts and CI on it.

## How it works

See [docs/how-it-works.md](docs/how-it-works.md) for the math, the
assumptions, and citations.

The short version:

- **KV cache** (the part everyone forgets): `2 × layers × kv_heads × head_dim × dtype_bytes`
  per token. DeepSeek-style MLA uses the absorbed formulation
  `2 × layers × (kv_lora_rank + qk_rope_head_dim) × dtype_bytes`. Sliding-window
  layers are bounded by the window; global layers see the full context.
- **Weights**: `params × dtype_bytes`, sharded by `--tensor-parallel`.
- **Conservative defaults**: 8% VRAM safety reserve, vLLM
  `gpu_memory_utilization=0.9`, engine static overhead. Override with
  `--reserve`, `--gpu-mem-util`.
- **Verdict**: `FITS` / `TIGHT` / `OOM`, plus the max safe context at your
  concurrency and copy-paste flags for vLLM (`--max-model-len`,
  `--max-num-seqs`, `--tensor-parallel-size`, `--kv-cache-dtype`) and
  llama.cpp (`--ctx-size`, `-ngl`, `--parallel`).

Built-in presets: `llama-3.1-8b`, `llama-3.3-70b`, `qwen3-32b`,
`mistral-7b-v0.3`, `gemma-3-27b`, `phi-4`, `deepseek-v3` — values verified
against the real HF configs. Built-in GPUs: 4090, 4080, 3090, 3080, 5090,
A100-40/80, H100-80, V100-16, T4-16, 6000-Ada.

## Why this exists

Weights math is blogged everywhere; the KV cache is the hidden killer. Two
real deployments that motivated this tool:

- A compliance-org deployment loaded ~130GB of weights onto 2×A100-80GB and
  found only ~11GB left for KV cache — vLLM capped them at ~92K tokens on a
  model advertising 196K.
- A widely-shared writeup: 7.5GB weights + 7.0GB KV at 128K = 15GB — an OOM
  crash on a 12GB card.

Both are encoded as regression tests (`tests/test_fitcheck.py`). Sources are
linked in [docs/how-it-works.md](docs/how-it-works.md).

## Limitations

- KV math assumes uniform layers and full caching; it does not model vLLM's
  paged/blocked allocation granularity, prefix-caching savings, or CUDA-graph
  memory.
- Activation memory during prefill is approximated by the static engine
  overhead, not computed per sequence — at very long contexts on small cards,
  prefill spikes can exceed this estimate.
- MLA `qk_nope_head_dim` is not cached separately in the absorbed formulation
  (correct per the DeepSeek-V3 paper), but custom MLA variants may differ.

## License

MIT — see [LICENSE](LICENSE).
