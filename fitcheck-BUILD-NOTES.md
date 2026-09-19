# fitcheck build notes (cycle-3 MVP, 2026-09-19)

## Layout
- `fitcheck/__init__.py`, `cli.py` (argparse, human + JSON output), `__main__.py`
- `fitcheck/kv.py` — per-token/KV math (standard MHA/GQA/MQA, MLA absorbed, sliding window)
- `fitcheck/models.py` — preset loader, HF config.json parser, local dir reader (safetensors index), `--from-hf` urllib fetch
- `fitcheck/presets/*.json` — 7 presets with config values verified against real HF configs
- `fitcheck/gpus.py` — 10-entry GPU DB
- `fitcheck/engine.py` — vLLM / llama.cpp profiles + flag recommendation
- `fitcheck/budget.py` — budget, verdict, ASCII bar, max-safe-context
- `tests/test_fitcheck.py` — 21 unittest tests, stdlib only, no network
- `docs/how-it-works.md`, `README.md`, `LICENSE` (MIT), `pyproject.toml`

## Preset values (all verified against the real HF config.json for each repo)
- llama-3.1-8b: 32L / 32H / 8KV / d128 / bf16 / 8.03B
- llama-3.3-70b: 80L / 64H / 8KV / d128 / bf16 / 70.55B
- qwen3-32b: 64L / 64H / 8KV / d128 / bf16 / 32.5B
- mistral-7b-v0.3: 32L / 32H / 8KV / d128 / window 4096 / 7.25B
- gemma-3-27b: 62L / 32H / 16KV / d168 / window 1024, 10 global layers (5:1 local:global pattern) / 27.38B
- phi-4: 32L / 24H / 8KV / d128 / bf16 / 14.7B
- deepseek-v3: MLA 61L, kv_lora_rank 512, qk_rope_head_dim 64 / bf16 / 671B

## Validation anchors (credibility gate)
- (a) Llama-3.1-8B: 131072 B/token -> exactly 16 GiB KV at 128K. Pinned.
- (b) Compliance-org war story reconstruction: 128B-param fp8 model (61L/8KV/d128),
  ~119 GiB weights on 2xA100-80, ctx 196K -> OOM, max safe ctx ~88K (well under 196K).
  War story's real numbers: ~130GB weights -> ~11GB KV headroom, vLLM capped at 92,544 tokens.
- (c) "Context kills VRAM" reconstruction: 7.5 GiB weights (weight_bytes) + GQA
  (32L/4KV/d128/fp16 -> 65536 B/token -> 8 GiB at 128K) on RTX 3080 12GB -> OOM.
  Total 16 GiB vs article's 15GB on 12GB card. Documented in test docstring.

## Bugs caught during build
1. `__main__.py` imported `main` but never called it -> `python -m fitcheck` printed
   nothing with rc=0. Fixed to call main().
2. My own arithmetic in two MLA test expectations (140288 vs correct 140544);
   the code was right, the expectation wrong. Fixed expectations.
3. False alarm: qwen3-32b fp8 shows 131072 B/token == llama-3.1-8b fp16's figure
   by coincidence (64 layers x 1 byte == 32 layers x 2 bytes). Verified not a bug.

## Design decisions
- GiB (2^30) everywhere, matching vLLM/llama.cpp runtime printouts.
- Verdict OOM triggers when requested-context KV exceeds the KV budget (vLLM fails
  at startup then) OR raw usage exceeds VRAM. TIGHT at >85% raw usage.
- Defaults: 8% reserve, vLLM util 0.9 / llama.cpp 0.95, static overhead
  1.5/0.5 GiB per GPU. All overridable.
- Exit code 1 on OOM for scripting; --json for machine use.
- Tests never touch network: fixtures for HF config parse; --from-hf is CLI-only.

## Not yet (future scope, out of MVP)
- GGUF local file parsing (dtype flag accepted for weights sizing, but no
  gguf reader from a local file).
- vLLM paged-block granularity, prefix-cache savings, prefill activation spikes.
- MoE expert-parallel / pipeline-parallel layouts (TP only).
- Live HF model metadata (params) for --from-hf (config lacks num_parameters;
  currently errors and asks for --params).
