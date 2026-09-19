"""CLI: fitcheck <model> --gpu 4090 --ctx 128k --engine vllm"""

import argparse
import json
import os
import re
import sys

from . import __version__
from .budget import budget, ascii_bar, GIB
from .engine import engine_profile, recommended_flags, known_engines
from .gpus import lookup as gpu_lookup, known_names as known_gpu_names
from .kv import kv_cache_bytes
from .models import (
    load_preset, list_presets, read_local_config, fetch_hf_config,
)


def parse_count(s: str) -> int:
    """Parse 128k / 1m / 32000 / 32B style counts."""
    m = re.fullmatch(r"\s*([\d.]+)\s*([kmgbKMGB]?)\s*", str(s))
    if not m:
        raise argparse.ArgumentTypeError(f"cannot parse count {s!r}")
    val, suffix = float(m.group(1)), m.group(2).lower()
    mult = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "b": 1e9}.get(suffix)
    if mult is None:
        raise argparse.ArgumentTypeError(f"cannot parse count {s!r}")
    return int(val * mult)


def parse_gb(s: str) -> float:
    """Parse a VRAM size; treats '24' as GiB, '24GB'/'24G' likewise."""
    m = re.fullmatch(r"\s*([\d.]+)\s*(gb?|gib)?\s*", str(s), re.IGNORECASE)
    if not m:
        raise argparse.ArgumentTypeError(f"cannot parse size {s!r}")
    return float(m.group(1))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fitcheck",
        description=(
            "Will this model fit on your GPU(s) at this context length with this engine?\n"
            "Computes real KV-cache math from the HF config — no trial-and-error OOMs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  fitcheck llama-3.1-8b --gpu 4090 --ctx 128k\n"
            "  fitcheck qwen3-32b --gpu a100-80 --gpus 2 --ctx 64k --engine vllm --kv-dtype fp8\n"
            "  fitcheck ./my-model-dir --gpu-vram 48 --ctx 1m --engine llama.cpp\n"
            "  fitcheck --from-hf Qwen/Qwen3-32B --gpu 5090 --ctx 32k\n"
            "  fitcheck custom --layers 32 --kv-heads 8 --head-dim 128 --params 7B \\\n"
            "           --gpu 3080 --ctx 8k\n"
            f"\nknown presets: {', '.join(list_presets())}\n"
            f"known GPUs: {', '.join(known_gpu_names())}"
        ),
    )
    p.add_argument("model", nargs="?",
                   help="preset name, path to HF config.json, or a model directory")
    p.add_argument("--version", action="version", version=f"fitcheck {__version__}")

    # Model sources
    src = p.add_argument_group("model sources")
    src.add_argument("--from-hf", metavar="ORG/REPO",
                     help="fetch config.json from huggingface.co (needs network)")
    src.add_argument("--layers", type=int, help="manual: transformer layers")
    src.add_argument("--kv-heads", type=int, help="manual: key/value heads (GQA)")
    src.add_argument("--attention-heads", type=int, help="manual: query heads")
    src.add_argument("--head-dim", type=int, help="manual: per-head dim")
    src.add_argument("--attention", choices=["standard", "mla"], default="standard")
    src.add_argument("--kv-lora-rank", type=int, help="manual MLA: kv_lora_rank")
    src.add_argument("--qk-rope-head-dim", type=int, help="manual MLA: qk_rope_head_dim")
    src.add_argument("--sliding-window", type=parse_count,
                     help="manual: sliding window size (KV bounded by window)")
    src.add_argument("--params", type=parse_count, help="manual: param count, e.g. 32B")
    src.add_argument("--w-dtype", default=None,
                     help="weight dtype: bf16/fp16/fp8/int8/int4/gguf-q4... (default from preset/config)")
    src.add_argument("--kv-dtype", default="fp16",
                     help="KV cache dtype: fp16/bf16/fp8/int4 (default fp16)")

    # Hardware
    hw = p.add_argument_group("hardware")
    hw.add_argument("--gpu", default="4090",
                    help="GPU name from the built-in DB (default 4090)")
    hw.add_argument("--gpu-vram", type=parse_gb, default=None,
                    help="override VRAM in GiB (skips the DB)")
    hw.add_argument("--gpus", type=int, default=1, help="number of GPUs")
    hw.add_argument("--tensor-parallel", type=int, default=None,
                    help="TP degree (default: = --gpus)")

    # Run config
    run = p.add_argument_group("run config")
    run.add_argument("--ctx", type=parse_count, default=8192,
                     help="context length, e.g. 128k (default 8192)")
    run.add_argument("--engine", default="vllm",
                     help=f"engine: {', '.join(known_engines())} (default vllm)")
    run.add_argument("--concurrency", type=int, default=1,
                     help="concurrent sequences (vLLM max_num_seqs / llama.cpp --parallel)")
    run.add_argument("--gpu-mem-util", type=float, default=None,
                     help="override engine memory utilization (default 0.9 vLLM, 0.95 llama.cpp)")
    run.add_argument("--reserve", type=float, default=0.08,
                     help="fraction of VRAM held back as safety reserve (default 0.08)")

    out = p.add_argument_group("output")
    out.add_argument("--json", action="store_true",
                     help="machine-readable JSON output")
    out.add_argument("--list-presets", action="store_true",
                     help="list built-in model presets and exit")
    out.add_argument("--list-gpus", action="store_true",
                     help="list built-in GPUs and exit")
    return p


def resolve_model(args) -> dict:
    """Resolve the model dict from preset / path / --from-hf / manual flags."""
    manual = any([args.layers, args.kv_heads, args.head_dim,
                  args.kv_lora_rank, args.params, args.attention == "mla"])

    model = None
    if args.from_hf:
        model = fetch_hf_config(args.from_hf)
    elif args.model:
        # Try preset, then local path, in that order.
        try:
            model = load_preset(args.model)
        except ValueError:
            if os.path.exists(args.model):
                model = read_local_config(args.model)
            else:
                raise
    elif not manual:
        raise SystemExit("error: give a preset name, a config path, --from-hf, or manual flags "
                         "(try --help)")

    if manual:
        model = dict(model or {})
        if args.layers:
            model["layers"] = args.layers
        if args.attention_heads:
            model["attention_heads"] = args.attention_heads
        if args.kv_heads:
            model["kv_heads"] = args.kv_heads
        if args.head_dim:
            model["head_dim"] = args.head_dim
        model["attention"] = args.attention
        if args.attention == "mla":
            if not args.kv_lora_rank or not args.qk_rope_head_dim:
                raise SystemExit("error: MLA needs --kv-lora-rank and --qk-rope-head-dim")
            model["kv_lora_rank"] = args.kv_lora_rank
            model["qk_rope_head_dim"] = args.qk_rope_head_dim
        if args.sliding_window:
            model["sliding_window"] = args.sliding_window
        if args.params:
            model["params"] = args.params
        model.setdefault("name", "custom")

    if args.w_dtype:
        model["weight_dtype"] = args.w_dtype
    if not model.get("params") and not model.get("weight_bytes"):
        raise SystemExit("error: model has no param count; add --params")
    return model


def human(model_name: str, b: dict, engine: str, flags: list,
          kv_dtype: str, w_dtype: str, ptb: float) -> str:
    L = []
    L.append(f"fitcheck {__version__}  |  {model_name}")
    L.append("=" * 64)
    L.append(f"engine: {engine}    ctx: {b['ctx']:,} tokens    "
             f"concurrency: {b['concurrency']}    GPUs: {b['gpus']}x{b['vram_per_gpu_gib']:g} GiB"
             + (f" (TP={b['tensor_parallel']})" if b["tensor_parallel"] > 1 else ""))
    L.append(f"weights: {w_dtype}    KV dtype: {kv_dtype}    "
             f"KV/token: {ptb:,.0f} B ({ptb / 1024:.1f} KiB)")
    L.append("")
    L.append(ascii_bar(b))
    L.append("")
    verdict = b["verdict"]
    icon = {"FITS": "[OK]", "TIGHT": "[!]", "OOM": "[X]"}[verdict]
    L.append(f"verdict: {icon} {verdict}")
    if verdict == "OOM":
        L.append(f"  needs {b['used_gib']:.2f} GiB but the engine will only allocate {b['vram_per_gpu_gib'] * b['utilization']:.2f} GiB per GPU")
        L.append(f"  max safe context at this concurrency: {b['max_safe_ctx']:,} tokens")
    elif verdict == "TIGHT":
        L.append(f"  only {b['pool_headroom_gib']:.2f} GiB inside the engine pool — servable but tight;")
        L.append(f"  lower --ctx/--concurrency or raise --gpu-mem-util for margin")
        L.append(f"  max safe context at this concurrency: {b['max_safe_ctx']:,} tokens")
    else:
        L.append(f"  headroom: {b['headroom_gib']:.2f} GiB — you could push to ~{b['max_safe_ctx']:,} tokens at this concurrency")
    L.append("")
    L.append(f"recommended {engine} flags:")
    for f in flags:
        L.append(f"  {f}")
    return "\n".join(L)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        print("\n".join(list_presets()))
        return 0
    if args.list_gpus:
        for n in known_gpu_names():
            vram, arch = gpu_lookup(n)
            print(f"{n:14} {vram:>3} GiB  {arch}")
        return 0

    try:
        model = resolve_model(args)
    except ValueError as e:
        print(f"fitcheck: error: {e}", file=sys.stderr)
        return 2

    if args.gpu_vram:
        vram_gib = args.gpu_vram
    else:
        hit = gpu_lookup(args.gpu)
        if not hit:
            print(f"fitcheck: error: unknown GPU {args.gpu!r}; use --gpu-vram or one of: "
                  + ", ".join(known_gpu_names()), file=sys.stderr)
            return 2
        vram_gib = hit[0]

    try:
        eng = engine_profile(args.engine)
    except ValueError as e:
        print(f"fitcheck: error: {e}", file=sys.stderr)
        return 2
    utilization = args.gpu_mem_util if args.gpu_mem_util is not None else eng["default_utilization"]

    tp = args.tensor_parallel if args.tensor_parallel is not None else args.gpus
    try:
        b = budget(model, vram_gib, args.gpus, tp, args.ctx, args.kv_dtype,
                   args.w_dtype, args.concurrency, eng, utilization, args.reserve)
    except ValueError as e:
        print(f"fitcheck: error: {e}", file=sys.stderr)
        return 2

    w_dtype = args.w_dtype or model.get("weight_dtype", "bf16")
    flags = recommended_flags(args.engine, args.ctx, args.concurrency, tp,
                              args.kv_dtype, utilization)
    # Effective per-token KV at this context (sliding-window layers are
    # bounded by the window, so the nominal figure would mislead).
    ptb = kv_cache_bytes(model, args.ctx, args.kv_dtype, 1) / args.ctx if args.ctx else 0

    if args.json:
        out = dict(b)
        out.update({
            "model": model.get("name"),
            "weight_dtype": w_dtype,
            "kv_dtype": args.kv_dtype,
            "engine": args.engine,
            "recommended_flags": flags,
        })
        print(json.dumps(out, indent=2))
    else:
        print(human(model.get("name", "?"), b, eng["label"], flags,
                    args.kv_dtype, w_dtype, ptb))
    return 0 if b["verdict"] != "OOM" else 1


if __name__ == "__main__":
    sys.exit(main())
