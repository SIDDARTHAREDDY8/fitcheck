"""Model definitions: built-in presets, HuggingFace config parsing, HF fetch.

Presets are JSON fixtures verified against the actual published HF configs.
Tests use the fixtures (no network). `--from-hf <org/repo>` fetches the live
config.json via urllib only when the user explicitly asks.
"""

import json
import os
import urllib.request

PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")

WEIGHT_DTYPE_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "int8": 1.0,
    "int4": 0.5,
    "nf4": 0.5,
    "gguf-q8": 1.05,
    "gguf-q6": 0.78,
    "gguf-q5": 0.69,
    "gguf-q4": 0.55,
    "gguf-q3": 0.43,
    "gguf-q2": 0.34,
    # GGUF family aliases users actually type
    "q8_0": 1.05, "q6_k": 0.78, "q5_k_m": 0.69, "q4_k_m": 0.55,
    "q3_k_m": 0.43, "q2_k": 0.34,
}


def weight_dtype_bytes(name: str) -> float:
    norm = name.strip().lower().replace("-", "").replace("_", "")
    table = {k.replace("-", "").replace("_", ""): v for k, v in WEIGHT_DTYPE_BYTES.items()}
    if norm not in table:
        raise ValueError(f"unknown weight dtype {name!r}; known: {sorted(WEIGHT_DTYPE_BYTES)}")
    return table[norm]


def list_presets():
    names = []
    for f in sorted(os.listdir(PRESET_DIR)):
        if f.endswith(".json"):
            names.append(f[:-5])
    return names


def load_preset(name: str) -> dict:
    key = name.strip().lower().replace("_", "-")
    path = os.path.join(PRESET_DIR, key + ".json")
    if not os.path.isfile(path):
        raise ValueError(
            f"unknown preset {name!r}; known presets: {', '.join(list_presets())}"
        )
    with open(path) as fh:
        return json.load(fh)


def model_from_hf_config(cfg: dict, repo: str = "") -> dict:
    """Translate a HuggingFace config.json dict into fitcheck's model schema."""
    n_layers = cfg.get("num_hidden_layers")
    n_heads = cfg.get("num_attention_heads")
    n_kv = cfg.get("num_key_value_heads")
    hidden = cfg.get("hidden_size")
    head_dim = cfg.get("head_dim") or cfg.get("qk_head_dim")
    if head_dim is None and hidden and n_heads:
        head_dim = hidden // n_heads

    # MLA detection (DeepSeek V3 / V2 family)
    if "kv_lora_rank" in cfg or "qk_rope_head_dim" in cfg:
        return {
            "name": repo or cfg.get("_name_or_path", "unknown"),
            "description": f"MLA model ({repo})" if repo else "MLA model",
            "attention": "mla",
            "layers": n_layers,
            "kv_lora_rank": cfg["kv_lora_rank"],
            "qk_rope_head_dim": cfg["qk_rope_head_dim"],
            "weight_dtype": (cfg.get("torch_dtype") or "bf16").replace("torch.", ""),
            "params": cfg.get("num_parameters"),
        }

    model = {
        "name": repo or cfg.get("_name_or_path", "unknown"),
        "description": f"HF config ({repo})" if repo else "HF config",
        "attention": "standard",
        "layers": n_layers,
        "attention_heads": n_heads,
        "kv_heads": n_kv,
        "head_dim": head_dim,
        "weight_dtype": (cfg.get("torch_dtype") or "bf16").replace("torch.", ""),
        "params": cfg.get("num_parameters"),
        "sliding_window": cfg.get("sliding_window"),
        "global_layers": 0,
    }
    # Gemma 3 style: 5 local + 1 global attention layers repeating
    att_types = cfg.get("layer_types") or []
    if att_types and "full_attention" in att_types:
        model["global_layers"] = sum(1 for t in att_types if t == "full_attention")
    return model


def read_local_config(path: str) -> dict:
    """Read a local HF config.json (file path or model directory)."""
    if os.path.isdir(path):
        cfg_path = os.path.join(path, "config.json")
    else:
        cfg_path = path
    if not os.path.isfile(cfg_path):
        raise ValueError(f"no config.json found at {cfg_path!r}")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    model = model_from_hf_config(cfg, repo=os.path.basename(os.path.abspath(path)))

    # If the dir has safetensors, use real on-disk weight size.
    model_dir = os.path.dirname(cfg_path)
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    seen, total = set(), 0
    for candidate in (index_path,):
        if os.path.isfile(candidate):
            with open(candidate) as fh:
                index = json.load(fh)
            files = index.get("weight_map", {})
            for f in set(files.values()):
                fp = os.path.join(model_dir, f)
                if fp not in seen and os.path.isfile(fp):
                    seen.add(fp)
                    total += os.path.getsize(fp)
            break
    else:
        for f in sorted(os.listdir(model_dir)):
            if f.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(model_dir, f))
    if total:
        model["weight_bytes"] = total
    return model


def fetch_hf_config(repo: str) -> dict:
    """Fetch config.json for <org/repo> from huggingface.co (needs network)."""
    url = f"https://huggingface.co/{repo}/resolve/main/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "fitcheck/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        cfg = json.loads(resp.read().decode("utf-8"))
    return model_from_hf_config(cfg, repo=repo)
