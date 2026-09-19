"""Validation suite — pins fitcheck's math to published/measured figures.

(a) Llama 3.1 8B: 128 KiB/token -> ~16 GiB KV at 128K tokens.
(b) Compliance-org war story: 130GB-class weights on 2xA100-80 must NOT get a
    FITS verdict at 196K context.
(c) "Context kills VRAM" article: 7.5GB weights + ~7GB KV at 128K = OOM on a
    12GB card.
"""
import io
import json
import unittest
from contextlib import redirect_stdout

from fitcheck.cli import main, parse_count
from fitcheck.budget import budget
from fitcheck.engine import engine_profile
from fitcheck.kv import per_token_bytes, kv_cache_bytes
from fitcheck.models import load_preset, list_presets, model_from_hf_config

GIB = 2 ** 30


def make_budget(model, vram_gib, gpus=1, ctx=8192, kv_dtype="fp16",
                w_dtype=None, concurrency=1, engine="vllm", util=None,
                reserve=0.08):
    eng = engine_profile(engine)
    util = eng["default_utilization"] if util is None else util
    tp = gpus
    return budget(model, vram_gib, gpus, tp, ctx, kv_dtype, w_dtype,
                  concurrency, eng, util, reserve)


class TestPerTokenMath(unittest.TestCase):
    def test_llama31_8b_per_token(self):
        # 2 * 32 layers * 8 kv heads * 128 head_dim * 2 bytes = 131072 B/token
        m = load_preset("llama-3.1-8b")
        self.assertEqual(per_token_bytes(m, "fp16"), 131072)

    def test_llama31_8b_kv_at_128k(self):
        # Published ballpark: ~16 GB KV at 128K context.
        m = load_preset("llama-3.1-8b")
        self.assertEqual(kv_cache_bytes(m, 131072, "fp16"), 16 * GIB)

    def test_deepseek_v3_mla(self):
        # MLA absorbed KV: 2 * 61 layers * (512 + 64) * 2 bytes = 140544 B/token
        m = load_preset("deepseek-v3")
        self.assertEqual(per_token_bytes(m, "fp16"), 140544)
        self.assertEqual(per_token_bytes(m, "fp8"), 70272)

    def test_sliding_window_bounds_kv(self):
        # Sliding-window math (synthetic SWA model): KV at 128K == KV at 4096
        # Note: no built-in preset uses SWA as of 2026-09-19 (v0.3 dropped it),
        # so this pins the math on a synthetic model, not a preset.
        m = {"name": "swa-synth", "attention": "standard", "layers": 32,
             "attention_heads": 32, "kv_heads": 8, "head_dim": 128,
             "sliding_window": 4096}
        self.assertEqual(kv_cache_bytes(m, 131072, "fp16"),
                         kv_cache_bytes(m, 4096, "fp16"))
        # 2*32*8*128*2*4096 = 536870912 B = 0.5 GiB
        self.assertEqual(kv_cache_bytes(m, 131072, "fp16"), 536870912)

    def test_mistral_v03_no_sliding_window(self):
        # Mistral-7B-v0.3 dropped the sliding window (unlike v0.1's 4K SWA):
        # full-context KV. Regression guard for the preset.
        m = load_preset("mistral-7b-v0.3")
        self.assertIsNone(m["sliding_window"])
        self.assertEqual(kv_cache_bytes(m, 32768, "fp16"),
                         2 * 32 * 8 * 128 * 2 * 32768)

    def test_phi4_arch(self):
        # Phi-4 (not Phi-4-Mini): 40 layers, 40 heads, 10 KV heads, d128.
        m = load_preset("phi-4")
        self.assertEqual((m["layers"], m["attention_heads"], m["kv_heads"],
                          m["head_dim"]), (40, 40, 10, 128))
        self.assertEqual(per_token_bytes(m, "bf16"), 2 * 40 * 10 * 128 * 2)

    def test_gemma3_global_layers_full_ctx(self):
        # 10 of 62 layers attend globally -> they see the full context
        m = load_preset("gemma-3-27b")
        at_128k = kv_cache_bytes(m, 131072, "bf16")
        # hand check: 2*16*168*2 = 10752 B/layer/token
        exp = 10752 * (52 * 1024 + 10 * 131072)
        self.assertEqual(at_128k, exp)


class TestWarStories(unittest.TestCase):
    def test_compliance_org_2xa100(self):
        """Reconstruction of the compliance-org war story
        (medium.com/@leonid_91859/running-local-ai-models-for-compliance-sensitive-organizations):
        ~130GB of weights on 2xA100-80GB (160GB) left only ~11GB for KV in
        practice; vLLM capped the hardware at ~92K tokens while the model
        advertised 196K. Reconstruction: 128B-param fp8 model (61 layers,
        8 kv heads, head_dim 128) ~= 119 GiB of weights on 2xA100-80.
        fitcheck must NOT bless 196K context here."""
        model = {
            "name": "compliance-128b", "attention": "standard",
            "layers": 61, "kv_heads": 8, "head_dim": 128,
            "weight_dtype": "fp8", "params": 128e9,
        }
        b = make_budget(model, 80, gpus=2, ctx=196608, kv_dtype="fp8",
                        engine="vllm")
        self.assertEqual(b["verdict"], "OOM")
        # weights ~= 128e9 * 1 B = 119.2 GiB total, 59.6 GiB per GPU (TP=2)
        self.assertAlmostEqual(b["weights_gib"], 128e9 / GIB / 2, places=1)
        # max safe context must be well under the advertised 196K
        self.assertLess(b["max_safe_ctx"], 196608)
        self.assertGreater(b["max_safe_ctx"], 0)

    def test_context_kills_vram(self):
        """Reconstruction of the "Context kills VRAM" article
        (medium.com/@lyx_62906/context-kills-vram-running-llms-on-a-local-gpu):
        7.5GB weights + ~7.0GB KV at 128K = ~15GB, OOM on a 12GB card.
        The article doesn't name the model; we reconstruct with a GQA model
        (32 layers, 4 kv heads, head_dim 128, fp16 -> 65536 B/token -> 8 GiB
        at 128K), which lands at the same ~15GB total."""
        model = {
            "name": "gqa-reconstruction", "attention": "standard",
            "layers": 32, "kv_heads": 4, "head_dim": 128,
            "weight_dtype": "fp16", "weight_bytes": int(7.5 * GIB),
        }
        ptb = per_token_bytes(model, "fp16")
        self.assertEqual(ptb, 65536)  # ~64 KiB/token, article's ~54-64 KiB ballpark
        kv = kv_cache_bytes(model, 131072, "fp16")
        self.assertAlmostEqual(kv / GIB, 8.0, places=1)
        b = make_budget(model, 12, gpus=1, ctx=131072, engine="vllm")
        self.assertEqual(b["verdict"], "OOM")
        self.assertAlmostEqual(b["used_gib"], 7.5 + 8.0 + 1.5, places=1)


class TestBudgetsAndVerdicts(unittest.TestCase):
    def test_fits_small_model(self):
        m = load_preset("llama-3.1-8b")
        b = make_budget(m, 24, gpus=1, ctx=8192, engine="vllm")
        self.assertEqual(b["verdict"], "FITS")
        self.assertGreater(b["headroom_gib"], 0)

    def test_tensor_parallel_halves(self):
        m = load_preset("llama-3.3-70b")
        b1 = make_budget(m, 80, gpus=1, ctx=8192, engine="vllm")
        eng = engine_profile("vllm")
        b2 = budget(m, 80, 2, 2, 8192, "fp16", None, 1, eng, 0.9, 0.08)
        self.assertAlmostEqual(b2["weights_gib"], b1["weights_gib"] / 2, places=2)
        self.assertAlmostEqual(b2["kv_gib"], b1["kv_gib"] / 2, places=2)

    def test_quantized_kv_halves_cache(self):
        m = load_preset("llama-3.1-8b")
        full = kv_cache_bytes(m, 131072, "fp16")
        q = kv_cache_bytes(m, 131072, "fp8")
        self.assertAlmostEqual(q, full / 2)

    def test_max_safe_ctx_math(self):
        m = load_preset("llama-3.1-8b")
        b = make_budget(m, 24, gpus=1, ctx=131072, engine="vllm")
        # KV budget = 24*0.92*0.9 - 16.0(weights) - 1.5(static) GiB
        weights = 8030261248 * 2 / GIB
        kv_budget = 24 * 0.92 * 0.9 - weights - 1.5
        expected = int(kv_budget * GIB / 131072)
        self.assertEqual(b["max_safe_ctx"], expected)

    def test_tight_boundary(self):
        m = load_preset("llama-3.1-8b")
        b = make_budget(m, 24, gpus=1, ctx=131072, engine="vllm")
        # used/total = (16 + 16 + 1.5)/24 = 1.39 -> OOM, sanity on boundary logic
        self.assertEqual(b["verdict"], "OOM")
        self.assertGreater(b["used_ratio"], 1.0)

    def test_headline_case_is_tight_not_oom(self):
        # Llama-3.1-8B at 32k on a 4090 is servable in practice (~20.5 GiB
        # used vs 21.6 GiB vLLM pool). Must be TIGHT, never OOM — a flat OOM
        # here would be the publicly-wrong verdict this tool exists to avoid.
        m = load_preset("llama-3.1-8b")
        b = make_budget(m, 24, gpus=1, ctx=32768, engine="vllm")
        self.assertEqual(b["verdict"], "TIGHT")
        self.assertGreater(b["pool_headroom_gib"], 0)
        self.assertLess(b["max_safe_ctx"], 32768)

    def test_comfortable_fit(self):
        m = load_preset("llama-3.1-8b")
        b = make_budget(m, 24, gpus=1, ctx=8192, engine="vllm")
        self.assertEqual(b["verdict"], "FITS")


class TestCLIOperations(unittest.TestCase):
    def test_parse_count(self):
        self.assertEqual(parse_count("128k"), 131072)
        self.assertEqual(parse_count("1m"), 1048576)
        self.assertEqual(parse_count("32000"), 32000)
        self.assertEqual(parse_count("32B"), 32000000000)

    def test_preset_list(self):
        names = list_presets()
        self.assertIn("llama-3.1-8b", names)
        self.assertIn("deepseek-v3", names)
        self.assertGreaterEqual(len(names), 7)

    def test_hf_config_parse(self):
        # Fixture shaped like Qwen/Qwen3-32B config.json (offline).
        cfg = {
            "num_hidden_layers": 64, "num_attention_heads": 64,
            "num_key_value_heads": 8, "hidden_size": 5120,
            "torch_dtype": "bfloat16", "sliding_window": None,
        }
        m = model_from_hf_config(cfg, repo="Qwen/Qwen3-32B")
        self.assertEqual(m["layers"], 64)
        self.assertEqual(m["kv_heads"], 8)
        self.assertEqual(m["head_dim"], 80)  # 5120 // 64
        self.assertEqual(m["weight_dtype"], "bfloat16")

    def test_hf_config_mla_detected(self):
        cfg = {"num_hidden_layers": 61, "kv_lora_rank": 512,
               "qk_rope_head_dim": 64, "torch_dtype": "bfloat16"}
        m = model_from_hf_config(cfg, repo="deepseek-ai/DeepSeek-V3")
        self.assertEqual(m["attention"], "mla")
        self.assertEqual(per_token_bytes(m, "fp16"), 140544)

    def test_json_output(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["llama-3.1-8b", "--gpu", "4090", "--ctx", "8k", "--json"])
        self.assertEqual(rc, 0)
        out = json.loads(buf.getvalue())
        self.assertIn(out["verdict"], ("FITS", "TIGHT", "OOM"))
        self.assertIn("--max-model-len", " ".join(out["recommended_flags"]))
        self.assertIn("max_safe_ctx", out)

    def test_oom_exit_code(self):
        with redirect_stdout(io.StringIO()):
            rc = main(["qwen3-32b", "--gpu", "3080", "--ctx", "128k"])
        self.assertEqual(rc, 1)  # OOM -> nonzero exit for scripting

    def test_llamacpp_flags(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["mistral-7b-v0.3", "--gpu", "4090", "--ctx", "8k",
                  "--engine", "llama.cpp", "--json"])
        out = json.loads(buf.getvalue())
        flags = " ".join(out["recommended_flags"])
        self.assertIn("--ctx-size", flags)
        self.assertIn("-ngl", flags)

    def test_unknown_preset_errors(self):
        with redirect_stdout(io.StringIO()):
            rc = main(["definitely-not-a-model", "--gpu", "4090"])
        self.assertEqual(rc, 2)

    def test_manual_flags(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--layers", "32", "--kv-heads", "4", "--head-dim", "128",
                       "--params", "3.75B", "--gpu", "3080", "--ctx", "8k", "--json"])
        self.assertEqual(rc, 0)
        out = json.loads(buf.getvalue())
        self.assertEqual(out["kv_per_token_bytes"], 65536)


if __name__ == "__main__":
    unittest.main()
