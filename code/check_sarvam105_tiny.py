"""Free local check of Sarvam-105B's Hub code under transformers 5.14.1 (CPU, ~1 min).

    python code/check_sarvam105_tiny.py <dir with configuration_sarvam_moe.py, modeling_sarvam_moe.py, config.json> <out dir>

Builds a tiny random SarvamMLA model from the real config, applies the three transformers-5 shims
the 105B needs, round-trips it through save/from_pretrained, and checks: eager vs sdpa support, the
RoPE cache before and after the first forward, logits equality, and jlens layout auto-detection.
Verified 2026-10-18: imports need 2 shims; sdpa unsupported; RoPE OK after the first forward."""
import importlib, json, sys, types
from pathlib import Path
import torch as t

code = Path(sys.argv[1]); out = Path(sys.argv[2])

# transformers-5 shims for Sarvam-105B's 4.57-era imports: fill in ONLY names that are missing,
# with their transformers-4 behaviour (torch>=1.13 and torch.fx are always available on our torch).
import transformers.utils as _tu, transformers.utils.import_utils as _iu, transformers.pytorch_utils as _pu
_missing = []
if not hasattr(_iu, "is_torch_fx_available"):
    _iu.is_torch_fx_available = lambda: True; _missing.append("is_torch_fx_available")
if not hasattr(_pu, "is_torch_greater_or_equal_than_1_13"):
    _pu.is_torch_greater_or_equal_than_1_13 = True; _missing.append("is_torch_greater_or_equal_than_1_13")
for _name in ("add_start_docstrings", "add_start_docstrings_to_model_forward", "replace_return_docstrings"):
    if not hasattr(_tu, _name):
        setattr(_tu, _name, lambda *a, **k: (lambda f: f)); _missing.append(_name)
print("shimmed (missing in transformers 5):", _missing)
pkg = types.ModuleType("sarvam_hub_mla"); pkg.__path__ = [str(code)]; sys.modules["sarvam_hub_mla"] = pkg
modeling = importlib.import_module("sarvam_hub_mla.modeling_sarvam_moe")
cfg_mod = sys.modules["sarvam_hub_mla.configuration_sarvam_moe"]
# transformers 5.14's validate_rope() takes no ignore_keys; Sarvam's config override passes one.
# The model builds its own YaRN rotary module from config scalars, so config-side validation is
# not needed for correctness: accept and drop the argument, then run the stock validation.
_stock_validate = cfg_mod.SarvamMLAConfig.validate_rope
def _validate_rope(self, ignore_keys=None, **_):
    try:
        return _stock_validate(self)
    except Exception as exc:  # e.g. an unknown "deepseek_yarn" rope_type in 5.14's validator
        print("  (validate_rope skipped:", repr(exc)[:120], ")")
cfg_mod.SarvamMLAConfig.validate_rope = _validate_rope
print("import OK; tied keys:", modeling.SarvamMLAForCausalLM._tied_weights_keys)
real = json.loads((code / "config.json").read_text())
tiny = {k: v for k, v in real.items() if k not in {"architectures", "auto_map", "transformers_version", "dtype", "attn_implementation"}}
tiny.update(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, qk_nope_head_dim=16,
            qk_rope_head_dim=8, q_head_dim=24, v_head_dim=16, kv_lora_rank=32, head_dim=40, num_experts=8, num_experts_per_tok=2,
            moe_intermediate_size=32, max_position_embeddings=512, initializer_range=0.08)
config = cfg_mod.SarvamMLAConfig(**tiny)
print("config q_head_dim:", getattr(config, "q_head_dim", None), "| q_lora_rank:", getattr(config, "q_lora_rank", None))
if isinstance(modeling.SarvamMLAForCausalLM._tied_weights_keys, list):  # same shim as the 30B
    modeling.SarvamMLAForCausalLM._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
t.manual_seed(0)
model = modeling.SarvamMLAForCausalLM(config)
model.save_pretrained(out)
ref = model.model.layers[0].self_attn.rotary_emb
ref_cos = None
ids = t.randint(10, 1000, (1, 24))
with t.no_grad():
    model(input_ids=ids, use_cache=False)
ref_cos = ref.cos_cached[:24].clone()
for attn_impl in ("eager", "sdpa"):
    try:
        loaded = modeling.SarvamMLAForCausalLM.from_pretrained(out, dtype=t.float32, attn_implementation=attn_impl)
        rot = loaded.model.layers[0].self_attn.rotary_emb
        before = {k: (None if getattr(rot, k, None) is None else float(getattr(rot, k).abs().sum())) for k in ("inv_freq", "cos_cached", "sin_cached")}
        with t.no_grad():
            logits = loaded(input_ids=ids, use_cache=False).logits
        after_cos = rot.cos_cached[:24]
        print(f"[{attn_impl}] load OK | buffers |sum| before first forward: {before} | max_seq_len_cached={rot.max_seq_len_cached}")
        print(f"[{attn_impl}] after first forward: cos matches fresh model: {bool(t.allclose(after_cos.float(), ref_cos.float(), atol=1e-5))} | logits finite: {bool(t.isfinite(logits).all())}")
        with t.no_grad():
            ref_logits = model(input_ids=ids, use_cache=False).logits
        print(f"[{attn_impl}] loaded logits == original logits: {bool(t.allclose(logits, ref_logits, atol=1e-4))}")
    except Exception as exc:
        print(f"[{attn_impl}] FAILED: {exc!r}"[:400])
from jlens.hf import _find_layout
print("jlens layout auto-detect:", _find_layout(loaded))
