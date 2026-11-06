"""Sarvam-30B support for jlens and the J++ pipeline (safety-research/jpp_lens).

Stock J++ (and stock transformers 5) cannot handle ``sarvamai/sarvam-30b`` in four places:

1. Loading. The architecture ships as Hub custom code (``SarvamMoEForCausalLM``),
   and the token embedding is called ``word_embeddings``, so jlens' layout
   auto-detection fails. We import the code from a pinned Hub revision and
   instantiate the class directly, so ``trust_remote_code`` is never needed.
2. BOS. The tokenizer has two start tokens: ``<bos>``, which its post-processor
   adds, and ``[@BOS@]``, the configured ``bos_token``, which the chat template
   writes. jlens' ``force_bos`` rewrites the post-processor and drops ``<bos>``.
   :class:`SarvamLensModel` builds the prefix explicitly. The default, ``[@BOS@]``
   alone, is the measured best (see :data:`BOS_MODES`).
2b. RoPE. Under transformers 5, ``from_pretrained`` leaves Sarvam's non-persistent
   ``inv_freq`` buffer zeroed. The model then has no position information and reads
   its context as a bag of words. :func:`_rebuild_rotary_embedding` recomputes it.
3. LRP surgery (R-Lens / J++ Lens). jpp's ``apply_lrp_rules`` only has patchers
   for Qwen/Gemma/OLMo/DeepSeek classes and raises on Sarvam's. Its q/k-norm
   exclusion also matches by name (``q_norm``/``k_norm`` or under ``self_attn``),
   while Sarvam's attention norms are ``attention.query_layernorm`` /
   ``attention.key_layernorm``. :func:`apply_sarvam_lrp_rules` mirrors jpp's
   rule semantics for Sarvam's modules:

   ============================  ======================================================
   jpp rule flag                 Sarvam module it acts on
   ============================  ======================================================
   ln_rule                       residual-stream ``SarvamMoERMSNorm`` (input_layernorm,
                                 post_attention_layernorm); q/k norms keep stock grads
   identity_rule / half_rule     dense layer-0 ``SarvamMoEMLP`` and the shared expert
   routed_experts                the same rules inside every routed ``SarvamMoEMLP``
   router_detach                 ``topk_weight`` detached in the sparse-MoE block
   shared_expert_scale           backward-only gradient scale on the shared expert
   shared_gate_detach            no-op (Sarvam's shared expert has no gate)
   ============================  ======================================================

Every patch is bit-exact in the forward; only the gradients change. That includes the
identity rule, which jpp computes as ``z * sigmoid(z)`` and accepts as "kernel dust"
(see :func:`_make_mlp_lrp_forward` for why that dust is not harmless on an MoE).

Sarvam-105B (``sarvamai/sarvam-105b``, ``SarvamMLAForCausalLM``) loads through the same
:func:`load_sarvam_lens_model`, picked by the checkpoint's ``model_type`` (see
:data:`SARVAM_MODELS`). It needs import- and config-time shims for transformers 5, eager
attention, and no RoPE rebuild (:func:`verify_mla_rope` checks that at runtime). Its code files
have the 30B's names, so it imports under its own package. LRP surgery is 30B-only.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import json
import logging
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch as t
from torch import Tensor, nn

logger = logging.getLogger(__name__)

SARVAM_REPO = "sarvamai/sarvam-30b"
# Pinned so the custom modeling code we import is the code these patchers were checked against.
SARVAM_REVISION = "071ae95e933605ca1104a6b4524a36a98488efa4"
SARVAM_MODEL_TYPE = "sarvam_moe"
SARVAM_CODE_FILES = ("configuration_sarvam_moe.py", "modeling_sarvam_moe.py")
# Sarvam's attention-internal norms; everything else named *layernorm is on the residual stream.
SARVAM_QK_NORM_NAMES = ("query_layernorm", "key_layernorm")

# Both checkpoints ship code files with the SAME names, so each imports under its own package.
#   cls          the *ForCausalLM class in modeling_sarvam_moe.py
#   attn         attention implementation; ``force_attn`` overrides whatever the caller asks for
#                (the 105B's MLA is eager-only: ``attn_implementation="sdpa"`` raises)
#   layout       jlens ``Layout`` kwargs, or None to let jlens auto-detect
#   embed_key    the source of the tied-weights pair (nothing is tied: tie_word_embeddings is false)
#   rebuild_rope the 30B's global rotary module needs rebuilding after load; the 105B's per-layer
#                YaRN modules rebuild their cache on the first forward (see verify_mla_rope)
SARVAM_MODELS: dict[str, dict[str, Any]] = {
    "sarvam_moe": dict(repo=SARVAM_REPO, revision=SARVAM_REVISION, package="sarvam_moe_hub_code",
                       cls="SarvamMoEForCausalLM", config_cls="SarvamMoEConfig", attn="sdpa", force_attn=False,
                       layout=dict(path="model", embed="word_embeddings"),
                       embed_key="model.word_embeddings.weight", rebuild_rope=True),
    "sarvam_mla": dict(repo="sarvamai/sarvam-105b", revision="abe264bdb8fb2244daadaa59ebc2020b3c139e89",
                       package="sarvam_hub_sarvam_mla", cls="SarvamMLAForCausalLM", config_cls="SarvamMLAConfig",
                       attn="eager", force_attn=True, layout=None,
                       embed_key="model.embed_tokens.weight", rebuild_rope=False),
}
_CODE_PACKAGE = SARVAM_MODELS["sarvam_moe"]["package"]
# JSON device map that replaces device_map="auto" (the 2-GPU smoke test splits a tiny model with it).
DEVICE_MAP_ENV = "SARVAM_DEVICE_MAP_JSON"


### LOADING


def load_sarvam_code(
    repo: str | None = None, *, revision: str | None = None, code_dir: str | None = None,
    model_type: str = SARVAM_MODEL_TYPE,
) -> types.ModuleType:
    """Import Sarvam's modeling module from ``code_dir``, or from ``repo`` at ``revision``
    (default: the registry's pinned repo and revision for ``model_type``), as a package
    (the modeling file imports its config relatively). Idempotent per model type."""
    spec = SARVAM_MODELS[model_type]
    module_name = f"{spec['package']}.modeling_sarvam_moe"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if code_dir is None:
        from huggingface_hub import hf_hub_download

        paths = [hf_hub_download(repo or spec["repo"], f, revision=revision or spec["revision"]) for f in SARVAM_CODE_FILES]
        code_dir = str(Path(paths[0]).parent)
    if model_type == "sarvam_mla":
        _apply_mla_import_shims()
    package = types.ModuleType(spec["package"])
    package.__path__ = [str(code_dir)]
    sys.modules[spec["package"]] = package
    modeling = importlib.import_module(module_name)
    if model_type == "sarvam_mla":
        _apply_mla_config_shim(sys.modules[f"{spec['package']}.configuration_sarvam_moe"])
    _apply_transformers5_shims(modeling, model_type)
    return modeling


def _apply_transformers5_shims(modeling: types.ModuleType, model_type: str = SARVAM_MODEL_TYPE) -> None:
    """Sarvam's code targets transformers 4.57; jpp pins 5.14. transformers 5 reads
    ``_tied_weights_keys`` as a ``{target: source}`` dict, while Sarvam declares the 4.x list
    ``["lm_head.weight"]``, which breaks ``save_pretrained`` and weight tying. The checkpoint
    sets ``tie_word_embeddings: false``, so the dict only names the pair. Nothing gets tied."""
    spec = SARVAM_MODELS[model_type]
    causal_lm = getattr(modeling, spec["cls"])
    if isinstance(causal_lm._tied_weights_keys, list):
        causal_lm._tied_weights_keys = {"lm_head.weight": spec["embed_key"]}


def _apply_mla_import_shims() -> None:
    """Sarvam-105B's modeling file imports four transformers-4 names that transformers 5
    removed. Fill in ONLY the missing ones, with their 4.x behaviour (torch >= 1.13 and torch.fx
    are always available on our torch; the docstring decorators are no-ops)."""
    import transformers.pytorch_utils as pu
    import transformers.utils as tu
    import transformers.utils.import_utils as iu

    if not hasattr(iu, "is_torch_fx_available"):
        iu.is_torch_fx_available = lambda: True
    if not hasattr(pu, "is_torch_greater_or_equal_than_1_13"):
        pu.is_torch_greater_or_equal_than_1_13 = True
    for name in ("add_start_docstrings", "add_start_docstrings_to_model_forward", "replace_return_docstrings"):
        if not hasattr(tu, name):
            setattr(tu, name, lambda *a, **k: (lambda f: f))


def _apply_mla_config_shim(config_module: types.ModuleType) -> None:
    """``SarvamMLAConfig.convert_rope_params_to_dict`` calls ``validate_rope(ignore_keys=...)``,
    a keyword transformers 5.14 doesn't take. Accept and drop it, then run the stock validator,
    which may not know the ``deepseek_yarn`` type. The model builds its YaRN module from the
    config's own ``rope_scaling`` (whose ``type`` key survives standardisation), so config-side
    validation isn't needed for correctness; :func:`verify_mla_rope` checks the result."""
    config_cls = config_module.SarvamMLAConfig
    if getattr(config_cls, "_sarvam_rope_shim", False):
        return
    stock_validate = config_cls.validate_rope

    def validate_rope(self: Any, ignore_keys: Any = None, **_: Any) -> Any:
        try:
            return stock_validate(self)
        except Exception as exc:  # e.g. an unknown "deepseek_yarn" rope_type
            logger.info("SarvamMLAConfig.validate_rope skipped: %r", exc)
            return None

    config_cls.validate_rope = validate_rope
    config_cls._sarvam_rope_shim = True


def verify_mla_rope(hf_model: nn.Module, raw_config: dict, modeling: types.ModuleType, *, seq_len: int) -> dict:
    """After a forward of ``seq_len`` tokens, check every 105B attention's rotary cache and
    softmax scale against values built from the raw ``config.json`` (not from the loaded config
    object, so a config-processing bug can't hide on both sides). Raises on any mismatch."""
    rope = raw_config["rope_scaling"]
    reference = modeling.SarvamMLAYarnRotaryEmbedding(
        raw_config["qk_rope_head_dim"], max_position_embeddings=seq_len, base=raw_config["rope_theta"],
        scaling_factor=rope["factor"], original_max_position_embeddings=rope["original_max_position_embeddings"],
        beta_fast=rope["beta_fast"], beta_slow=rope["beta_slow"], mscale=rope["mscale"],
        mscale_all_dim=rope["mscale_all_dim"])
    ref_cos = reference.cos_cached[:seq_len].float()
    head_dim = raw_config["qk_rope_head_dim"] + raw_config["qk_nope_head_dim"]
    mscale = modeling.yarn_get_mscale(rope["factor"], rope["mscale_all_dim"])
    expected_scale = head_dim ** -0.5 * mscale * mscale
    worst, problems = 0.0, []
    for i, layer in enumerate(hf_model.model.layers):
        attn = layer.self_attn
        rot = attn.rotary_emb
        if type(rot).__name__ != "SarvamMLAYarnRotaryEmbedding":
            problems.append(f"layer {i}: rotary is {type(rot).__name__}, not YaRN")
            continue
        if rot.max_seq_len_cached is None or rot.max_seq_len_cached < seq_len:
            problems.append(f"layer {i}: cache not rebuilt (max_seq_len_cached={rot.max_seq_len_cached})")
            continue
        cos = rot.cos_cached[:seq_len].float().cpu()
        if not t.isfinite(cos).all() or bool((cos == 1).all()):
            problems.append(f"layer {i}: cos cache degenerate")
            continue
        worst = max(worst, float((cos - ref_cos).abs().max()))
        if abs(attn.softmax_scale - expected_scale) > 1e-6 * expected_scale:
            problems.append(f"layer {i}: softmax_scale {attn.softmax_scale} != {expected_scale}")
    if worst > 2e-2:  # bf16 cache vs an fp32 reference
        problems.append(f"cos cache differs from the config.json reference by {worst:.3g}")
    if problems:
        raise RuntimeError("Sarvam-105B RoPE check failed: " + "; ".join(problems[:6]))
    return {"layers_checked": len(hf_model.model.layers), "cos_max_abs_diff": round(worst, 5),
            "softmax_scale": round(float(expected_scale), 6)}


def _rebuild_rotary_embedding(hf_model: nn.Module, modeling: types.ModuleType) -> None:
    """transformers 5 builds a model on the meta device and then loads checkpoint tensors.
    It re-derives non-persistent buffers only for its own classes, so Sarvam's
    ``inv_freq`` (``persistent=False``) comes back as all zeros. Every rotary angle is
    then 0 and the model gets no position information: it reads context as a bag of
    words and loops on greedy decoding. Rebuild the module from the config, after loading."""
    text_model = hf_model.model
    device = text_model.word_embeddings.weight.device
    text_model.rotary_emb = modeling.SarvamMoERotaryEmbedding(config=hf_model.config, device=device)


@functools.lru_cache(maxsize=None)
def _raw_config(name_or_path: str) -> dict | None:
    """``config.json`` of a local dir or a Hub repo, or None when there is none."""
    config_path = Path(name_or_path) / "config.json"
    if not config_path.is_file():
        if os.path.isdir(name_or_path):
            return None
        try:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(name_or_path, "config.json"))
        except Exception:  # not a Hub repo either: let the stock loader report it
            return None
    return json.loads(config_path.read_text(encoding="utf-8"))


def sarvam_model_type(name_or_path: str) -> str | None:
    """``"sarvam_moe"`` (30B), ``"sarvam_mla"`` (105B), or None for anything else."""
    model_type = (_raw_config(name_or_path) or {}).get("model_type")
    return model_type if model_type in SARVAM_MODELS else None


def is_sarvam_checkpoint(name_or_path: str) -> bool:
    """True when ``name_or_path`` (a local dir or a Hub repo) is a Sarvam checkpoint we support."""
    return sarvam_model_type(name_or_path) is not None


def is_sarvam_module(module: nn.Module) -> bool:
    return type(module).__name__.startswith("SarvamMoE")


# How a prompt starts. Sarvam's tokenizer has two "BOS" tokens: its post-processor prepends
# ``<bos>`` (id 2, the Gemma-style tokenizer default) to every encode, while the
# ``bos_token`` setting is ``[@BOS@]`` (id 6), which its chat template writes as
# ``[@BOS@]\n``. A chat request therefore reaches the model as ``<bos>[@BOS@]\n...``.
#   hf          <bos> text            the tokenizer's own default
#   chat        <bos>[@BOS@]\n text   exactly the chat template's prefix
#   sarvam_bos  [@BOS@] text          the configured bos_token alone
# Measured on Sarvam-30B (H200, 2026-10-18; 5-shot word translation, 5 pairs x 32 words):
# sarvam_bos 96.3%, chat 91.2%, hf 60.6% (with hf, greedy Hindi decoding also loops).
BOS_MODES = ("hf", "chat", "sarvam_bos")
DEFAULT_BOS_MODE = "sarvam_bos"
BOS_MODE_ENV = "SARVAM_BOS_MODE"


def _hf_lens_model_class() -> type:
    from jlens.hf import HFLensModel

    class SarvamLensModel(HFLensModel):
        """``HFLensModel`` whose prompts start the way :attr:`bos_mode` says (see
        :data:`BOS_MODES`). The attribute can be switched on a loaded model."""

        bos_mode: str = DEFAULT_BOS_MODE

        def encode(self, text: str, *, max_length: int = 512) -> Tensor:
            if self.bos_mode not in BOS_MODES:
                raise ValueError(f"bos_mode {self.bos_mode!r} not in {BOS_MODES}")
            if self.bos_mode == "sarvam_bos":
                body = self.tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length - 1).input_ids
                ids = [self.tokenizer.bos_token_id, *body]
            else:  # the post-processor prepends <bos>
                prefix = "[@BOS@]\n" if self.bos_mode == "chat" else ""
                ids = self.tokenizer(prefix + text, truncation=True, max_length=max_length).input_ids
            return t.tensor([ids], dtype=t.long, device=self.input_device)

    return SarvamLensModel


def load_sarvam_lens_model(
    name_or_path: str = SARVAM_REPO,
    *,
    revision: str | None = None,
    attn_implementation: str | None = "sdpa",
    device_map: str | None = None,
    dtype: t.dtype = t.bfloat16,
    device: str | None = None,
    code_dir: str | None = None,
    bos_mode: str | None = None,
) -> Any:
    """Load Sarvam-30B or -105B (by the checkpoint's ``model_type``) as a jlens ``LensModel``.
    ``revision`` defaults to the registry's pinned revision for the official repo and to
    ``None`` for anything else. ``bos_mode`` defaults to ``$SARVAM_BOS_MODE``, else
    :data:`DEFAULT_BOS_MODE`. The 105B always loads with eager attention, and its code comes
    from ``name_or_path`` when the dir holds it (``prepare`` copies the pinned revision's files
    there). The returned model carries ``sarvam_model_type`` and ``loading_info``."""
    from jlens.hf import Layout
    from transformers import AutoTokenizer

    model_type = sarvam_model_type(name_or_path) or SARVAM_MODEL_TYPE
    spec = SARVAM_MODELS[model_type]
    if revision is None and name_or_path == spec["repo"]:
        revision = spec["revision"]
    if code_dir is None and model_type == "sarvam_mla" and all((Path(name_or_path) / f).is_file() for f in SARVAM_CODE_FILES):
        code_dir = name_or_path
    modeling = load_sarvam_code(code_dir=code_dir, model_type=model_type)
    if spec["force_attn"] and attn_implementation != spec["attn"]:
        logger.info("%s: attn_implementation %r -> %r (the only one it supports)", model_type, attn_implementation, spec["attn"])
        attn_implementation = spec["attn"]
    if device_map == "auto" and os.environ.get(DEVICE_MAP_ENV):
        device_map = json.loads(os.environ[DEVICE_MAP_ENV])
    hf_model, loading_info = getattr(modeling, spec["cls"]).from_pretrained(
        name_or_path,
        revision=revision,
        dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        output_loading_info=True,
    )
    if device_map is None:
        hf_model = hf_model.to(device or ("cuda" if t.cuda.is_available() else "cpu"))
    if spec["rebuild_rope"]:
        _rebuild_rotary_embedding(hf_model, modeling)
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
    layout = Layout(**spec["layout"]) if spec["layout"] else None  # None: jlens auto-detects
    # force_bos=False: jlens' flag rewrites the tokenizer's post-processor to [@BOS@],
    # silently dropping the <bos> the tokenizer would otherwise add (the first pilot's bug).
    model = _hf_lens_model_class()(hf_model, tokenizer, layout=layout, force_bos=False)
    model.bos_mode = bos_mode or os.environ.get(BOS_MODE_ENV, DEFAULT_BOS_MODE)
    model.sarvam_model_type = model_type
    model.sarvam_modeling = modeling
    model.loading_info = {key: sorted(str(item) for item in value) for key, value in loading_info.items()
                          if isinstance(value, (list, set, tuple))}
    return model


def load_any_lens_model(name_or_path: str, **kwargs: Any) -> Any:
    """Sarvam through :func:`load_sarvam_lens_model`; any other model through
    jpp's ``get_hf_model``."""
    if is_sarvam_checkpoint(name_or_path):
        keep = ("attn_implementation", "device_map", "revision", "dtype", "device", "code_dir", "bos_mode")
        return load_sarvam_lens_model(name_or_path, **{k: v for k, v in kwargs.items() if k in keep})
    from workspace_lens import get_hf_model

    return get_hf_model(name_or_path, **kwargs)


### LRP SURGERY


def _make_sparse_moe_block_lrp_forward(router_detach: bool, shared_expert_scale: float) -> Any:
    """Value-exact replica of ``SarvamMoESparseMoeBlock.forward`` (revision
    :data:`SARVAM_REVISION`) with two backward-only knobs: ``router_detach`` treats
    the top-k mixing weights as constants (expert *selection* is already
    non-differentiable), and ``shared_expert_scale`` multiplies the gradient
    through the shared expert by c via ``s + (c-1)*(s - s.detach())``. The
    correction term is exactly zero in the forward."""

    def forward(self: Any, hidden_states: Tensor) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight, router_logits = self.gate(hidden_states)
        if router_detach:
            topk_weight = topk_weight.detach()
        flat_hidden = hidden_states.view(-1, h)
        flat_topk_idx = topk_idx.view(-1, topk_idx.shape[-1])
        flat_topk_weight = topk_weight.view(-1, topk_weight.shape[-1])
        y = self.experts(flat_hidden, flat_topk_idx, flat_topk_weight)
        y = y.view(bsz, seq_len, h)
        if self.config.num_shared_experts is not None:
            shared = self.shared_experts(identity)
            if shared_expert_scale != 1.0:
                shared = shared + (shared_expert_scale - 1.0) * (shared - shared.detach())
            y = y + shared
        router_info = (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))
        return y, router_info

    return forward


def _make_mlp_lrp_forward(identity_rule: bool, half_rule: bool) -> Any:
    """``SarvamMoEMLP.forward`` (``down(silu(gate(x)) * up(x))``) with the identity and
    half rules, **bit-exact in the forward**. jpp's identity rule computes
    ``z * sigmoid(z).detach()``, which rounds differently from ``F.silu`` in bf16. On an MoE
    that difference flips top-k expert choices and compounds over layers: on Sarvam-30B the
    final residual moved by up to 20 on values up to 135. Here the value is ``silu(z)``
    exactly and only the gradient is the rule's: ``silu(z).detach() + (z - z.detach()) *
    sigmoid(z).detach()`` adds an exact zero in the forward and has d/dz = sigmoid(z)."""

    def forward(self: Any, x: Tensor) -> Tensor:
        z = self.gate_proj(x)
        act = self.act_fn(z)
        if identity_rule:
            act = act.detach() + (z - z.detach()) * t.sigmoid(z).detach()
        product = act * self.up_proj(x)
        if half_rule:
            product = 0.5 * product + 0.5 * product.detach()  # exact: equal halves of an exact scaling
        return self.down_proj(product)

    return forward


def _mlp_role(qualified_name: str) -> str:
    """'routed' (a routed expert), 'shared' (the shared expert) or 'dense' (layer-0 FFN)."""
    parts = qualified_name.split(".")
    if "experts" in parts:
        return "routed"
    if "shared_experts" in parts:
        return "shared"
    return "dense"


@contextlib.contextmanager
def apply_sarvam_lrp_rules(model: nn.Module, lrp_config: Any) -> Iterator[Any]:
    """Reversibly patch Sarvam's module instances for the rules in
    ``lrp_config`` (a jpp ``LrpRuleConfig``). Yields jpp's ``LrpPatchReport``.
    Raises on a compiled or gradient-checkpointed model, an unexpected
    activation, or a requested rule family that patched nothing."""
    from workspace_lens.lrp.lrp import (
        LrpPatchReport,
        _find_compiled_block,
        _qwen2_rmsnorm_lrp_forward,
        patch_instance_forward,
        restore_instance_forwards,
    )

    if getattr(model, "is_gradient_checkpointing", False):
        raise RuntimeError("disable gradient checkpointing before an LRP fit")
    compiled_block = _find_compiled_block(model)
    if compiled_block is not None:
        raise RuntimeError(f"{compiled_block} is torch.compile'd; compiled forwards bypass the patches")

    cfg = lrp_config
    saved_forwards: list[tuple[nn.Module, object]] = []
    report = LrpPatchReport(lrp_config=cfg)
    block_knobs = cfg.router_detach or cfg.shared_expert_scale != 1.0
    try:
        for name, module in model.named_modules():
            cls = type(module).__name__
            if cls == "SarvamMoERMSNorm":
                if not cfg.ln_rule or name.rsplit(".", 1)[-1] in SARVAM_QK_NORM_NAMES:
                    continue
                # Sarvam's RMSNorm forward is Qwen2's line for line (fp32 rsqrt,
                # weight applied after the downcast, eps named variance_epsilon).
                patch_instance_forward(saved_forwards, module, _qwen2_rmsnorm_lrp_forward)
                report.ln_rule.append(name)
            elif cls == "SarvamMoESparseMoeBlock":
                if not (block_knobs or cfg.shared_gate_detach):
                    continue
                patch_instance_forward(
                    saved_forwards,
                    module,
                    _make_sparse_moe_block_lrp_forward(cfg.router_detach, cfg.shared_expert_scale),
                )
                report.moe_gates.append(name)
            elif cls == "SarvamMoEMLP":
                role = _mlp_role(name)
                if role == "routed" and not cfg.routed_experts:
                    continue
                if not (cfg.identity_rule or cfg.half_rule):
                    continue
                act_name = type(module.act_fn).__name__
                if cfg.identity_rule and act_name not in ("SiLU", "SiLUActivation"):
                    raise RuntimeError(f"identity rule expects SiLU at {name!r}, found {act_name}")
                patch_instance_forward(
                    saved_forwards,
                    module,
                    _make_mlp_lrp_forward(cfg.identity_rule, cfg.half_rule),
                )
                (report.moe_experts if role == "routed" else report.mlp_rule).append(name)

        requested_and_unpatched = [
            family
            for family, requested, patched in (
                ("ln_rule", cfg.ln_rule, report.ln_rule),
                ("identity_rule/half_rule", cfg.identity_rule or cfg.half_rule, report.mlp_rule),
                ("routed_experts", cfg.routed_experts, report.moe_experts),
                ("router_detach/shared_expert_scale", block_knobs, report.moe_gates),
                ("mhc_detach", cfg.mhc_detach, []),
                ("shared_expert_forward_scale", cfg.shared_expert_forward_scale != 1.0, []),
            )
            if requested and not patched
        ]
        if requested_and_unpatched:
            raise RuntimeError(
                f"LRP rule families {requested_and_unpatched} matched no Sarvam module; "
                "the fit would silently use standard gradients for them"
            )
        yield report
    finally:
        restore_instance_forwards(saved_forwards)


### HOOKING INTO JPP


def install_jpp_hooks() -> None:
    """Route jpp's LRP surgery to :func:`apply_sarvam_lrp_rules` when the model is
    Sarvam (every other model still gets the stock surgery). Idempotent."""
    import workspace_lens.fitting.relp_fitting as relp_fitting
    from workspace_lens.lrp.lrp import hf_module_to_patch, lrp_rule_config_for_mode

    if getattr(relp_fitting, "_sarvam_hooks_installed", False):
        return
    stock_apply = relp_fitting.apply_lrp_rules_for_mode

    @contextlib.contextmanager
    def apply_lrp_rules_for_mode(model: object, lrp_mode: str) -> Iterator[Any]:
        hf_module = hf_module_to_patch(model)
        if not is_sarvam_module(hf_module):
            with stock_apply(model, lrp_mode) as report:
                yield report
            return
        rule_config = lrp_rule_config_for_mode(lrp_mode)
        if not rule_config.requests_any_rule():
            yield None
            return
        with apply_sarvam_lrp_rules(hf_module, rule_config) as report:
            yield report

    relp_fitting.apply_lrp_rules_for_mode = apply_lrp_rules_for_mode
    relp_fitting._sarvam_hooks_installed = True
