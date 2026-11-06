"""End-to-end check of the Sarvam-105B path on a tiny random ``SarvamMLAForCausalLM`` (the real
code, config shape and tokenizer of sarvamai/sarvam-105b; 4 layers, d_model 64, 8 experts):

    JPP_DIR=/path/to/jpp_lens python smoke_test_mla.py [--workdir DIR] [--two-gpu]

Checks:
 1. the 105B and 30B code import side by side (separate packages, transformers-5 shims);
 2. the adapter: model-type detection, forced eager attention, no missing/unexpected weights,
    jlens layout auto-detection, BOS modes, an exact save/load round trip;
 3. the RoPE runtime check passes after a forward, and fails on a corrupted cache;
 4. logit-lens ablation directions are the gain-scaled unembedding rows; batched forwards (the
    right-padded ``Reader.forward_batch``, ``RowAblation`` rows) match lone ones;
 5. the 105B pilot stages (load, sanity, bos_probe, pivot_logit, causal_logit), the reader-question
    stages, and the expert-routing / layer-0 neuron stages (``neuron_experiment``, batched) through
    ``modal_pilot.run_pilot``.
``--two-gpu`` loads in bf16 split over cuda:0 / cuda:1 (embeddings and the first half of the
layers on GPU 0; the rest, the final norm and the LM head on GPU 1, as device_map="auto" places
the real model) and runs stage 5 there. Results on a random model are meaningless; this only
exercises the code paths.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch as t

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_jpp  # noqa: E402
import sarvam_support  # noqa: E402

SPEC = sarvam_support.SARVAM_MODELS["sarvam_mla"]
HUB_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
N_LAYERS = 4
STAGES = ["load", "sanity", "bos_probe", "pivot_logit", "causal_logit"]
# reader-question stages (docs/09): chat formats, collapse errors, clozes, Chinese skill, steering, thinking;
# then expert routing by language and layer-0 neurons (neuron_experiment.py)
RQ_STAGES = ["rq_chat_word_src", "rq_chat_sentence", "rq_chat_word_en", "rq_collapse", "rq_cloze", "rq_zh",
             "causal_steer", "rq_think", "routing", "neurons"]


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message, flush=True)
    if not condition:
        raise SystemExit(f"smoke test failed: {message}")


def hub_file(name: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(SPEC["repo"], name, revision=SPEC["revision"]))


def build_tiny_mla(workdir: Path) -> tuple[Path, t.nn.Module, t.Tensor, t.Tensor]:
    """Save a tiny random 105B-shaped checkpoint laid out like ``prepare``'s output (weights,
    the original-style config.json, tokenizer, and the two code files). Returns the dir, the
    in-memory model, and reference input ids and logits for the round-trip check."""
    modeling = sarvam_support.load_sarvam_code(model_type="sarvam_mla")
    config_module = sys.modules[f"{SPEC['package']}.configuration_sarvam_moe"]
    real = json.loads(hub_file("config.json").read_text(encoding="utf-8"))
    drop = {"architectures", "auto_map", "transformers_version", "dtype", "attn_implementation"}
    tiny = {k: v for k, v in real.items() if k not in drop}
    tiny.update(hidden_size=64, intermediate_size=128, num_hidden_layers=N_LAYERS, num_attention_heads=4,
                qk_nope_head_dim=16, qk_rope_head_dim=8, q_head_dim=24, v_head_dim=16, kv_lora_rank=32, head_dim=40,
                num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, max_position_embeddings=512,
                initializer_range=0.08)
    t.manual_seed(0)
    model = getattr(modeling, SPEC["cls"])(getattr(config_module, SPEC["config_cls"])(**tiny))
    for layer in model.model.layers[tiny["first_k_dense_replace"]:]:  # a nonzero router bias, as trained
        layer.mlp.gate.e_score_correction_bias.normal_(0, 0.1)
    model.eval()
    out = workdir / "tiny-sarvam105"
    model.save_pretrained(out)
    # The real checkpoint's config.json style (rope_scaling / rope_theta at the top level), which is
    # what prepare copies and what verify_mla_rope reads as its independent reference.
    # (Keeping transformers_version also keeps transformers 5 from flagging the tokenizer as Mistral's.)
    (out / "config.json").write_text(json.dumps({"architectures": [SPEC["cls"]], **tiny, "dtype": "float32",
                                                 "transformers_version": real["transformers_version"]}, indent=2),
                                     encoding="utf-8")
    for name in (*HUB_FILES[1:], *sarvam_support.SARVAM_CODE_FILES):
        shutil.copy(hub_file(name), out / name)
    ids = t.randint(10, 1000, (1, 24), generator=t.Generator().manual_seed(1))
    with t.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    return out, model, ids, logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--two-gpu", action="store_true")
    args = parser.parse_args()
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="sarvam_mla_smoke_"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"workdir: {workdir}")
    jpp_cli = run_jpp._import_jpp_cli()
    run_jpp.install(jpp_cli)

    print("1. imports")
    mla = sarvam_support.load_sarvam_code(model_type="sarvam_mla")
    moe = sarvam_support.load_sarvam_code(model_type="sarvam_moe")
    check(mla is not moe and hasattr(mla, "SarvamMLAForCausalLM") and hasattr(moe, "SarvamMoEForCausalLM"),
          f"105B and 30B code side by side ({mla.__name__}, {moe.__name__})")
    check(mla.SarvamMLAForCausalLM._tied_weights_keys == {"lm_head.weight": "model.embed_tokens.weight"},
          "tied-weights shim maps lm_head to embed_tokens")

    print("2. adapter")
    tiny_dir, reference_model, ref_ids, ref_logits = build_tiny_mla(workdir)
    check(sarvam_support.sarvam_model_type(str(tiny_dir)) == "sarvam_mla" and sarvam_support.is_sarvam_checkpoint(str(tiny_dir)),
          "checkpoint detected as sarvam_mla")
    model = sarvam_support.load_sarvam_lens_model(str(tiny_dir), dtype=t.float32, device="cpu", attn_implementation="sdpa")
    hf = model._hf_model
    check(hf.config._attn_implementation == "eager", f"sdpa request forced to eager ({hf.config._attn_implementation})")
    bad = {k: v for k, v in model.loading_info.items() if k in ("missing_keys", "unexpected_keys", "mismatched_keys") and v}
    check(not bad, f"no missing/unexpected/mismatched weights ({bad or model.loading_info.keys()})")
    check((model.n_layers, model.d_model) == (N_LAYERS, 64) and model._embed_tokens is hf.model.embed_tokens
          and model._final_norm is hf.model.norm, f"jlens layout auto-detected: {model!r}")
    tok = model.tokenizer
    starts = {}
    for mode in sarvam_support.BOS_MODES:
        model.bos_mode = mode
        starts[mode] = tok.convert_ids_to_tokens(model.encode("पानी", max_length=8)[0].tolist()[:3])
    model.bos_mode = sarvam_support.DEFAULT_BOS_MODE
    check(starts["hf"][0] == "<bos>" and starts["chat"][:2] == ["<bos>", "[@BOS@]"] and starts["sarvam_bos"][0] == "[@BOS@]",
          f"BOS prefixes per mode (same tokenizer as the 30B): {starts}")
    with t.no_grad():
        logits = hf(input_ids=ref_ids, use_cache=False).logits
    check(t.allclose(logits, ref_logits, atol=1e-4), "save/load round trip: logits match the in-memory model")

    print("3. RoPE runtime check")
    raw = json.loads((tiny_dir / "config.json").read_text(encoding="utf-8"))
    rope = sarvam_support.verify_mla_rope(hf, raw, model.sarvam_modeling, seq_len=ref_ids.shape[1])
    check(rope["layers_checked"] == N_LAYERS and rope["cos_max_abs_diff"] < 1e-5, f"RoPE caches match config.json: {rope}")
    rot = hf.model.layers[1].self_attn.rotary_emb
    saved = rot.cos_cached.clone()
    rot.cos_cached.fill_(1.0)
    try:
        sarvam_support.verify_mla_rope(hf, raw, model.sarvam_modeling, seq_len=ref_ids.shape[1])
        caught = False
    except RuntimeError as exc:
        caught = "layer 1" in str(exc)
    rot.cos_cached.copy_(saved)
    check(caught, "RoPE check fails loudly on a corrupted cache")
    wrong_scale = dict(raw, rope_scaling=dict(raw["rope_scaling"], mscale_all_dim=0.0))
    try:
        sarvam_support.verify_mla_rope(hf, wrong_scale, model.sarvam_modeling, seq_len=ref_ids.shape[1])
        caught = False
    except RuntimeError as exc:
        caught = "softmax_scale" in str(exc)
    check(caught, "RoPE check catches a wrong attention temperature (YaRN mscale)")

    print("4. logit-lens ablation directions")
    import causal_test

    lens = jpp_cli.load_lenses([jpp_cli.LOGIT_LENS_SPEC], model, hf_model_name=str(tiny_dir),
                               layers=list(range(N_LAYERS - 1)))[jpp_cli.LOGIT_LENS_SPEC]
    W = (model._lm_head.weight.float() * causal_test.final_norm_gain(model)).detach()
    directions = causal_test.readout_directions(lens, 1, [5, 9], W)
    check(t.equal(directions, W[[5, 9]]), "logit-lens directions are g * W_U rows")
    Q = causal_test.subspace(lens, 1, [5, 9, 11], W)
    check(Q.shape == (64, 3) and t.allclose(Q.T @ Q, t.eye(3), atol=1e-5), "subspace basis is orthonormal (d x k)")
    import reader_questions

    reader = reader_questions.Reader(model, {"logit": lens}, list(range(N_LAYERS - 1)), jpp_cli.vocab_size_of(model), None)
    prompts = [model.encode(p) for p in ("पानी", 'Tamil: "நீர்" - Telugu: "', "one two three four five six seven")]
    pairs = list(zip((reader.forward(ids) for ids in prompts), reader.forward_batch(prompts)))
    acts_diff = max((a[l] - b[l]).abs().max().item() for (a, _), (b, _) in pairs for l in a)
    logits_diff = max((x - y).abs().max().item() for (_, x), (_, y) in pairs)
    check(acts_diff < 1e-4 and logits_diff < 1e-4 and len({int(p.shape[1]) for p in prompts}) == 3,
          f"Reader.forward_batch (right-padded, lengths {[int(p.shape[1]) for p in prompts]}) matches lone forwards: "
          f"acts {acts_diff:.1e}, logits {logits_diff:.1e}")
    ids, plans = prompts[1], [None, ({1: Q, 2: Q}, 1.0), ({2: Q}, 3.0)]
    with t.no_grad():
        with causal_test.RowAblation(model, plans):
            batched = hf(input_ids=ids.repeat(len(plans), 1), use_cache=False).logits[:, -1]
        lone = [hf(input_ids=ids, use_cache=False).logits[0, -1]]
        for bases, alpha in plans[1:]:
            with causal_test.Ablation(model, bases, alpha=alpha):
                lone.append(hf(input_ids=ids, use_cache=False).logits[0, -1])
    row_diff = max((batched[i] - lone[i]).abs().max().item() for i in range(len(plans)))
    effect = min((lone[i] - lone[0]).abs().max().item() for i in (1, 2))
    check(row_diff < 1e-4 and effect > 1e-3, f"RowAblation rows match one Ablation forward each: {row_diff:.1e} "
          f"(the ablations move the logits by >= {effect:.1e})")

    print(f"5. pilot stages ({'2 GPUs, bf16' if args.two_gpu else 'CPU, fp32'})")
    import modal_pilot

    del model, hf
    if args.two_gpu:
        check(t.cuda.device_count() >= 2, f"{t.cuda.device_count()} GPUs visible")
        split = {"model.embed_tokens": 0, **{f"model.layers.{i}": int(i >= N_LAYERS // 2) for i in range(N_LAYERS)},
                 "model.norm": 1, "lm_head": 1}
        os.environ[sarvam_support.DEVICE_MAP_ENV] = json.dumps(split)
        device_map = "auto"
    else:
        device_map = None
    os.environ.pop(sarvam_support.BOS_MODE_ENV, None)
    report = modal_pilot.run_pilot(str(tiny_dir), str(workdir / "runs"), budget_minutes=30, commit=lambda: None,
                                   device_map=device_map, stages=STAGES + RQ_STAGES, tag="mla_smoke", smoke=True,
                                   bos_modes=["sarvam_bos", "chat"], seconds_per_forward_guess=0.05)
    by_stage = {s["stage"]: s for s in report["stages"]}
    for name in STAGES + RQ_STAGES:
        entry = by_stage.get(name, {})
        check(entry.get("status") == "ok", f"stage {name}: {entry.get('status')}"
              + ("" if entry.get("status") == "ok" else f"\n{entry.get('traceback', '')[-2500:]}"))
    load = by_stage["load"]["result"]
    if args.two_gpu:
        check(set(load.get("modules_per_device", {})) == {"0", "1"}, f"model split over 2 GPUs: {load.get('modules_per_device')}")
    sanity = by_stage["sanity"]["result"]
    check("rope" in sanity and "error" not in sanity["rope"] and "rlens" not in sanity,
          f"sanity: RoPE checked, LRP skipped, {sanity['seconds_per_forward']} s/forward")
    bos = by_stage["bos_probe"]["result"]
    ran = {k for k, v in bos.items() if isinstance(v, dict)}
    check(ran == {"sarvam_bos", "chat"}, f"bos_probe ran only the requested modes ({sorted(ran)})")
    check("stop_rule" in bos, f"a random model trips the BOS stop rule ({bos.get('stop_rule')}); ignored in smoke mode")
    pivot = by_stage["pivot_logit"]["result"]
    check(pivot["exit"] == 0 and pivot["all_indic_mean_prob_by_layer"], f"pivot_logit: kept {pivot['items_kept']}")
    causal = by_stage["causal_logit"]["result"]
    finite = all(isinstance(causal[c]["mean_drop"], float) and causal[c]["mean_drop"] == causal[c]["mean_drop"]
                 for c in ("english_mid", "control_mid", "english_early"))
    check(causal["n_items"] > 0 and finite and causal["mid_layers"] and causal["early_layers"],
          f"causal_logit: {causal['n_items']} items, mid {causal['mid_layers']}, early {causal['early_layers']} "
          f"({causal['band']['source']})")
    for name in ("rq_chat_word_src", "rq_chat_sentence", "rq_chat_word_en", "rq_cloze"):
        res = by_stage[name]["result"]
        check(res["kept"] > 0 and res["all_indic"], f"{name}: {res['kept']}/{res['n_items']} items, steps {res['found_at_step']}")
    collapse = by_stage["rq_collapse"]["result"]
    check(all(collapse[t_]["n_items"] > 0 for t_ in ("homonyms", "kinship")),
          f"rq_collapse: homonyms {collapse['homonyms']['outcome_rate']}, kinship n {collapse['kinship']['n_items']}")
    check(by_stage["rq_zh"]["result"]["accuracy_by_pair"], f"rq_zh: {by_stage['rq_zh']['result']['accuracy_by_pair']}")
    steer = by_stage["causal_steer"]["result"]
    check(all(k in steer for k in ("english_mid@2", "control_mid@3")), f"causal_steer: {steer['n_items']} items, alphas 1/2/3")
    think = by_stage["rq_think"]["result"]
    check(think["n_items"] > 0, f"rq_think: {think['n_items']} item(s), {think['mean_tokens']} tokens (KV-cached)")
    routing = by_stage["routing"]["result"]["routing"]
    consistency = routing["batch_consistency"] or {}
    check(routing["moe_layers"] == list(range(1, N_LAYERS)) and routing["n_words"] == 80 and routing["n_prompts"] > 0
          and routing["words"]["band"].get("lang_acc") is not None and consistency.get("mean_jaccard", 0) >= 0.9,
          f"routing: {routing['n_words']} words, {routing['n_prompts']} prompts, top-{routing['top_k']} of "
          f"{routing['n_experts']} at layers {routing['moe_layers']}; batched vs lone expert sets: {consistency}")
    neurons = by_stage["neurons"]["result"]["neurons"]
    check(neurons["d_ff"] == 128 and neurons["n_words"] == 80 and neurons["concept_selective"]["n_permutations"] > 0,
          f"neurons: layer-0 d_ff {neurons['d_ff']}, concept-selective {neurons['concept_selective']['n']} "
          f"(null max {neurons['concept_selective']['null_max']}), script-selective {neurons['script_selective']['n']}")
    check(report.get("usd_per_hour") and "stopped" not in report, "report has a price and no stop rule fired (smoke)")
    print(f"\nALL CHECKS PASSED  (artifacts in {workdir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
