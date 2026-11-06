"""End-to-end check on a tiny random-weight Sarvam (same code and tokenizer as
sarvam-30b, 4 layers, d_model 64, 8 experts). It runs on CPU in a few minutes. Run it
before spending GPU time:

    JPP_DIR=/path/to/jpp_lens python smoke_test.py [--workdir DIR]

Checks:
 1. loading: the pinned Sarvam code, the word_embeddings layout, BOS prepended;
 2. LRP surgery: the forward is unchanged under every rule set (bit-exact for the
    LN-rule); the right modules are patched, with attention q/k norms left alone;
    gradients change;
 3. stock autograd works through Sarvam's eval-mode expert dispatch (its in-place mul_);
 4. the full jpp pipeline through run_jpp: fit-router, fit-shard (J and R+MoE), merge,
    prepare-evals, grade, fit-weights (J++), evaluate;
 5. pivot_experiment on the resulting lenses, writing CSVs and plots;
 6. activation patching: the hooks' exact invariants, then causal_test.run_patch end to end.
Results on a random model are meaningless; this only exercises the code paths.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
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

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
FIT_TEXTS = [
    "पानी जीवन का आधार है। नदियाँ, झीलें और समुद्र पृथ्वी के बड़े हिस्से को ढकते हैं, और हर जीव को जीने के लिए पानी चाहिए। " * 2,
    "தமிழ் உலகின் மிகப் பழமையான மொழிகளில் ஒன்று. சங்க இலக்கியம் இரண்டாயிரம் ஆண்டுகளுக்கு முன்பே எழுதப்பட்டது. " * 2,
    "The river flows past the old village, where farmers grow rice and wheat along its banks every season of the year. " * 2,
    "తెలుగు భాష దక్షిణ భారతదేశంలో విస్తృతంగా మాట్లాడబడుతుంది. ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాల అధికార భాష. " * 2,
    "বাংলা ভাষা পৃথিবীর অন্যতম প্রধান ভাষা। কলকাতা ও ঢাকা এই ভাষার সাহিত্যের দুই প্রধান কেন্দ্র। " * 2,
    "ಕನ್ನಡ ಕರ್ನಾಟಕದ ಅಧಿಕೃತ ಭಾಷೆ. ಈ ಭಾಷೆಗೆ ಸಾವಿರಾರು ವರ್ಷಗಳ ಸಾಹಿತ್ಯ ಪರಂಪರೆ ಇದೆ ಮತ್ತು ಅನೇಕ ಕವಿಗಳು ಇದನ್ನು ಬೆಳೆಸಿದ್ದಾರೆ. " * 2,
]


def check(condition: bool, message: str) -> None:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        raise SystemExit(f"smoke test failed: {message}")


def build_tiny_sarvam(workdir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    modeling = sarvam_support.load_sarvam_code()
    real = json.loads(Path(hf_hub_download(sarvam_support.SARVAM_REPO, "config.json", revision=sarvam_support.SARVAM_REVISION)).read_text())
    drop = {"architectures", "auto_map", "transformers_version", "dtype", "attn_implementation"}
    tiny = {k: v for k, v in real.items() if k not in drop}
    tiny.update(
        hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, max_position_embeddings=512, max_window_layers=4,
        initializer_range=0.08,
    )
    config_cls = sys.modules["sarvam_moe_hub_code.configuration_sarvam_moe"].SarvamMoEConfig
    t.manual_seed(0)
    model = modeling.SarvamMoEForCausalLM(config_cls(**tiny))
    out = workdir / "tiny-sarvam"
    model.save_pretrained(out)
    for name in TOKENIZER_FILES:
        shutil.copy(hf_hub_download(sarvam_support.SARVAM_REPO, name, revision=sarvam_support.SARVAM_REVISION), out / name)
    return out


def last_hidden(model, ids):
    with t.no_grad():
        return model.forward(ids).last_hidden_state


def vector_jacobian(model, ids, cotangent, recorder_cls) -> list[t.Tensor]:
    """``cotangent^T dh_final / dh_l`` at layers 0..2: the backward pass a Jacobian fit runs."""
    sources = [0, 1, 2]
    final = model.n_layers - 1
    with recorder_cls(model.layers, at=[*sources, final], start_graph_at=0) as recorder, t.enable_grad():
        model.forward(ids)
        target = recorder.activations[final]
        return list(t.autograd.grad(target, [recorder.activations[l] for l in sources], grad_outputs=cotangent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()
    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="sarvam_smoke_"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"workdir: {workdir}")

    jpp_cli = run_jpp._import_jpp_cli()
    run_jpp.install(jpp_cli)
    from jlens.hooks import ActivationRecorder
    from workspace_lens.lrp.lrp import LrpRuleConfig, lrp_rule_config_for_mode

    print("1. loading")
    tiny_dir = build_tiny_sarvam(workdir)
    check(sarvam_support.is_sarvam_checkpoint(str(tiny_dir)), "checkpoint detected as sarvam_moe")
    model = sarvam_support.load_sarvam_lens_model(str(tiny_dir), dtype=t.float32, device="cpu")
    check((model.n_layers, model.d_model) == (4, 64), f"layout resolved: {model!r}")
    # transformers 5 leaves the non-persistent RoPE buffer zeroed (no positions at all); the
    # loader rebuilds it. Random weights can't reveal a positionless model, so check directly.
    config = model._hf_model.config
    expected_inv_freq = 1.0 / (config.rope_theta ** (t.arange(0, config.head_dim, 2, dtype=t.float32) / config.head_dim))
    inv_freq = model._hf_model.model.rotary_emb.inv_freq.float().cpu()
    check(t.allclose(inv_freq, expected_inv_freq, rtol=1e-5), f"RoPE inv_freq rebuilt (first {inv_freq[:3].tolist()})")
    ids = model.encode("पानी - தண்ணீர் - water, " * 6, max_length=64)
    tok = model.tokenizer
    starts = {}
    for mode in sarvam_support.BOS_MODES:
        model.bos_mode = mode
        starts[mode] = tok.convert_ids_to_tokens(model.encode("पानी", max_length=8)[0].tolist()[:3])
    model.bos_mode = sarvam_support.DEFAULT_BOS_MODE
    check(starts["hf"][0] == "<bos>" and starts["chat"][:2] == ["<bos>", "[@BOS@]"] and starts["sarvam_bos"][0] == "[@BOS@]",
          f"BOS prefixes per mode: {starts}")
    check(ids.shape[1] <= 64, "encode respects max_length")

    print("2. LRP surgery")
    stock = last_hidden(model, ids)
    with sarvam_support.apply_sarvam_lrp_rules(model._hf_model, LrpRuleConfig(ln_rule=True)):
        check(t.equal(last_hidden(model, ids), stock), "LN-rule forward is bit-identical")
    for mode, n_mlp, n_routed, n_blocks in (("rlens", 4, 0, 0), ("r+moe", 4, 3 * 8, 3)):
        with sarvam_support.apply_sarvam_lrp_rules(model._hf_model, lrp_rule_config_for_mode(mode)) as report:
            patched = last_hidden(model, ids)
        check(t.equal(patched, stock), f"{mode}: fp32 forward bit-identical")
        # 2 residual norms per layer plus the final norm (as jpp does; it sits past the fit's
        # target block, so it never enters the backward pass); attention q/k norms untouched.
        check(len(report.ln_rule) == 9 and "model.norm" in report.ln_rule
              and not any("query_layernorm" in n or "key_layernorm" in n for n in report.ln_rule),
              f"{mode}: LN-rule on the 8 residual norms + final norm, q/k norms untouched")
        check((len(report.mlp_rule), len(report.moe_experts), len(report.moe_gates)) == (n_mlp, n_routed, n_blocks),
              f"{mode}: dense+shared MLPs {len(report.mlp_rule)}, routed experts {len(report.moe_experts)}, MoE blocks {len(report.moe_gates)}")
    check(not any("forward" in m.__dict__ for m in model._hf_model.modules()), "every instance forward restored on exit")

    # bf16 is where jpp's z*sigmoid(z) identity rule drifts (and flips MoE routing on the real
    # model); ours must be bit-exact there too.
    model_bf16 = sarvam_support.load_sarvam_lens_model(str(tiny_dir), dtype=t.bfloat16, device="cpu")
    stock_bf16 = last_hidden(model_bf16, ids)
    for mode in ("rlens", "r+moe"):
        with sarvam_support.apply_sarvam_lrp_rules(model_bf16._hf_model, lrp_rule_config_for_mode(mode)):
            check(t.equal(last_hidden(model_bf16, ids), stock_bf16), f"{mode}: bf16 forward bit-identical")
    # Same gradient as jpp's identity+half rule (which is exact in fp32): d/dz = sigmoid(z), halves.
    from workspace_lens.lrp.lrp import _make_gated_mlp_lrp_forward, _silu_with_detached_sigmoid

    mlp = model._hf_model.model.layers[0].mlp
    x = t.randn(5, model.d_model, generator=t.Generator().manual_seed(1), requires_grad=True)
    grads = []
    for make in (lambda: sarvam_support._make_mlp_lrp_forward(True, True),
                 lambda: _make_gated_mlp_lrp_forward(True, True, _silu_with_detached_sigmoid)):
        (grad,) = t.autograd.grad(make()(mlp, x).sum(), x)
        grads.append(grad)
    check(t.allclose(grads[0], grads[1], atol=1e-6), "exact identity rule has jpp's gradient")

    print("3. gradients through the eval-mode expert dispatch")
    fit_ids = model.encode(FIT_TEXTS[0], max_length=48)
    cotangent = t.randn(1, fit_ids.shape[1], model.d_model, generator=t.Generator().manual_seed(0))
    grads_std = vector_jacobian(model, fit_ids, cotangent, ActivationRecorder)
    with sarvam_support.apply_sarvam_lrp_rules(model._hf_model, lrp_rule_config_for_mode("r+moe")):
        grads_rel = vector_jacobian(model, fit_ids, cotangent, ActivationRecorder)
    check(all(t.isfinite(g).all() and g.abs().sum() > 0 for g in grads_std), "standard gradients finite and nonzero")
    rel_diff = max(float((r - s).norm() / s.norm()) for r, s in zip(grads_rel, grads_std))
    check(rel_diff > 1e-4, f"R+MoE gradients differ from standard (max relative diff {rel_diff:.3f})")

    print("4. jpp pipeline through run_jpp")
    os.chdir(run_jpp.JPP_DIR)  # jpp reads data/ relative to its root
    prompts_json = workdir / "prompts.json"
    prompts_json.write_text(json.dumps(FIT_TEXTS, ensure_ascii=False), encoding="utf-8")
    os.environ[run_jpp.FIT_PROMPTS_ENV] = str(prompts_json)
    name = str(tiny_dir)
    common = ["--hf-model-name", name, "--layers", "0,1,2"]
    router = workdir / "router.pt"
    check(run_jpp.main(["fit-router", *common, "--num-prompts", "6", "--num-clusters", "2", "--projection-dim", "4",
                        "--max-seq-len", "48", "--skip-first-n-positions", "4", "--out", str(router)]) == 0, "fit-router")
    lenses = {}
    for mode, label in (("none", "j"), ("r+moe", "r")):
        art = workdir / f"artifacts_{label}"
        check(run_jpp.main(["fit-shard", *common, "--router-path", str(router), "--num-prompts", "6", "--prompts-json", str(prompts_json),
                            "--lrp-mode", mode, "--max-seq-len", "48", "--skip-first-n-positions", "4",
                            "--artifacts-base-dir", str(art), "--checkpoint-name", f"tiny_{label}"]) == 0, f"fit-shard --lrp-mode {mode}")
        shard_dirs = sorted({str(Path(p).parent) for p in glob.glob(str(art / "**" / "*.pt"), recursive=True)})
        check(len(shard_dirs) >= 1, f"shard checkpoint written ({shard_dirs})")
        experts, pooled = workdir / f"experts_{label}.pt", workdir / f"tiny_{label}_lens.pt"
        check(run_jpp.main(["merge-experts", "--checkpoint-dirs", *shard_dirs, "--num-clusters", "2", "--min-kept-positions", "1",
                            "--checkpoint-name", f"tiny_{label}", "--device", "cpu", "--out", str(experts),
                            "--pooled-lens-out", str(pooled)]) == 0, f"merge-experts ({label})")
        lenses[label] = (experts, pooled)

    evals_dir = workdir / "evals_no_multilingual"
    check(run_jpp.main(["prepare-evals", "--out", str(evals_dir)]) == 0, "prepare-evals")
    check(json.loads((evals_dir / "lens-eval-multilingual.json").read_text(encoding="utf-8"))["items"] == [], "multilingual items emptied")
    correctness = workdir / "correctness.csv"
    check(run_jpp.main(["grade", "--hf-model-name", name, "--out", str(correctness)]) == 0, "grade")
    jpp_lens = workdir / "tiny_jpp.pt"
    check(run_jpp.main(["fit-weights", *common, "--experts", str(lenses["r"][0]), "--checkpoint-name", "tiny_jpp",
                        "--items", "fit:0", "--eval-data-dir", str(evals_dir), "--correctness-csv", str(correctness),
                        "--steps", "3", "--top-n", "2", "--out", str(jpp_lens)]) == 0, "fit-weights (J++ from the R+MoE experts)")
    lens_args = ["--lens", "logit", "--lens", str(lenses["j"][1]), "--lens", str(lenses["r"][1]), "--lens", str(jpp_lens)]
    leaky = ["evaluate", *common, *lens_args, "--items", "held-out:0", "--correctness-csv", str(correctness),
             "--out", str(workdir / "eval_leaky" / "ranks.csv")]  # default eval dir != the weights' dir
    try:
        run_jpp.main(leaky)
        refused = False
    except SystemExit as exc:
        refused = "refusing to evaluate" in str(exc)
    check(refused, "evaluate refuses held-out items from a different list than fit-weights used")
    for wrong in (["--items", "held-out:1"], ["--items", "held-out:0", "--fit-fraction", "0.1"]):  # same dir, another split
        try:
            run_jpp.main(["evaluate", *common, *lens_args, *wrong, "--eval-data-dir", str(evals_dir),
                          "--correctness-csv", str(correctness), "--out", str(workdir / "eval_leaky2" / "ranks.csv")])
            refused = False
        except SystemExit as exc:
            refused = "refusing to evaluate" in str(exc)
        check(refused, f"evaluate refuses {' '.join(wrong)} on the weights' own dir (fitted on fit:0, fraction 0.2)")
    # the item-level guard alone (what evaluate runs once it has its scored items): a fitting item is refused
    stamp = json.loads((workdir / "tiny_jpp_stamp.json").read_text(encoding="utf-8"))
    fitted_item = type("Item", (), {"name": stamp["fitting_item_names"][0]})()
    try:
        run_jpp.refuse_in_sample_items(run_jpp._import_jpp_cli(), [str(jpp_lens)], [fitted_item])
        refused = False
    except SystemExit as exc:
        refused = "were used to fit" in str(exc)
    check(refused, "the item-level guard refuses a scored item named in fitting_item_names")
    check(run_jpp.main(["evaluate", *common, *lens_args, "--items", "held-out:0", "--eval-data-dir", str(evals_dir),
                        "--correctness-csv", str(correctness), "--out", str(workdir / "eval" / "ranks.csv")]) == 0,
          "evaluate on the held-out split of the weights' own item list")
    multilingual_dir = workdir / "evals_multilingual_only"
    run_jpp.main(["prepare-evals", "--out", str(multilingual_dir), "--exclude", "association,multihop,order-ops,poetry,typo"])
    check(run_jpp.main(["evaluate", *common, *lens_args, "--items", "all", "--eval-data-dir", str(multilingual_dir),
                        "--correctness-csv", str(correctness), "--out", str(workdir / "eval_multilingual" / "ranks.csv")]) == 0,
          "evaluate all multilingual items (never seen by the weights)")

    print("5. pivot experiment")
    import pivot_experiment

    results = workdir / "pivot"
    check(pivot_experiment.main(["--hf-model-name", name, "--lens", "logit", "--lens", str(lenses["j"][1]),
                                 "--lens", str(lenses["r"][1]), "--lens", str(jpp_lens), "--pairs", "ta-te,mr-bn,hi_latn-ta",
                                 "--no-correctness-filter", "--max-items", "4", "--out", str(results)]) == 0, "pivot_experiment")
    with (results / "rows.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    check(len(rows) > 0 and {r["lens"] for r in rows} == {"logit", "tiny_j_lens", "tiny_r_lens", "tiny_jpp"}, f"{len(rows)} rows over 4 lenses")
    pngs = sorted(p.name for p in results.glob("*.png"))
    check(len(pngs) == 8, f"plots written: {pngs}")

    print("6. activation patching")
    import causal_test

    final = model.n_layers - 1
    concepts = pivot_experiment.load_concepts(HERE / "data/concepts.tsv")
    prompts = {c: model.encode(pivot_experiment.build_prompt(concepts, c, "ta", "te", shots=5, seed=0)) for c in ("water", "fire")}

    def final_logits(ids, sources=None, bases=None):
        with contextlib.ExitStack() as stack:
            if sources is not None:
                stack.enter_context(causal_test.Patch(model, sources, bases))
            with t.no_grad():
                return model._hf_model(input_ids=ids, use_cache=False).logits[0, -1].float()

    clean = {}
    for c, ids in prompts.items():
        with causal_test.Capture(model, [1, 2, final]) as capture:
            clean[c] = (final_logits(ids), capture.residuals)
    A, (logits_A, res_A), (logits_B, res_B) = prompts["water"], clean["water"], clean["fire"]
    check(t.allclose(final_logits(A, {final: res_B[final]}, {final: None}), logits_B, atol=1e-5),
          "whole-state patch after the last block reproduces B's logits")
    W = (model._lm_head.weight.float() * causal_test.final_norm_gain(model)).detach()
    j_lens = jpp_cli.load_lens_file(str(lenses["j"][1]), hf_model_name=name)
    Q = causal_test.subspace(j_lens, 1, [100, 200, 300], W)
    check(t.equal(final_logits(A, {1: res_A[1], 2: res_A[2]}, {1: Q, 2: None}), logits_A),
          "patching a run's own residual is a bit-exact no-op")
    whole_1 = final_logits(A, {1: res_B[1]}, {1: None})
    check(t.allclose(final_logits(A, {1: res_B[1]}, {1: t.eye(model.d_model)}), whole_1, atol=1e-5),
          "a full-rank basis equals the whole-state patch")
    check(t.allclose(final_logits(A, {1: res_B[1], 2: res_B[2]}, {1: None, 2: None}),
                     final_logits(A, {2: res_B[2]}, {2: None}), atol=1e-6),
          "a whole-state patch over a band equals patching its last layer alone")
    patch_dir = workdir / "patch"
    summary = causal_test.run_patch(model, j_lens, concepts_path=str(HERE / "data/concepts.tsv"), pairs="ta-te,mr-bn",
                                    layers=[1, 2], out_dir=str(patch_dir), vocab_size=jpp_cli.vocab_size_of(model),
                                    require_correct=False)
    with (patch_dir / "patch_rows.csv").open(encoding="utf-8") as handle:
        patch_rows = list(csv.DictReader(handle))
    check(summary["n_items"] == len(patch_rows) > 0 and all(summary[c]["n"] > 0 for c in causal_test.PATCH_CONDITIONS),
          f"run_patch: {len(patch_rows)} items, every condition scored ({summary['skipped']})")
    check(all(r["random_rank"] == r["english_rank"] and int(r["english_other_rank"]) <= int(r["english_rank"])
              <= int(r["english_ab_rank"]) for r in patch_rows),
          "random rank = English rank >= control rank; English(A)+English(B) rank >= it")
    shares = [float(r[f"{c}_share"]) for r in patch_rows for c in causal_test.PATCH_CONDITIONS if r.get(f"{c}_share")]
    check(all(float(r["whole_share"]) == 1.0 for r in patch_rows) and all(-1e-6 <= s <= 1 + 1e-6 for s in shares),
          "share of the B - A difference: 1 for the whole state, within [0, 1] for every subspace")
    split_dir = workdir / "patch_split"
    split = causal_test.run_patch(model, j_lens, concepts_path=str(HERE / "data/concepts.tsv"), pairs="ta-te,mr-bn",
                                  layers=[1, 2], out_dir=str(split_dir), vocab_size=jpp_cli.vocab_size_of(model),
                                  require_correct=False, conditions=causal_test.PATCH_SPLIT_CONDITIONS)
    with (split_dir / "patch_rows.csv").open(encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    check(len(split_rows) == len(patch_rows) and all(split[c]["n"] > 0 for c in causal_test.PATCH_SPLIT_CONDITIONS)
          and "whole_pB" not in split_rows[0] and len(split["contrasts"]) == 6,
          f"split run: {len(split_rows)} items, only the 5 split conditions, their 6 contrasts")
    check(all(a["english_ab_pB"] == b["english_ab_pB"] for a, b in zip(patch_rows, split_rows)),
          "english_ab reproduces bit-exactly across the two runs")
    check(all(r["random_ab_rank"] == r["english_ab_rank"] and int(r["english_a_mean_rank"]) < int(r["english_ab_rank"])
              for r in split_rows), "random_ab rank = English(A)+English(B) rank > English(A) rank")
    check(causal_test.most_specific_first([10, 13, 12, 11], {10: "wat", 11: "water", 12: "water", 13: "wate"}) == [11, 12, 13, 10]
          and causal_test.balanced_pick([[1, 2, 3], [4, 5]], 4) == [1, 4, 2, 5]
          and causal_test.balanced_pick([[1, 2], [1, 3]], 3) == [1, 3, 2]
          and causal_test.balanced_pick([[1], [2, 3, 4]], 3) == [1, 2, 3],
          "token rule: longest prefix first, A and B in turn, no duplicates")
    rank_dir = workdir / "patch_rank"
    ranked = causal_test.run_patch(model, j_lens, concepts_path=str(HERE / "data/concepts.tsv"), pairs="ta-te,mr-bn",
                                   layers=[1, 2], out_dir=str(rank_dir), vocab_size=jpp_cli.vocab_size_of(model),
                                   require_correct=False, conditions=causal_test.PATCH_RANK_CONDITIONS)
    with (rank_dir / "patch_rows.csv").open(encoding="utf-8") as handle:
        rank_rows = list(csv.DictReader(handle))
    check(len(rank_rows) == len(split_rows) and all(ranked[c]["n"] > 0 for c in causal_test.PATCH_RANK_CONDITIONS)
          and len(ranked["contrasts"]) == 6, f"rank run: {len(rank_rows)} items, 4 conditions, 6 contrasts")
    check(all(len({r[f"{c}_rank"] for c in causal_test.PATCH_RANK_CONDITIONS}) == 1
              and int(r["english_eq_rank"]) == min(int(s["english_ab_rank"]), int(s["hindi_ab_rank"]), int(s["target_ab_rank"]))
              for r, s in zip(rank_rows, split_rows)),
          "equal rank: English = Hindi = target = random = the smallest A + B span, per item")
    print(f"\nALL CHECKS PASSED  (artifacts in {workdir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
