"""The budget-capped H200 pilot, run inside Modal by ``modal_app.pilot``.

Stages, in priority order. Each starts only if its estimate fits the remaining budget:
  load          Sarvam-30B bf16 from the Volume straight onto the GPU (once, cached)
  sanity        BOS; LRP surgery on the real model (forward change, patch counts); greedy text
  fit_router    the k-means router on 128 WikiText prompts (forward passes only)
  fit_shard_j   a 2-prompt J-Lens fit: measures seconds per prompt, the number the budget needs
  pivot_logit   the pivot-language experiment with the logit lens (forward passes only)
  fit_shard_r   a 2-prompt R+MoE fit (timing; LRP surgery under real backward passes)
  grade         the correctness CSV that fit-weights / evaluate need later
  causal_logit  the English ablation with logit-lens directions (the 105B's causal test; no lens fit)
  patch         activation patching from a concept-B prompt at the mid band: whole state vs its
                English / Hindi / target / random subspace component (J-lens directions)
  patch_split   the follow-up: A + B Hindi / target spans, a rank-matched random control for the
                A + B English span, and A's English reset to the other concepts' mean
  patch_rank    the rank-matched test: the A + B swap with English, Hindi and target spans of equal
                rank per item, plus a random subspace of that rank
``report.json`` (saved to the Volume after every stage) holds timings, results, the
running cost estimate and a projection for a full 64-prompt fit.

Sarvam-105B: ``sanity`` checks the RoPE caches and measures seconds per forward; the
forward-bound stages then size their estimates from it. A stage whose result carries a
``stop_rule`` (missing weights, broken RoPE, BOS-probe accuracy < 60%, too few pivot items)
ends the run: every later stage is skipped.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
import traceback
from pathlib import Path

# Modal list prices on 2026-10-18: H200 $4.54/h + 8 cores x $0.0472 + 32 GiB x $0.0080.
# A 2xH200 container (the 105B) is 2 x 4.54 + the same CPU/memory: pass usd_per_hour.
USD_PER_HOUR = 4.54 + 8 * 0.0472 + 32 * 0.0080
# The cheaper causal test (5 pairs x 32 concepts, logit-lens directions).
CAUSAL_LOGIT_PAIRS = "ta-te,bn-gu,kn-ml,mr-bn,te-kn"
# The 105B's forward is launch-bound (a Python loop over ~100 touched experts per layer), so rq_collapse
# reads RQ_BATCH prompts per forward and causal_steer runs an item's 8 conditions as one batch. The 30B
# keeps batch 1 (its published numbers stay bit-reproducible). A batched forward is estimated at
# BATCH_FORWARD_COST single forwards until a run measures it.
RQ_BATCH = 16
BATCH_FORWARD_COST = 2.0
BOS_PROBE_PAIRS = ("ta-te", "hi-ta", "ta-en", "bn-gu", "en-hi")
GREEDY_PROMPTS = (
    'हिन्दी: "पानी" - English: "',
    'தமிழ்: "நீர்" - తెలుగు: "',
    "भारत की राजधानी",
    "The capital of Tamil Nadu is",
)


class _Collect(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.name}: {record.getMessage()}")


def run_pilot(model_dir: str, runs_dir: str, *, budget_minutes: float, commit, device_map: str | None = "cuda",
              stages: list[str] | None = None, tag: str = "pilot", fit_prompts: int = 2,
              fit_rows: int = 16, shard_idx: int = 0, num_shards: int = 1, usd_per_hour: float = USD_PER_HOUR,
              bos_modes: list[str] | None = None, causal_pairs: str = CAUSAL_LOGIT_PAIRS, smoke: bool = False,
              seconds_per_forward_guess: float = 0.5, extra_lenses: list[str] | None = None,
              fit_name: str | None = None, stage_args: dict | None = None) -> dict:
    """``device_map=None`` runs on whatever device the loader picks (the local CPU dry run).
    ``extra_lenses`` are lens files for ``pivot_ext``; ``fit_name`` names ``fit_j_shard``'s fit
    (default ``j_<fit_prompts>p``). ``stage_args`` reaches the cross-lingual modules as ``ctx.args``.
    ``stages`` picks a subset (in the fixed order below); ``tag`` names the output dir;
    ``fit_prompts`` sizes the real lens fits; ``fit_rows`` is jpp's --jacobian-rows-per-pass;
    ``shard_idx``/``num_shards`` split ``fit_j_shard`` across containers. ``usd_per_hour`` prices
    the container; ``bos_modes`` limits the BOS probe (default: every mode); ``causal_pairs`` are
    ``causal_logit``'s pairs. ``smoke`` (a random tiny model) keeps every item and ignores the stop
    rules; ``seconds_per_forward_guess`` sizes estimates until ``sanity`` measures it."""
    start = time.monotonic()
    os.chdir(os.environ.get("JPP_DIR", "/jpp"))  # jpp reads data/ relative to its root
    runs = Path(runs_dir)
    pilot_dir = runs / tag
    pilot_dir.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(logging.FileHandler(pilot_dir / "pilot.log", encoding="utf-8"))
    collector = _Collect()
    root_logger.addHandler(collector)

    import torch as t

    import pivot_experiment
    import run_jpp
    import sarvam_support

    jpp_cli = run_jpp._import_jpp_cli()  # puts jpp's src/ and scripts/ on sys.path
    run_jpp.install(jpp_cli)
    from workspace_lens.lrp.lrp import lrp_rule_config_for_mode
    # Load once: every stage (and pivot_experiment) goes through jpp_cli.get_hf_model.
    loader, cache = jpp_cli.get_hf_model, {}

    def cached_loader(name: str, **kwargs):
        if name not in cache:
            cache[name] = loader(name, **kwargs)
        return cache[name]

    jpp_cli.get_hf_model = cached_loader

    def model():
        return jpp_cli.get_hf_model(model_dir, attn_implementation="sdpa", device_map=device_map)

    device_args = ["--device-map", device_map] if device_map else []

    def layers() -> str:
        """Every block below the final one (0..17 on Sarvam-30B's 19 blocks)."""
        return ",".join(str(i) for i in range(model().n_layers - 1))

    model_type = sarvam_support.sarvam_model_type(model_dir)
    is_mla = model_type == "sarvam_mla"
    report: dict = {"usd_per_hour": round(usd_per_hour, 3), "budget_minutes": budget_minutes,
                    "model_type": model_type, "stages": []}
    # Seconds per forward pass on a ~100-token prompt: guessed until ``sanity`` measures it.
    speed = {"seconds_per_forward": seconds_per_forward_guess, "measured": False}
    # Random-weight dry runs (CPU, or smoke) would trip every accuracy rule.
    enforce_stop_rules = device_map is not None and not smoke
    require_correct = device_map is not None and not smoke

    def forward_minutes(n_forwards: int, overhead_minutes: float = 0.5) -> float:
        """A forward-bound stage's estimate: 25% headroom over the measured speed."""
        return round(overhead_minutes + 1.25 * n_forwards * speed["seconds_per_forward"] / 60, 2)

    def save() -> None:
        minutes = (time.monotonic() - start) / 60
        report["elapsed_minutes"] = round(minutes, 2)
        report["est_cost_usd_so_far"] = round(minutes / 60 * usd_per_hour, 2)
        (pilot_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        commit()

    def stage(name: str, est_minutes, fn) -> None:
        if report.get("stopped"):
            report["stages"].append({"stage": name, "status": f"skipped: stop rule ({report['stopped']})"})
            save()
            return
        est_minutes = est_minutes() if callable(est_minutes) else est_minutes
        elapsed = (time.monotonic() - start) / 60
        if elapsed + est_minutes > budget_minutes:
            report["stages"].append({"stage": name, "status": "skipped: over budget", "est_minutes": est_minutes})
            save()
            return
        stage_start, first_message = time.monotonic(), len(collector.messages)
        entry: dict = {"stage": name, "est_minutes": est_minutes}
        try:
            entry["result"] = fn()
            entry["status"] = "ok"
        except BaseException as exc:  # SystemExit from a CLI stage included
            entry["status"] = f"error: {exc!r}"
            entry["traceback"] = traceback.format_exc()[-3000:]
            if is_mla and name in ("load", "sanity"):  # nothing after these can work
                report["stopped"] = f"{name} failed"
        entry["minutes"] = round((time.monotonic() - stage_start) / 60, 2)
        entry["log_tail"] = collector.messages[first_message:][-8:]
        stop = entry.get("result", {}).get("stop_rule") if isinstance(entry.get("result"), dict) else None
        if stop and enforce_stop_rules:
            report["stopped"] = f"{name}: {stop}"
        report["stages"].append(entry)
        save()

    def next_token(hf, ids):
        """Greedy next id as a [1, 1] tensor on the input's device, wherever the logits land
        (accelerate's device_map hooks return them on the input's GPU; don't rely on it)."""
        return hf(input_ids=ids, use_cache=False).logits[0, -1].argmax().view(1, 1).to(ids.device)

    def load() -> dict:
        m = model()
        result: dict = {"model": repr(m), "gpu_gb_allocated": round(t.cuda.memory_allocated() / 1e9, 1) if t.cuda.is_available() else 0.0}
        if t.cuda.is_available():
            result["gpu_gb_by_device"] = {i: round(t.cuda.memory_allocated(i) / 1e9, 1) for i in range(t.cuda.device_count())}
        device_map_used = getattr(m._hf_model, "hf_device_map", None)
        if device_map_used:
            result["modules_per_device"] = {str(d): sum(1 for v in device_map_used.values() if v == d)
                                            for d in sorted(set(device_map_used.values()), key=str)}
        info = getattr(m, "loading_info", {})
        result["loading_info"] = {key: {"n": len(value), "first": value[:5]} for key, value in info.items()}
        bad = {key: len(info.get(key, [])) for key in ("missing_keys", "unexpected_keys", "mismatched_keys") if info.get(key)}
        if is_mla and bad:  # a missing weight would be silently random-initialised
            result["stop_rule"] = f"checkpoint/code mismatch {bad}"
        return result

    def sanity() -> dict:
        m = model()
        hf, tokenizer = m._hf_model, m.tokenizer
        ids = m.encode("भारत की राजधानी नई दिल्ली है। தமிழ்நாட்டின் தலைநகரம் சென்னை. The river flows.", max_length=64)
        result: dict = {"bos_first": int(ids[0, 0]) == tokenizer.bos_token_id, "n_tokens": int(ids.shape[1])}
        with t.no_grad():
            stock = m.forward(ids).last_hidden_state.float()
            rerun = m.forward(ids).last_hidden_state.float()
        # GPU run-to-run noise: the floor any patched-forward difference is judged against.
        result["stock_rerun_max_abs_diff"] = float((rerun - stock).abs().max())
        if is_mla:
            raw = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
            try:
                result["rope"] = sarvam_support.verify_mla_rope(hf, raw, m.sarvam_modeling, seq_len=int(ids.shape[1]))
            except RuntimeError as exc:
                result["rope"] = {"error": str(exc)[:600]}
                result["stop_rule"] = "RoPE check failed"
        else:  # LRP surgery exists only for the 30B's modules
            for mode in ("rlens", "r+moe"):
                with sarvam_support.apply_sarvam_lrp_rules(hf, lrp_rule_config_for_mode(mode)) as patch_report:
                    with t.no_grad():
                        patched = m.forward(ids).last_hidden_state.float()
                result[mode] = {
                    "max_abs_diff": float((patched - stock).abs().max()),
                    "stock_abs_max": float(stock.abs().max()),
                    "patched": {"ln_rule": len(patch_report.ln_rule), "mlp_rule": len(patch_report.mlp_rule),
                                "routed_experts": len(patch_report.moe_experts), "moe_blocks": len(patch_report.moe_gates)},
                }
        # Speed on a 5-shot translation prompt, the shape every later stage runs.
        concepts = pivot_experiment.load_concepts(pivot_experiment.HERE / "data/concepts.tsv")
        timing_ids = m.encode(pivot_experiment.build_prompt(concepts, sorted(concepts)[0], "ta", "te", shots=5, seed=0))
        def sync_all() -> None:  # every GPU: with device_map="auto" the LM head runs on the last one
            for device in range(t.cuda.device_count() if t.cuda.is_available() else 0):
                t.cuda.synchronize(device)

        with t.no_grad():
            hf(input_ids=timing_ids, use_cache=False)  # warm-up
            sync_all()
            timing_start = time.monotonic()
            for _ in range(3):
                hf(input_ids=timing_ids, use_cache=False)
            sync_all()
        speed.update(seconds_per_forward=(time.monotonic() - timing_start) / 3, measured=True)
        result["seconds_per_forward"] = round(speed["seconds_per_forward"], 3)
        result["timing_prompt_tokens"] = int(timing_ids.shape[1])
        report["seconds_per_forward"] = result["seconds_per_forward"]
        greedy = {}
        for prompt in GREEDY_PROMPTS:
            ids = m.encode(prompt)
            with t.no_grad():
                for _ in range(12):
                    ids = t.cat([ids, next_token(hf, ids)], dim=1)
            greedy[prompt] = tokenizer.decode(ids[0, -12:])
        result["greedy"] = greedy
        result["greedy_water_ok"] = "water" in greedy[GREEDY_PROMPTS[0]].lower()
        return result

    router_path = runs / "wiki" / "router.pt"

    def fit_router() -> dict:
        code = run_jpp.main(["fit-router", "--hf-model-name", model_dir, *device_args, "--layers", layers(),
                             "--num-prompts", "128", "--out", str(router_path)])
        return {"exit": code}

    def fit_shard(mode: str):
        def run() -> dict:
            shard_start = time.monotonic()
            code = run_jpp.main(["fit-shard", "--hf-model-name", model_dir, *device_args, "--layers", layers(),
                                 "--router-path", str(router_path), "--lrp-mode", mode, "--num-prompts", "2",
                                 "--checkpoint-every-n-prompts", "100", "--artifacts-base-dir", f"/tmp/pilot_{mode}",
                                 "--checkpoint-name", f"pilot_{mode}"])
            seconds_per_prompt = (time.monotonic() - shard_start) / 2
            return {
                "exit": code,
                "seconds_per_prompt_upper_bound": round(seconds_per_prompt, 1),
                "projected_64_prompt_fit": {
                    "h200_hours": round(64 * seconds_per_prompt / 3600, 2),
                    "usd": round(64 * seconds_per_prompt / 3600 * USD_PER_HOUR, 2),
                },
            }
        return run

    def pivot_logit() -> dict:
        out = pilot_dir / "pivot_logit"
        smoke_args = ["--no-correctness-filter", "--max-items", "3"] if smoke else []
        code = pivot_experiment.main(["--hf-model-name", model_dir, *device_args, "--lens", "logit", *smoke_args,
                                      "--out", str(out)])
        summary_path = out / "summary.csv"  # absent when the correctness filter kept no item
        rows = []
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8") as handle:
                rows = [r for r in csv.DictReader(handle) if r["pair"] == "ALL_INDIC"]
        curves: dict[str, list[float]] = {}
        top10: dict[str, list[float]] = {}
        for r in sorted(rows, key=lambda r: int(r["layer"])):
            curves.setdefault(r["role"], []).append(round(float(r["mean_prob"]), 4))
            top10.setdefault(r["role"], []).append(round(float(r["top10_rate"]), 3))
        with (out / "items.csv").open(encoding="utf-8") as handle:
            items = list(csv.DictReader(handle))
        kept = sum(1 for i in items if i["kept"] == "True")
        result = {"exit": code, "items_kept": f"{kept}/{len(items)}", "all_indic_mean_prob_by_layer": curves,
                  "all_indic_top10_by_layer": top10}
        shuffled_peak = max(top10.get("english_shuffled", [0.0]), default=0.0)
        if is_mla and (kept < 300 or shuffled_peak > 0.1):
            result["stop_rule"] = f"pivot sanity: kept {kept} (< 300?), english_shuffled top-10 peak {shuffled_peak} (> 0.1?)"
        return result

    def grade() -> dict:
        csv_path = runs / "correctness.csv"
        code = run_jpp.main(["grade", "--hf-model-name", model_dir, *device_args, "--out", str(csv_path)])
        return {"exit": code}

    def bos_probe() -> dict:
        """Few-shot translation accuracy (top-1 begins the target word) under each BOS mode;
        the best becomes the mode for every later stage."""
        m = model()
        hf, tokenizer = m._hf_model, m.tokenizer
        index = pivot_experiment.build_vocab_index(tokenizer, jpp_cli.vocab_size_of(m))
        concepts = pivot_experiment.load_concepts(pivot_experiment.HERE / "data/concepts.tsv")
        result: dict = {}
        run_modes = list(modes)
        # Keep the later stages inside the budget: drop all but the first mode (the 30B's best,
        # on the same tokenizer) rather than lose the pivot or the ablation to the BOS probe.
        later = sum(est() for name, est in (("pivot_logit", pivot_minutes), ("causal_logit", causal_logit_minutes))
                    if stages is None or name in stages)
        elapsed = (time.monotonic() - start) / 60
        if len(run_modes) > 1 and elapsed + bos_minutes(len(run_modes)) + later > budget_minutes:
            result["dropped_modes"] = run_modes[1:]
            run_modes = run_modes[:1]
        for mode in run_modes:
            m.bos_mode = mode
            per_pair, hits, total = {}, 0, 0
            for pair in BOS_PROBE_PAIRS:
                src, tgt = pair.split("-")
                pair_hits = pair_total = 0
                for concept in sorted(concepts):
                    prompt = pivot_experiment.build_prompt(concepts, concept, src, tgt, shots=5, seed=0)
                    target_ids = pivot_experiment.prefix_token_ids(index, concepts[concept][tgt], tokenizer)
                    with t.no_grad():
                        top = int(hf(input_ids=m.encode(prompt), use_cache=False).logits[0, -1].argmax())
                    pair_hits += top in target_ids
                    pair_total += 1
                per_pair[pair] = f"{pair_hits}/{pair_total}"
                hits, total = hits + pair_hits, total + pair_total
            ids = m.encode("भारत की राजधानी")
            with t.no_grad():
                for _ in range(10):
                    ids = t.cat([ids, next_token(hf, ids)], dim=1)
            result[mode] = {"accuracy": round(hits / total, 3), "per_pair": per_pair,
                            "greedy_hindi": tokenizer.decode(ids[0, -10:])}
        best = max(run_modes, key=lambda mode: result[mode]["accuracy"])
        m.bos_mode = best
        os.environ[sarvam_support.BOS_MODE_ENV] = best
        result["chosen"] = best
        if result[best]["accuracy"] < 0.6:  # the 30B reached 96%; its broken-RoPE run ~0%
            result["stop_rule"] = f"best BOS-probe accuracy {result[best]['accuracy']} < 0.6: the model is broken"
        return result

    modes = tuple(bos_modes or sarvam_support.BOS_MODES)
    unknown_modes = set(modes) - set(sarvam_support.BOS_MODES)
    if unknown_modes:
        raise ValueError(f"unknown BOS modes {sorted(unknown_modes)}")

    def bos_minutes(n_modes: int) -> float:
        return max(2.0, forward_minutes(n_modes * (32 * len(BOS_PROBE_PAIRS) + 10)))

    def pivot_minutes() -> float:
        return max(3.0, forward_minutes(14 * 32, overhead_minutes=1.0))

    import glob

    lens_dir = runs / "lenses"
    r_name = f"r_moe_{fit_prompts}p"
    r_experts, r_pooled, jpp_lens = (lens_dir / f"{r_name}_experts.pt", lens_dir / f"{r_name}.pt",
                                     lens_dir / f"jpp_{fit_prompts}p.pt")

    def fit_r_lens() -> dict:
        """The real R+MoE fit: ``fit_prompts`` WikiText prompts, merged into expert Jacobians
        (for J++) and their pooled average (the R-Lens), both saved to the Volume."""
        art, fit_start = f"/tmp/fit_{r_name}", time.monotonic()
        code = run_jpp.main(["fit-shard", "--hf-model-name", model_dir, *device_args, "--layers", layers(),
                             "--router-path", str(router_path), "--lrp-mode", "r+moe", "--num-prompts", str(fit_prompts),
                             "--checkpoint-every-n-prompts", "1000", "--artifacts-base-dir", art, "--checkpoint-name", r_name])
        fit_minutes = (time.monotonic() - fit_start) / 60
        shard_dirs = sorted(glob.glob(f"{art}/*/{r_name}/shard*"))
        lens_dir.mkdir(parents=True, exist_ok=True)
        merged = run_jpp.main(["merge-experts", "--checkpoint-dirs", *shard_dirs, "--checkpoint-name", r_name,
                               "--out", str(r_experts), "--pooled-lens-out", str(r_pooled)])
        commit()
        return {"fit_exit": code, "merge_exit": merged, "fit_minutes": round(fit_minutes, 1),
                "seconds_per_prompt": round(fit_minutes * 60 / fit_prompts, 1), "pooled_lens": str(r_pooled)}

    def fit_jpp() -> dict:
        """J++ weights over the R+MoE experts, learned WITHOUT the multilingual eval (whose
        English-target items would tune the lens to find English in non-English prompts)."""
        evals = "/tmp/evals_no_multilingual"
        run_jpp.main(["prepare-evals", "--out", evals])
        code = run_jpp.main(["fit-weights", "--hf-model-name", model_dir, *device_args, "--layers", layers(),
                             "--experts", str(r_experts), "--checkpoint-name", f"jpp_{fit_prompts}p",
                             "--eval-data-dir", evals, "--correctness-csv", str(runs / "correctness.csv"),
                             "--out", str(jpp_lens)])
        commit()
        return {"exit": code, "lens": str(jpp_lens)}

    def pivot_lenses() -> dict:
        lens_args = ["--lens", "logit"]
        for path in (r_pooled, jpp_lens):
            if path.is_file():
                lens_args += ["--lens", str(path)]
        out = pilot_dir / "pivot_lenses"
        code = pivot_experiment.main(["--hf-model-name", model_dir, *device_args, *lens_args, "--out", str(out)])
        peaks: dict[str, dict] = {}
        summary_path = out / "summary.csv"
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8") as handle:
                for r in csv.DictReader(handle):
                    if r["pair"] == "ALL_INDIC":
                        peaks.setdefault(r["lens"], {}).setdefault(r["role"], []).append(round(float(r["mean_prob"]), 3))
        return {"exit": code, "lenses": lens_args[1::2], "all_indic_mean_prob_by_layer": peaks}

    def rows_probe() -> dict:
        """Seconds/prompt and peak GPU memory of a 2-prompt J fit at 32 and 64 Jacobian rows
        per backward pass (jpp default 16: 256 passes per prompt on d_model 4096)."""
        out: dict = {}
        for rows in (32, 64):
            t.cuda.empty_cache()
            t.cuda.reset_peak_memory_stats()
            probe_start = time.monotonic()
            try:
                code = run_jpp.main(["fit-shard", "--hf-model-name", model_dir, *device_args, "--layers", layers(),
                                     "--router-path", str(router_path), "--lrp-mode", "none", "--num-prompts", "2",
                                     "--jacobian-rows-per-pass", str(rows), "--checkpoint-every-n-prompts", "1000",
                                     "--artifacts-base-dir", f"/tmp/rows_{rows}", "--checkpoint-name", f"rows_{rows}"])
                out[rows] = {"exit": code, "seconds_per_prompt": round((time.monotonic() - probe_start) / 2, 1),
                             "peak_gpu_gb": round(t.cuda.max_memory_allocated() / 1e9, 1)}
            except BaseException as exc:  # OOM at the larger row count is an answer, not a failure
                out[rows] = {"error": repr(exc)[:300]}
            t.cuda.empty_cache()
        return out

    j_name = fit_name or f"j_{fit_prompts}p"

    def fit_j_shard() -> dict:
        """One shard of the plain J-lens fit (lrp-mode none); checkpoints on the Volume so the
        CPU ``merge_j`` step can sum all shards."""
        shard_start = time.monotonic()
        code = run_jpp.main(["fit-shard", "--hf-model-name", model_dir, *device_args, "--layers", layers(),
                             "--router-path", str(router_path), "--lrp-mode", "none",
                             "--num-prompts", str(fit_prompts), "--num-shards", str(num_shards), "--shard-idx", str(shard_idx),
                             "--jacobian-rows-per-pass", str(fit_rows), "--checkpoint-every-n-prompts", "20",
                             "--artifacts-base-dir", str(runs / "jfit" / j_name), "--checkpoint-name", j_name])
        commit()
        minutes = (time.monotonic() - shard_start) / 60
        return {"exit": code, "shard": f"{shard_idx}/{num_shards}", "minutes": round(minutes, 1)}

    j_lens_path = lens_dir / "j_200p.pt"

    def pivot_all() -> dict:
        """The pivot experiment with every fitted lens plus the shuffled-concept controls."""
        lens_args = ["--lens", "logit"]
        for path in (j_lens_path, r_pooled, jpp_lens):
            if path.is_file():
                lens_args += ["--lens", str(path)]
        out = pilot_dir / "pivot_all"
        code = pivot_experiment.main(["--hf-model-name", model_dir, *device_args, *lens_args, "--out", str(out)])
        commit()
        return {"exit": code, "lenses": lens_args[1::2]}

    def causal() -> dict:
        import causal_test

        m = model()
        results = {}
        n = m.n_layers
        mid = [l for l in (13, 14, 15, 16) if l < n - 1] if n == 19 else list(range(int(0.68 * n), int(0.89 * n)))
        early = [l for l in (5, 6, 7, 8) if l < n - 1] if n == 19 else list(range(int(0.26 * n), int(0.47 * n)))
        for lens_path in (j_lens_path, r_pooled):
            if not lens_path.is_file():
                continue
            lens = jpp_cli.load_lens_file(str(lens_path), hf_model_name=model_dir)
            results[lens_path.stem] = causal_test.run(
                m, lens, concepts_path=str(pivot_experiment.HERE / "data/concepts.tsv"), pairs=causal_test.DEFAULT_PAIRS,
                mid_layers=mid, early_layers=early, out_dir=str(pilot_dir / f"causal_{lens_path.stem}"),
                vocab_size=jpp_cli.vocab_size_of(m), require_correct=device_map is not None)
            commit()
        return results

    def bands() -> tuple[list[int], list[int]]:
        n = model().n_layers
        if n == 19:  # Sarvam-30B: English readable at 13-16, nothing before 11
            return [13, 14, 15, 16], [5, 6, 7, 8]
        mid = [l for l in range(int(0.68 * n), int(0.89 * n)) if l < n - 1][:6] or [max(0, n - 2)]
        return mid, early_band(len(mid), n)

    def early_band(width: int, n: int) -> list[int]:
        """``width`` layers ending at 0.47 x depth, where no lens reads any language."""
        end = int(0.47 * n)
        return list(range(max(0, end - width + 1), end + 1))

    def auto_bands() -> tuple[list[int], list[int], dict]:
        """The mid band from THIS container's logit-lens pivot: the contiguous layers (at most
        6) where English is in the top 10 for >= 80% of Indic->Indic items. Falls back to
        :func:`bands` without a pivot result or such a layer. The early band has the same width."""
        summary_path = pilot_dir / "pivot_logit" / "summary.csv"
        n = model().n_layers
        rates: dict[int, float] = {}
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8") as handle:
                rates = {int(r["layer"]): float(r["top10_rate"]) for r in csv.DictReader(handle)
                         if r["pair"] == "ALL_INDIC" and r["lens"] == "logit" and r["role"] == "english"}
        runs, current = [], []
        for layer in sorted(rates):
            if rates[layer] >= 0.8 and layer < n - 1 and (not current or layer == current[-1] + 1):
                current.append(layer)
            else:
                if current:
                    runs.append(current)
                current = [layer] if rates[layer] >= 0.8 and layer < n - 1 else []
        if current:
            runs.append(current)
        if not runs:
            mid, early = bands()
            return mid, early, {"source": "fallback fractions", "english_top10": rates}
        best = max(runs, key=lambda run: sum(rates[l] for l in run))
        if len(best) > 6:  # the 6-layer window with the most English
            best = max((best[i:i + 6] for i in range(len(best) - 5)), key=lambda w: sum(rates[l] for l in w))
        return best, early_band(len(best), n), {"source": "pivot_logit english top-10 >= 0.8",
                                                "english_top10": {l: rates[l] for l in best}}

    def causal_logit() -> dict:
        """English ablation with LOGIT-lens directions (g * W_U rows, no fitted lens): the
        30B keeps its J-lens bands (13-16 / 5-8) for an apples-to-apples calibration; any other
        model takes the band its own logit-lens pivot found (:func:`auto_bands`)."""
        import causal_test

        m = model()
        if m.n_layers == 19:
            mid, early = bands()
            band_info = {"source": "Sarvam-30B J-lens bands"}
        else:
            mid, early, band_info = auto_bands()
        lens = jpp_cli.load_lenses([jpp_cli.LOGIT_LENS_SPEC], m, hf_model_name=model_dir,
                                   layers=list(range(m.n_layers - 1)))[jpp_cli.LOGIT_LENS_SPEC]
        pairs = ",".join(causal_pairs.split(",")[:2]) if smoke else causal_pairs
        summary = causal_test.run(
            m, lens, concepts_path=str(pivot_experiment.HERE / "data/concepts.tsv"), pairs=pairs,
            mid_layers=mid, early_layers=early, out_dir=str(pilot_dir / "causal_logit"),
            vocab_size=jpp_cli.vocab_size_of(m), require_correct=require_correct)
        commit()
        return {"pairs": pairs, "band": band_info, **summary}

    def causal_logit_minutes() -> float:
        # base + 3 conditions per item; items the model gets wrong stop after the base pass
        return forward_minutes(4 * 32 * len(causal_pairs.split(",")), overhead_minutes=1.0)

    def primary_lens():
        path = j_lens_path if j_lens_path.is_file() else r_pooled
        return path, jpp_cli.load_lens_file(str(path), hf_model_name=model_dir)

    def swap() -> dict:
        import causal_test

        mid, early = bands()
        path, lens = primary_lens()
        summary = causal_test.run_swap(
            model(), lens, concepts_path=str(pivot_experiment.HERE / "data/concepts.tsv"), pairs=causal_test.DEFAULT_PAIRS,
            mid_layers=mid, early_layers=early, out_dir=str(pilot_dir / f"swap_{path.stem}"),
            vocab_size=jpp_cli.vocab_size_of(model()), require_correct=device_map is not None)
        commit()
        return {"lens": path.stem, **summary}

    def patch_conditions(kind: str) -> tuple[str, ...]:
        import causal_test

        return {"": causal_test.PATCH_CONDITIONS, "split": causal_test.PATCH_SPLIT_CONDITIONS,
                "rank": causal_test.PATCH_RANK_CONDITIONS}[kind]

    def patch(kind: str):
        """``patch`` (kind ""): the 7 first-run conditions; ``patch_split``: the follow-up's 5;
        ``patch_rank``: the rank-matched test's 4."""
        def run() -> dict:
            import causal_test

            mid, _ = bands()
            path, lens = primary_lens()
            # Dry runs (random weights): 2 pairs exercise every code path.
            pairs = causal_test.DEFAULT_PAIRS if require_correct else ",".join(causal_test.DEFAULT_PAIRS.split(",")[:2])
            summary = causal_test.run_patch(
                model(), lens, concepts_path=str(pivot_experiment.HERE / "data/concepts.tsv"), pairs=pairs,
                layers=mid, out_dir=str(pilot_dir / f"patch{f'_{kind}' if kind else ''}_{path.stem}"),
                vocab_size=jpp_cli.vocab_size_of(model()), require_correct=require_correct,
                conditions=patch_conditions(kind))
            commit()
            result = {"lens": path.stem, **summary}
            if require_correct and not kind and summary["whole"]["flip_to_B_rate"] < 0.5:  # a weak ceiling
                result["warning"] = f"whole-state ceiling flips only {summary['whole']['flip_to_B_rate']:.2f}"
            return result
        return run

    def patch_minutes(kind: str):
        def est() -> float:
            import causal_test

            # one clean pass per (pair, concept), then one per condition for the items kept
            n_pairs = len(causal_test.DEFAULT_PAIRS.split(","))
            return forward_minutes(n_pairs * 32 * (1 + len(patch_conditions(kind))), overhead_minutes=1.0)
        return est

    def hindi_ablation() -> dict:
        import causal_test

        mid, early = bands()
        path, lens = primary_lens()
        summary = causal_test.run(
            model(), lens, concepts_path=str(pivot_experiment.HERE / "data/concepts.tsv"), pairs=causal_test.DEFAULT_PAIRS,
            mid_layers=mid, early_layers=early, out_dir=str(pilot_dir / f"hindi_ablation_{path.stem}"),
            vocab_size=jpp_cli.vocab_size_of(model()), require_correct=device_map is not None, latent_lang="hi")
        commit()
        return {"lens": path.stem, **summary}

    def sentences() -> dict:
        import sentence_experiment

        specs = ["logit"] + [str(p) for p in (j_lens_path, r_pooled) if p.is_file()]
        n = model().n_layers
        lenses = jpp_cli.load_lenses(specs, model(), hf_model_name=model_dir, layers=list(range(n - 1)))
        summary = sentence_experiment.run(model(), lenses, layers=list(range(n - 1)), vocab_size=jpp_cli.vocab_size_of(model()),
                                          out_dir=str(pilot_dir / "sentences"), require_correct=device_map is not None)
        commit()
        return summary

    ### READER QUESTIONS (docs/09_reader_questions.md; code in reader_questions.py)

    def own_j_lens() -> bool:
        """The fitted J-lens on the Volume is the 30B's (19 blocks): never apply it to another model."""
        return not is_mla and model().n_layers == 19 and j_lens_path.is_file()

    def rq_lenses():
        """The J-lens first (where fitted: its top tokens feed the meta-token count), then logit."""
        m = model()
        all_layers = list(range(m.n_layers - 1))
        specs = ([str(j_lens_path)] if own_j_lens() else []) + ["logit"]
        return jpp_cli.load_lenses(specs, m, hf_model_name=model_dir, layers=all_layers), all_layers

    def rq_band() -> list[int]:
        """Each model's English band (30B: 13-16; any other model: the generic mid band of ``bands``)."""
        n = model().n_layers
        return {19: [13, 14, 15, 16]}.get(n) or bands()[0]

    def rq_items(kind: str, instr: str = "src", max_steps: int = 2):
        def run() -> dict:
            import reader_questions as rqm

            lenses, all_layers = rq_lenses()
            label = f"rq_{kind}" + (f"_{instr}" if kind == "chat_word" else "")
            extra = {} if require_correct else {"max_items": 2, "require_correct": False}
            result = rqm.run_items(model(), lenses, kind=kind, instr=instr, out_dir=str(pilot_dir / label),
                                   vocab_size=jpp_cli.vocab_size_of(model()), layers=all_layers, max_steps=max_steps,
                                   meta_band=rq_band(), batch_size=RQ_BATCH if is_mla else 1, **extra)
            commit()
            return result
        return run

    def rq_collapse() -> dict:
        import reader_questions as rqm

        lenses, all_layers = rq_lenses()
        extra = {} if require_correct else {"max_pairs": 2}
        result = {table: rqm.run_collapse(model(), lenses, out_dir=str(pilot_dir / "rq_collapse"), table=table,
                                          vocab_size=jpp_cli.vocab_size_of(model()), layers=all_layers,
                                          batch_size=RQ_BATCH if is_mla else 1, **extra)
                  for table in ("homonyms", "kinship")}
        commit()
        return result

    def rq_zh() -> dict:
        """How well does the model translate INTO Chinese? (reader question 5: how much Chinese it learned)"""
        out = pilot_dir / "rq_zh"
        zh_pairs = "ta-zh,te-zh,bn-zh,hi-zh,kn-zh,ml-zh,en-zh"
        smoke_args = [] if require_correct else ["--max-items", "2", "--pairs", "ta-zh,en-zh"]
        code = pivot_experiment.main(["--hf-model-name", model_dir, *device_args, "--lens", "logit", "--pairs", zh_pairs,
                                      *smoke_args, "--out", str(out)])
        with (out / "items.csv").open(encoding="utf-8") as handle:
            items = list(csv.DictReader(handle))
        per_pair: dict = {}
        for i in items:
            per_pair.setdefault(i["pair"], [0, 0])
            per_pair[i["pair"]][0] += i["kept"] == "True"
            per_pair[i["pair"]][1] += 1
        commit()
        return {"exit": code, "accuracy_by_pair": {p: f"{k}/{n}" for p, (k, n) in per_pair.items()},
                "accuracy": round(sum(k for k, _ in per_pair.values()) / max(1, sum(n for _, n in per_pair.values())), 3)}

    def causal_steer() -> dict:
        """The English ablation with negative steering (alpha 2: reflect, 3: push to -2x) beside the
        plain projection (alpha 1), on the J-lens bands where fitted (30B), else the logit lens."""
        import causal_test

        m = model()
        path_lens = (j_lens_path.stem, jpp_cli.load_lens_file(str(j_lens_path), hf_model_name=model_dir)) \
            if own_j_lens() else ("logit",jpp_cli.load_lenses([jpp_cli.LOGIT_LENS_SPEC], m, hf_model_name=model_dir,
                                                                          layers=list(range(m.n_layers - 1)))["logit"])
        mid = rq_band()
        early = bands()[1] if m.n_layers == 19 else early_band(len(mid), m.n_layers)
        pairs = causal_pairs if require_correct else ",".join(causal_pairs.split(",")[:1])
        summary = causal_test.run(m, path_lens[1], concepts_path=str(pivot_experiment.HERE / "data/concepts.tsv"), pairs=pairs,
                                  mid_layers=mid, early_layers=early, out_dir=str(pilot_dir / f"causal_steer_{path_lens[0]}"),
                                  vocab_size=jpp_cli.vocab_size_of(m), require_correct=require_correct, alphas=(1.0, 2.0, 3.0),
                                  batch_conditions=is_mla)
        commit()
        return {"lens": path_lens[0], "pairs": pairs, **summary}

    def rq_think() -> dict:
        import reader_questions as rqm

        kw = {"per_pair": 2 if is_mla else 4, "max_new_tokens": 160 if is_mla else 200} if require_correct else \
             {"pairs": "ta-te", "per_pair": 1, "max_new_tokens": 6}
        result = rqm.run_think(model(), out_dir=str(pilot_dir / "rq_think"), **kw)
        commit()
        return result

    def pivot_ext() -> dict:
        """The pivot (pivot_experiment, all 14 pairs, shuffled controls) with the lens files passed as
        ``extra_lenses`` beside the logit lens: Qwen's published J-lens, or an Indic-corpus J-lens
        against WikiText ones. Layers: those every file lens covers."""
        files = [p for p in (extra_lenses or []) if Path(p).is_file()]
        missing = sorted(set(extra_lenses or []) - set(files))
        lens_args = ["--lens", "logit"] + [a for p in files for a in ("--lens", p)]
        smoke_args = [] if require_correct else ["--no-correctness-filter", "--max-items", "2"]
        out = pilot_dir / "pivot_ext"
        code = pivot_experiment.main(["--hf-model-name", model_dir, *device_args, *lens_args, *smoke_args, "--out", str(out)])
        commit()
        peaks: dict = {}
        summary_path = out / "summary.csv"
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8") as handle:
                for r in csv.DictReader(handle):
                    if r["pair"] == "ALL_INDIC":
                        role = peaks.setdefault(r["lens"], {}).setdefault(r["role"], [0.0, 0.0])
                        role[0], role[1] = max(role[0], float(r["top10_rate"])), max(role[1], float(r["mean_prob"]))
        with (out / "items.csv").open(encoding="utf-8") as handle:
            items = list(csv.DictReader(handle))
        return {"exit": code, "lenses": lens_args[1::2], "missing": missing,
                "items_kept": f"{sum(i['kept'] == 'True' for i in items)}/{len(items)}",
                "all_indic_peak_top10_and_prob": peaks}

    def neuron_stage(analysis: str):
        """``neuron_experiment.run`` (expert routing by language / layer-0 neurons), written separately."""
        def run() -> dict:
            import neuron_experiment

            result = neuron_experiment.run(model(), out_dir=str(pilot_dir / analysis), analyses=(analysis,),
                                           vocab_size=jpp_cli.vocab_size_of(model()),
                                           batch_size=RQ_BATCH if is_mla else 1, smoke=not require_correct)
            commit()
            return result
        return run

    n_pairs = len(pivot_experiment.DEFAULT_PAIRS.split(","))
    rq_scale = 1.0 if require_correct else 0.05  # dry runs cap items per pair (see each stage)

    def rq_minutes(n_forwards: float, overhead: float = 1.0):
        return lambda: forward_minutes(int(n_forwards * rq_scale), overhead)

    rq_plan = (  # priority order: real-life chat first; thinking mode (slowest per item) last
        ("rq_chat_word_src", rq_minutes(n_pairs * 32 * 1.3, 1.5), rq_items("chat_word", "src")),
        ("rq_chat_sentence", rq_minutes(12 * 32 * 1.3), rq_items("chat_sentence")),
        # batched (105B): greedy steps per item as measured on its rq_chat_word_src
        ("rq_chat_word_en", rq_minutes(-(-int(n_pairs * 32 * 1.7) // RQ_BATCH) * BATCH_FORWARD_COST if is_mla
                                       else n_pairs * 32 * 1.3), rq_items("chat_word", "en")),
        ("rq_collapse", rq_minutes(-(-(650 + 192) // RQ_BATCH) * BATCH_FORWARD_COST if is_mla else 650 + 192), rq_collapse),
        # batched (105B): ~5 greedy steps per item, as the 30B's (43 found early, 85 x 7 steps)
        ("rq_cloze", rq_minutes(-(-(128 * 5) // RQ_BATCH) * BATCH_FORWARD_COST if is_mla else 128 * 3),
         rq_items("cloze", max_steps=6)),
        ("rq_zh", rq_minutes(7 * 32), rq_zh),
        ("causal_steer", rq_minutes(5 * 32 * (BATCH_FORWARD_COST if is_mla else 8)), causal_steer),
        # cached decoding: ~0.5 of a full forward per token
        ("rq_think", rq_minutes(6 * (2 if is_mla else 4) * (160 if is_mla else 200) * 0.5, 0.5), rq_think),
        # pivot_ext: one forward per lens per item (pivot_experiment), 14 pairs x 32 concepts
        ("pivot_ext", lambda: forward_minutes(int(14 * 32 * (1 + len(extra_lenses or [])) * rq_scale), 1.5), pivot_ext),
        # routing / neurons: ~800 / ~320 forwards (32 concepts x 10 languages + the 448 pivot prompts)
        ("routing", rq_minutes(-(-800 // RQ_BATCH) * BATCH_FORWARD_COST if is_mla else 800), neuron_stage("routing")),
        ("neurons", rq_minutes(-(-320 // RQ_BATCH) * BATCH_FORWARD_COST if is_mla else 320), neuron_stage("neurons")),
    )

    ### CROSS-LINGUAL FOLLOW-UPS (docs/11, 12): prompt domains, the 22 languages.
    # A stage named "<module>.<analysis>" (module in XL_MODULES) calls module.run(ctx, analysis), with
    # module.n_forwards(ctx, analysis) (single-forward equivalents) sizing the budget gate. ctx is
    # built here, so the modules never touch Modal; they batch on every model (RQ_BATCH), unlike the
    # published 30B stages above. Run ``batch_check`` first on a model they haven't run on.

    model_kind = {"sarvam_moe": "sarvam30", "sarvam_mla": "sarvam105"}.get(model_type) or \
        ("qwen" if "qwen" in Path(model_dir).name.lower() else "other")
    qwen_j_path = runs.parent / "ext_lenses/qwen3.6-35b-a3b/j-lens/lens.pt"
    xl_lens_cache: dict = {}

    def xl_lens_specs() -> list[str]:
        """Each model's J-lens where one exists (30B: ours; Qwen: its published one), then logit."""
        if own_j_lens():
            return [str(j_lens_path), "logit"]
        if model_kind == "qwen" and qwen_j_path.is_file():
            return [str(qwen_j_path), "logit"]
        return ["logit"]

    def xl_ctx(name: str):
        from types import SimpleNamespace

        m = model()
        n = m.n_layers
        all_layers = list(range(n - 1))

        def load_lenses(specs: list[str] | None = None) -> dict:
            key = tuple(specs or xl_lens_specs())
            if key not in xl_lens_cache:
                xl_lens_cache[key] = jpp_cli.load_lenses(list(key), m, hf_model_name=model_dir, layers=all_layers)
            return xl_lens_cache[key]

        band = rq_band()
        return SimpleNamespace(
            model=m, model_kind=model_kind, model_dir=model_dir, is_mla=is_mla, n_layers=n, layers=all_layers,
            band=band, early_band=bands()[1] if n == 19 else early_band(len(band), n),
            lens_specs=xl_lens_specs(), load_lenses=load_lenses, vocab_size=jpp_cli.vocab_size_of(m),
            out_dir=str(pilot_dir / name.replace(".", "_")), data_dir=str(pivot_experiment.HERE / "data"),
            batch_size=RQ_BATCH, smoke=not require_correct, args=dict(stage_args or {}), commit=commit,
            seconds_per_forward=speed["seconds_per_forward"])

    XL_MODULES = ("domain_experiment", "newlang_experiment", "quant_check", "routing_ablation", "post1_experiment")

    def xl_stage(name: str):
        module_name, _, analysis = name.partition(".")

        def run() -> dict:
            import importlib

            result = importlib.import_module(module_name).run(xl_ctx(name), analysis)
            commit()
            return result

        def est() -> float:
            import importlib

            n_forwards = importlib.import_module(module_name).n_forwards(xl_ctx(name), analysis)
            return forward_minutes(int(n_forwards), overhead_minutes=1.0)
        return name, est, run

    def batch_check() -> dict:
        """Batched vs lone on THIS model in its real dtype: the readout (right-padded forward_batch:
        top-1 agreement and max |delta log-prob| of the final logits) and greedy decoding
        (left-padded batch_gen.generate: exact token match over 16 steps)."""
        import batch_gen
        import reader_questions as rqm

        m = model()
        reader = rqm.Reader(m, {}, [], jpp_cli.vocab_size_of(m), None)
        ids_list = [m.tokenizer(p, return_tensors="pt").input_ids.to(m.input_device) for p in batch_gen.CHECK_PROMPTS]
        lone = [reader.forward(ids)[1] for ids in ids_list]
        batched = [logits for _, logits in reader.forward_batch(ids_list)]
        top1 = [int(a.argmax()) == int(b.argmax()) for a, b in zip(lone, batched)]
        delta = max(float((t.log_softmax(a, -1) - t.log_softmax(b, -1)).abs().max()) for a, b in zip(lone, batched))
        result = {"readout_top1_agree": round(sum(top1) / len(top1), 3), "readout_max_abs_dlogprob": round(delta, 4),
                  "generate": batch_gen.check(m, batch_gen.CHECK_PROMPTS, max_new_tokens=16, batch_size=8),
                  "model_kind": model_kind}
        if result["readout_top1_agree"] < 1.0 or result["generate"]["exact_match_rate"] < 0.75:
            # batch_probe.py: the batching logic is exact (fp32 and bf16) on the tiny 30B and the tiny 105B
            # split over 2 GPUs, so a gap here is bf16 kernel choice by batch shape, amplified by MoE
            # routing flips. Per-item greedy text is then not reproducible across batch shapes;
            # batch_check_deep measures whether aggregates move.
            result["warning"] = ("batched differs from lone (bf16 batch-variance, not a padding bug: see batch_probe): "
                                 "per-item greedy outputs are batch-shape dependent; run batch_check_deep for the aggregate effect")
        return result

    def batch_check_deep() -> dict:
        """The aggregate effect of batching on THIS model, for the stages that batch. 48 chat prompts
        (6 per language x 8 languages from data/xling, Sarvam's template + empty think block) read lone
        and batched: final-layer top-1 agreement and KL; the logit lens's English mass (ASCII-letter
        tokens) at the English band, as lone-minus-batched per prompt (mean = bias, mean |.| = noise),
        against the lone-vs-lone rerun floor; and 16-token greedy decoding of 16 prompts, lone vs batched."""
        import numpy as np

        import batch_gen
        import reader_questions as rqm

        m = model()
        tok = m.tokenizer
        vocab = jpp_cli.vocab_size_of(m)
        lens = jpp_cli.load_lenses(["logit"], m, hf_model_name=model_dir, layers=list(range(m.n_layers - 1)))["logit"]
        band = rq_band()
        n_tok = min(vocab, len(tok))  # a padded LM head (Qwen) has rows past the tokenizer's vocabulary
        pieces = list(tok.convert_ids_to_tokens(list(range(n_tok)))) + [None] * (vocab - n_tok)
        word = lambda p: (p or "").lstrip("▁Ġ ")
        english = t.tensor([len(word(p)) >= 2 and word(p).isascii() and word(p).isalpha() for p in pieces],
                           device=m._lm_head.weight.device)
        langs = ("en", "hi", "ta", "bn", "ur", "or", "mni", "sat")
        sets = {name: json.loads((pivot_experiment.HERE / f"data/xling/{name}.json").read_text(encoding="utf-8"))
                for name in ("mgsm", "everyday", "xstest")}
        prompts = []
        for lang in langs:
            for name, k in (("mgsm", 2), ("everyday", 2), ("xstest", 2)):
                for item in sets[name]["items"][:k]:
                    msg = item["en"] if lang == "en" else item["tr"][lang]
                    text = rqm.chat_text(tok, msg) + (rqm.NO_THINK_REPLY if model_kind.startswith("sarvam") else "")
                    prompts.append(text)
        if not require_correct:
            prompts = prompts[:6]
        reader = rqm.Reader(m, {}, list(band), vocab, None)
        ids_list = [rqm.chat_ids(m, p) for p in prompts]

        def eng_mass(acts) -> float:
            masses = []
            for layer in band:
                logits = m.unembed(lens.transport(acts[layer][None], layer)).float()[0][:vocab]
                masses.append(float(t.softmax(logits, -1)[english].sum()))
            return float(np.mean(masses))

        lone = [reader.forward(ids) for ids in ids_list]
        rerun = [reader.forward(ids) for ids in ids_list[:8]]
        batched = []
        for start in range(0, len(ids_list), RQ_BATCH):
            batched += reader.forward_batch(ids_list[start: start + RQ_BATCH])
        kl = lambda a, b: float(t.sum(t.softmax(a, -1) * (t.log_softmax(a, -1) - t.log_softmax(b, -1))))
        d_batch = [eng_mass(a[0]) - eng_mass(b[0]) for a, b in zip(lone, batched)]
        d_rerun = [eng_mass(a[0]) - eng_mass(b[0]) for a, b in zip(lone[:8], rerun)]
        lone_mass = [eng_mass(a[0]) for a in lone]
        n_gen = min(16, len(ids_list))
        end_ids = batch_gen.end_ids_of(tok)
        gen_b = batch_gen.generate(m, ids_list[:n_gen], max_new_tokens=16, end_ids=end_ids, batch_size=RQ_BATCH)
        gen_l = [batch_gen.generate_one(m, ids, max_new_tokens=16, end_ids=end_ids) for ids in ids_list[:n_gen]]
        prefix = [next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))) for a, b in zip(gen_l, gen_b)]
        return {"n_prompts": len(prompts), "band": band, "langs": langs,
                "final_top1_agree": round(float(np.mean([int(a[1].argmax()) == int(b[1].argmax()) for a, b in zip(lone, batched)])), 3),
                "final_kl_mean": round(float(np.mean([kl(a[1], b[1]) for a, b in zip(lone, batched)])), 5),
                "final_kl_max": round(max(kl(a[1], b[1]) for a, b in zip(lone, batched)), 5),
                "english_mass_lone_mean": round(float(np.mean(lone_mass)), 4),
                "english_mass_bias_lone_minus_batched": round(float(np.mean(d_batch)), 5),
                "english_mass_abs_diff_mean": round(float(np.mean(np.abs(d_batch))), 5),
                "english_mass_rerun_abs_diff_mean": round(float(np.mean(np.abs(d_rerun))), 5),
                "generate_exact": round(float(np.mean([a == b for a, b in zip(gen_l, gen_b)])), 3),
                "generate_mean_matching_prefix": round(float(np.mean(prefix)), 2)}

    xl_plan = [xl_stage(name) for name in (stages or ()) if name.partition(".")[0] in XL_MODULES]

    plan = (
        ("load", 5 if is_mla else 2, load),
        ("sanity", lambda: max(1.0, forward_minutes(60)), sanity),
        ("bos_probe", lambda: bos_minutes(1), bos_probe),  # it drops modes itself to fit (see bos_probe)
        ("fit_router", 2, fit_router),
        ("fit_shard_j", 8, fit_shard("none")),
        ("pivot_logit", pivot_minutes, pivot_logit),
        ("fit_shard_r", 8, fit_shard("r+moe")),
        ("rows_probe", 6, rows_probe),
        ("fit_j_shard", 1.2 * fit_prompts / num_shards + 2, fit_j_shard),  # ~70 s/prompt at 64 rows
        ("fit_r_lens", 1.25 * fit_prompts + 3, fit_r_lens),
        ("fit_jpp", 6, fit_jpp),
        ("pivot_lenses", 5, pivot_lenses),
        ("pivot_all", 9, pivot_all),
        ("causal", 8, causal),
        ("swap", 6, swap),
        ("patch", patch_minutes(""), patch("")),
        ("patch_split", patch_minutes("split"), patch("split")),
        ("patch_rank", patch_minutes("rank"), patch("rank")),
        ("hindi_ablation", 3, hindi_ablation),
        ("sentences", 3, sentences),
        ("grade", 2, grade),
        ("causal_logit", causal_logit_minutes, causal_logit),
        *rq_plan,
        ("batch_check", lambda: forward_minutes(8 * 2 + 8 * 17 * 2, overhead_minutes=0.5), batch_check),
        # 48 lone + 8 rerun + 3 batched reads, 16 lone x 16-step decodes (~0.5 forward per cached step) + 1 batched
        ("batch_check_deep", lambda: forward_minutes(48 + 8 + 3 * 2 + 16 * 16 // 2 + 16 * 2, overhead_minutes=1.0), batch_check_deep),
        *xl_plan,  # in the order given on the command line
    )
    unknown = set(stages or ()) - {name for name, _, _ in plan}
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}")
    for name, est_minutes, fn in plan:
        if stages is None or name in stages:
            stage(name, est_minutes, fn)
    report["done"] = True
    save()
    return report
