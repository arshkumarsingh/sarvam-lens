"""The 22 scheduled languages in the word-translation pivot (docs/11): Wiktionary-sourced word lists for the 14 new languages, then the pivot with shuffled controls.

Interface (called by modal_pilot for a stage named "newlang_experiment.<analysis>"):
  run(ctx, analysis) -> dict          the stage's result (JSON-serialisable; written into report.json)
  n_forwards(ctx, analysis) -> float  single-forward equivalents, for the budget gate
ctx (built in modal_pilot.xl_ctx): model (jlens HFLensModel), model_kind ("sarvam30" | "sarvam105" |
"qwen"), model_dir, is_mla, n_layers, layers (0..n-2), band / early_band (the model's English band
and a matched early control), lens_specs + load_lenses(specs=None) (J-lens where one exists, then
logit), vocab_size, out_dir, data_dir, batch_size (RQ_BATCH: batch on every model), smoke (tiny
random model: keep everything tiny, skip correctness filters), args (--stage-args JSON), commit(),
seconds_per_forward. Shared helpers: languages.py (registry, scripts, digits), batch_gen.py
(batched greedy decoding), reader_questions.Reader (lens readouts, forward_batch).

Analyses:
- ``pivot``: :mod:`pivot_experiment`'s word-translation pivot on ``data/concepts_new.tsv`` (the
  original 11 columns plus 17 Wiktionary-sourced ones, ``data/concepts_new_sources.tsv``) over
  :data:`PAIRS`. Same prompts, token sets, roles, shuffled controls, correctness filter, readout
  filtering and output files (``rows.csv``, ``items.csv``, ``summary.csv``, plots) as
  ``pivot_experiment.main``, but every lens is read off ONE forward per item, ``batch_size`` items
  per forward (:meth:`reader_questions.Reader.forward_batch`), and all layers of a lens are scored
  at once. ``pivot_experiment.main`` runs one unbatched forward per lens per item, which on the
  105B (0.55 s/forward) would cost ~4x this stage. ``python newlang_experiment.py check <tiny dir>``
  checks the rows against ``pivot_experiment.main`` on a tiny model. The items the model gets wrong
  are greedy-decoded (``decode_steps``) and their answer matched against every column of the list,
  so a miss can be told apart: a listed variant's later token, the Hindi or English word, or
  something else.
  Writes ``newlang_summary.csv`` (per language and direction: items kept, i.e. the model's own
  accuracy, and band-mean English/Hindi top-10 rate and probability vs the shuffled controls, with
  95% bootstrap CIs over concepts) and returns the same table.
- ``causal``: the published English ablation (``causal_test.run``, unbatched: the English word's
  lens directions projected out at the English band vs a different concept's English vs an early
  band, plus negative steering at alpha 3) on :data:`CAUSAL_PAIRS`. Each pair runs on the concept list
  cut to its complete concepts. Writes each pair's ``causal_rows.csv`` and the pooled
  ``causal_newlang_summary.json`` (paired bootstrap CIs, in points).
- ``misses``: the pivot on :data:`MISS_PAIRS` (the targets the 30B mostly misses, plus English->X)
  with readout rows for the missed items too (``kept`` column), the misses decoded and typed by what the
  model wrote. Writes ``kept_missed.csv``: band-mean English, source, target and Hindi top-10 and
  probability, kept vs missed and by failure type (``python newlang_experiment.py misses-report``).

Stage args (``--stage-args``): ``pairs`` (comma list; default :data:`PAIRS`), ``lenses`` (specs;
default ``ctx.lens_specs``), ``batch_size``, ``decode`` ("all", "wrong" (default) or "none"),
``decode_steps`` (default 6), ``max_items`` (cap per pair, on candidate items), ``plots`` (default
true; the per-pair figures can be redrawn locally from ``summary.csv``).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch as t

import languages
import pivot_experiment as pe

logger = logging.getLogger("newlang")
HERE = Path(__file__).resolve().parent
CONCEPTS = HERE / "data/concepts_new.tsv"

NEW = languages.NEW_INDIC + languages.WORLD
# Each new language is a source once and a target once, against an original language of another
# family (Indo-Aryan <-> Dravidian; Tibeto-Burman / Austroasiatic <-> Indo-Aryan or Dravidian) and,
# where possible, another script, so the target's tokens don't collide with the source's or Hindi's.
# Dogri is left out: Wiktionary has 5 of the 32 words (2 identical to Hindi), and Wikidata has no
# Dogri labels (docs/11_new_languages.md).
CROSS_PAIRS = (
    "as-te,ta-as,or-ta,kn-or,pa-ml,te-pa,ur-ml,kn-ur,ne-ta,ml-ne,sa-kn,te-sa,mai-ml,kn-mai,"
    "sd-kn,ml-sd,ks-ta,te-ks,kok-te,ta-kok,mni-bn,te-mni,sat-hi,bn-sat,brx-ta,bn-brx,"
    "ja-ta,kn-ja,ru-ml,te-ru,ar-kn,ml-ar"
)
# new <-> new: the two rare native scripts both ways, and Perso-Arabic into Devanagari
NEW_NEW_PAIRS = "mni-sat,sat-mni,ur-ne"
# English -> X: the model's translation skill into the lowest-resource targets (and Japanese)
EN_PAIRS = "en-mni,en-sat,en-brx,en-ks,en-ja"
# original pairs from the published run, for a same-run, same-code baseline
BASELINE_PAIRS = "ta-te,bn-ta,te-gu"
PAIRS = ",".join((CROSS_PAIRS, NEW_NEW_PAIRS, EN_PAIRS, BASELINE_PAIRS))
# The causal check (analysis ``causal``): new-language pairs whose 30B pivot kept most items, one per class
CAUSAL_PAIRS = "or-ta,ur-ml,ks-ta,mni-bn,sat-hi,ja-ta"
CAUSAL_ALPHAS = (1.0, 3.0)  # 1: project the English directions out (published); 3: push them to -2x
# Is English there when translation FAILS? (analysis ``misses``): the targets the 30B mostly misses,
# from an Indian source and from English
MISS_PAIRS = "te-ks,bn-sat,mni-sat,te-mni,sat-mni,bn-brx,ta-kok,ml-sd,en-ks,en-sat,en-mni,en-brx"
MISS_ROLES = ("english", "english_shuffled", "source", "target", "hindi", "hindi_shuffled")
# a miss's decoded answer, by kind (miss_kind) -> the failure type reported
FAILURE_TYPE = {"source word": "source copy", "source script": "source copy", "other language's word": "neighbour language",
                "unlisted target-script word": "target script, unlisted",
                "listed target word": "listed word (first-token miss)"}
LATENT_ROLES = ("english", "english_shuffled", "hindi", "hindi_shuffled", "chinese", "chinese_shuffled", "target", "source")
# Measured on the 30B and 105B (docs/05): a 16-prompt forward costs ~2 single forwards; scoring
# one lens's layers for one item is ~10 ms once vectorised over layers.
BATCH_FORWARD_COST = 2.0
SCORE_SECONDS = 0.01
SETUP_SECONDS = 15.0  # vocab index (262k decodes), readout mask, token sets: 2.3 s for a whole tiny-model stage


### ITEMS


def build_items(concepts: dict, pairs: list[str], index: dict, tokenizer, *, min_chars: int = 2,
                max_items: int | None = None, kanji_rule: bool = True) -> tuple[list[dict], list[dict]]:
    """``pivot_experiment.main``'s items, in its order: (specs to forward, items skipped before any
    forward). Each spec carries the scored sets (roles, then the shuffled controls) and the raw target set.
    ``kanji_rule``: a pair with Japanese tracks no Chinese role (or Chinese control). Japanese words are
    mostly the Chinese column's characters (水, 山, 手, 猫), so the disjoint-set rule would drop most
    Japanese targets as shared with Chinese, and the two can't be told apart at the token level anyway."""
    specs, skipped = [], []
    ordered = sorted(concepts)
    for pair in pairs:
        src, tgt = pair.split("-")
        n = 0
        for concept in ordered:
            if max_items is not None and n >= max_items:
                break
            words = concepts[concept]
            roles = pe.tracked_languages(src, tgt)
            no_zh = kanji_rule and "ja" in (src, tgt)
            if no_zh:
                roles.pop("chinese", None)
            if any(lang not in words for lang in roles.values()):
                continue
            raw = {role: pe.prefix_token_ids(index, words[lang], tokenizer, min_chars=min_chars) for role, lang in roles.items()}
            sets, shared = pe.disjoint_sets(raw)
            control = ordered[(ordered.index(concept) + len(ordered) // 2) % len(ordered)]
            claimed = set().union(*raw.values())
            role_langs = dict(roles)
            scored = dict(sets)
            for control_role, lang in pe.CONTROL_ROLES.items():
                if lang == "zh" and no_zh:
                    continue
                if lang in concepts[control]:
                    scored[control_role] = pe.prefix_token_ids(index, concepts[control][lang], tokenizer, min_chars=min_chars) - claimed
                    role_langs[control_role] = f"{lang}:{control}"
            item = {"pair": pair, "concept": concept, "target_word": words[tgt][0], "n_shared_tokens": len(shared)}
            if not sets["target"]:
                skipped.append({**item, "kept": False, "reason": "target tokens all shared with another role"})
                continue
            n += 1
            specs.append({"item": item, "src": src, "tgt": tgt, "concept": concept, "sets": scored,
                          "role_langs": role_langs, "raw_target": raw["target"]})
    return specs, skipped


### READOUT


def score_layers(model, lenses: dict, layers: list[int], acts: dict[int, t.Tensor], role_ids: dict[str, set[int]],
                 mask_V: t.Tensor | None, vocab_size: int):
    """:meth:`reader_questions.Reader.score` with every layer of a lens decoded in one unembed:
    yields (lens, layer, role, prob, best_rank, top10) in the same order, one host sync per lens."""
    device = model._lm_head.weight.device
    index = {role: t.tensor(sorted(ids), dtype=t.long, device=device) for role, ids in role_ids.items() if ids}
    with t.no_grad():
        for name, lens in lenses.items():
            # layers sit on different GPUs under device_map="auto" (105B): gather on the LM head's
            h = t.cat([lens.transport(acts[layer][None], layer).to(device) for layer in layers])
            logits = model.unembed(h).float()[:, :vocab_size]
            if mask_V is not None:
                logits = logits.masked_fill(mask_V, float("-inf"))
            probs = t.softmax(logits, -1)
            stats = {}
            for role, ids in index.items():
                best = logits[:, ids].max(1).values
                stats[role] = t.stack([probs[:, ids].sum(1), ((logits > best[:, None]).sum(1) + 1).float()])
            host = {role: s.cpu().numpy() for role, s in stats.items()}
            for i, layer in enumerate(layers):
                for role in role_ids:
                    if role not in host:
                        yield name, layer, role, float("nan"), -1, False
                        continue
                    rank = int(host[role][1, i])
                    yield name, layer, role, float(host[role][0, i]), rank, rank <= 10


def answer_text(tokenizer, ids: list[int]) -> str:
    """The decoded answer up to its closing quote (or a line break)."""
    text = tokenizer.decode(ids, skip_special_tokens=True)
    for stop in ('"', "\n", "”"):
        text = text.split(stop)[0]
    return unicodedata.normalize("NFC", text.strip())


def match_answer(answer: str, concepts: dict, concept: str, tgt: str) -> str:
    """Which listed word the decoded answer is: the target list (a variant whose later tokens
    differ), another column's word for the same concept (``hi``, ``en``...), the target-language word
    of ANOTHER concept, or nothing listed."""
    key = pe.normalize_token_text(answer)
    if not key:
        return "empty"
    norm = lambda words: {pe.normalize_token_text(w) for w in words}
    listed = norm(concepts[concept].get(tgt, []))
    if key in listed:
        return "target"
    if any(len(w) >= 2 and key.startswith(w) for w in listed):  # an inflected form (Sanskrit जलम् for जल)
        return "target_inflected"
    hits = [lang for lang, words in concepts[concept].items() if lang != tgt and key in norm(words)]
    if hits:
        return "same_concept:" + "|".join(sorted(hits))
    others = [c for c in concepts if c != concept and key in norm(concepts[c].get(tgt, []))]
    if others:
        return "other_concept:" + others[0]
    return "unlisted"


def pivot(model, lenses: dict, *, concepts_path: Path, pairs: list[str], layers: list[int], vocab_size: int,
          out_dir: Path, batch_size: int = 16, require_correct: bool = True, max_items: int | None = None,
          decode_steps: int = 6, decode: str = "wrong", min_chars: int = 2, kanji_rule: bool = True,
          plots: bool = True, rows_for_missed: bool = False, commit=None) -> tuple[list[dict], list[dict]]:
    """The pivot over ``pairs``; writes ``rows.csv``, ``items.csv``, ``summary.csv`` and plots in
    ``pivot_experiment``'s format and returns (rows, items). ``rows_for_missed`` also reads out the
    items the correctness filter drops: every row then carries ``kept``, and ``summary.csv`` and the
    plots still use the kept rows only."""
    import batch_gen
    import reader_questions as rqm

    tokenizer = model.tokenizer
    concepts = pe.load_concepts(concepts_path)
    index = pe.build_vocab_index(tokenizer, vocab_size)
    tracked: set[int] = set()
    for concept in concepts.values():
        for spellings in concept.values():
            tracked |= pe.prefix_token_ids(index, spellings, tokenizer, min_chars=min_chars)
    mask_V = pe.readout_mask(tokenizer, vocab_size, tracked)
    reader = rqm.Reader(model, lenses, layers, vocab_size, mask_V)
    specs, skipped = build_items(concepts, pairs, index, tokenizer, min_chars=min_chars, max_items=max_items,
                                 kanji_rule=kanji_rule)
    logger.info("%d items to read over %d pairs (%d skipped: target tokens all shared)", len(specs), len(pairs), len(skipped))
    last_layer = model.n_layers - 1
    rows: list[dict] = []
    results: list[dict] = []
    for start in range(0, len(specs), batch_size):
        chunk = specs[start: start + batch_size]
        ids_list = [model.encode(pe.build_prompt(concepts, s["concept"], s["src"], s["tgt"], shots=5, seed=0)) for s in chunk]
        for spec, ids, (acts, logits) in zip(chunk, ids_list, reader.forward_batch(ids_list)):
            top = int(logits.argmax())
            item = {**spec["item"], "model_top_token": tokenizer.decode([top]),
                    "model_top5": " | ".join(tokenizer.decode([i]) for i in logits[:vocab_size].topk(5).indices.tolist())}
            correct = top in spec["raw_target"]
            results.append({"spec": spec, "ids": ids, "item": item, "correct": correct})
            kept = correct or not require_correct
            if not kept and not rows_for_missed:
                continue
            extra = {"kept": kept} if rows_for_missed else {}
            for lens_name, layer, role, prob, rank, top10 in score_layers(model, lenses, layers, acts, spec["sets"], reader.mask, vocab_size):
                rows.append({"pair": spec["item"]["pair"], "src": spec["src"], "tgt": spec["tgt"], "concept": spec["concept"],
                             "lens": lens_name, "layer": layer, "depth": layer / last_layer, "role": role,
                             "lang": spec["role_langs"][role], "prob": prob, "best_rank": rank, "top10": top10, **extra})
        if start // batch_size % 20 == 0:
            logger.info("read %d / %d items", min(start + batch_size, len(specs)), len(specs))
    # What the model actually wrote (``decode``: "all", "wrong" or "none"): its first token can be a
    # bare letter that begins many words (Ol Chiki, Tamil), so the whole answer is matched against the
    # list, and a miss can be told apart (a later-token variant, the Hindi/English word, or other).
    todo = [] if decode == "none" or not decode_steps else results if decode == "all" else [r for r in results if not r["correct"]]
    if todo:
        outs = batch_gen.generate(model, [r["ids"] for r in todo], max_new_tokens=decode_steps, batch_size=batch_size)
        for r, out in zip(todo, outs):
            r["item"]["model_answer"] = answer_text(tokenizer, out)
            r["item"]["answer_match"] = match_answer(r["item"]["model_answer"], concepts, r["spec"]["concept"], r["spec"]["tgt"])
    items = [{**r["item"], "kept": r["correct"] or not require_correct,
              "reason": "" if r["correct"] or not require_correct else "model's next token does not begin the target word"}
             for r in results] + skipped
    indic_pairs = {p for p in pairs if all(pe.FAMILY.get(l) in ("IA", "Dr") and l not in ("hi", "hi_latn") for l in p.split("-"))}
    summary = pe.summarize([r for r in rows if r.get("kept", True)], indic_pairs)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("rows.csv", rows), ("items.csv", items), ("summary.csv", summary)):
        write_csv(out_dir / name, table)
    if commit:
        commit()
    for lens_name in lenses if plots else ():
        for metric in ("mean_prob", "top10_rate"):
            pe.plot_lens(summary, lens_name, metric, out_dir / f"{lens_name}_{metric}.png", pairs)
    return rows, items


def write_csv(path: Path, table: list[dict]) -> None:
    if not table:
        return
    fields = list(dict.fromkeys(k for row in table for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)


### PER-LANGUAGE SUMMARY


def groups_of(pair: str) -> list[str]:
    """The summary groups an item counts in: ``<lang>>`` (language as source), ``><lang>``
    (as target, from another Indian or world language), ``en><lang>``, and ``baseline``."""
    src, tgt = pair.split("-")
    if pair in BASELINE_PAIRS.split(","):
        return ["baseline"]
    if src == "en":
        return [f"en>{tgt}"]
    return [g for g in (f"{src}>" if src in NEW else None, f">{tgt}" if tgt in NEW else None) if g]


# Pooled classes (docs/11): by script and resource level of the new endpoint.
CLASSES = {
    "new Indo-Aryan, Brahmic script": ("as", "or", "pa", "ne", "sa", "mai", "kok"),
    "Perso-Arabic script (ur, sd, ks)": ("ur", "sd", "ks"),
    "own script, non-Indo-Aryan (mni Meitei Mayek, sat Ol Chiki)": ("mni", "sat"),
    "Bodo (Devanagari, Tibeto-Burman)": ("brx",),
    "world (ja, ru, ar)": ("ja", "ru", "ar"),
}


def class_groups_of(pair: str) -> list[str]:
    """:func:`groups_of` pooled by :data:`CLASSES`: ``<class> >`` (as source) and ``> <class>``."""
    src, tgt = pair.split("-")
    if pair in BASELINE_PAIRS.split(","):
        return ["baseline"]
    if src == "en":
        return [f"en > {name}" for name, langs in CLASSES.items() if tgt in langs]
    return ([f"{name} >" for name, langs in CLASSES.items() if src in langs]
            + [f"> {name}" for name, langs in CLASSES.items() if tgt in langs])


def cluster_ci(values_by_concept: dict[str, list[float]], *, resamples: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI of the item mean, resampling concepts (an item's pairs share its concept)."""
    keys = sorted(values_by_concept)
    if len(keys) < 2:
        return float("nan"), float("nan")
    sums = np.array([np.sum(values_by_concept[k]) for k in keys])
    counts = np.array([len(values_by_concept[k]) for k in keys])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(keys), size=(resamples, len(keys)))
    means = sums[draws].sum(1) / np.maximum(counts[draws].sum(1), 1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def language_table(rows: list[dict], items: list[dict], band: list[int], layers: list[int], groups=None) -> list[dict]:
    """Per group and lens: items kept (the model's own accuracy), then for each latent role the
    band-mean top-10 rate and probability (each item's mean over the band layers; CI over
    concepts; items whose set for that role is empty, e.g. a Devanagari target's Hindi tokens
    all shared, are left out and counted in ``<role>_n``), the English onset (first layer whose
    top-10 rate reaches 0.5) and the English peak layer. ``groups``: pair -> group names
    (default :func:`groups_of`; :func:`class_groups_of` pools by class)."""
    groups = groups or groups_of
    rows = [r for r in rows if str(r.get("kept", True)) != "False"]  # readouts of kept items only
    tested = defaultdict(lambda: [0, 0])
    exact = defaultdict(lambda: [0, 0])  # decoded answers that are a listed target spelling
    for item in items:
        if item.get("reason") == "target tokens all shared with another role":
            continue
        for g in groups(item["pair"]):
            tested[g][0] += item["kept"] in (True, "True")
            tested[g][1] += 1
            if item.get("answer_match"):
                exact[g][0] += item["answer_match"] in ("target", "target_inflected")
                exact[g][1] += 1
    per_item = defaultdict(lambda: defaultdict(list))  # (group, lens, role) -> concept -> [(top10, prob)]
    per_layer = defaultdict(list)  # (group, lens, role, layer) -> top10s
    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["pair"], r["concept"], r["lens"], r["role"])].append(r)
    for (pair, concept, lens, role), rr in by_key.items():
        in_band = [r for r in rr if int(r["layer"]) in band]
        if not in_band or int(in_band[0]["best_rank"]) < 0:  # empty set: no score
            continue
        top10 = float(np.mean([str(r["top10"]) == "True" for r in in_band]))
        prob = float(np.mean([float(r["prob"]) for r in in_band]))
        for g in groups(pair):
            per_item[(g, lens, role)][concept].append((top10, prob))
            for r in rr:
                per_layer[(g, lens, role, int(r["layer"]))].append(str(r["top10"]) == "True")
    lenses = sorted({r["lens"] for r in rows})
    out = []
    for g in sorted(tested, key=group_order) if groups is groups_of else list(dict.fromkeys(tested)):
        kept, n = tested[g]
        for lens in lenses:
            row = {"group": g, "lens": lens, "band": f"L{band[0]}-{band[-1]}", "kept": kept, "tested": n,
                   "accuracy": round(kept / n, 3) if n else float("nan")}
            if exact[g][1]:
                row["exact_word"] = f"{exact[g][0]}/{exact[g][1]}"
            for role in LATENT_ROLES:
                cells = per_item.get((g, lens, role))
                if not cells:
                    continue
                for j, metric in enumerate(("top10", "prob")):
                    by_concept = {c: [v[j] for v in vals] for c, vals in cells.items()}
                    flat = [v for vals in by_concept.values() for v in vals]
                    lo, hi = cluster_ci(by_concept)
                    row[f"{role}_{metric}"] = round(float(np.mean(flat)), 4)
                    row[f"{role}_{metric}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
                row[f"{role}_n"] = sum(len(v) for v in cells.values())
            rate = {l: float(np.mean(per_layer[(g, lens, "english", l)])) for l in layers if per_layer.get((g, lens, "english", l))}
            if rate:
                onset = next((l for l in sorted(rate) if rate[l] >= 0.5), None)
                peak = max(rate, key=rate.get)
                row["english_onset"] = "none" if onset is None else f"L{onset}"
                row["english_peak"] = f"{rate[peak]:.2f} @L{peak}"
            out.append(row)
    return out


def group_order(g: str) -> tuple:
    lang = g.replace("en>", "").strip(">")
    order = list(NEW) + ["baseline"]
    return (order.index(lang) if lang in order else len(order), g.startswith("en>"), g.startswith(">"))


### MODAL HOOK


def _pairs(ctx) -> list[str]:
    pairs = ctx.args.get("pairs") or PAIRS
    pairs = [p for p in pairs.split(",") if p]
    return pairs[:2] if ctx.smoke else pairs


def run(ctx, analysis: str) -> dict:
    if analysis == "hello":  # the hook's own check (modal_pilot dry run)
        return {"model_kind": ctx.model_kind, "lens_specs": ctx.lens_specs, "n_layers": ctx.n_layers,
                "band": ctx.band, "batch_size": ctx.batch_size, "args": ctx.args}
    if analysis == "causal":
        return run_causal(ctx)
    if analysis == "misses":
        return run_misses(ctx)
    if analysis == "geometry":
        return run_geometry(ctx)
    if analysis != "pivot":
        raise ValueError(f"unknown analysis {analysis!r}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    specs = ctx.args.get("lenses") or ctx.lens_specs
    lenses = ctx.load_lenses(specs)
    pairs = _pairs(ctx)
    out = Path(ctx.out_dir)
    rows, items = pivot(
        ctx.model, lenses, concepts_path=Path(ctx.data_dir) / "concepts_new.tsv", pairs=pairs, layers=ctx.layers,
        vocab_size=ctx.vocab_size, out_dir=out, batch_size=int(ctx.args.get("batch_size", ctx.batch_size)),
        require_correct=not ctx.smoke, max_items=2 if ctx.smoke else ctx.args.get("max_items"),
        decode_steps=int(ctx.args.get("decode_steps", 2 if ctx.smoke else 6)), decode=ctx.args.get("decode", "wrong"),
        plots=bool(ctx.args.get("plots", True)), commit=ctx.commit)
    table = language_table(rows, items, ctx.band, ctx.layers)
    write_csv(out / "newlang_summary.csv", table)
    ctx.commit()
    tested = [i for i in items if i.get("reason") != "target tokens all shared with another role"]
    misses = defaultdict(int)
    for i in tested:
        if i.get("answer_match"):
            misses[i["answer_match"].split(":")[0]] += 1
    return {"pairs": len(pairs), "lenses": list(lenses), "band": ctx.band, "items_tested": len(tested),
            "items_kept": sum(1 for i in tested if i["kept"]), "skipped_shared": len(items) - len(tested),
            "miss_answers": dict(misses), "out": str(out),
            "by_group": [{k: row[k] for k in ("group", "lens", "kept", "tested", "english_top10", "english_shuffled_top10",
                                              "hindi_top10", "english_onset") if k in row} for row in table]}


def count_items(concepts_path: Path, pairs: list[str]) -> int:
    """Candidate items (both words and the tracked roles' words listed), before the token-set step."""
    concepts = pe.load_concepts(concepts_path)
    n = 0
    for pair in pairs:
        src, tgt = pair.split("-")
        langs = set(pe.tracked_languages(src, tgt).values())
        n += sum(all(lang in words for lang in langs) for words in concepts.values())
    return n


def n_forwards(ctx, analysis: str) -> float:
    if analysis == "causal":
        pairs = _causal_pairs(ctx)
        n_items = 2 * len(pairs) if ctx.smoke else count_items(Path(ctx.data_dir) / "concepts_new.tsv", pairs)
        setup = 30.0 / max(ctx.seconds_per_forward, 1e-3)  # a vocab index per pair
        if _causal_batched(ctx):
            # base + every condition as rows of ONE forward per item. Measured on the 105B (rq105c's
            # causal_steer, 8 rows per item): 155 items in 0.7 min, ~0.27 s = ~0.5 forward per item.
            return n_items * 0.6 + setup
        # unbatched: a base forward per item, then one per condition for the ~85% kept
        conditions = 3 + 2 * sum(a != 1.0 for a in CAUSAL_ALPHAS)
        return n_items * (1 + 0.85 * conditions) + setup
    if analysis == "geometry":
        spf = max(ctx.seconds_per_forward, 1e-3)
        batch = int(ctx.args.get("batch_size", ctx.batch_size))
        cells = 40 if ctx.smoke else len(geometry_cells(pe.load_concepts(Path(ctx.data_dir) / "concepts_new.tsv")))
        carriers = len(ctx.args.get("carriers") or CARRIERS)
        vectors = carriers * math.ceil(cells / batch) * BATCH_FORWARD_COST
        fair, setup = 0.0, 20.0  # without the fair readout: saving the vectors; the metrics take ~1 s per carrier
        if ctx.args.get("fair", ctx.model_kind != "qwen"):
            setup = 60.0  # + the vocab index and script masks (262k decodes each)
            items = 4 if ctx.smoke else count_items(Path(ctx.data_dir) / "concepts_new.tsv",
                                                    list(dict.fromkeys((ORIGINAL_INDIC_PAIRS + "," + PAIRS).split(","))))
            n_lenses = len(ctx.args.get("lenses") or ctx.lens_specs)
            fair = math.ceil(items / batch) * BATCH_FORWARD_COST + items * n_lenses * 0.03 / spf
        return vectors + fair + setup / spf
    if analysis not in ("pivot", "misses"):
        return 1.0
    misses = analysis == "misses"
    pairs = _miss_pairs(ctx) if misses else _pairs(ctx)
    n_items = 2 * len(pairs) if ctx.smoke else count_items(Path(ctx.data_dir) / "concepts_new.tsv", pairs)
    batch = int(ctx.args.get("batch_size", ctx.batch_size))
    n_lenses = len(ctx.args.get("lenses") or ctx.lens_specs)
    spf = max(ctx.seconds_per_forward, 1e-3)
    per_forward = BATCH_FORWARD_COST if batch > 1 else 1.0
    reads = math.ceil(n_items / batch) * per_forward
    # decoding (all items, or the misses: half by default, ~70% on the misses analysis's pairs). Per
    # batch, a prefill and each cached step cost ~1.5 batched forwards (xl_new_30b: 1,208 items decoded
    # for 8 steps took most of its 5.9-minute stage, ~3.2 s per 16-item batch).
    steps = int(ctx.args.get("decode_steps", 2 if ctx.smoke else 8 if misses else 6))
    mode = "wrong" if misses else ctx.args.get("decode", "wrong")
    share = {"all": 1.0, "wrong": 0.7 if misses else 0.5, "none": 0.0}[mode]
    decode = math.ceil(n_items * share / batch) * per_forward * (1 + steps) * 1.5 if steps else 0
    scoring = (n_items * n_lenses * SCORE_SECONDS + SETUP_SECONDS) / spf
    return reads + decode + scoring


### CAUSAL CHECK


def _causal_pairs(ctx) -> list[str]:
    pairs = [p for p in (ctx.args.get("pairs") or CAUSAL_PAIRS).split(",") if p]
    return pairs[:1] if ctx.smoke else pairs


def _concepts_for(pair: str, src_path: Path, out: Path) -> Path:
    """``concepts_new.tsv`` cut to the concepts with a word in the pair's languages and English:
    :func:`causal_test.run` indexes every concept's target word. The demos and the control concept
    are then exactly the pivot's for a complete pair (an incomplete concept is never a demo)."""
    src, tgt = pair.split("-")
    with src_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields, rows = reader.fieldnames, [r for r in reader if all((r.get(l) or "").strip() for l in (src, tgt, "en"))]
    path = out / f"concepts_{pair}.tsv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def paired_drop(rows: list[dict], cond: str, *, resamples: int = 2000, seed: int = 0) -> dict:
    """Mean drop in P(target) under ``cond`` and its paired 95% bootstrap CI, in points (x100),
    plus the share still correct (``causal_test.run``'s statistic, pooled over pairs)."""
    drop = np.array([float(r["base_p"]) - float(r[f"{cond}_p"]) for r in rows])
    if not len(drop):
        return {"n": 0}
    boots = np.random.default_rng(seed).choice(drop, size=(resamples, len(drop)), replace=True).mean(1)
    out = {"n": len(drop), "drop_pts": round(100 * float(drop.mean()), 2),
           "ci95_pts": [round(100 * float(np.percentile(boots, 2.5)), 2), round(100 * float(np.percentile(boots, 97.5)), 2)]}
    if f"{cond}_correct" in rows[0]:
        out["still_correct"] = round(float(np.mean([str(r[f"{cond}_correct"]) == "True" for r in rows])), 3)
    return out


def causal_tables(rows: list[dict]) -> dict:
    conds = ["english_mid", "control_mid", "english_early"] + [f"{n}@{a:g}" for a in CAUSAL_ALPHAS if a != 1.0
                                                                for n in ("english_mid", "control_mid")]
    conds = [c for c in conds if rows and f"{c}_p" in rows[0]]
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["pair"]].append(r)
    table = {pair: {"base_p": round(float(np.mean([float(r["base_p"]) for r in rr])), 3),
                    **{c: paired_drop(rr, c) for c in conds}} for pair, rr in by_pair.items()}
    # English's drop minus the control's drop, per item (paired_drop takes base_p - <cond>_p)
    extra = [dict(r, extra_p=float(r["base_p"]) - (float(r["control_mid_p"]) - float(r["english_mid_p"]))) for r in rows]
    table["ALL"] = {"base_p": round(float(np.mean([float(r["base_p"]) for r in rows])), 3) if rows else None,
                    **{c: paired_drop(rows, c) for c in conds},
                    "english_minus_control": paired_drop(extra, "extra") if rows else {}}
    return table


def _causal_batched(ctx) -> bool:
    """Conditions as rows of one forward (``causal_test.run(batch_conditions=True)``): the default on
    the 105B, whose forward is launch-bound (batched and unbatched ablations agree), never on
    the 30B, whose published ablation is unbatched."""
    return bool(ctx.args.get("batch_conditions", ctx.is_mla))


def run_causal(ctx) -> dict:
    """The published English ablation (``causal_test.run``: the lens directions of the English word's
    tokens projected out of the final position at the English band, vs a different concept's English
    word in a same-size subspace, vs an early band) on new-language pairs, plus negative steering.
    The lens is ``ctx.lens_specs[0]``: the J-lens on the 30B, the logit lens on the 105B (which has no
    fitted lens)."""
    import causal_test

    spec = ctx.args.get("lens") or ctx.lens_specs[0]
    name, lens = next(iter(ctx.load_lenses([spec]).items()))
    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for pair in _causal_pairs(ctx):
        causal_test.run(ctx.model, lens, concepts_path=str(_concepts_for(pair, Path(ctx.data_dir) / "concepts_new.tsv", out)),
                        pairs=pair, mid_layers=ctx.band, early_layers=ctx.early_band, out_dir=str(out / pair),
                        vocab_size=ctx.vocab_size, require_correct=not ctx.smoke, latent_lang="en",
                        alphas=CAUSAL_ALPHAS if not ctx.smoke else (1.0,), batch_conditions=_causal_batched(ctx))
        path = out / pair / "causal_rows.csv"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                rows += list(csv.DictReader(handle))
        ctx.commit()
    table = causal_tables(rows)
    (out / "causal_newlang_summary.json").write_text(json.dumps(table, indent=1))
    ctx.commit()
    return {"lens": name, "mid_layers": ctx.band, "early_layers": ctx.early_band, "batched": _causal_batched(ctx),
            "n_items": len(rows), "by_pair": table}


### IS ENGLISH THERE WHEN TRANSLATION FAILS?


def _miss_pairs(ctx) -> list[str]:
    pairs = [p for p in (ctx.args.get("pairs") or MISS_PAIRS).split(",") if p]
    return pairs[:2] if ctx.smoke else pairs


def item_status(item: dict) -> str:
    if str(item.get("kept")) == "True":
        return "kept"
    if not item.get("model_answer") and not item.get("answer_match"):
        return "missed, not decoded"
    return "missed: " + FAILURE_TYPE.get(miss_kind(item), "other")


def kept_missed_table(rows: list[dict], items: list[dict], band: list[int]) -> list[dict]:
    """Band-mean top-10 rate and probability per role (:data:`MISS_ROLES`), for each target language
    and source kind (Indian/world vs English) split by item status: kept, missed (all), and missed
    by failure type. 95% bootstrap CIs over concepts; ``<role>_n`` items have a non-empty set."""
    status = {}
    for item in items:
        if item.get("reason") == "target tokens all shared with another role":
            continue
        status[(item["pair"], item["concept"])] = item_status(item)
    band_vals = defaultdict(lambda: defaultdict(list))  # (pair, concept) -> (lens, role) -> band rows
    for r in rows:
        if int(r["layer"]) in band:
            band_vals[(r["pair"], r["concept"])][(r["lens"], r["role"])].append(r)
    per = defaultdict(lambda: defaultdict(list))  # (group, lens, role) -> concept -> [(top10, prob)]
    counts = defaultdict(int)
    for (pair, concept), st in status.items():
        src, tgt = pair.split("-")
        kind = "English" if src == "en" else "Indian/world"
        groups = [(tgt, kind, st)] + ([(tgt, kind, "missed")] if st != "kept" else [])
        groups += [("ALL", kind, g[2]) for g in groups]
        for g in groups:
            counts[g] += 1
        for (lens, role), rr in band_vals.get((pair, concept), {}).items():
            if int(rr[0]["best_rank"]) < 0:  # empty set: no score
                continue
            value = (float(np.mean([str(r["top10"]) == "True" for r in rr])), float(np.mean([float(r["prob"]) for r in rr])))
            for g in groups:
                per[(g, lens, role)][concept].append(value)
    lenses = sorted({r["lens"] for r in rows})
    order = lambda g: (g[0] == "ALL", (NEW + ("ALL",)).index(g[0]) if g[0] in NEW + ("ALL",) else 99, g[1], g[2] != "kept", g[2])
    out = []
    for g in sorted(counts, key=order):
        for lens in lenses:
            row = {"target": g[0], "from": g[1], "status": g[2], "lens": lens, "items": counts[g]}
            for role in MISS_ROLES:
                cells = per.get((g, lens, role))
                if not cells:
                    continue
                for j, metric in enumerate(("top10", "prob")):
                    by_concept = {c: [v[j] for v in vals] for c, vals in cells.items()}
                    lo, hi = cluster_ci(by_concept)
                    row[f"{role}_{metric}"] = round(float(np.mean([v for vals in by_concept.values() for v in vals])), 4)
                    row[f"{role}_{metric}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
                row[f"{role}_n"] = sum(len(v) for v in cells.values())
            out.append(row)
    return out


def run_misses(ctx) -> dict:
    """The pivot on :data:`MISS_PAIRS` with readout rows for the items the model gets wrong as well,
    decoding the misses to type them; writes ``kept_missed.csv`` (:func:`kept_missed_table`)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    lenses = ctx.load_lenses(ctx.args.get("lenses") or ctx.lens_specs)
    out = Path(ctx.out_dir)
    rows, items = pivot(
        ctx.model, lenses, concepts_path=Path(ctx.data_dir) / "concepts_new.tsv", pairs=_miss_pairs(ctx), layers=ctx.layers,
        vocab_size=ctx.vocab_size, out_dir=out, batch_size=int(ctx.args.get("batch_size", ctx.batch_size)),
        require_correct=True, max_items=2 if ctx.smoke else None, decode_steps=int(ctx.args.get("decode_steps", 2 if ctx.smoke else 8)),
        decode="wrong", plots=False, rows_for_missed=True, commit=ctx.commit)
    table = kept_missed_table(rows, items, ctx.band)
    write_csv(out / "kept_missed.csv", table)
    ctx.commit()
    keep = ("target", "from", "status", "lens", "items", "english_top10", "english_shuffled_top10", "source_top10", "target_top10",
            "hindi_top10")
    return {"pairs": len(_miss_pairs(ctx)), "lenses": list(lenses), "band": ctx.band, "items": len(items),
            "kept": sum(1 for i in items if str(i.get("kept")) == "True"),
            "by_group": [{k: r[k] for k in keep if k in r} for r in table if r["target"] == "ALL"]}


def misses_report(run_dir: str, band: list[int]) -> str:
    """Markdown tables for the ``misses`` analysis from a downloaded run."""
    run = Path(run_dir)
    with (run / "rows.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (run / "items.csv").open(encoding="utf-8") as handle:
        items = list(csv.DictReader(handle))
    table = kept_missed_table(rows, items, band)
    lines = [f"Band L{band[0]}–{band[-1]}: band-mean top-10 rate [95% CI over concepts] (probability).", "",
             "| target | from | status | lens | items | English | English, other concept | source word | target word | Hindi | Hindi, other concept |",
             "|" + "---|" * 11]
    for r in table:
        cell = lambda role: (f"{r[role + '_top10']:.2f} {r[role + '_top10_ci'].replace(' ', '')} ({r[role + '_prob']:.3f})"
                             if role + "_top10" in r else "–")
        short = lambda role: f"{r[role + '_top10']:.2f}" if role + "_top10" in r else "–"
        lines.append(f"| {r['target']} | {r['from']} | {r['status']} | {r['lens']} | {r['items']} | {cell('english')} | "
                     f"{short('english_shuffled')} | {cell('source')} | {cell('target')} | {cell('hindi')} | {short('hindi_shuffled')} |")
    return "\n".join(lines)


### CONCEPT GEOMETRY: LANGUAGE-NEUTRAL OR ENGLISH-CENTRIC?

GEOM_LANGS = ("en",) + languages.SCHEDULED + ("zh", "ja", "ru", "ar")
CARRIERS = ("word", "line")  # the word alone after BOS; the pivot's source line `label: "word` (no target)
# the published pivot's 10 Indic->Indic pairs (ALL_INDIC), for the vocabulary-fair readout
ORIGINAL_INDIC_PAIRS = "ta-te,te-kn,kn-ml,ml-ta,bn-gu,gu-mr,mr-bn,bn-ta,ta-bn,te-gu"
FAIR_ROLES = ("english", "english_shuffled", "hindi", "hindi_shuffled", "target", "source")


def _lead_ids(model) -> list[int]:
    """What precedes a carrier: nothing where ``encode`` adds the model's BOS (Sarvam's [@BOS@]),
    else Qwen's document separator <|endoftext|>, so a one-token word never sits at position 0
    (an attention sink, whose residual is degenerate)."""
    tok = model.tokenizer
    probe = model.encode("a")[0].tolist()
    if tok.bos_token_id is not None and probe and probe[0] == tok.bos_token_id:
        return []
    sep = tok.convert_tokens_to_ids("<|endoftext|>")
    return [sep] if isinstance(sep, int) and sep >= 0 and sep != tok.unk_token_id else []


def geometry_cells(concepts: dict) -> list[tuple[str, str, str]]:
    """(concept, lang, word) for every listed word in :data:`GEOM_LANGS` (first spelling)."""
    return [(c, lang, concepts[c][lang][0]) for c in sorted(concepts) for lang in GEOM_LANGS if lang in concepts[c]]


def collect_vectors(model, cells, carrier: str, batch_size: int) -> tuple[t.Tensor, list[int]]:
    """``[n_cells, n_layers, d]`` fp16: the residual after every block at the word's last token, in
    one carrier (right-padded batches, read at each row's last real token), and each word's token count."""
    import reader_questions as rqm

    lead = _lead_ids(model)
    layers = list(range(model.n_layers))
    reader = rqm.Reader(model, {}, layers, 1, None)
    ids_list, n_tokens = [], []
    for _, lang, word in cells:
        text = word if carrier == "word" else f'{languages.LANGS[lang]["label"]}: "{word}'
        ids = model.encode(text)
        if lead:
            ids = t.cat([t.tensor([lead], dtype=ids.dtype, device=ids.device), ids], dim=1)
        ids_list.append(ids)
        n_tokens.append(len(model.tokenizer(word, add_special_tokens=False).input_ids))
    out = None
    for start in range(0, len(ids_list), batch_size):
        for row, (acts, _) in enumerate(reader.forward_batch(ids_list[start: start + batch_size])):
            stack = t.stack([acts[l].float().cpu() for l in layers]).half()
            if out is None:
                out = t.empty((len(ids_list), len(layers), stack.shape[-1]), dtype=t.float16)
            out[start + row] = stack
    return out, n_tokens


def geometry_metrics(V: t.Tensor, cells, *, resamples: int = 1000, seed: int = 0, device: str = "cpu") -> dict:
    """Per layer, from ``V`` (:func:`collect_vectors`), with 95% bootstrap CIs over concepts:
    - retrieval: each language's vectors minus that language's mean, unit-normed; for each ordered
      language pair (A, B) and concept c in both, is c's nearest neighbour among B's concepts c?
      Pooled top-1 accuracy, chance (mean 1/|B's concepts|), and the full A x B accuracy matrix.
    - centroid proximity (centred, and raw unit vectors): for each language X, the mean cosine of its
      concept vectors to the leave-X-out cross-language centroid of the same concept; the languages'
      rank; English minus Hindi, paired over shared concepts."""
    concepts = sorted({c for c, _, _ in cells})
    langs = [l for l in GEOM_LANGS if any(lang == l for _, lang, _ in cells)]
    C, K = len(concepts), len(langs)
    ci_, li_ = {c: i for i, c in enumerate(concepts)}, {l: j for j, l in enumerate(langs)}
    present = t.zeros((K, C), dtype=t.bool)
    pos = t.full((K, C), -1, dtype=t.long)
    for row, (c, lang, _) in enumerate(cells):
        present[li_[lang], ci_[c]] = True
        pos[li_[lang], ci_[c]] = row
    rng = np.random.default_rng(seed)
    draws = t.tensor(rng.integers(0, C, size=(resamples, C)))
    present_d, eye = present.to(device), t.eye(K, dtype=t.bool, device=device)
    n_b = present.sum(1).float().to(device)  # concepts per language
    layers_out = []

    def boot(num_c: t.Tensor, den_c: t.Tensor) -> list[float]:
        num, den = num_c.cpu()[draws].sum(1), den_c.cpu()[draws].sum(1).clamp(min=1)
        vals = (num / den).numpy()
        return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

    for layer in range(V.shape[1]):
        X = V[:, layer, :].float().to(device)
        U = t.zeros((K, C, X.shape[1]), device=device)
        R = t.zeros_like(U)
        for j in range(K):
            rows = pos[j][present[j]].to(device)
            Xj = X[rows]
            U[j, present_d[j]] = t.nn.functional.normalize(Xj - Xj.mean(0, keepdim=True), dim=-1)
            R[j, present_d[j]] = t.nn.functional.normalize(Xj, dim=-1)
        # retrieval
        sims = t.einsum("acd,bed->abce", U, U)
        sims = sims.masked_fill(~present_d[None, :, None, :], float("-inf"))
        pred = sims.argmax(-1)  # [K, K, C]
        arange_c = t.arange(C, device=device)
        valid = present_d[:, None, :] & present_d[None, :, :] & ~eye[:, :, None]
        correct = (pred == arange_c[None, None, :]) & valid
        chance = (valid.float() / n_b[None, :, None]).sum() / valid.sum()
        acc_matrix = (correct.sum(-1).float() / valid.sum(-1).clamp(min=1)).cpu()
        num_c, den_c = correct.sum((0, 1)).float(), valid.sum((0, 1)).float()
        retrieval = {"acc": round(float(num_c.sum() / den_c.sum()), 4), "ci95": boot(num_c, den_c),
                     "chance": round(float(chance), 4)}
        # centroid proximity
        prox = {}
        for name, M in (("centred", U), ("raw", R)):
            S, n = M.sum(0), present_d.sum(0).float()  # [C, d], [C]
            cos = t.zeros((K, C), device=device)
            ok = present_d & (n[None, :] >= 2)
            for j in range(K):
                cen = t.nn.functional.normalize((S - M[j]) / (n - 1).clamp(min=1)[:, None], dim=-1)
                cos[j] = (M[j] * cen).sum(-1)
            vals = cos.masked_fill(~ok, 0.0).cpu()
            okc = ok.cpu()
            per_lang = {}
            for j, lang in enumerate(langs):
                num, den = vals[j], okc[j].float()
                per_lang[lang] = {"mean": round(float(num.sum() / den.sum().clamp(min=1)), 4), "n": int(den.sum()),
                                  "ci95": boot(num, den)}
            ranked = sorted(langs, key=lambda l: -per_lang[l]["mean"])
            en, hi = li_.get("en"), li_.get("hi")
            both = okc[en] & okc[hi] if en is not None and hi is not None else None
            diff = (vals[en] - vals[hi]) * both if both is not None else None
            prox[name] = {"by_lang": per_lang, "order": ranked, "en_rank": ranked.index("en") + 1 if "en" in ranked else None,
                          "en_minus_hi": ({"mean": round(float(diff.sum() / both.sum()), 4), "ci95": boot(diff, both.float())}
                                          if diff is not None else None)}
        layers_out.append({"layer": layer, "retrieval": retrieval, "proximity": prox,
                           "acc_matrix": [[round(float(x), 3) for x in row] for row in acc_matrix]})
    return {"langs": langs, "concepts": concepts, "layers": layers_out}


def script_masks(tokenizer, vocab_size: int) -> dict[str, t.Tensor]:
    """``{script: bool [V]}``: tokens whose letters are all in one script (languages.SCRIPTS)."""
    decoded = tokenizer.batch_decode([[i] for i in range(vocab_size)])
    masks = {name: t.zeros(vocab_size, dtype=t.bool) for name in languages.SCRIPTS}
    for i, text in enumerate(decoded):
        scripts = {languages.script_of(ch) for ch in text if unicodedata.category(ch).startswith("L")}
        if len(scripts) == 1 and None not in scripts:
            masks[scripts.pop()][i] = True
    return masks


def _role_script_mask(role_lang: str, masks: dict[str, t.Tensor]) -> t.Tensor:
    lang = role_lang.split(":")[0]
    if lang == "ja":
        return masks["kana"] | masks["han"]
    return masks[languages.LANGS[lang]["script"]]


def fair_readout(model, lenses: dict, *, concepts_path: Path, pairs: list[str], band: list[int], vocab_size: int,
                 batch_size: int, kanji_rule: bool = True, require_correct: bool = True) -> list[dict]:
    """The pivot's prompts and correctness filter; for each kept item, lens, band layer and role, the
    best rank of the role's tokens over the whole (readout-filtered) vocabulary AND within the role
    language's own script, as a percentile of that script's token count, so a language isn't helped
    by how much of the vocabulary its script owns."""
    import reader_questions as rqm

    tokenizer = model.tokenizer
    concepts = pe.load_concepts(concepts_path)
    index = pe.build_vocab_index(tokenizer, vocab_size)
    tracked: set[int] = set()
    for concept in concepts.values():
        for spellings in concept.values():
            tracked |= pe.prefix_token_ids(index, spellings, tokenizer)
    mask_V = pe.readout_mask(tokenizer, vocab_size, tracked)
    device = model._lm_head.weight.device
    masks = {k: (v & ~mask_V).to(device) for k, v in script_masks(tokenizer, vocab_size).items()}
    reader = rqm.Reader(model, lenses, band, vocab_size, mask_V)
    specs, _ = build_items(concepts, pairs, index, tokenizer, kanji_rule=kanji_rule)
    rows = []
    for start in range(0, len(specs), batch_size):
        chunk = specs[start: start + batch_size]
        ids_list = [model.encode(pe.build_prompt(concepts, s["concept"], s["src"], s["tgt"], shots=5, seed=0)) for s in chunk]
        for spec, (acts, logits) in zip(chunk, reader.forward_batch(ids_list)):
            if require_correct and int(logits.argmax()) not in spec["raw_target"]:
                continue
            with t.no_grad():
                for name, lens in lenses.items():
                    h = t.cat([lens.transport(acts[l][None], l).to(device) for l in band])
                    lg = model.unembed(h).float()[:, :vocab_size].masked_fill(reader.mask, float("-inf"))
                    for role in FAIR_ROLES:
                        ids = spec["sets"].get(role)
                        if not ids:
                            continue
                        m = _role_script_mask(spec["role_langs"][role], masks)
                        best = lg[:, t.tensor(sorted(ids), device=device)].max(1).values
                        rank_all = ((lg > best[:, None]).sum(1) + 1).cpu()
                        rank_sub = ((lg.masked_fill(~m, float("-inf")) > best[:, None]).sum(1) + 1).cpu()
                        n_sub = int(m.sum())
                        for i, layer in enumerate(band):
                            rows.append({"pair": spec["item"]["pair"], "concept": spec["concept"], "lens": name, "layer": layer,
                                         "role": role, "lang": spec["role_langs"][role], "rank_all": int(rank_all[i]),
                                         "rank_script": int(rank_sub[i]), "script_size": n_sub,
                                         "pct_script": round(1 - (int(rank_sub[i]) - 1) / max(n_sub, 1), 6)})
    return rows


def fair_table(rows: list[dict], groups: dict[str, set[str]]) -> list[dict]:
    """Per group of pairs, lens and role: band-mean percentile within the role's own script, the
    top-10 rate within it, and the ordinary top-10 rate over the whole vocabulary (CIs over concepts)."""
    out = []
    for gname, gpairs in groups.items():
        for lens in sorted({r["lens"] for r in rows}):
            for role in FAIR_ROLES:
                per_item = defaultdict(list)
                for r in rows:
                    if r["pair"] in gpairs and r["lens"] == lens and r["role"] == role:
                        per_item[(r["pair"], r["concept"])].append(r)
                if not per_item:
                    continue
                row = {"group": gname, "lens": lens, "role": role, "items": len(per_item),
                       "script_size": int(np.median([rr[0]["script_size"] for rr in per_item.values()]))}
                for key, fn in (("pct_script", lambda r: float(r["pct_script"])),
                                ("top10_script", lambda r: float(int(r["rank_script"]) <= 10)),
                                ("top10_all", lambda r: float(int(r["rank_all"]) <= 10))):
                    by_concept = defaultdict(list)
                    for (pair, concept), rr in per_item.items():
                        by_concept[concept].append(float(np.mean([fn(r) for r in rr])))
                    lo, hi = cluster_ci(by_concept)
                    row[key] = round(float(np.mean([v for vals in by_concept.values() for v in vals])), 4)
                    row[f"{key}_ci"] = f"[{lo:.3f}, {hi:.3f}]"
                out.append(row)
    return out


def run_geometry(ctx) -> dict:
    """Is the middle-layer concept space language-neutral or English-centric? Vectors per (concept,
    language) in two carriers (saved), :func:`geometry_metrics` on each, and on Sarvam the
    vocabulary-fair readout (:func:`fair_readout`) on the pivot prompts at the English band."""
    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    concepts = pe.load_concepts(Path(ctx.data_dir) / "concepts_new.tsv")
    cells = geometry_cells(concepts)
    if ctx.smoke:
        keep = {c for c in sorted(concepts)[:4]}
        cells = [x for x in cells if x[0] in keep and x[1] in ("en", "hi", "ta", "mni", "ja")]
    device = str(ctx.model._lm_head.weight.device) if t.cuda.is_available() else "cpu"
    result = {"model_kind": ctx.model_kind, "n_cells": len(cells), "band": ctx.band, "carriers": {}}
    batch = int(ctx.args.get("batch_size", ctx.batch_size))
    for carrier in ctx.args.get("carriers") or CARRIERS:
        V, n_tokens = collect_vectors(ctx.model, cells, carrier, batch)
        t.save({"vectors": V, "cells": cells, "n_tokens": n_tokens, "model_kind": ctx.model_kind}, out / f"vectors_{carrier}.pt")
        ctx.commit()
        metrics = geometry_metrics(V, cells, resamples=200 if ctx.smoke else 1000, device=device)
        metrics["n_tokens_by_lang"] = {l: round(float(np.mean([n for (_, lang, _), n in zip(cells, n_tokens) if lang == l])), 2)
                                       for l in metrics["langs"]}
        (out / f"geometry_{carrier}.json").write_text(json.dumps(metrics))
        band = [m for m in metrics["layers"] if m["layer"] in ctx.band]
        result["carriers"][carrier] = {
            "retrieval_band": round(float(np.mean([m["retrieval"]["acc"] for m in band])), 4),
            "chance": band[0]["retrieval"]["chance"] if band else None,
            "centred_order_band": [m["proximity"]["centred"]["order"][:5] for m in band],
            "en_rank_centred_band": [m["proximity"]["centred"]["en_rank"] for m in band]}
        ctx.commit()
    if ctx.args.get("fair", ctx.model_kind != "qwen"):
        spec = ctx.args.get("lenses") or ctx.lens_specs
        lenses = ctx.load_lenses(spec)
        pairs = (ORIGINAL_INDIC_PAIRS + "," + PAIRS).split(",")
        pairs = pairs[:2] if ctx.smoke else list(dict.fromkeys(pairs))
        rows = fair_readout(ctx.model, lenses, concepts_path=Path(ctx.data_dir) / "concepts_new.tsv", pairs=pairs,
                            band=ctx.band, vocab_size=ctx.vocab_size, batch_size=batch, require_correct=not ctx.smoke)
        write_csv(out / "fair_rows.csv", rows)
        groups = {"original Indic pairs": set(ORIGINAL_INDIC_PAIRS.split(",")),
                  "new-language pairs": set(PAIRS.split(",")) - set(BASELINE_PAIRS.split(",")) - set(EN_PAIRS.split(","))}
        table = fair_table(rows, groups)
        write_csv(out / "fair_summary.csv", table)
        result["fair"] = [{k: r[k] for k in ("group", "lens", "role", "items", "pct_script", "top10_script", "top10_all")}
                          for r in table]
        ctx.commit()
    return result


### LOCAL CHECKS AND RE-ANALYSIS


def _local_model(model_dir: str):
    import run_jpp
    import sarvam_support

    run_jpp.install(run_jpp._import_jpp_cli())
    return sarvam_support.load_sarvam_lens_model(model_dir, dtype=t.float32, device="cpu")


def check(model_dir: str, lens_path: str | None, out_dir: str, pairs: str, concepts: Path = CONCEPTS) -> int:
    """Rows from :func:`pivot` (batched, all layers at once) vs ``pivot_experiment.main`` on the
    same tiny model, lenses, pairs and concepts, without the correctness filter."""
    import run_jpp

    jpp_cli = run_jpp._import_jpp_cli()
    run_jpp.install(jpp_cli)
    out = Path(out_dir)
    specs = ["logit"] + ([lens_path] if lens_path else [])
    reference = out / "reference"
    code = pe.main(["--hf-model-name", model_dir, *sum((["--lens", s] for s in specs), []), "--pairs", pairs,
                    "--concepts", str(concepts), "--no-correctness-filter", "--out", str(reference)])
    assert code == 0
    model = jpp_cli.get_hf_model(model_dir, attn_implementation="sdpa", device_map=None)
    layers = list(range(model.n_layers - 1))
    lenses = jpp_cli.load_lenses(specs, model, hf_model_name=model_dir, layers=layers)
    vocab = jpp_cli.vocab_size_of(model)
    with (reference / "rows.csv").open(encoding="utf-8") as handle:
        ref = list(csv.DictReader(handle))
    ok = True
    for batch_size in (1, 5):
        rows, items = pivot(model, lenses, concepts_path=concepts, pairs=pairs.split(","), layers=layers, vocab_size=vocab,
                            out_dir=out / f"batched_{batch_size}", batch_size=batch_size, require_correct=False, decode_steps=3,
                            kanji_rule=False)
        same_keys = [(r["pair"], r["concept"], r["lens"], str(r["layer"]), r["role"]) for r in rows] == \
                    [(r["pair"], r["concept"], r["lens"], r["layer"], r["role"]) for r in ref]
        dprob = max((abs(float(a["prob"]) - float(b["prob"])) for a, b in zip(rows, ref)
                     if not (math.isnan(float(a["prob"])) and math.isnan(float(b["prob"])))), default=0.0)
        rank_diff = sum(int(a["best_rank"]) != int(b["best_rank"]) for a, b in zip(rows, ref))
        top10_diff = sum(str(a["top10"]) != b["top10"] for a, b in zip(rows, ref))
        print(f"batch {batch_size}: {len(rows)} rows vs {len(ref)}; same keys {same_keys}; max |dprob| {dprob:.2e}; "
              f"rank diffs {rank_diff}; top10 diffs {top10_diff}; decoded misses {sum('model_answer' in i for i in items)}")
        ok &= same_keys and len(rows) == len(ref) and dprob < 1e-4 and top10_diff == 0
    print("EQUIVALENT" if ok else "MISMATCH")
    return 0 if ok else 1


def miss_kind(item: dict) -> str:
    """What a missed item's decoded answer was: a copy of the source word's script, the same
    concept's word in another listed language, English, a target-script word not on the list
    (a synonym, a spelling variant or a wrong word), a listed target spelling whose first token the
    filter missed, or other."""
    src, tgt = item["pair"].split("-")
    match, answer = item.get("answer_match") or "", item.get("model_answer") or ""
    shares = languages.script_shares(answer)
    script = max(shares, key=shares.get) if any(shares.values()) else None
    want = {"kana", "han"} if tgt == "ja" else {languages.LANGS[tgt]["script"]}
    if match in ("target", "target_inflected"):
        return "listed target word"
    if match.startswith("same_concept:"):
        langs = match.split(":")[1].split("|")
        return "source word" if src in langs else ("English word" if langs == ["en"] else "other language's word")
    if script and script not in want and script == languages.LANGS[src]["script"]:
        return "source script"
    if script == "latin" and tgt != "en":
        return "English/Latin"
    if script in want:
        return "unlisted target-script word"
    return "other"


def misses_table(items: list[dict]) -> list[dict]:
    """Per target language and source kind (Indian/world vs English): items, first-token kept,
    exact word, and the misses by :func:`miss_kind` (needs a run with ``decode``)."""
    out = defaultdict(lambda: defaultdict(int))
    for item in items:
        if item.get("reason") == "target tokens all shared with another role":
            continue
        src, tgt = item["pair"].split("-")
        if tgt not in NEW:
            continue
        key = (tgt, "en" if src == "en" else "xx")
        out[key]["tested"] += 1
        out[key]["kept"] += item["kept"] in (True, "True")
        out[key]["exact"] += item.get("answer_match") in ("target", "target_inflected")
        if item["kept"] not in (True, "True") and "model_answer" in item:
            out[key][miss_kind(item)] += 1
    return [{"target": tgt, "from": "English" if kind == "en" else "Indian/world", **dict(v)}
            for (tgt, kind), v in sorted(out.items(), key=lambda kv: (NEW.index(kv[0][0]), kv[0][1]))]


def _fmt(row: dict, role: str, metric: str = "top10") -> str:
    if f"{role}_{metric}" not in row:
        return "–"
    value, ci = row[f"{role}_{metric}"], row.get(f"{role}_{metric}_ci", "")
    return f"{float(value):.2f} {ci.replace(' ', '')}" if metric == "top10" else f"{float(value):.3f}"


def report(run_dir: str, band: list[int], n_layers: int) -> str:
    """Markdown tables for docs/11_new_languages.md from a downloaded run (rows.csv, items.csv)."""
    run = Path(run_dir)
    with (run / "rows.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (run / "items.csv").open(encoding="utf-8") as handle:
        items = list(csv.DictReader(handle))
    layers = list(range(n_layers - 1))
    lines = []
    head = ("| group | lens | kept / tested | exact word | EN top-10 [95% CI] | EN P | EN shuffled | HI top-10 (n) | "
            "HI shuffled | ZH top-10 | EN onset | EN peak |")
    sep = "|" + "---|" * 12
    for title, groups in (("Per language", None), ("Pooled by class", class_groups_of)):
        table = language_table(rows, items, band, layers, groups=groups)
        if groups is class_groups_of:
            order = ["baseline"] + [f"{n} >" for n in CLASSES] + [f"> {n}" for n in CLASSES] + [f"en > {n}" for n in CLASSES]
            table.sort(key=lambda r: (order.index(r["group"]) if r["group"] in order else 99, r["lens"]))
        lines += [f"**{title}** (band L{band[0]}–{band[-1]}; `X>` = X as source, `>X` = X as target)", "", head, sep]
        for r in table:
            lines.append(f"| {r['group']} | {r['lens']} | {r['kept']}/{r['tested']} | {r.get('exact_word', '–')} | "
                         f"{_fmt(r, 'english')} | {_fmt(r, 'english', 'prob')} | {_fmt(r, 'english_shuffled').split(' ')[0]} | "
                         f"{_fmt(r, 'hindi').split(' ')[0]} ({r.get('hindi_n', 0)}) | {_fmt(r, 'hindi_shuffled').split(' ')[0]} | "
                         f"{_fmt(r, 'chinese').split(' ')[0]} | {r.get('english_onset', '–')} | {r.get('english_peak', '–')} |")
        lines.append("")
    misses = misses_table(items)
    if any("exact" in m for m in misses) and any(i.get("model_answer") for i in items):
        kinds = ["listed target word", "unlisted target-script word", "source word", "source script",
                 "other language's word", "English word", "English/Latin", "other"]
        lines += ["**What the model wrote** (decoded answers; misses = first token not on the list)", "",
                  "| target | from | kept | exact word | " + " | ".join(kinds) + " |", "|" + "---|" * (4 + len(kinds))]
        for m in misses:
            lines.append(f"| {m['target']} | {m['from']} | {m['kept']}/{m['tested']} | {m['exact']} | "
                         + " | ".join(str(m.get(k, 0)) for k in kinds) + " |")
    return "\n".join(lines)


def summarize_dir(run_dir: str, band: list[int], n_layers: int) -> list[dict]:
    """Recompute ``newlang_summary.csv`` from a downloaded run's rows/items."""
    run = Path(run_dir)
    with (run / "rows.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (run / "items.csv").open(encoding="utf-8") as handle:
        items = list(csv.DictReader(handle))
    table = language_table(rows, items, band, list(range(n_layers - 1)))
    write_csv(run / "newlang_summary.csv", table)
    return table


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="batched pivot vs pivot_experiment.main on a tiny model")
    c.add_argument("model_dir")
    c.add_argument("--lens", default=None)
    c.add_argument("--pairs", default="mni-bn,sat-hi,en-ja")
    c.add_argument("--out", required=True)
    c.add_argument("--concepts", default=str(CONCEPTS))
    mr = sub.add_parser("misses-report", help="kept-vs-missed tables (analysis misses) from a run's rows.csv / items.csv")
    mr.add_argument("run_dir")
    mr.add_argument("--band", required=True, help="e.g. 13-16")
    mr.add_argument("--n-layers", type=int, default=19)
    r = sub.add_parser("report", help="markdown tables for docs/11_new_languages.md from a run's rows.csv / items.csv")
    r.add_argument("run_dir")
    r.add_argument("--band", required=True, help="e.g. 13-16")
    r.add_argument("--n-layers", type=int, required=True)
    s = sub.add_parser("summarize", help="rewrite newlang_summary.csv from a run's rows.csv / items.csv")
    s.add_argument("run_dir")
    s.add_argument("--band", required=True, help="e.g. 13-16")
    s.add_argument("--n-layers", type=int, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.cmd == "check":
        return check(args.model_dir, args.lens, args.out, args.pairs, Path(args.concepts))
    lo, hi = (int(x) for x in args.band.split("-"))
    if args.cmd == "report":
        print(report(args.run_dir, list(range(lo, hi + 1)), args.n_layers))
        return 0
    if args.cmd == "misses-report":
        print(misses_report(args.run_dir, list(range(lo, hi + 1))))
        return 0
    table = summarize_dir(args.run_dir, list(range(lo, hi + 1)), args.n_layers)
    print(json.dumps(table[:3], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
