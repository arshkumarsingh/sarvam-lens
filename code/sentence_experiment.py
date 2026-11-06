"""Does the English pivot extend beyond word translation, to plain sentences?

``data/sentences.tsv`` holds natural Hindi/Tamil/Telugu/Bengali sentences that stop right
before a concept word, with no translation instruction ("I'm thirsty, please give me a glass
of ___"). Indic SOV order puts the verb last, so these clozes are natural. At the final
position, each lens reads out every layer; we score the native target word, its English and
Hindi translations, and shuffled-concept controls. Token sets and scoring are the same as in
:mod:`pivot_experiment`. Items the model does not complete with the target are dropped.
"""

from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

import pivot_experiment as pe

HERE = Path(__file__).resolve().parent


def load_sentences(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["target"] = [unicodedata.normalize("NFC", w.strip()) for w in row["target"].split("|") if w.strip()]
        row["prompt"] = unicodedata.normalize("NFC", row["prompt"].strip())
    return rows


def run(model, lenses: dict, *, layers: list[int], vocab_size: int, out_dir: str,
        sentences_path: str | None = None, require_correct: bool = True) -> dict:
    tokenizer = model.tokenizer
    index = pe.build_vocab_index(tokenizer, vocab_size)
    concepts = pe.load_concepts(HERE / "data/concepts.tsv")
    ordered = sorted(concepts)
    sentences = load_sentences(Path(sentences_path or HERE / "data/sentences.tsv"))
    tracked = set()
    for s in sentences:
        tracked |= pe.prefix_token_ids(index, s["target"], tokenizer)
    for c in concepts.values():
        for lang in ("en", "hi"):
            tracked |= pe.prefix_token_ids(index, c[lang], tokenizer)
    mask_V = pe.readout_mask(tokenizer, vocab_size, tracked)
    last_layer = model.n_layers - 1

    rows, items = [], []
    for s in sentences:
        lang, concept = s["lang"], s["concept"]
        control = ordered[(ordered.index(concept) + len(ordered) // 2) % len(ordered)]
        raw = {"target": pe.prefix_token_ids(index, s["target"], tokenizer),
               "english": pe.prefix_token_ids(index, concepts[concept]["en"], tokenizer)}
        if lang != "hi":
            raw["hindi"] = pe.prefix_token_ids(index, concepts[concept]["hi"], tokenizer)
        sets, _ = pe.disjoint_sets(raw)
        claimed = set().union(*raw.values())
        sets["english_shuffled"] = pe.prefix_token_ids(index, concepts[control]["en"], tokenizer) - claimed
        if lang != "hi":
            sets["hindi_shuffled"] = pe.prefix_token_ids(index, concepts[control]["hi"], tokenizer) - claimed
        readouts, model_top = {}, None
        for name, lens in lenses.items():
            lens_logits, model_logits, _ = lens.apply(model, s["prompt"], layers=layers, token_positions_for_residuals=[-1])
            readouts[name] = lens_logits
            model_top = int(model_logits[0].argmax())
        ok = model_top in raw["target"]
        items.append({"lang": lang, "concept": concept, "model_top": tokenizer.decode([model_top]), "kept": ok or not require_correct})
        if require_correct and not ok:
            continue
        for name, lens_logits in readouts.items():
            for layer in layers:
                for role, ids in sets.items():
                    r = pe.score(lens_logits[layer][0], ids, mask_V)
                    rows.append({"lang": lang, "concept": concept, "lens": name, "layer": layer,
                                 "depth": layer / last_layer, "role": role, "prob": r.prob, "top10": r.top10})

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fname, table in (("sentence_rows.csv", rows), ("sentence_items.csv", items)):
        if table:
            with (out / fname).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(table)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["lens"], r["layer"], r["role"])].append(r)
    summary: dict = {"n_kept": sum(i["kept"] for i in items), "n_items": len(items), "top10_rate": {}, "mean_prob": {}}
    for (lens_name, layer, role), g in sorted(groups.items()):
        summary["top10_rate"].setdefault(lens_name, {}).setdefault(role, {})[layer] = round(float(np.mean([x["top10"] for x in g])), 3)
        summary["mean_prob"].setdefault(lens_name, {}).setdefault(role, {})[layer] = round(float(np.mean([x["prob"] for x in g])), 4)
    return summary
