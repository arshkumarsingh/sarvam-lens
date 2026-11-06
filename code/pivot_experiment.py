"""Which language does the model pass through when it translates between Indian languages?

For each source -> target pair and concept, a few-shot word-translation prompt stops just
before the target word (Wendler et al. 2024, "Do Llamas Work in English?"):

    தமிழ்: "மரம்" - తెలుగు: "చెట్టు"
    ...
    தமிழ்: "நீர்" - తెలుగు: "

At the final position every lens (``logit``, J-Lens, R-Lens, J++ Lens files) reads out
every layer. For each tracked language role (target, source, English, Hindi, Chinese) we record:
the lens probability of tokens that begin the concept's word in that language, the best
rank of any such token, and whether one lands in the top 10.

Token control. A language's token set for a concept is every vocab token whose text
(NFC, case-folded, without leading whitespace or quotes) is a prefix of at least
``--min-prefix-chars`` code points of one of the word's spellings, plus the tokenizer's
own first token of the word. Sarvam splits most Indic words into ~2 tokens, so scoring
whole-word tokens only would miss most of them. Tokens in two tracked languages' sets (Hindi and Marathi
share Devanagari, e.g. पानी / पाणी share "पा") are dropped from both. An item whose
target set becomes empty, or whose final-layer prediction is not a target-word token
(unless ``--no-correctness-filter``), is skipped and logged.

    python pivot_experiment.py --hf-model-name sarvamai/sarvam-30b \
        --lens logit --lens lenses/sarvam_j.pt --lens lenses/sarvam_r.pt --lens lenses/sarvam_jpp.pt \
        --out results/sarvam

Writes ``rows.csv`` (one row per item x lens x layer x role), ``items.csv`` (kept/skipped
and why), ``summary.csv`` (means with 95% bootstrap CIs over concepts) and one PNG per lens.
"""

from __future__ import annotations

import argparse
import csv
import logging
import unicodedata
import zlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as t

import languages

logger = logging.getLogger("pivot")
HERE = Path(__file__).resolve().parent

# Each language's own name for itself, used as the prompt's line label, and its family. Both come
# from languages.py (the original 11 are unchanged; ALL_INDIC still keeps IA/Dr pairs only).
LANG_LABELS = {code: spec["label"] for code, spec in languages.LANGS.items()}
FAMILY = {code: spec["family"] for code, spec in languages.LANGS.items()}
# Pivot test pairs: Dravidian<->Dravidian, Indo-Aryan<->Indo-Aryan, across families, then
# the script test (Devanagari vs romanized Hindi into Tamil) and two English controls.
DEFAULT_PAIRS = "ta-te,te-kn,kn-ml,ml-ta,bn-gu,gu-mr,mr-bn,bn-ta,ta-bn,te-gu,hi-ta,hi_latn-ta,en-ta,ta-en"
ROLES = ("target", "english", "hindi", "source", "chinese")
# Shuffled-concept controls: the same language's tokens for a DIFFERENT concept. If a control
# ranks as high as its real role, the real signal is generic script/language, not the concept.
# (The Chinese control was added after the first runs.)
CONTROL_ROLES = {"english_shuffled": "en", "hindi_shuffled": "hi", "chinese_shuffled": "zh"}
ALL_ROLES = ROLES + tuple(CONTROL_ROLES)
QUOTE_CHARS = "\"'“”‘’«»"


### CONCEPTS AND PROMPTS


def load_concepts(path: Path) -> dict[str, dict[str, list[str]]]:
    """``{concept: {lang: [spellings...]}}`` from the TSV (spellings split on ``|``)."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        row["concept"]: {
            lang: [unicodedata.normalize("NFC", s.strip()) for s in cell.split("|") if s.strip()]
            for lang, cell in row.items()
            if lang != "concept" and cell
        }
        for row in rows
    }


def build_prompt(concepts, concept: str, src: str, tgt: str, *, shots: int, seed: int) -> str:
    """``shots`` demonstration lines over other concepts (fixed per pair and concept), then
    the query line ending at the target word's opening quote."""
    others = sorted(c for c in concepts if c != concept and src in concepts[c] and tgt in concepts[c])
    # crc32, not hash(): str hashes are salted per process, which would change the demos run to run.
    rng = np.random.default_rng([seed, zlib.crc32(f"{src}|{tgt}|{concept}".encode())])
    demos = rng.choice(others, size=min(shots, len(others)), replace=False)
    lines = [f'{LANG_LABELS[src]}: "{concepts[c][src][0]}" - {LANG_LABELS[tgt]}: "{concepts[c][tgt][0]}"' for c in demos]
    lines.append(f'{LANG_LABELS[src]}: "{concepts[concept][src][0]}" - {LANG_LABELS[tgt]}: "')
    return "\n".join(lines)


### TOKEN SETS


def normalize_token_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).lstrip().lstrip(QUOTE_CHARS).casefold()


def build_vocab_index(tokenizer, vocab_size: int) -> dict[str, list[int]]:
    """``{normalized token text: ids}`` over the LM head's ids, without special tokens."""
    special = set(getattr(tokenizer, "all_special_ids", []))
    decoded = tokenizer.batch_decode([[i] for i in range(vocab_size)])
    index: dict[str, list[int]] = defaultdict(list)
    for token_id, text in enumerate(decoded):
        key = normalize_token_text(text)
        if key and token_id not in special:
            index[key].append(token_id)
    return index


def prefix_token_ids(index: dict[str, list[int]], spellings: list[str], tokenizer, *, min_chars: int = 2) -> set[int]:
    """Tokens that begin one of ``spellings``: every vocab token whose text is a prefix of
    at least ``min_chars`` code points, plus the tokenizer's own first token of the word
    (with and without a leading space) even when it is shorter. Without the floor, bare
    letters like "w" or a lone consonant "प", which begin countless words, would count."""
    ids: set[int] = set()
    for word in spellings:
        key = normalize_token_text(word)
        for end in range(min_chars, len(key) + 1):
            ids.update(index.get(key[:end], ()))
        for variant in (word, f" {word}"):
            first_ids = tokenizer(variant, add_special_tokens=False).input_ids
            first_text = normalize_token_text(tokenizer.decode(first_ids[:1])) if first_ids else ""
            # A byte-level BPE (Qwen) can start an Indic word with a partial-UTF-8 token, which
            # decodes to U+FFFD and so is no text prefix, yet it is the word's real first token.
            if first_text and (key.startswith(first_text) or "�" in first_text):
                ids.add(int(first_ids[0]))
    return ids


def tracked_languages(src: str, tgt: str) -> dict[str, str]:
    """``{role: lang}``. A role whose language is already tracked under an earlier role is
    dropped (``hi -> ta`` has no separate "hindi" role: it is the source)."""
    roles: dict[str, str] = {}
    # Chinese is tracked for Chinese-centric baselines (Qwen); for Sarvam it should stay near 0.
    for role, lang in (("target", tgt), ("source", src), ("english", "en"), ("hindi", "hi"), ("chinese", "zh")):
        if lang not in roles.values():
            roles[role] = lang
    return roles


def disjoint_sets(sets: dict[str, set[int]]) -> tuple[dict[str, set[int]], set[int]]:
    """Remove tokens that more than one role claims; return the cleaned sets and the shared ids."""
    counts: dict[int, int] = defaultdict(int)
    for ids in sets.values():
        for token_id in ids:
            counts[token_id] += 1
    shared = {token_id for token_id, n in counts.items() if n > 1}
    return {role: ids - shared for role, ids in sets.items()}, shared


### READOUT


@dataclass
class Readout:
    prob: float
    best_rank: int
    top10: bool


def score(logits_V: t.Tensor, ids: set[int], mask_V: t.Tensor | None) -> Readout:
    logits_V = logits_V.float()
    if mask_V is not None:
        logits_V = logits_V.masked_fill(mask_V, float("-inf"))
    if not ids:
        return Readout(float("nan"), -1, False)
    id_tensor = t.tensor(sorted(ids), dtype=t.long)
    probs_V = t.softmax(logits_V, dim=-1)
    best_logit = logits_V[id_tensor].max()
    best_rank = int((logits_V > best_logit).sum()) + 1
    return Readout(float(probs_V[id_tensor].sum()), best_rank, best_rank <= 10)


def readout_mask(tokenizer, vocab_size: int, keep_ids: set[int]) -> t.Tensor:
    """jpp's Readout Filtering (drop tokens with no letter or digit), sparing ``keep_ids``."""
    from lens_evals.eval_utils import all_non_semantic_token_ids

    mask_V = t.zeros(vocab_size, dtype=t.bool)
    excluded = sorted(all_non_semantic_token_ids(tokenizer, vocab_size) - keep_ids)
    mask_V[t.tensor(excluded, dtype=t.long)] = True
    return mask_V


### SUMMARY AND PLOTS


def bootstrap_ci(values: np.ndarray, *, resamples: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict], indic_pairs: set[str]) -> list[dict]:
    """Mean probability and top-10 rate with bootstrap CIs over concepts, per
    (pair, lens, layer, role), plus an ``ALL_INDIC`` pseudo-pair pooling the pairs
    with no English or Hindi endpoint."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["pair"], row["lens"], row["layer"], row["role"])].append(row)
        if row["pair"] in indic_pairs:
            groups[("ALL_INDIC", row["lens"], row["layer"], row["role"])].append(row)
    summary = []
    for (pair, lens, layer, role), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], ALL_ROLES.index(kv[0][3]))):
        probs = np.array([r["prob"] for r in group if not np.isnan(r["prob"])])
        top10 = np.array([float(r["top10"]) for r in group])
        ranks = np.array([r["best_rank"] for r in group if r["best_rank"] > 0])
        lo, hi = bootstrap_ci(probs)
        top_lo, top_hi = bootstrap_ci(top10)
        summary.append(
            {
                "pair": pair, "lens": lens, "layer": layer, "depth": group[0]["depth"], "role": role,
                "n": len(group), "mean_prob": float(probs.mean()) if len(probs) else float("nan"),
                "prob_ci_lo": lo, "prob_ci_hi": hi,
                "top10_rate": float(top10.mean()), "top10_ci_lo": top_lo, "top10_ci_hi": top_hi,
                "median_rank": float(np.median(ranks)) if len(ranks) else float("nan"),
            }
        )
    return summary


# Validated categorical slots 1-5 (dataviz reference palette, light): fixed role -> hue,
# so a role keeps its color in every panel and every lens. Aqua, yellow and magenta sit below 3:1
# on the surface, so every line is also direct-labeled and dashed by role.
ROLE_STYLE = {
    "target": ("#2a78d6", "-"),
    "english": ("#eb6834", "-"),
    "hindi": ("#1baf7a", "--"),
    "source": ("#eda100", ":"),
    "chinese": ("#e87ba4", "-."),
}
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def plot_lens(summary: list[dict], lens: str, metric: str, out_path: Path, pairs: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [p for p in ["ALL_INDIC", *pairs] if any(s["pair"] == p and s["lens"] == lens for s in summary)]
    if not panels:
        return
    columns = min(4, len(panels))
    rows_n = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows_n, columns, figsize=(3.6 * columns, 2.9 * rows_n), squeeze=False, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    ylabel = {"mean_prob": "lens probability", "top10_rate": "share with a token in top 10"}[metric]
    lo_key, hi_key = ("prob_ci_lo", "prob_ci_hi") if metric == "mean_prob" else ("top10_ci_lo", "top10_ci_hi")
    # Lens probabilities are often far below 1: scale the shared y-axis to the data
    # (rates stay on 0-1 so they compare across lenses).
    upper = 1.0
    if metric == "mean_prob":
        peaks = [s[hi_key] for s in summary if s["lens"] == lens and not np.isnan(s[hi_key])]
        upper = min(1.0, max(0.05, 1.15 * max(peaks, default=0.0)))
    for ax, pair in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        ends: list[tuple[float, float, str]] = []
        for role in ROLES:
            series = sorted((s for s in summary if s["pair"] == pair and s["lens"] == lens and s["role"] == role), key=lambda s: s["layer"])
            if not series:
                continue
            color, dash = ROLE_STYLE[role]
            x = [s["depth"] for s in series]
            y = [s[metric] for s in series]
            ax.fill_between(x, [s[lo_key] for s in series], [s[hi_key] for s in series], color=color, alpha=0.12, linewidth=0)
            ax.plot(x, y, color=color, linestyle=dash, linewidth=2, label=role)
            ends.append((x[-1], y[-1], role))
        # Direct labels at the line ends, nudged apart so overlapping ends stay readable.
        min_gap, last_y = 0.07 * upper, -np.inf
        for x_end, y_end, role in sorted(ends, key=lambda e: e[1]):
            label_y = max(y_end, last_y + min_gap)
            ax.annotate(role, (x_end, y_end), xytext=(x_end + 0.02, label_y), textcoords="data",
                        fontsize=7, color=INK_MUTED, va="center")
            last_y = label_y
        n = max((s["n"] for s in summary if s["pair"] == pair and s["lens"] == lens), default=0)
        title = "all Indic pairs" if pair == "ALL_INDIC" else pair.replace("-", " → ")
        ax.set_title(f"{title}  (n={n})", fontsize=9, color=INK, loc="left")
        ax.set_ylim(0, upper)
        ax.set_xlim(0, 1.12)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(colors=INK_MUTED, labelsize=7)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8, color=INK_MUTED)
    for ax in axes[-1, :]:
        ax.set_xlabel("relative depth (layer / last layer)", fontsize=8, color=INK_MUTED)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=5, fontsize=8, frameon=False, labelcolor=INK)
    fig.suptitle(f"{lens}: which language is the next word in, layer by layer", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


### MAIN


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-model-name", required=True)
    parser.add_argument("--lens", action="append", required=True, help="'logit' or a lens .pt (repeatable)")
    parser.add_argument("--layers", default=None, help="comma-separated; default: every layer all file lenses cover")
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--concepts", default=str(HERE / "data/concepts.tsv"))
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--min-prefix-chars", type=int, default=2, help="shortest prefix token counted (the word's own first token is always counted)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-correctness-filter", action="store_true")
    parser.add_argument("--no-readout-filter", action="store_true", help="rank over the full vocab (default: jpp's Readout Filtering)")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-items", type=int, default=None, help="cap items per pair (smoke tests)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    import run_jpp
    import sarvam_support

    jpp_cli = run_jpp._import_jpp_cli()
    run_jpp.install(jpp_cli)
    model = jpp_cli.get_hf_model(args.hf_model_name, attn_implementation=args.attn_implementation, device_map=args.device_map)
    tokenizer = model.tokenizer
    vocab_size = jpp_cli.vocab_size_of(model)

    file_specs = [s for s in args.lens if s != jpp_cli.LOGIT_LENS_SPEC]
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    elif file_specs:
        layer_sets = [set(jpp_cli.load_lens_file(s, hf_model_name=args.hf_model_name).source_layers) for s in file_specs]
        layers = sorted(set.intersection(*layer_sets))
    else:
        layers = list(range(model.n_layers - 1))
    lenses = jpp_cli.load_lenses(args.lens, model, hf_model_name=args.hf_model_name, layers=layers)
    logger.info("lenses %s at layers %s", list(lenses), layers)

    concepts = load_concepts(Path(args.concepts))
    pairs = [p for p in args.pairs.split(",") if p]
    index = build_vocab_index(tokenizer, vocab_size)
    last_layer = model.n_layers - 1

    # Every tracked token set, so Readout Filtering never makes a scored token unrankable.
    all_tracked: set[int] = set()
    for concept in concepts.values():
        for spellings in concept.values():
            all_tracked |= prefix_token_ids(index, spellings, tokenizer, min_chars=args.min_prefix_chars)
    mask_V = None if args.no_readout_filter else readout_mask(tokenizer, vocab_size, all_tracked)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    items: list[dict] = []
    for pair in pairs:
        src, tgt = pair.split("-")
        kept = 0
        for concept in sorted(concepts):
            if args.max_items is not None and kept >= args.max_items:
                break
            words = concepts[concept]
            roles = tracked_languages(src, tgt)
            if any(lang not in words for lang in roles.values()):
                continue
            raw_sets = {
                role: prefix_token_ids(index, words[lang], tokenizer, min_chars=args.min_prefix_chars)
                for role, lang in roles.items()
            }
            sets, shared = disjoint_sets(raw_sets)
            # Control concept: a fixed derangement (half-way round the sorted list), minus any
            # token a real role claims, so a control can never borrow the real signal.
            ordered = sorted(concepts)
            control_concept = ordered[(ordered.index(concept) + len(ordered) // 2) % len(ordered)]
            claimed = set().union(*raw_sets.values())
            scored_sets = dict(sets)
            role_langs = dict(roles)
            for control_role, lang in CONTROL_ROLES.items():
                if lang in concepts[control_concept]:
                    scored_sets[control_role] = prefix_token_ids(
                        index, concepts[control_concept][lang], tokenizer, min_chars=args.min_prefix_chars) - claimed
                    role_langs[control_role] = f"{lang}:{control_concept}"
            prompt = build_prompt(concepts, concept, src, tgt, shots=args.shots, seed=args.seed)
            item = {"pair": pair, "concept": concept, "target_word": words[tgt][0], "n_shared_tokens": len(shared)}
            if not sets["target"]:
                items.append({**item, "kept": False, "reason": "target tokens all shared with another role"})
                continue

            readouts = {}
            model_top = None
            for lens_name, lens in lenses.items():
                lens_logits, model_logits, _ = lens.apply(model, prompt, layers=layers, token_positions_for_residuals=[-1])
                readouts[lens_name] = lens_logits
                model_top = int(model_logits[0].argmax())
            item["model_top_token"] = tokenizer.decode([model_top])
            if not args.no_correctness_filter and model_top not in raw_sets["target"]:
                items.append({**item, "kept": False, "reason": "model's next token does not begin the target word"})
                continue
            items.append({**item, "kept": True, "reason": ""})
            kept += 1
            for lens_name, lens_logits in readouts.items():
                for layer in layers:
                    for role, lang in role_langs.items():
                        r = score(lens_logits[layer][0], scored_sets[role], mask_V)
                        rows.append(
                            {
                                "pair": pair, "src": src, "tgt": tgt, "concept": concept, "lens": lens_name,
                                "layer": layer, "depth": layer / last_layer, "role": role, "lang": lang,
                                "prob": r.prob, "best_rank": r.best_rank, "top10": r.top10,
                            }
                        )
        logger.info("%s: kept %d items", pair, kept)

    indic_pairs = {p for p in pairs if all(FAMILY.get(l) in ("IA", "Dr") and l not in ("hi", "hi_latn") for l in p.split("-"))}
    summary = summarize(rows, indic_pairs)
    for name, table in (("rows.csv", rows), ("items.csv", items), ("summary.csv", summary)):
        if table:
            fieldnames = list(dict.fromkeys(k for row in table for k in row))
            with (out / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(table)
    for lens_name in lenses:
        for metric in ("mean_prob", "top10_rate"):
            plot_lens(summary, lens_name, metric, out / f"{lens_name}_{metric}.png", pairs)
    kept_total = sum(1 for i in items if i["kept"])
    logger.info("kept %d / %d items; wrote %s", kept_total, len(items), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
