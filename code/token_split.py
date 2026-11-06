"""Reader question 1: is English easier to read out only because English words are single tokens?

Free: re-slices the existing pivot results by how many tokens each word takes in Sarvam's tokenizer
(the 30B and 105B tokenizers are byte-identical). For each model, lens and Indic→Indic item:
- English, Hindi and target top-10 rates in the model's English band, split by the token count
  of the word itself (1 token vs 2+);
- English minus Hindi on the same items, restricted to items whose Hindi word is ONE token
  (Hindi then has no tokenization handicap at all).

    python token_split.py <tokenizer dir> <out json>
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import pivot_experiment as pe  # noqa: E402

# (label, rows.csv, lens, English band, last readout layer)
RUNS = (
    ("Sarvam-30B, J-lens", ROOT / "results/final/pivot_all/rows.csv", "j_200p", (14, 15, 16), 17),
    ("Sarvam-30B, logit lens", ROOT / "results/final/pivot_all/rows.csv", "logit", (14, 15, 16), 17),
)
INDIC = {"ta-te", "te-kn", "kn-ml", "ml-ta", "bn-gu", "gu-mr", "mr-bn", "bn-ta", "ta-bn", "te-gu"}


def n_tokens(tokenizer, word: str) -> int:
    return len(tokenizer(word, add_special_tokens=False).input_ids)


def bootstrap(values: list[float], seed: int = 0) -> list[float]:
    if len(values) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = rng.choice(np.array(values, dtype=float), size=(2000, len(values)), replace=True).mean(1)
    return [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


def main(tokenizer_dir: str, out_path: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    concepts = pe.load_concepts(HERE / "data/concepts.tsv")
    lengths = {(c, lang): n_tokens(tokenizer, spellings[0]) for c, langs in concepts.items() for lang, spellings in langs.items()}
    out: dict = {"token_counts": {}}
    for lang in ("en", "hi", "ta", "te", "kn", "ml", "bn", "gu", "mr", "zh"):
        counts = [lengths[(c, lang)] for c in concepts if (c, lang) in lengths]
        out["token_counts"][lang] = {"single_token_share": round(float(np.mean([n == 1 for n in counts])), 3),
                                     "mean_tokens": round(float(np.mean(counts)), 2)}
    for label, path, lens, band, last in RUNS:
        # item -> role -> {layer: top10}
        items: dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
        with path.open(encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                if r["lens"] != lens or r["pair"] not in INDIC:
                    continue
                items[(r["pair"], r["concept"])][r["role"]][int(r["layer"])] = r["top10"] == "True"
        band_rate = lambda roles, role: float(np.mean([roles[role][l] for l in band])) if role in roles else float("nan")
        per_item = []
        for (pair, concept), roles in items.items():
            tgt = pair.split("-")[1]
            per_item.append({"tgt_tokens": lengths[(concept, tgt)], "hi_tokens": lengths[(concept, "hi")],
                             "en_tokens": lengths[(concept, "en")],
                             "english": band_rate(roles, "english"), "hindi": band_rate(roles, "hindi"),
                             "target_band": band_rate(roles, "target"),
                             "target_last": float(roles["target"][last]) if last in roles["target"] else float("nan")})
        res: dict = {"n_items": len(per_item), "band": list(band)}
        for key, word in (("by_target_tokens", "tgt_tokens"), ("by_hindi_tokens", "hi_tokens"), ("by_english_tokens", "en_tokens")):
            res[key] = {}
            for bucket, test in (("1 token", lambda n: n == 1), ("2+ tokens", lambda n: n >= 2)):
                sub = [x for x in per_item if test(x[word])]
                res[key][bucket] = {"n": len(sub), **{m: round(float(np.mean([x[m] for x in sub])), 3) if sub else None
                                                      for m in ("english", "hindi", "target_band", "target_last")}}
        one_tok_hi = [x for x in per_item if x["hi_tokens"] == 1]
        diff = [x["english"] - x["hindi"] for x in one_tok_hi]
        res["english_minus_hindi_when_hindi_is_1_token"] = {"n": len(diff), "mean": round(float(np.mean(diff)), 3) if diff else None,
                                                            "ci95": bootstrap(diff)}
        out[label] = res
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
