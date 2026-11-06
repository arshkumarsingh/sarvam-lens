"""Quality checks on the machine-translated prompt sets (docs/11), CPU only:
``python check_translations.py [data/xling]`` -> data/xling/translation_qa.json + a printed table.

Per language, over every item of every set:
- empty: no text came back;
- untranslated: the output is the English input, or its letters are mostly Latin when the language
  isn't written in Latin;
- script_ok: the share of the output's letters in the language's own script (languages.SCRIPTS),
  averaged; Japanese counts kana and Han together;
- len_ratio: output characters / English characters (median), a crude truncation/run-on flag;
- mgsm_numbers_kept: the share of MGSM problems whose every numeric VALUE in the English is written
  in digits in the translation (native digits mapped back; thousands separators, decimal commas and
  "1:00 PM" normalised). A miss is either a number written as a word ("4 times" -> "четыре раза",
  "80,000" -> "80 ألف"; the problem is unchanged) or a real omission (the problem changed). Flags:
  ``flags.mgsm_number_missing`` (strict: any value missing) and ``flags.mgsm_number_missing_gt10``
  (a value above 10 missing; small numbers are the ones translators write as words). Stages that
  grade MGSM should report accuracy on the strict-pass items (cleanest) beside all items.
- chrf_vs_human (bn, te, ja, ru): chrF2 between the machine and the human MGSM translation.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import languages

NUMBER = re.compile(r"\d+(?:\.\d+)?")
# "80,000" / "80 000" (space, no-break or narrow no-break) / "80, 000" (an IndicTrans2 artifact)
THOUSANDS = re.compile(r"(?<=\d)(?:,|, |[ \u00a0\u202f])(?=\d{3}(?!\d))")
DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d{1,2}(?!\d))")  # Russian "1,5"
CLOCK = re.compile(r"\b(\d{1,2}):00\b")  # "1:00 PM" -> 1 (translations write "1 o'clock")


def _ascii_digits(text: str) -> str:
    """Every Unicode decimal digit (native scripts included) as ASCII."""
    return "".join(str(int(ch)) if ch.isdecimal() else ch for ch in text)


def _numbers(text: str) -> Counter:
    """Numeric values written in digits, as rounded floats (so "$1.50" equals "1.5")."""
    text = CLOCK.sub(r"\1", DECIMAL_COMMA.sub(".", THOUSANDS.sub("", _ascii_digits(text))))
    return Counter(round(float(n), 6) for n in NUMBER.findall(text))


def _script_ok(text: str, lang: str) -> float:
    shares = languages.script_shares(text)
    if lang == "ja":
        return shares["kana"] + shares["han"]
    return shares[languages.LANGS[lang]["script"]]


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """Sentence chrF (character n-grams 1..6, beta 2), spaces removed as in sacrebleu's default."""
    hyp, ref = hyp.replace(" ", ""), ref.replace(" ", "")
    precisions, recalls = [], []
    for k in range(1, n + 1):
        h = Counter(hyp[i: i + k] for i in range(len(hyp) - k + 1))
        r = Counter(ref[i: i + k] for i in range(len(ref) - k + 1))
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = statistics.mean(precisions), statistics.mean(recalls)
    return 0.0 if p + r == 0 else 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def main(folder: str = "data/xling") -> dict:
    root = Path(folder)
    sets = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*.json"))
            if p.stem not in ("en_sets", "translation_qa")}
    langs = [l for l in next(iter(sets.values()))["langs"] if l != "en"]
    report: dict = {"per_lang": {}, "flags": {"mgsm_number_missing": {}}}
    for lang in langs:
        rows = [(name, item) for name, s in sets.items() for item in s["items"]]
        outs = [(item["en"], item["tr"].get(lang, "")) for _, item in rows]
        latin_lang = languages.LANGS[lang]["script"] == "latin"
        empty = sum(not o.strip() for _, o in outs)
        untranslated = sum(o.strip() == e.strip() or (not latin_lang and languages.script_shares(o)["latin"] > 0.5)
                           for e, o in outs if o.strip())
        script_ok = statistics.mean(_script_ok(o, lang) for _, o in outs if o.strip()) if len(outs) > empty else 0.0
        ratios = [len(o) / max(1, len(e)) for e, o in outs if o.strip()]
        entry = {"n": len(outs), "empty": empty, "untranslated": untranslated, "script_ok": round(script_ok, 3),
                 "len_ratio_median": round(statistics.median(ratios), 2) if ratios else None}
        if "mgsm" in sets:
            lost = {item["id"]: _numbers(item["en"]) - _numbers(item["tr"].get(lang, "")) for item in sets["mgsm"]["items"]}
            missing = [i for i, l in lost.items() if l]
            big = [i for i, l in lost.items() if any(v > 10 for v in l)]
            entry["mgsm_numbers_kept"] = round(1 - len(missing) / len(lost), 3)
            entry["mgsm_numbers_gt10_kept"] = round(1 - len(big) / len(lost), 3)
            report["flags"]["mgsm_number_missing"][lang] = missing
            report["flags"].setdefault("mgsm_number_missing_gt10", {})[lang] = big
            human = [(item["tr"].get(lang, ""), item["meta"]["human"][lang]) for item in sets["mgsm"]["items"]
                     if lang in item["meta"].get("human", {})]
            if human:
                entry["chrf_vs_human"] = round(statistics.mean(chrf(h, r) for h, r in human), 1)
        report["per_lang"][lang] = entry
    (root / "translation_qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'lang':5s} {'empty':>5s} {'untr':>5s} {'script':>6s} {'len':>5s} {'mgsm#':>6s} {'#>10':>6s} {'chrF':>5s}")
    for lang, e in report["per_lang"].items():
        print(f"{lang:5s} {e['empty']:5d} {e['untranslated']:5d} {e['script_ok']:6.3f} {e['len_ratio_median'] or 0:5.2f} "
              f"{e.get('mgsm_numbers_kept', float('nan')):6.3f} {e.get('mgsm_numbers_gt10_kept', float('nan')):6.3f} "
              f"{e.get('chrf_vs_human', '')!s:>5s}")
    return report


if __name__ == "__main__":
    main(*sys.argv[1:])
