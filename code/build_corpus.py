"""Build a lens-fitting prompt pool as a JSON list (for ``JPP_FIT_PROMPTS_JSON`` /
``fit-shard --prompts-json``).

    python build_corpus.py --langs en,hi,bn,ta,te,mr,gu,kn,ml --per-lang 120 --out corpora/indic_wiki.json

Each prompt is a ~1,500-character passage of one Wikipedia article in one language
(``wikimedia/wikipedia``, streamed). Languages are interleaved round-robin, so every
prefix of the list is language-balanced. jpp fits on the first ``--num-prompts`` and
shards contiguous slices, so a block-ordered list would give each shard one language.
Template-generated articles are near-duplicates (Telugu's dump, past the first 1,000
articles, is mostly calendar-day pages), so an article whose opening ``--dedupe-chars``
share half their words with an earlier passage's is skipped.

Alongside the JSON, ``<out>.manifest.json`` records the per-language counts, the
duplicates skipped, and the dump. The paper recipe (WikiText-103, English only) needs no file: leave
``JPP_FIT_PROMPTS_JSON`` unset.
"""

from __future__ import annotations

import argparse
import itertools
import json
import unicodedata
from pathlib import Path

DEFAULT_LANGS = "en,hi,bn,ta,te,mr,gu,kn,ml"


def opening_words(passage: str, n_chars: int) -> frozenset[str]:
    """The words of the passage's first ``n_chars``, digits (any script) removed. Template-generated
    articles share nearly all of them, e.g. Telugu Wikipedia's calendar days
    ("మే 25, గ్రెగొరియన్‌ క్యాలెండర్‌ ప్రకారము సంవత్సరములో 145వ రోజు ...", one per date)."""
    return frozenset("".join(ch for ch in passage[:n_chars] if not ch.isdigit()).split())


def is_template_duplicate(words: frozenset[str], seen: list[frozenset[str]], threshold: float = 0.5) -> bool:
    return any(len(words & other) >= threshold * len(words | other) for other in seen)


def passages(lang: str, dump: str, *, min_chars: int, max_chars: int, skip_articles: int, dedupe_chars: int,
             skipped: dict[str, int]):
    """Yield one passage per article: the first run of paragraphs reaching ``min_chars``,
    cut to ``max_chars``. An article whose opening ``dedupe_chars`` share half their words with an
    earlier passage's (Jaccard >= 0.5, digits removed) is skipped and counted in ``skipped[lang]``
    (``dedupe_chars=0`` turns this off)."""
    from datasets import load_dataset

    stream = load_dataset("wikimedia/wikipedia", f"{dump}.{lang}", split="train", streaming=True)
    seen: list[frozenset[str]] = []
    for record in itertools.islice(stream, skip_articles, None):
        paragraphs = [p.strip() for p in unicodedata.normalize("NFC", record["text"]).split("\n")]
        paragraphs = [p for p in paragraphs if len(p) >= 80]  # drop headings, list stubs
        passage = ""
        for paragraph in paragraphs:
            if len(passage) >= min_chars:
                break
            passage = f"{passage}\n\n{paragraph}" if passage else paragraph
        if len(passage) < min_chars:
            continue
        if dedupe_chars:
            words = opening_words(passage, dedupe_chars)
            if is_template_duplicate(words, seen):
                skipped[lang] = skipped.get(lang, 0) + 1
                continue
            seen.append(words)
        yield passage[:max_chars]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", default=DEFAULT_LANGS)
    parser.add_argument("--per-lang", type=int, default=120)
    parser.add_argument("--dump", default="20231101")
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--skip-articles", type=int, default=1000, help="skip the dump's first N articles (often stubs/dates)")
    parser.add_argument("--dedupe-chars", type=int, default=100,
                        help="skip articles whose first N chars share half their words with an earlier one's; 0 = off")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    langs = [lang for lang in args.langs.split(",") if lang]
    by_lang: dict[str, list[str]] = {}
    skipped: dict[str, int] = {}
    for lang in langs:
        source = passages(lang, args.dump, min_chars=args.min_chars, max_chars=args.max_chars, skip_articles=args.skip_articles,
                          dedupe_chars=args.dedupe_chars, skipped=skipped)
        by_lang[lang] = list(itertools.islice(source, args.per_lang))
        print(f"{lang}: {len(by_lang[lang])} passages ({skipped.get(lang, 0)} template duplicates skipped)")
        if len(by_lang[lang]) < args.per_lang:
            print(f"  warning: {lang} ran out after {len(by_lang[lang])}")

    interleaved = [p for group in itertools.zip_longest(*by_lang.values()) for p in group if p is not None]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(interleaved, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "dataset": "wikimedia/wikipedia",
        "dump": args.dump,
        "order": "round-robin over languages",
        "langs": langs,
        "counts": {lang: len(p) for lang, p in by_lang.items()},
        "min_chars": args.min_chars,
        "max_chars": args.max_chars,
        "skip_articles": args.skip_articles,
        "dedupe_chars": args.dedupe_chars,
        "template_duplicates_skipped": {lang: skipped.get(lang, 0) for lang in langs},
    }
    Path(f"{out}.manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {len(interleaved)} prompts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
