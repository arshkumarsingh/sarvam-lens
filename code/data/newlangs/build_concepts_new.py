"""Build ``data/concepts_new.tsv`` and ``data/concepts_new_sources.tsv`` from cited sources (docs/11).

Inputs, all in this folder:
- ``wiktionary_raw/<concept>.txt``: the English Wiktionary translation-table lines for the 17 new
  languages (plus Hindi), copied verbatim with the page's revision id (via firecrawl).
- ``swadesh_appendix.tsv``: the hand-written Wiktionary Swadesh-list appendices for Santali and
  Manipuri (Meitei Mayek), used where the table has no usable word (the first listed word).
- ``overrides.tsv``: per (concept, lang), the chosen spellings in order (the first is the prompt
  word), their source URL and a note. Used where the table's first word is not the common everyday
  word, where it is in the wrong script (Manipuri in Bengali script, Bodo in Latin), or where the
  table has no entry and another cited page does. A cell with the word ``-`` is left empty.

Rules (the parser): keep single words in the language's script (languages.py), NFC; Arabic, Urdu
and Sindhi get their short-vowel marks (harakat) stripped, as everyday text omits them (the
Wiktionary form is kept as an alternate spelling); Russian stress marks are stripped. Templates
marked ``t-check`` (unverified on Wiktionary) are kept but flagged.

    python data/newlangs/build_concepts_new.py            # writes candidates.tsv for review
    python data/newlangs/build_concepts_new.py --write    # + concepts_new.tsv, concepts_new_sources.tsv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE))
import languages  # noqa: E402

NEW = languages.NEW_INDIC + languages.WORLD
# Wiktionary language names (and codes) -> our codes
NAMES = {"Assamese": "as", "Odia": "or", "Oriya": "or", "Punjabi": "pa", "Panjabi": "pa", "Urdu": "ur", "Nepali": "ne",
         "Sanskrit": "sa", "Maithili": "mai", "Sindhi": "sd", "Kashmiri": "ks", "Konkani": "kok", "Goan Konkani": "kok",
         "Dogri": "doi", "Manipuri": "mni", "Meitei": "mni", "Bodo": "brx", "Bodo (India)": "brx", "Santali": "sat",
         "Japanese": "ja", "Russian": "ru", "Arabic": "ar", "Hindi": "hi"}
SCRIPT_SUBLINES = {"Arabic", "Perso-Arabic", "Devanagari", "Gurmukhi", "Shahmukhi", "Bengali", "Bangla", "Meitei",
                   "Meetei Mayek", "Ol Chiki", "Latin", "Roman", "Kannada", "Malayalam"}
TEMPLATE = re.compile(r"\{\{(t\+?|tt\+?|t\+?-check|tt\+?-check|t\+check|tt\+check)\|([^|}]+)\|((?:\[\[[^\]]*\]\]|[^|}])+)([^}]*)\}\}")
QUALIFIER = re.compile(r"\{\{(?:q|qual|qualifier|i|gloss)\|([^}]*)\}\}")
# fathatan .. sukun (maddah above, 0x0653, stays), plus the optional noon-ghunna mark (U+0658)
HARAKAT = dict.fromkeys([*range(0x064B, 0x0653), 0x0658])
STRIP_HARAKAT = {"ar", "ur", "sd"}
CONCEPTS = ("water fire sun moon tree flower bird dog cat house book eye hand mother milk river mountain sky rain "
            "horse cow fish salt gold night heart village king star tooth name road").split()


def unlink(word: str) -> str:
    return re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", word)


def in_script(word: str, lang: str) -> bool:
    want = {"kana", "han"} if lang == "ja" else {languages.LANGS[lang]["script"]}
    scripts = {languages.script_of(ch) for ch in word if unicodedata.category(ch).startswith("L")}
    return bool(scripts) and scripts <= want


def clean(word: str, lang: str) -> list[str]:
    """The everyday spelling first, then the Wiktionary spelling if it differs. Invisible format
    characters (e.g. a stray U+2069 direction mark) go, except ZWJ/ZWNJ, which spell."""
    word = "".join(ch for ch in unicodedata.normalize("NFC", unlink(word).strip())
                   if unicodedata.category(ch) != "Cf" or ch in "‌‍")
    out = word
    if lang in STRIP_HARAKAT:
        out = word.translate(HARAKAT)
    elif lang == "ru":
        out = unicodedata.normalize("NFD", word).replace("́", "").replace("̀", "")
    return list(dict.fromkeys([unicodedata.normalize("NFC", out), word]))


def _outer(text: str) -> str:
    """``text`` with the commas inside templates masked, so splitting on "," splits terms only."""
    return re.sub(r"\{\{[^{}]*\}\}", lambda m: m.group(0).replace(",", "\x00"), text)


def parse_file(path: Path) -> dict:
    head, blocks, lines = {}, [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("# ") and ":" in raw:
            key, _, value = raw[2:].partition(":")
            head[key.strip()] = value.strip()
        elif raw.startswith("## block:"):
            blocks.append(raw[len("## block:"):].strip())
        elif raw.strip():
            lines.append((len(blocks) - 1, raw))
    cells: dict[str, list[dict]] = {}
    missing: dict[str, str] = {}
    parent = None
    for block, line in lines:
        if line.startswith("MISSING"):
            name = line[len("MISSING"):].strip()
            if name in NAMES:
                missing.setdefault(NAMES[name], "no line in the table")
            continue
        m = re.match(r"^\*(:?)\s*([^:{]+?):\s*(.*)$", line)
        if not m:
            continue
        nested, name, body = m.group(1) == ":", m.group(2).strip(), m.group(3)
        if not nested:
            parent = NAMES.get(name)
            lang = parent
            sub = ""
        elif name in SCRIPT_SUBLINES and parent:
            lang, sub = parent, name
        else:
            continue  # a related language nested under a parent (Old Japanese, Bishnupriya Manipuri...)
        if lang is None:
            continue
        if "t-needed" in body and not TEMPLATE.search(body):
            missing.setdefault(lang, "{{t-needed}}: the table asks for a translation")
            continue
        matches = list(TEMPLATE.finditer(body))
        for i, tm in enumerate(matches):
            kind, code, word, rest = tm.groups()
            # A qualifier belongs to the term in its own comma-separated segment: right before it
            # (", {{q|poetic}} {{t|..}}") or right after it ("{{t|..}} {{q|colloquial}},").
            before = _outer(body[matches[i - 1].end() if i else 0: tm.start()]).rsplit(",", 1)[-1].replace("\x00", ",")
            after = _outer(body[tm.end(): matches[i + 1].start() if i + 1 < len(matches) else len(body)]).split(",", 1)[0].replace("\x00", ",")
            quals = QUALIFIER.findall(before) + QUALIFIER.findall(after)
            cells.setdefault(lang, []).append({
                "word": unlink(word), "code": code, "check": "check" in kind, "sub": sub, "block": block,
                "qualifier": "; ".join(quals), "tr": (re.search(r"\|tr=([^|}]*)", rest) or [None, ""])[1],
                "raw": line.strip()})
    return {"head": head, "blocks": blocks, "cells": cells, "missing": missing}

def _tsv(name: str) -> list[dict]:
    path = HERE / name
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader((line for line in handle if not line.startswith("#")), delimiter="\t"))


# Wikidata labels are the last resort, and only for languages whose labels read as native words
# (Maithili's copy Hindi, e.g. कुत्ता for dog; Konkani's mix Latin, Kannada and Devanagari).
WIKIDATA_LANGS = {"ks", "mni", "sat", "sd"}


def resolve(concept: str, lang: str, parsed: dict, url: str, overrides, swadesh, search, wikidata) -> dict:
    """The cell's spellings and its source, by tier: a manual override, the translation table's
    first usable word, the Swadesh appendix, an entry found by search, a Wikidata label."""
    entries = parsed["cells"].get(lang, [])
    usable = [e for e in entries if " " not in e["word"].strip() and in_script(e["word"], lang)]
    rejected = [e for e in entries if e not in usable]
    why = parsed["missing"].get(lang) or ("table has only " + "; ".join(f"{e['word']} ({e['sub'] or 'other script/multiword'})" for e in rejected)
                                         if rejected else "no line in the table")
    spell = lambda words: "|".join(dict.fromkeys(s for w in words for s in clean(w, lang)))
    o = overrides.get((concept, lang))
    if o:
        words = [w for w in o["words"].split("|") if w.strip()]
        return {"word": "" if words == ["-"] else spell(words), "tier": "override", "source": o.get("source") or url,
                "note": o.get("note", "")}
    if usable:
        e = usable[0]
        flags = ["first word of the table"] + (["t-check (unverified on Wiktionary)"] if e["check"] else []) + \
                ([f"qualifier: {e['qualifier']}"] if e["qualifier"] else []) + \
                (["other listed: " + ", ".join(x["word"] for x in usable[1:])] if len(usable) > 1 else [])
        return {"word": spell([e["word"]]), "tier": "table", "source": url, "note": "; ".join(flags)}
    if (concept, lang) in swadesh:
        sw = swadesh[(concept, lang)]
        words = sw["words_in_list_order"].split("|")
        return {"word": spell(words[:1]), "tier": "swadesh", "source": sw["source"],
                "note": f"Swadesh list no. {sw['swadesh_no']}, first listed word ({why})"
                        + (f"; other listed: {', '.join(words[1:])}" if len(words) > 1 else "")}
    hit = search.get((concept, lang))
    if hit:
        return {"word": spell([hit["word"]]), "tier": "entry", "source": hit["source"],
                "note": f"Wiktionary entry, definition {hit['definition']!r} ({why})" + (f"; {hit['note']}" if hit.get("note") else "")}
    label = wikidata.get((concept, lang))
    if label and lang in WIKIDATA_LANGS and " " not in label["label"] and in_script(label["label"], lang):
        return {"word": spell([label["label"]]), "tier": "wikidata", "source": f"https://www.wikidata.org/wiki/{label['item']}",
                "note": f"Wikidata label of {label['item']} ({label['en_label']}), fetched 2026-10-19 ({why})"}
    return {"word": "", "tier": "empty", "source": url, "note": why}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    with (CODE / "data/concepts.tsv").open(encoding="utf-8") as handle:
        base = {r["concept"]: r for r in csv.DictReader(handle, delimiter="\t")}
    overrides = {(r["concept"], r["lang"]): r for r in _tsv("overrides.tsv")}
    swadesh = {(r["concept"], r["lang"]): r for r in _tsv("swadesh_appendix.tsv")}
    search = {}
    for r in _tsv("search_hits.tsv"):
        if r.get("word", "").strip() and (r["concept"], r["lang"]) not in search:
            title = r.get("page_title") or r["word"]
            r["source"] = f"https://en.wiktionary.org/w/index.php?title={title.replace(' ', '_')}&oldid={r.get('revid', '')}"
            search[(r["concept"], r["lang"])] = r
    wikidata = {(r["concept"], r["lang"]): r for r in _tsv("wikidata_labels.tsv")}
    sources, table = [], []
    for concept in CONCEPTS:
        parsed = parse_file(HERE / "wiktionary_raw" / f"{concept}.txt")
        page, revid = parsed["head"].get("page", concept), parsed["head"].get("revid", "")
        url = (f"https://en.wiktionary.org/w/index.php?title={page.replace(' ', '_')}&oldid={revid}#Translations"
               if revid.isdigit() else f"https://en.wiktionary.org/wiki/{page.replace(' ', '_')}#Translations")
        row = dict(base[concept])
        for lang in NEW:
            cell = resolve(concept, lang, parsed, url, overrides, swadesh, search, wikidata)
            row[lang] = cell["word"]
            sources.append({"concept": concept, "lang": lang, **cell})
        table.append(row)
    tiers = {lang: {} for lang in NEW}
    for s in sources:
        tiers[s["lang"]][s["tier"]] = tiers[s["lang"]].get(s["tier"], 0) + 1
    for lang in NEW:
        print(f"{lang:4s} filled {sum(1 for r in table if r.get(lang)):2d}/32  {tiers[lang]}")
    if args.write:
        columns = list(next(iter(base.values())).keys()) + list(NEW)
        with (CODE / "data/concepts_new.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(table)
        with (CODE / "data/concepts_new_sources.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["concept", "lang", "word", "tier", "source", "note"], delimiter="\t",
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(sources)
        print("wrote data/concepts_new.tsv and data/concepts_new_sources.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
