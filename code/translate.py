"""Machine translation of the cross-lingual prompt sets (docs/11), run inside Modal by
``modal_translate.py``. English (``data/xling/en_sets.json``) into:
- the 22 scheduled Indian languages with IndicTrans2 (ai4bharat/indictrans2-en-indic-1B, MIT, gated:
  needs an HF token) and its IndicProcessor pre/post-processing, as the model card shows;
- Japanese, Russian and Arabic with NLLB-200 3.3B (facebook/nllb-200-3.3B, CC-BY-NC-4.0).
Beam search 4, fp16. Each language is written to the Volume as soon as it's done
(``<out>/parts/<lang>.json``), so a rerun resumes. Each part records an ordered fingerprint of the
English source it was translated from (:func:`source_fingerprint`); a rerun redoes a part whose
fingerprint differs, and ``assemble`` refuses one, so reordering or editing ``en_sets.json`` can't
attach a translation to the wrong item (until 2026-10-20 only the file's existence and the row count
were checked). ``assemble`` then writes one file per set:
``{"set", "source", "license", "citation", "translators", "langs", "items": [{"id", "en", "meta",
"tr": {lang: text}}]}``. MGSM's human translations stay in ``meta.human``, beside the machine ones.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

IT2_MODEL = "ai4bharat/indictrans2-en-indic-1B"
NLLB_MODEL = "facebook/nllb-200-3.3B"
# Pinned to the revisions the smoke test ran (2026-10-19): IndicTrans2 runs remote code, so a new
# upstream modeling file must never be picked up silently.
REVISIONS = {IT2_MODEL: "89cd0007fd7b9e0693a2fd4404d187e07dac79fd", NLLB_MODEL: "1a07f7d195896b2114afcb79b7b57ab512e7b43e"}
NUM_BEAMS = 4
MAX_NEW_TOKENS = 512
FIXES: Counter = Counter()  # post-fixes applied, reported with the run


def _max_new(n_input_tokens: int) -> int:
    """Room for a long translation, but not for a runaway repeat."""
    return min(MAX_NEW_TOKENS, 3 * n_input_tokens + 16)


MIN_REPEAT_CHARS = 15  # shorter repeats are real reduplication ("धीरे धीरे"), never collapsed


def _postfix(text: str, *, collapse_repeats: bool = False) -> str:
    """Two artifacts the smoke test showed. NLLB sometimes emits the translation twice ("A? A?"):
    with ``collapse_repeats`` (NLLB only), keep one copy when the output is the same sentence of at
    least MIN_REPEAT_CHARS repeated 2-3 times. IndicTrans2's Meitei Mayek output puts spaces around
    the apun iyek (U+ABED), which joins consonants: remove them."""
    text = text.strip()
    if collapse_repeats:
        for k in (2, 3):
            unit = text[: len(text) // k].strip()
            if len(unit) >= MIN_REPEAT_CHARS and re.fullmatch(rf"(?:{re.escape(unit)}\s*){{{k}}}", text):
                FIXES["repeat_collapsed"] += 1
                return unit
    fixed = re.sub("\\s*\uABED\\s*", "\uABED", text)
    if fixed != text:
        FIXES["meitei_apun_iyek_spacing"] += 1
    return fixed


def _sentences(en_sets: dict) -> tuple[list[str], list[tuple[str, int]]]:
    """Every English item once, with (set, index) back-references."""
    texts, refs = [], []
    for name, s in en_sets["sets"].items():
        for i, item in enumerate(s["items"]):
            texts.append(item["en"])
            refs.append((name, i))
    return texts, refs


def source_fingerprint(en_sets: dict, smoke: bool = False) -> dict:
    """The ordered (set, id, English text) sequence a part's ``texts`` index into: its length and a
    sha256 over the exact sequence (any reorder, insertion or edit changes it)."""
    keys = [[name, item["id"], item["en"]] for name, s in en_sets["sets"].items() for item in s["items"]]
    if smoke:
        keys = [keys[0], keys[500], keys[-1]]
    digest = hashlib.sha256(json.dumps(keys, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"n": len(keys), "sha256": digest, "first": keys[0][:2] if keys else None, "last": keys[-1][:2] if keys else None}


def part_matches(path: Path, fingerprint: dict) -> bool:
    """A saved part can be reused only if it was translated from exactly this source sequence."""
    if not path.is_file():
        return False
    part = json.loads(path.read_text(encoding="utf-8"))
    return part.get("source") == fingerprint and len(part.get("texts", [])) == fingerprint["n"]


# A batch holds at most this many padded English characters (items x longest item). Beam search
# keeps 4 hypotheses per item with a KV cache that grows with the (2-3x longer) Indic output, so
# 96 MGSM problems (~300 chars each) ran an L40S out of memory; ~40 of them fit with room to spare.
MAX_PADDED_CHARS = 12000


def _batched(texts: list[str], batch_size: int, max_padded_chars: int = MAX_PADDED_CHARS):
    """Length-sorted batches (less padding), capped by item count and padded size; yields
    (original indices, texts)."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    idx: list[int] = []
    for i in order:
        if idx and (len(idx) == batch_size or (len(idx) + 1) * len(texts[i]) > max_padded_chars):
            yield idx, [texts[j] for j in idx]
            idx = []
        idx.append(i)
    if idx:
        yield idx, [texts[j] for j in idx]


def _split_on_oom(translate_chunk, chunk: list[str]) -> list[str]:
    """``translate_chunk(chunk)``; on CUDA out-of-memory, free the cache and do each half alone."""
    import torch

    try:
        return translate_chunk(chunk)
    except torch.OutOfMemoryError:
        if len(chunk) == 1:
            raise
        torch.cuda.empty_cache()
        FIXES["oom_batch_split"] += 1
        half = len(chunk) // 2
        return _split_on_oom(translate_chunk, chunk[:half]) + _split_on_oom(translate_chunk, chunk[half:])


def translate_it2(texts: list[str], tgt: str, *, model, tokenizer, processor, batch_size: int = 96) -> list[str]:
    import torch

    def one(chunk: list[str]) -> list[str]:
        batch = processor.preprocess_batch(chunk, src_lang="eng_Latn", tgt_lang=tgt)
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt",
                           return_attention_mask=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, use_cache=True, min_length=0, max_new_tokens=_max_new(inputs["input_ids"].shape[1]),
                                       num_beams=NUM_BEAMS, num_return_sequences=1)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return [_postfix(text) for text in processor.postprocess_batch(decoded, lang=tgt)]

    out = [""] * len(texts)
    for idx, chunk in _batched(texts, batch_size):
        for i, text in zip(idx, _split_on_oom(one, chunk)):
            out[i] = text
    return out


def translate_nllb(texts: list[str], tgt: str, *, model, tokenizer, batch_size: int = 64) -> list[str]:
    import torch

    forced = tokenizer.convert_tokens_to_ids(tgt)

    def one(chunk: list[str]) -> list[str]:
        inputs = tokenizer(chunk, truncation=True, padding="longest", return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(**inputs, forced_bos_token_id=forced, max_new_tokens=_max_new(inputs["input_ids"].shape[1]),
                                       num_beams=NUM_BEAMS)
        return [_postfix(text, collapse_repeats=True) for text in tokenizer.batch_decode(generated, skip_special_tokens=True)]

    out = [""] * len(texts)
    for idx, chunk in _batched(texts, batch_size, max_padded_chars=MAX_PADDED_CHARS // 2):  # 3.3B: bigger cache
        for i, text in zip(idx, _split_on_oom(one, chunk)):
            out[i] = text
    return out


def run(en_sets_path: str, out_dir: str, *, langs: list[str], commit=lambda: None, smoke: bool = False) -> dict:
    """Translate every English item into ``langs`` (codes from languages.py). ``smoke``: 3 items."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    import languages

    run_start = time.monotonic()
    en_sets = json.loads(Path(en_sets_path).read_text(encoding="utf-8"))
    texts, _ = _sentences(en_sets)
    if smoke:
        texts = [texts[0], texts[500], texts[-1]]
    fingerprint = source_fingerprint(en_sets, smoke)
    parts = Path(out_dir) / ("parts_smoke" if smoke else "parts")
    parts.mkdir(parents=True, exist_ok=True)
    # resume only parts translated from this exact source sequence; a stale or unfingerprinted part is redone
    timings, todo = {}, [l for l in langs if not part_matches(parts / f"{l}.json", fingerprint)]
    stale = [l for l in todo if (parts / f"{l}.json").is_file()]
    it2_langs = [l for l in todo if languages.LANGS[l]["it2"]]
    nllb_langs = [l for l in todo if not languages.LANGS[l]["it2"]]
    revisions = {}

    def save(lang: str, outs: list[str], model_name: str, seconds: float) -> None:
        (parts / f"{lang}.json").write_text(json.dumps({"lang": lang, "model": model_name, "source": fingerprint, "texts": outs},
                                                       ensure_ascii=False), encoding="utf-8")
        timings[lang] = round(seconds, 1)
        commit()

    if it2_langs:
        from IndicTransToolkit.processor import IndicProcessor

        tokenizer = AutoTokenizer.from_pretrained(IT2_MODEL, trust_remote_code=True, revision=REVISIONS[IT2_MODEL])
        model = AutoModelForSeq2SeqLM.from_pretrained(IT2_MODEL, trust_remote_code=True, torch_dtype=torch.float16,
                                                      revision=REVISIONS[IT2_MODEL]).to("cuda")
        revisions[IT2_MODEL] = getattr(model.config, "_commit_hash", None)
        processor = IndicProcessor(inference=True)
        for lang in it2_langs:
            start = time.monotonic()
            outs = translate_it2(texts, languages.LANGS[lang]["it2"], model=model, tokenizer=tokenizer, processor=processor)
            save(lang, outs, IT2_MODEL, time.monotonic() - start)
        del model
        torch.cuda.empty_cache()
    if nllb_langs:
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="eng_Latn", revision=REVISIONS[NLLB_MODEL])
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL, torch_dtype=torch.float16,
                                                      revision=REVISIONS[NLLB_MODEL]).to("cuda")
        revisions[NLLB_MODEL] = getattr(model.config, "_commit_hash", None)
        for lang in nllb_langs:
            start = time.monotonic()
            outs = translate_nllb(texts, languages.LANGS[lang]["nllb"], model=model, tokenizer=tokenizer)
            save(lang, outs, NLLB_MODEL, time.monotonic() - start)
    (parts / "_revisions.json").write_text(json.dumps(revisions), encoding="utf-8") if revisions else None
    commit()
    sample = {l: json.loads((parts / f"{l}.json").read_text(encoding="utf-8"))["texts"][0][:120]
              for l in langs if (parts / f"{l}.json").is_file()}
    return {"n_texts": len(texts), "minutes": round((time.monotonic() - run_start) / 60, 1), "fixes": dict(FIXES), "done": sorted(p.stem for p in parts.glob("*.json") if not p.stem.startswith("_")),
            "seconds_per_lang": timings, "revisions": revisions, "first_item_sample": sample, "redone_stale_parts": stale,
            "source": fingerprint}


def assemble(en_sets_path: str, parts_dir: str, out_dir: str, langs: list[str], *, allow_unfingerprinted: bool = False) -> dict:
    """One file per set from the per-language parts (CPU; runs locally after ``modal volume get``).
    Every part must carry the fingerprint of the current ``en_sets`` (same items, same order). Parts
    from before 2026-10-20 have none: they are refused unless ``allow_unfingerprinted`` (then only the
    row count is checked, as before; run :func:`verify_assembled` afterwards)."""
    en_sets = json.loads(Path(en_sets_path).read_text(encoding="utf-8"))
    texts, refs = _sentences(en_sets)
    fingerprint = source_fingerprint(en_sets)
    parts = {l: json.loads((Path(parts_dir) / f"{l}.json").read_text(encoding="utf-8")) for l in langs}
    revisions_path = Path(parts_dir) / "_revisions.json"
    revisions = json.loads(revisions_path.read_text(encoding="utf-8")) if revisions_path.is_file() else {}
    for lang, part in parts.items():
        if "source" in part:
            if part["source"] != fingerprint:
                raise ValueError(f"part {lang} was translated from a different source sequence "
                                 f"({part['source'].get('n')} items, sha {str(part['source'].get('sha256'))[:12]}) than {en_sets_path} "
                                 f"({fingerprint['n']}, {fingerprint['sha256'][:12]}): re-translate it")
        elif not allow_unfingerprinted:
            raise ValueError(f"part {lang} has no source fingerprint (made before 2026-10-20); pass allow_unfingerprinted=True "
                             "only if en_sets.json is unchanged since, then run verify_assembled")
        if len(part["texts"]) != len(texts):
            raise ValueError(f"part {lang} has {len(part['texts'])} texts, en_sets {len(texts)}")
    out = {}
    for name, s in en_sets["sets"].items():
        items = [dict(item, tr={}) for item in s["items"]]
        out[name] = {**{k: v for k, v in s.items() if k != "items"}, "set": name,
                     "translators": {"indic": {"model": IT2_MODEL, "revision": revisions.get(IT2_MODEL)},
                                     "world": {"model": NLLB_MODEL, "revision": revisions.get(NLLB_MODEL)},
                                     "decoding": f"beam {NUM_BEAMS}, fp16, max {MAX_NEW_TOKENS} new tokens"},
                     "langs": ["en", *langs], "source_fingerprint": fingerprint, "items": items}
    for k, (name, i) in enumerate(refs):
        for lang, part in parts.items():
            out[name]["items"][i]["tr"][lang] = part["texts"][k]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, data in out.items():
        (Path(out_dir) / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return {name: len(d["items"]) for name, d in out.items()}


def translate_list(texts: list[str], langs: list[str]) -> dict[str, list[str]]:
    """Translate a short list of English sentences into ``langs`` with the same models, revisions and
    decoding as :func:`run` (used for prompt pieces such as doc 12's hint sentence)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    import languages

    out: dict[str, list[str]] = {}
    it2_langs = [l for l in langs if languages.LANGS[l]["it2"]]
    nllb_langs = [l for l in langs if not languages.LANGS[l]["it2"]]
    if it2_langs:
        from IndicTransToolkit.processor import IndicProcessor

        tokenizer = AutoTokenizer.from_pretrained(IT2_MODEL, trust_remote_code=True, revision=REVISIONS[IT2_MODEL])
        model = AutoModelForSeq2SeqLM.from_pretrained(IT2_MODEL, trust_remote_code=True, torch_dtype=torch.float16,
                                                      revision=REVISIONS[IT2_MODEL]).to("cuda")
        processor = IndicProcessor(inference=True)
        for lang in it2_langs:
            out[lang] = translate_it2(texts, languages.LANGS[lang]["it2"], model=model, tokenizer=tokenizer, processor=processor)
        del model
        torch.cuda.empty_cache()
    if nllb_langs:
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="eng_Latn", revision=REVISIONS[NLLB_MODEL])
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL, torch_dtype=torch.float16,
                                                      revision=REVISIONS[NLLB_MODEL]).to("cuda")
        for lang in nllb_langs:
            out[lang] = translate_nllb(texts, languages.LANGS[lang]["nllb"], model=model, tokenizer=tokenizer)
    return out


_NUMBER = re.compile(r"\d{2,}")


def verify_assembled(en_sets_path: str, parts_dir: str, out_dir: str, langs: list[str]) -> dict:
    """Check saved per-set files against their source, for files assembled before fingerprints existed:
    1. each set's items have the source's ids and English texts, in the source's order;
    2. each translation equals its part's text at the item's index in the current source sequence;
    3. alignment that doesn't trust the parts' history: per language, the share of items whose English
       numbers (2+ digits) all reappear in the translation (IndicTrans2 and NLLB keep ASCII digits), on
       the true pairing and on the pairing shifted by one item, and the length correlation of the two.
       A mis-attached part would score like the shifted pairing."""
    import numpy as np

    en_sets = json.loads(Path(en_sets_path).read_text(encoding="utf-8"))
    texts, refs = _sentences(en_sets)
    parts = {l: json.loads((Path(parts_dir) / f"{l}.json").read_text(encoding="utf-8"))["texts"] for l in langs}
    report: dict = {"order_mismatch": {}, "part_mismatch": {}, "alignment": {}}
    flat: dict = {l: [] for l in langs}
    for name, s in en_sets["sets"].items():
        saved = json.loads((Path(out_dir) / f"{name}.json").read_text(encoding="utf-8"))
        src = [(it["id"], it["en"]) for it in s["items"]]
        got = [(it["id"], it["en"]) for it in saved["items"]]
        if src != got:
            report["order_mismatch"][name] = sum(a != b for a, b in zip(src, got)) + abs(len(src) - len(got))
        base = next(k for k, (n, _) in enumerate(refs) if n == name)
        for lang in langs:
            bad = sum(it["tr"].get(lang) != parts[lang][base + i] for i, it in enumerate(saved["items"]))
            if bad:
                report["part_mismatch"][f"{name}/{lang}"] = bad
            flat[lang].extend(it["tr"].get(lang, "") for it in saved["items"])
    en = [t for t in texts]
    nums = [set(_NUMBER.findall(t)) for t in en]
    with_nums = [k for k, n in enumerate(nums) if n]
    for lang in langs:
        tr = flat[lang]
        if len(tr) != len(en):
            report["alignment"][lang] = {"error": f"{len(tr)} translations for {len(en)} items"}
            continue
        hit = lambda shift: float(np.mean([nums[k] <= set(_NUMBER.findall(tr[(k + shift) % len(tr)])) for k in with_nums]))  # noqa: E731
        corr = float(np.corrcoef([len(x) for x in en], [len(x) for x in tr])[0, 1])
        report["alignment"][lang] = {"numbers_kept": round(hit(0), 3), "numbers_kept_shifted": round(hit(1), 3),
                                     "length_corr": round(corr, 3), "n_with_numbers": len(with_nums)}
    report["ok"] = not report["order_mismatch"] and not report["part_mismatch"] and all(
        a.get("numbers_kept", 0) > a.get("numbers_kept_shifted", 1) + 0.3 for a in report["alignment"].values())
    return report


if __name__ == "__main__":
    import sys

    import languages

    if len(sys.argv) > 1 and sys.argv[1] == "verify":  # python translate.py verify  (from code/)
        print(json.dumps(verify_assembled("data/xling/en_sets.json", "data/xling/parts", "data/xling", list(languages.XLING[1:])),
                         ensure_ascii=False, indent=1))
