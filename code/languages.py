"""One registry for every language the study touches (docs/11): the original 8 Indic languages plus
romanized Hindi, English and Chinese, the 14 other scheduled Indian languages, and three world
controls (Japanese, Russian, Arabic).

Per language:
- ``label``: its own name for itself (the few-shot line label in :mod:`pivot_experiment`).
- ``family``: ``IA`` Indo-Aryan, ``Dr`` Dravidian, ``TB`` Tibeto-Burman, ``AA`` Austroasiatic, or
  the world controls' own tags. ``pivot_experiment``'s ``ALL_INDIC`` aggregate keeps ``IA``/``Dr``
  only, so the published numbers don't change.
- ``script``: a key of :data:`SCRIPTS` (Unicode ranges). Scripts are shared (Bengali by bn/as;
  Devanagari by hi/mr/ne/sa/mai/brx/doi/kok; Arabic by ur/sd/ks/ar; Han by zh/ja), so a script
  share says "this script", not "this language".
- ``it2`` / ``nllb``: the translator's language code. IndicTrans2 covers the 22 scheduled
  languages; NLLB-200 the world controls.
- ``mgsm``: the MGSM config with a HUMAN translation, if any (the MT check for the math set).
- ``digit0``: the code point of the language's own zero, if it has native digits.

Tokenizer coverage, measured on Sarvam's 262k vocabulary (2026-10-19): every Indic script here is
native (0.2-0.5 tokens per character, no byte fallback), including Meitei Mayek (0.21) and Ol Chiki
(0.43). Japanese is NOT: 15 of 18 tokens for a short sentence were byte fallbacks, so Japanese is a
language Sarvam barely saw, not a like-for-like world control. Qwen's tokenizer is the reverse: it
shreds Meitei Mayek (2.5 tokens/char) and Ol Chiki (2.8).
"""

from __future__ import annotations

import unicodedata
from collections import Counter

SCRIPTS: dict[str, tuple[tuple[int, int], ...]] = {
    "latin": ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F)),
    "devanagari": ((0x900, 0x97F), (0xA8E0, 0xA8FF)),
    "bengali": ((0x980, 0x9FF),),
    "gurmukhi": ((0xA00, 0xA7F),),
    "gujarati": ((0xA80, 0xAFF),),
    "oriya": ((0xB00, 0xB7F),),
    "tamil": ((0xB80, 0xBFF),),
    "telugu": ((0xC00, 0xC7F),),
    "kannada": ((0xC80, 0xCFF),),
    "malayalam": ((0xD00, 0xD7F),),
    "meitei": ((0xABC0, 0xABFF), (0xAAE0, 0xAAFF)),
    "olchiki": ((0x1C50, 0x1C7F),),
    "arabic": ((0x600, 0x6FF), (0x750, 0x77F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "cyrillic": ((0x400, 0x4FF),),
    "kana": ((0x3040, 0x30FF),),
    "han": ((0x4E00, 0x9FFF),),
}

LANGS: dict[str, dict] = {
    # the study's original languages (labels and families exactly as pivot_experiment had them)
    "en": {"name": "English", "label": "English", "family": "En", "script": "latin", "it2": "eng_Latn", "nllb": "eng_Latn", "mgsm": "en", "digit0": None},
    "hi": {"name": "Hindi", "label": "हिन्दी", "family": "IA", "script": "devanagari", "it2": "hin_Deva", "nllb": "hin_Deva", "mgsm": None, "digit0": 0x966},
    "hi_latn": {"name": "Hindi (romanized)", "label": "Hindi", "family": "IA", "script": "latin", "it2": None, "nllb": None, "mgsm": None, "digit0": None},
    "bn": {"name": "Bengali", "label": "বাংলা", "family": "IA", "script": "bengali", "it2": "ben_Beng", "nllb": "ben_Beng", "mgsm": "bn", "digit0": 0x9E6},
    "ta": {"name": "Tamil", "label": "தமிழ்", "family": "Dr", "script": "tamil", "it2": "tam_Taml", "nllb": "tam_Taml", "mgsm": None, "digit0": 0xBE6},
    "te": {"name": "Telugu", "label": "తెలుగు", "family": "Dr", "script": "telugu", "it2": "tel_Telu", "nllb": "tel_Telu", "mgsm": "te", "digit0": 0xC66},
    "mr": {"name": "Marathi", "label": "मराठी", "family": "IA", "script": "devanagari", "it2": "mar_Deva", "nllb": "mar_Deva", "mgsm": None, "digit0": 0x966},
    "gu": {"name": "Gujarati", "label": "ગુજરાતી", "family": "IA", "script": "gujarati", "it2": "guj_Gujr", "nllb": "guj_Gujr", "mgsm": None, "digit0": 0xAE6},
    "kn": {"name": "Kannada", "label": "ಕನ್ನಡ", "family": "Dr", "script": "kannada", "it2": "kan_Knda", "nllb": "kan_Knda", "mgsm": None, "digit0": 0xCE6},
    "ml": {"name": "Malayalam", "label": "മലയാളം", "family": "Dr", "script": "malayalam", "it2": "mal_Mlym", "nllb": "mal_Mlym", "mgsm": None, "digit0": 0xD66},
    "zh": {"name": "Chinese", "label": "中文", "family": "Zh", "script": "han", "it2": None, "nllb": "zho_Hans", "mgsm": "zh", "digit0": None},
    # the other 14 scheduled languages. Labels are each language's Wikipedia self-name; check
    # them before using them as prompt labels (the word-list session verifies).
    "as": {"name": "Assamese", "label": "অসমীয়া", "family": "IA", "script": "bengali", "it2": "asm_Beng", "nllb": "asm_Beng", "mgsm": None, "digit0": 0x9E6},
    "or": {"name": "Odia", "label": "ଓଡ଼ିଆ", "family": "IA", "script": "oriya", "it2": "ory_Orya", "nllb": "ory_Orya", "mgsm": None, "digit0": 0xB66},
    "pa": {"name": "Punjabi", "label": "ਪੰਜਾਬੀ", "family": "IA", "script": "gurmukhi", "it2": "pan_Guru", "nllb": "pan_Guru", "mgsm": None, "digit0": 0xA66},
    "ur": {"name": "Urdu", "label": "اردو", "family": "IA", "script": "arabic", "it2": "urd_Arab", "nllb": "urd_Arab", "mgsm": None, "digit0": 0x6F0},
    "ne": {"name": "Nepali", "label": "नेपाली", "family": "IA", "script": "devanagari", "it2": "npi_Deva", "nllb": "npi_Deva", "mgsm": None, "digit0": 0x966},
    "sa": {"name": "Sanskrit", "label": "संस्कृतम्", "family": "IA", "script": "devanagari", "it2": "san_Deva", "nllb": "san_Deva", "mgsm": None, "digit0": 0x966},
    "mai": {"name": "Maithili", "label": "मैथिली", "family": "IA", "script": "devanagari", "it2": "mai_Deva", "nllb": "mai_Deva", "mgsm": None, "digit0": 0x966},
    "sd": {"name": "Sindhi", "label": "سنڌي", "family": "IA", "script": "arabic", "it2": "snd_Arab", "nllb": "snd_Arab", "mgsm": None, "digit0": 0x6F0},
    "ks": {"name": "Kashmiri", "label": "کٲشُر", "family": "IA", "script": "arabic", "it2": "kas_Arab", "nllb": "kas_Arab", "mgsm": None, "digit0": 0x6F0},
    "kok": {"name": "Konkani", "label": "कोंकणी", "family": "IA", "script": "devanagari", "it2": "gom_Deva", "nllb": None, "mgsm": None, "digit0": 0x966},
    "doi": {"name": "Dogri", "label": "डोगरी", "family": "IA", "script": "devanagari", "it2": "doi_Deva", "nllb": None, "mgsm": None, "digit0": 0x966},
    "mni": {"name": "Manipuri", "label": "ꯃꯤꯇꯩꯂꯣꯟ", "family": "TB", "script": "meitei", "it2": "mni_Mtei", "nllb": None, "mgsm": None, "digit0": 0xABF0},
    "brx": {"name": "Bodo", "label": "बर'", "family": "TB", "script": "devanagari", "it2": "brx_Deva", "nllb": None, "mgsm": None, "digit0": 0x966},
    "sat": {"name": "Santali", "label": "ᱥᱟᱱᱛᱟᱲᱤ", "family": "AA", "script": "olchiki", "it2": "sat_Olck", "nllb": None, "mgsm": None, "digit0": 0x1C50},
    # world controls (NLLB-200). Japanese: see the tokenizer note above.
    "ja": {"name": "Japanese", "label": "日本語", "family": "Ja", "script": "kana", "it2": None, "nllb": "jpn_Jpan", "mgsm": "ja", "digit0": None},
    "ru": {"name": "Russian", "label": "Русский", "family": "Ru", "script": "cyrillic", "it2": None, "nllb": "rus_Cyrl", "mgsm": "ru", "digit0": None},
    "ar": {"name": "Arabic", "label": "العربية", "family": "Ar", "script": "arabic", "it2": None, "nllb": "arb_Arab", "mgsm": None, "digit0": 0x660},
}

ORIGINAL_INDIC = ("hi", "bn", "ta", "te", "mr", "gu", "kn", "ml")
NEW_INDIC = ("as", "or", "pa", "ur", "ne", "sa", "mai", "sd", "ks", "kok", "doi", "mni", "brx", "sat")
SCHEDULED = ORIGINAL_INDIC + NEW_INDIC  # the 22 languages of the Eighth Schedule
WORLD = ("ja", "ru", "ar")
# Every language a translated prompt set exists for, in a fixed order (English first).
XLING = ("en",) + SCHEDULED + WORLD


def script_of(ch: str) -> str | None:
    cp = ord(ch)
    for name, ranges in SCRIPTS.items():
        if any(lo <= cp <= hi for lo, hi in ranges):
            return name
    return None


def letter_script(ch: str) -> str | None:
    """:func:`script_of` for Unicode letters and combining marks only (categories L*, M*). The SCRIPTS
    ranges also hold digits and symbols (Devanagari ०-९, Latin-1 × ÷), which are not writing."""
    return script_of(ch) if unicodedata.category(ch)[0] in "LM" else None


def script_shares(text: str) -> dict[str, float]:
    """Share of the text's letters (and combining marks) in each script. Digits, including native
    ones, symbols, spaces and punctuation don't count (until 2026-10-20 anything inside a script's
    code-point range did, so ०-९ counted as Devanagari and × ÷ as Latin)."""
    counts = Counter(s for s in map(letter_script, text) if s)
    total = sum(counts.values()) or 1
    return {name: round(counts[name] / total, 3) for name in SCRIPTS}


def native_digits(lang: str) -> str | None:
    """The language's own 0-9, or None."""
    zero = LANGS[lang]["digit0"]
    return None if zero is None else "".join(chr(zero + i) for i in range(10))


def to_native_digits(text: str, lang: str) -> str:
    digits = native_digits(lang)
    return text if digits is None else text.translate(str.maketrans("0123456789", digits))
