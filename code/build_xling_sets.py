"""The English prompt sets the cross-lingual stages use (docs/11), with sources and licenses.
Writes ``data/xling/en_sets.json``; ``translate.py`` adds the other 25 languages beside each item.

Sets (seed-0 samples where a source is larger; items keep their source order):
  xstest     XSTest (Röttger et al., NAACL 2024; CC-BY-4.0): all 450 prompts. The 250 safe ones look
             unsafe ("kill a Python process"); the 200 "contrast_*" ones are unsafe. Refusal on the
             first is over-refusal, on the second correct refusal.
  mgsm       MGSM (Shi et al., 2022; CC-BY-SA-4.0): the 250 test problems with their answers, plus the
             HUMAN translations that exist (bn, te, ja, ru, zh), kept beside the machine ones as the
             check on machine translation.
  science    ARC-Challenge test (Clark et al., 2018; CC-BY-SA-4.0): 150 question stems that stand alone
             without the answer options.
  technical  MMLU test (Hendrycks et al., 2021; MIT): 150 standalone stems from computer science,
             computer security, electrical engineering and machine learning.
  everyday   databricks-dolly-15k (Databricks, 2023; CC-BY-SA-3.0): 150 open_qa / general_qa /
             brainstorming instructions with no context passage.

Run from code/: ``python build_xling_sets.py`` (downloads from GitHub and the Hugging Face Hub).
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "xling" / "en_sets.json"
N_SAMPLE = 150
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
MGSM_HUMAN = ("bn", "te", "ja", "ru", "zh")
MMLU_SUBJECTS = ("college_computer_science", "high_school_computer_science", "computer_security",
                 "electrical_engineering", "machine_learning")
# A stem that needs its options, a figure or a table can't be asked on its own.
NEEDS_OPTIONS = re.compile(r"\b(following|these|above|below|shown|diagram|table|graph|picture|figure|statement|"
                           r"statements|option|options|choose|select)\b|_{2,}|\n", re.I)


def _parquet(repo: str, path: str) -> pd.DataFrame:
    return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset"))


def _sample(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    if len(rows) <= n:
        return rows
    keep = sorted(np.random.default_rng(seed).choice(len(rows), size=n, replace=False).tolist())
    return [rows[i] for i in keep]


def _standalone(text: str) -> bool:
    text = text.strip()
    return 25 <= len(text) <= 260 and text.endswith("?") and not NEEDS_OPTIONS.search(text)


def xstest() -> dict:
    text = requests.get(XSTEST_URL, timeout=60).text
    rows = list(csv.DictReader(io.StringIO(text)))
    items = [{"id": f"xstest-{r['id']}", "en": r["prompt"].strip(),
              "meta": {"type": r["type"], "label": r["label"], "unsafe": r["type"].startswith("contrast_"),
                       "focus": r.get("focus", ""), "note": r.get("note", "")}} for r in rows]
    return {"source": XSTEST_URL, "license": "CC-BY-4.0",
            "citation": "Röttger et al. 2024, XSTest (NAACL)", "items": items}


def mgsm() -> dict:
    en = _parquet("juletxara/mgsm", "en/test-00000-of-00001.parquet")
    human = {lang: _parquet("juletxara/mgsm", f"{lang}/test-00000-of-00001.parquet") for lang in MGSM_HUMAN}
    items = []
    for i, row in en.iterrows():
        assert all(int(h.loc[i, "answer_number"]) == int(row["answer_number"]) for h in human.values()), i
        items.append({"id": f"mgsm-{i:03d}", "en": row["question"].strip(),
                      "meta": {"answer": int(row["answer_number"]),
                               "human": {lang: h.loc[i, "question"].strip() for lang, h in human.items()}}})
    return {"source": "huggingface.co/datasets/juletxara/mgsm (test)", "license": "CC-BY-SA-4.0",
            "citation": "Shi et al. 2022, Language models are multilingual chain-of-thought reasoners", "items": items}


def science() -> dict:
    df = _parquet("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet")
    rows = [{"id": f"arc-{r['id']}", "en": r["question"].strip(), "meta": {"answer_key": r["answerKey"]}}
            for _, r in df.iterrows() if _standalone(r["question"])]
    return {"source": "huggingface.co/datasets/allenai/ai2_arc (ARC-Challenge test)", "license": "CC-BY-SA-4.0",
            "citation": "Clark et al. 2018, Think you have solved question answering? Try ARC",
            "n_standalone": len(rows), "items": _sample(rows, N_SAMPLE)}


def technical() -> dict:
    rows = []
    for subject in MMLU_SUBJECTS:
        df = _parquet("cais/mmlu", f"{subject}/test-00000-of-00001.parquet")
        rows += [{"id": f"mmlu-{subject}-{i:03d}", "en": r["question"].strip(), "meta": {"subject": subject}}
                 for i, r in df.iterrows() if _standalone(r["question"])]
    return {"source": "huggingface.co/datasets/cais/mmlu (test: " + ", ".join(MMLU_SUBJECTS) + ")", "license": "MIT",
            "citation": "Hendrycks et al. 2021, Measuring massive multitask language understanding",
            "n_standalone": len(rows), "items": _sample(rows, N_SAMPLE)}


def everyday() -> dict:
    path = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    rows = []
    with open(path, encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            r = json.loads(line)
            text = r["instruction"].strip()
            if (r["category"] in ("open_qa", "general_qa", "brainstorming") and not r["context"].strip()
                    and 20 <= len(text) <= 200 and "\n" not in text and text.isascii()):
                rows.append({"id": f"dolly-{i:05d}", "en": text, "meta": {"category": r["category"]}})
    return {"source": "huggingface.co/datasets/databricks/databricks-dolly-15k", "license": "CC-BY-SA-3.0",
            "citation": "Conover et al. 2023, Free Dolly", "n_standalone": len(rows), "items": _sample(rows, N_SAMPLE)}


def main() -> None:
    sets = {"xstest": xstest(), "mgsm": mgsm(), "science": science(), "technical": technical(), "everyday": everyday()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"sets": sets}, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, s in sets.items():
        print(f"{name:10s} {len(s['items']):4d} items (of {s.get('n_standalone', len(s['items']))}) {s['license']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
