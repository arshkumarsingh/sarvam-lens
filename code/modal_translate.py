"""Modal app for the prompt-set translations (docs/11). Separate from modal_app.py because it needs
an older transformers (IndicTrans2's remote code and IndicTransToolkit) and the Hugging Face secret
(IndicTrans2 is gated); keeping both out of the main app means a missing secret can't break it.

One-time setup by the user: accept the terms at
https://huggingface.co/ai4bharat/indictrans2-en-indic-1B, then create a Modal secret named
"huggingface-secret" holding a read token as HF_TOKEN (the dashboard's Hugging Face template), or
    modal secret create huggingface-secret HF_TOKEN=<read token>

    modal run modal_translate.py --step smoke     # 3 items x (hi, mni, sat, ja): checks both models
    modal run modal_translate.py --step run       # every item x 25 languages (resumes per language)
    modal volume get sarvam-lens corpora/xling/parts data/xling/
    python -c "import translate, languages as L; print(translate.assemble('data/xling/en_sets.json', 'data/xling/parts', 'data/xling', list(L.XLING[1:])))"
    modal run modal_translate.py --step fetch --repo SurakshaAI/SurakshaEval   # a gated dataset to the Volume (CPU)

L40S at $1.95/h (Modal list price 2026-10-19) + CPU/memory = $2.39/h; the run's hard timeout bounds its cost.
"""

from __future__ import annotations

import json

import modal

VOLUME_ROOT = "/vol"
OUT_DIR = f"{VOLUME_ROOT}/corpora/xling"
TIMEOUT_MIN = 75
HF_SECRET = "huggingface-secret"  # the Modal dashboard's Hugging Face template name
TOKEN_KEYS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")
USD_PER_HOUR = 1.95 + 4 * 0.0472 + 32 * 0.0080

app = modal.App("sarvam-lens-xlate")
volume = modal.Volume.from_name("sarvam-lens", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "sentencepiece==0.2.0", "sacremoses==0.1.1",
                 "numpy==2.2.6", "IndicTransToolkit==1.1.1", "accelerate==1.6.0")
    # weights cached on the Volume across runs; expandable segments: the first full run OOMed on fragmentation
    .env({"HF_HOME": f"{VOLUME_ROOT}/hf_cache", "PYTHONUTF8": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file("translate.py", "/xlate/translate.py")
    .add_local_file("languages.py", "/xlate/languages.py")
    .add_local_file("data/xling/en_sets.json", "/xlate/en_sets.json")
)


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, volumes={VOLUME_ROOT: volume},
              secrets=[modal.Secret.from_name(HF_SECRET)], timeout=TIMEOUT_MIN * 60, retries=0)
def translate_all(langs: list[str], smoke: bool = False) -> dict:
    import os
    import sys

    sys.path.insert(0, "/xlate")
    # Modal's Hugging Face template names the key HF_TOKEN; accept the other usual names too.
    key = next((k for k in TOKEN_KEYS if os.environ.get(k)), None)
    if key is None:
        raise RuntimeError(f"secret {HF_SECRET!r} has none of {TOKEN_KEYS}")
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ[key]
    import translate

    result = translate.run("/xlate/en_sets.json", OUT_DIR, langs=langs, commit=volume.commit, smoke=smoke)
    return {"token_key": key, **result}  # the key's name only, never its value


@app.function(image=image, gpu="L40S", cpu=4, memory=32768, volumes={VOLUME_ROOT: volume},
              secrets=[modal.Secret.from_name(HF_SECRET)], timeout=15 * 60, retries=0)
def translate_texts(texts: list[str], langs: list[str]) -> dict:
    """A short list of sentences through the same models (``--step list``)."""
    import os
    import sys

    sys.path.insert(0, "/xlate")
    key = next((k for k in TOKEN_KEYS if os.environ.get(k)), None)
    if key is None:
        raise RuntimeError(f"secret {HF_SECRET!r} has none of {TOKEN_KEYS}")
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ[key]
    import translate

    return translate.translate_list(texts, langs)


@app.function(image=image, cpu=1, memory=4096, volumes={VOLUME_ROOT: volume},
              secrets=[modal.Secret.from_name(HF_SECRET)], timeout=10 * 60, retries=0)
def fetch_dataset(repo_id: str, revision: str = "main") -> dict:
    """Snapshot a (possibly gated) Hugging Face dataset repo to the Volume, under
    corpora/hf/<owner>__<name>/<resolved revision>/ (``--step fetch``; CPU only). Returns the revision,
    licence and each file's size and SHA-256; the token's key name only, never its value."""
    import hashlib
    import os

    from huggingface_hub import HfApi, snapshot_download

    key = next((k for k in TOKEN_KEYS if os.environ.get(k)), None)
    if key is None:
        raise RuntimeError(f"secret {HF_SECRET!r} has none of {TOKEN_KEYS}")
    token = os.environ[key]
    info = HfApi().dataset_info(repo_id, revision=revision, token=token)
    dest = f"{VOLUME_ROOT}/corpora/hf/{repo_id.replace('/', '__')}/{info.sha}"
    snapshot_download(repo_id, repo_type="dataset", revision=info.sha, local_dir=dest, token=token)
    files = {}
    for root, _, names in os.walk(dest):
        if ".cache" in root:
            continue
        for name in names:
            path = os.path.join(root, name)
            with open(path, "rb") as handle:
                files[os.path.relpath(path, dest).replace(os.sep, "/")] = {
                    "bytes": os.path.getsize(path), "sha256": hashlib.sha256(handle.read()).hexdigest()}
    volume.commit()
    card = info.card_data.to_dict() if info.card_data else {}
    return {"repo_id": repo_id, "revision": info.sha, "license": card.get("license"), "gated": getattr(info, "gated", None),
            "volume_path": dest[len(VOLUME_ROOT) + 1:], "files": files, "token_key": key}


@app.local_entrypoint()
def main(step: str = "smoke", langs: str = "", texts_json: str = "", out_json: str = "", repo: str = "",
         revision: str = "main") -> None:
    import languages

    chosen = [l for l in langs.split(",") if l] or list(languages.XLING[1:])
    if step == "smoke":
        chosen = [l for l in langs.split(",") if l] or ["hi", "mni", "sat", "ja"]
        fn = translate_all.with_options(timeout=25 * 60)
        print(f"L40S at ${USD_PER_HOUR:.2f}/h, hard timeout 25 min -> worst case ${25 / 60 * USD_PER_HOUR:.2f}")
        print(json.dumps(fn.remote(chosen, smoke=True), ensure_ascii=False, indent=1))
    elif step == "run":
        print(f"L40S at ${USD_PER_HOUR:.2f}/h, hard timeout {TIMEOUT_MIN} min -> worst case ${TIMEOUT_MIN / 60 * USD_PER_HOUR:.2f}")
        print(json.dumps(translate_all.remote(chosen), ensure_ascii=False, indent=1))
    elif step == "list":  # --texts-json a JSON list of English sentences; --out-json where to write {lang: [texts]}
        texts = json.loads(open(texts_json, encoding="utf-8").read())
        print(f"L40S at ${USD_PER_HOUR:.2f}/h, hard timeout 15 min -> worst case ${15 / 60 * USD_PER_HOUR:.2f}")
        result = translate_texts.remote(texts, chosen)
        text = json.dumps({"source_en": texts, "translations": result}, ensure_ascii=False, indent=1)
        if out_json:
            open(out_json, "w", encoding="utf-8").write(text)
        print(text)
    elif step == "fetch":  # --repo owner/name [--revision sha]: a dataset snapshot to the Volume (CPU, ~$0.01)
        result = fetch_dataset.remote(repo, revision)
        text = json.dumps(result, ensure_ascii=False, indent=1)
        if out_json:
            open(out_json, "w", encoding="utf-8").write(text)
        print(text)
    else:
        raise SystemExit("--step smoke | run | list | fetch")
