"""Upload the Sarvam-30B lenses to a Hugging Face model repo, in one commit.

Dry run (default): checks every file in lenses/hf_manifest.json exists, that the lens files match
lenses/SHA256SUMS, and that no TODO placeholders remain in the model card, then prints the plan.
Nothing is sent.

    python -I -X utf8 lenses/upload_to_hf.py --repo-id <you>/<repo>

Upload, after `hf auth login` (older huggingface_hub: `huggingface-cli login`) (needs huggingface_hub; the project's jppenv has it):

    python -I -X utf8 lenses/upload_to_hf.py --repo-id <you>/<repo> --go

--private creates the repo private. The upload is one commit; re-running it with unchanged files
uploads nothing new (Hugging Face de-duplicates LFS content by hash).
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "lenses" / "hf_manifest.json"
SUMS = ROOT / "lenses" / "SHA256SUMS"
PLACEHOLDER = re.compile(r"TODO-[A-Z-]+")


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def preflight(files: dict[str, str], repo_id: str) -> list[str]:
    problems = []
    for local in files:
        if not (ROOT / local).exists():
            problems.append(f"missing local file: {local}")
    expected = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            expected[name.strip()] = digest
    by_repo = {repo: local for local, repo in files.items()}
    for name, digest in expected.items():
        local = by_repo.get(name)
        if local is None:
            problems.append(f"SHA256SUMS lists {name}, which isn't in the manifest")
        elif (ROOT / local).exists() and sha256(ROOT / local) != digest:
            problems.append(f"{local} no longer matches SHA256SUMS (re-run make_checksums.py)")
    for local in ("lenses/MODEL_CARD.md", "lenses/NOTICE_HF.md"):
        text = (ROOT / local).read_text(encoding="utf-8")
        left = sorted(set(PLACEHOLDER.findall(text)))
        if left:
            problems.append(f"{local} still has placeholders: {', '.join(left)}")
        if "<REPO-ID>" in text and repo_id:
            problems.append(f"{local}: replace <REPO-ID> with {repo_id}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True, help="e.g. yourname/sarvam-30b-lenses")
    ap.add_argument("--go", action="store_true", help="actually upload (default: dry run)")
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    args = ap.parse_args()

    files = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    total = sum((ROOT / p).stat().st_size for p in files if (ROOT / p).exists())
    print(f"Plan: {len(files)} files, {total / 1e9:.2f} GB, to https://huggingface.co/{args.repo_id}")
    for local, repo in files.items():
        size = (ROOT / local).stat().st_size if (ROOT / local).exists() else 0
        print(f"  {local:42s} -> {repo:30s} {size / 1e6:10.1f} MB")

    problems = preflight(files, args.repo_id)
    if problems:
        print("\nNot ready:\n  " + "\n  ".join(problems))
        return 1
    print("\nPreflight passed.")
    if not args.go:
        print("Dry run only. Add --go to upload.")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=repo, path_or_fileobj=str(ROOT / local)) for local, repo in files.items()]
    info = api.create_commit(args.repo_id, operations=ops, repo_type="model",
                             commit_message="Sarvam-30B lenses: J (200p), R (25p), J++ (25p), Indic J (27p), router")
    print(f"Uploaded: {info.commit_url}")
    print("Check it: hf download", args.repo_id, "--local-dir sarvam-lenses && python sarvam-lenses/verify_lenses.py sarvam-lenses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
