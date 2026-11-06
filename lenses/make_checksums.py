"""Write lenses/SHA256SUMS for the Hugging Face lens repo.

Reads lenses/hf_manifest.json, hashes every file listed under "checksummed" (by its local path) and
writes "<sha256>  <path in repo>" lines, the format `sha256sum -c` reads. Run from the project root:

    python -I -X utf8 lenses/make_checksums.py
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "lenses" / "hf_manifest.json"
OUT = ROOT / "lenses" / "SHA256SUMS"


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_repo_path = {repo: local for local, repo in manifest["files"].items()}
    lines = []
    for repo_path in manifest["checksummed"]:
        local = ROOT / by_repo_path[repo_path]
        digest = sha256(local)
        lines.append(f"{digest}  {repo_path}")
        print(f"{digest}  {repo_path}  ({local.stat().st_size:,} bytes)")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(lines)} files)")


if __name__ == "__main__":
    main()
