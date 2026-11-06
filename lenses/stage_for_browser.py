"""Build a folder laid out exactly like the Hugging Face lens repo, for uploading through the website.

    python -I -X utf8 lenses/stage_for_browser.py            # final: refuses while TODO- placeholders remain
    python -I -X utf8 lenses/stage_for_browser.py --preview  # build it anyway, to look at

The folder (default C:\\Users\\<you>\\sarvam-lens-hf-upload, outside OneDrive so it isn't synced) gets
every file in lenses/hf_manifest.json under its repo name: MODEL_CARD.md becomes README.md,
NOTICE_HF.md becomes NOTICE.md, and so on. All files sit at the top level, so you can select them
all and drag them onto the website's "Upload files" page.

The four 1.2 GB lenses are hard links to the originals (no extra disk space, same bytes); the small
text files are copies, so re-run this after editing the card. The folder is checked against
SHA256SUMS before the script finishes.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "lenses" / "hf_manifest.json"
SUMS = ROOT / "lenses" / "SHA256SUMS"
PLACEHOLDER = re.compile(r"TODO-[A-Z-]+")
LINK_ABOVE = 10 * 1024 * 1024  # hard-link files bigger than 10 MB, copy the rest
MARKER = ".staged_by_sarvam_lens"  # a dot file, so selecting the files to upload skips it on most systems


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path.home() / "sarvam-lens-hf-upload"))
    ap.add_argument("--preview", action="store_true", help="build even if TODO- placeholders remain")
    args = ap.parse_args()

    files = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    left = {}
    for local in files:
        if local.endswith(".md"):
            found = sorted(set(PLACEHOLDER.findall((ROOT / local).read_text(encoding="utf-8"))))
            if found:
                left[local] = found
    if left and not args.preview:
        print("Fill these placeholders first (or pass --preview):")
        for local, found in left.items():
            print(f"  {local}: {', '.join(found)}")
        return 1

    out = Path(args.out)
    marker = out / MARKER
    if out.exists():
        if not marker.exists() and any(out.iterdir()):
            print(f"{out} exists and wasn't made by this script; pick another --out")
            return 1
        shutil.rmtree(out)  # our own staging folder: links and copies only, originals untouched
    out.mkdir(parents=True)
    marker.write_text("staged by lenses/stage_for_browser.py; don't upload this file\n", encoding="utf-8")
    for local, repo in files.items():
        src, dst = ROOT / local, out / repo
        if "/" in repo:
            print(f"refusing nested path {repo}: the browser upload expects a flat folder")
            return 1
        if src.stat().st_size > LINK_ABOVE:
            try:
                os.link(src, dst)
                how = "link"
            except OSError:
                shutil.copy2(src, dst)
                how = "copy"
        else:
            shutil.copy2(src, dst)
            how = "copy"
        print(f"  {how:4s} {repo:30s} <- {local}")

    bad = []
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            if sha256(out / name.strip()) != digest:
                bad.append(name.strip())
    if bad:
        print("sha256 mismatch: " + ", ".join(bad))
        return 1
    total = sum(p.stat().st_size for p in out.iterdir() if p.name != MARKER)
    print(f"\nStaged {len(files)} files, {total / 1e9:.2f} GB, at {out}; all checksums match.")
    if left:
        print("PREVIEW ONLY: placeholders remain in " + ", ".join(left) + ". Fill them and re-run before uploading.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
