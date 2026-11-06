"""Check a downloaded copy of the Sarvam-30B lens repo.

Usage, from the folder you downloaded into (or pass it as an argument):

    hf download <repo-id> --local-dir sarvam-lenses
    python verify_lenses.py sarvam-lenses

It checks, without loading Sarvam-30B:
1. every file in SHA256SUMS is present and its sha256 matches;
2. each lens loads with torch.load(weights_only=True) (no pickled code runs) and has 18 source layers,
   0-17, each a 4096 x 4096 float32 Jacobian, with a config matching its stamp;
3. the router has 18 layers of 8 centroids in a 64-dim PCA projection of the 4096-dim residual.

Hashes need only the standard library; steps 2-3 need torch (skipped with a note if it's missing).
Exit code 0 means every check passed.
"""

import hashlib
import json
import sys
from pathlib import Path

LENSES = {
    "j_200p.pt": {"checkpoint_name": "j_200p_pooled", "num_prompts_trained_on": 200, "lrp_mode": "none"},
    "r_moe_25p.pt": {"checkpoint_name": "r_moe_25p_pooled", "num_prompts_trained_on": 25, "lrp_mode": "r+moe"},
    "jpp_25p.pt": {"checkpoint_name": "jpp_25p", "num_prompts_trained_on": 25, "lrp_mode": "r+moe"},
    "j_indic_27p.pt": {"checkpoint_name": "j_indic_27p_pooled", "num_prompts_trained_on": 27, "lrp_mode": "none"},
}
LAYERS = list(range(18))
D_MODEL = 4096


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def check_hashes(root: Path) -> list[str]:
    errors = []
    sums = root / "SHA256SUMS"
    if not sums.exists():
        return ["SHA256SUMS is missing"]
    for line in sums.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        path = root / name.strip()
        if not path.exists():
            errors.append(f"missing: {name}")
            continue
        got = sha256(path)
        status = "ok" if got == digest else "MISMATCH"
        print(f"  {status:8s} {name}")
        if got != digest:
            errors.append(f"sha256 mismatch: {name}")
    return errors


def check_tensors(root: Path) -> list[str]:
    try:
        import torch
    except ImportError:
        print("  torch not installed: structure checks skipped (hashes above still verify the files)")
        return []
    errors = []
    for name, expect in LENSES.items():
        path = root / name
        if not path.exists():
            continue
        obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        jac = obj["parameters"]["jacobians"]
        cfg = obj["config"]
        problems = []
        if sorted(jac) != LAYERS or list(obj["source_layers"]) != LAYERS:
            problems.append(f"layers {sorted(jac)}")
        for layer, mat in jac.items():
            if tuple(mat.shape) != (D_MODEL, D_MODEL) or mat.dtype != torch.float32:
                problems.append(f"layer {layer}: {tuple(mat.shape)} {mat.dtype}")
            elif not torch.isfinite(mat).all():
                problems.append(f"layer {layer}: non-finite values")
        if cfg.get("d_model") != D_MODEL:
            problems.append(f"d_model {cfg.get('d_model')}")
        for key, value in expect.items():
            if cfg.get(key) != value:
                problems.append(f"config {key}={cfg.get(key)!r}, expected {value!r}")
        stamp = root / name.replace(".pt", "_stamp.json")
        if stamp.exists():
            s = json.loads(stamp.read_text(encoding="utf-8"))
            if s.get("num_prompts_trained_on") not in (None, expect["num_prompts_trained_on"]):
                problems.append(f"stamp says {s.get('num_prompts_trained_on')} prompts")
        print(f"  {'ok' if not problems else 'FAIL':8s} {name}: 18 x {D_MODEL}x{D_MODEL} float32, "
              f"{cfg.get('num_prompts_trained_on')} prompts, lrp_mode={cfg.get('lrp_mode')}")
        errors += [f"{name}: {p}" for p in problems]
    router = root / "router.pt"
    if router.exists():
        r = torch.load(router, map_location="cpu", weights_only=True, mmap=True)
        problems = []
        if r.get("num_clusters") != 8 or r.get("projection_dim") != 64:
            problems.append(f"num_clusters={r.get('num_clusters')} projection_dim={r.get('projection_dim')}")
        for key, shape in (("pca_means", (D_MODEL,)), ("pca_components", (D_MODEL, 64)), ("centroids", (8, 64))):
            part = r.get(key, {})
            if sorted(part) != LAYERS:
                problems.append(f"{key}: layers {sorted(part)}")
            bad = [k for k, t in part.items() if tuple(t.shape) != shape]
            if bad:
                problems.append(f"{key}: wrong shape at layers {bad}")
        print(f"  {'ok' if not problems else 'FAIL':8s} router.pt: 18 layers, 8 clusters, 64-dim PCA")
        errors += [f"router: {p}" for p in problems]
    return errors


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    print(f"Checking {root}")
    print("sha256:")
    errors = check_hashes(root)
    print("structure:")
    errors += check_tensors(root)
    if errors:
        print("\nFAILED:\n  " + "\n  ".join(errors))
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
