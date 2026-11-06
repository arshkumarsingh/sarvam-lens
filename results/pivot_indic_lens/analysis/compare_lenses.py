"""ALL_INDIC readouts by lens from a pivot run's rows.csv, plus paired lens differences.

    python compare_lenses.py <pivot_dir> --lenses logit,j_indic_27p,j_wiki_40p,j_200p --ref j_wiki_40p --out report.json

Per (lens, layer, role): top-10 rate and mean probability over the kept Indic->Indic items,
95% CIs from a bootstrap over concepts (items of one concept resampled together). Paired
differences (lens - ref) use the same items, so the CI excludes item variation that both share.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

INDIC = {"bn", "ta", "te", "mr", "gu", "kn", "ml"}  # Indic->Indic pairs exclude hi and hi_latn (doc 02 §2.3)
ROLES = ("english", "english_shuffled", "hindi", "hindi_shuffled", "target", "chinese", "chinese_shuffled", "source")


def load(pivot_dir: Path):
    data = defaultdict(dict)  # (lens, layer, role) -> {(pair, concept): (top10, prob)}
    for r in csv.DictReader((pivot_dir / "rows.csv").open(encoding="utf-8")):
        src, tgt = r["pair"].split("-")
        if src in INDIC and tgt in INDIC:
            data[(r["lens"], int(r["layer"]), r["role"])][(r["pair"], r["concept"])] = (
                float(r["top10"] == "True"), float(r["prob"]))
    return data


def boot(values: dict, n: int = 2000, seed: int = 0):
    """Mean and 95% CI of values {(pair, concept): x}, resampling concepts. NaN (a role with an
    empty token set: prob only, as pivot_experiment.summarize does) is dropped."""
    by_concept = defaultdict(list)
    for (_, concept), x in values.items():
        if not np.isnan(x):
            by_concept[concept].append(x)
    if not by_concept:
        return float("nan"), float("nan"), float("nan")
    concepts = sorted(by_concept)
    sums = np.array([sum(by_concept[c]) for c in concepts])
    counts = np.array([len(by_concept[c]) for c in concepts])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(concepts), size=(n, len(concepts)))
    means = sums[idx].sum(1) / counts[idx].sum(1)
    return float(sums.sum() / counts.sum()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pivot_dir")
    p.add_argument("--lenses", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--layers", default="9-17")
    p.add_argument("--band", default="14-16")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lo, hi = map(int, a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    blo, bhi = map(int, a.band.split("-"))
    band = list(range(blo, bhi + 1))
    lenses = a.lenses.split(",")
    data = load(Path(a.pivot_dir))
    out: dict = {"n_items": {}, "by_layer": {}, "band": {}, "paired_vs_ref": {}, "onset": {}}
    for lens in lenses:
        out["n_items"][lens] = len(data[(lens, layers[0], "target")])
        for role in ROLES:
            for layer in layers:
                vals = data[(lens, layer, role)]
                t = boot({k: v[0] for k, v in vals.items()})
                m = boot({k: v[1] for k, v in vals.items()})
                out["by_layer"].setdefault(lens, {}).setdefault(role, {})[layer] = {"top10": t, "prob": m}
            # band average per item, then bootstrap
            items = set.intersection(*(set(data[(lens, l, role)]) for l in band))
            out["band"].setdefault(lens, {})[role] = {
                "top10": boot({k: np.mean([data[(lens, l, role)][k][0] for l in band]) for k in items}),
                "prob": boot({k: np.mean([data[(lens, l, role)][k][1] for l in band]) for k in items}),
            }
        eng = out["by_layer"][lens]["english"]
        out["onset"][lens] = next((l for l in layers if eng[l]["top10"][0] >= 0.5), None)
    for lens in lenses:
        if lens == a.ref:
            continue
        for role in ROLES:
            for metric, j in (("top10", 0), ("prob", 1)):
                per_layer = {}
                for layer in layers + ["band"]:
                    ls = band if layer == "band" else [layer]
                    items = set.intersection(*(set(data[(x, l, role)]) for x in (lens, a.ref) for l in ls))
                    d = {k: np.mean([data[(lens, l, role)][k][j] - data[(a.ref, l, role)][k][j] for l in ls]) for k in items}
                    per_layer[layer] = boot(d)
                out["paired_vs_ref"].setdefault(lens, {}).setdefault(role, {})[metric] = per_layer
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")

    def fmt(x, pct=True):
        f = (lambda v: f"{100 * v:.0f}") if pct else (lambda v: f"{v:.3f}")
        return f"{f(x[0])} [{f(x[1])},{f(x[2])}]"

    print(f"items: {out['n_items']}  English top-10 onset (>=50%): {out['onset']}")
    for role in ROLES:
        print(f"\n== {role}: top-10 % [95% CI] / mean prob, layers {layers}")
        print("lens".ljust(13) + "".join(f"L{l}".ljust(15) for l in layers))
        for lens in lenses:
            print(lens.ljust(13) + "".join(fmt(out["by_layer"][lens][role][l]["top10"]).ljust(15) for l in layers))
            print("".ljust(13) + "".join(fmt(out["by_layer"][lens][role][l]["prob"], pct=False).ljust(15) for l in layers))
    print(f"\n== band L{blo}-{bhi} mean per item; paired difference vs {a.ref} (top-10 pts, prob)")
    for lens in lenses:
        line = lens.ljust(13)
        for role in ROLES:
            b = out["band"][lens][role]
            line += f"{role}: {fmt(b['top10'])} p={b['prob'][0]:.3f}  "
        print(line)
        if lens != a.ref:
            line = "  - ref".ljust(13)
            for role in ROLES:
                dt = out["paired_vs_ref"][lens][role]["top10"]["band"]
                dp = out["paired_vs_ref"][lens][role]["prob"]["band"]
                line += f"{role}: {100 * dt[0]:+.1f} [{100 * dt[1]:+.1f},{100 * dt[2]:+.1f}] dp={dp[0]:+.3f} [{dp[1]:+.3f},{dp[2]:+.3f}]  "
            print(line)


if __name__ == "__main__":
    main()
