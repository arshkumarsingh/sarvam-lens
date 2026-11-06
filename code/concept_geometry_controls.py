"""Controls for the concept-geometry run (docs/11 §4d), computed locally from a downloaded run's saved
vectors (``vectors_<carrier>.pt``) and fair-readout rows (``fair_rows.csv``):

- centroid proximity with the two confounds of the plain leave-one-out centroid removed: the sample is
  mostly Indo-Aryan (15 of 27 languages), and Devanagari languages share whole word strings with Hindi.
  ``xscript``: the centroid of the OTHER-SCRIPT languages only; ``xscript_family``: the same, averaged
  within language family first and then across families (each family weighs the same).
- hub retrieval: is a concept found more often in English than in Hindi (or Chinese) from every other
  language? (mean over source languages of top-1 accuracy, paired bootstrap over concepts)
- the fair readout at size-scaled thresholds: rank within the own script <= 0.1% / 1% of the script.

    python concept_geometry_controls.py ../results/xl_geom_30b/newlang_experiment_geometry --band 13-16
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch as t

import languages

FAMILY = {code: spec["family"] for code, spec in languages.LANGS.items()}
# Script MEMBERSHIP as sets. Japanese is written in kana and Han, so it shares Han with Chinese (31 of
# its 32 words here are pure Han, 13 identical to the Chinese word). Two languages are "other-script"
# only when their sets are disjoint. (Until 2026-10-20 these were strings, "kana+han" != "han", which
# left Chinese in Japanese's other-script centroid and vice versa.)
SCRIPTS_OF = {code: frozenset({"kana", "han"} if code == "ja" else {spec["script"]})
              for code, spec in languages.LANGS.items()}


def other_script(a: str, b: str) -> bool:
    return not (SCRIPTS_OF[a] & SCRIPTS_OF[b])


def partners(langs: list[str], j: int, mode: str) -> list[int]:
    """Indices of the languages whose vectors form language ``j``'s centroid."""
    return [k for k, l in enumerate(langs) if k != j and (mode == "all" or other_script(l, langs[j]))]


def unit_vectors(V: t.Tensor, cells, layer: int):
    """Per-language centred, unit-normed vectors: (langs, concepts, U [K, C, d], present [K, C])."""
    concepts = sorted({c for c, _, _ in cells})
    langs = list(dict.fromkeys(lang for _, lang, _ in cells))
    ci_, li_ = {c: i for i, c in enumerate(concepts)}, {l: j for j, l in enumerate(langs)}
    X = V[:, layer, :].float()
    U = t.zeros((len(langs), len(concepts), X.shape[1]))
    present = t.zeros((len(langs), len(concepts)), dtype=t.bool)
    rows = defaultdict(list)
    for row, (c, lang, _) in enumerate(cells):
        rows[lang].append((row, ci_[c]))
    for lang, rr in rows.items():
        idx = t.tensor([r for r, _ in rr])
        Xl = X[idx]
        U[li_[lang], t.tensor([c for _, c in rr])] = t.nn.functional.normalize(Xl - Xl.mean(0, keepdim=True), dim=-1)
        present[li_[lang], t.tensor([c for _, c in rr])] = True
    return langs, concepts, U, present


def proximity(U, present, langs, mode: str) -> np.ndarray:
    """``[K, C]`` cosine of each language's concept vector to a centroid of the other languages
    (NaN where absent). ``mode``: "all" (leave-one-out), "xscript" (other scripts only),
    "xscript_family" (other scripts, families weighted equally)."""
    K, C = present.shape
    out = np.full((K, C), np.nan)
    for j, lang in enumerate(langs):
        others = partners(langs, j, mode)
        if mode == "xscript_family":
            fams = defaultdict(list)
            for k in others:
                fams[FAMILY[langs[k]]].append(k)
            parts = []
            for ks in fams.values():
                m = present[ks].float()  # [F, C]
                s = (U[ks] * m[..., None]).sum(0) / m.sum(0).clamp(min=1)[:, None]
                parts.append((s, (m.sum(0) > 0).float()))
            cen = sum(p * w[:, None] for p, w in parts) / sum(w for _, w in parts).clamp(min=1)[:, None]
            ok = sum(w for _, w in parts) > 0
        else:
            m = present[others].float()
            cen = (U[others] * m[..., None]).sum(0) / m.sum(0).clamp(min=1)[:, None]
            ok = m.sum(0) > 0
        cos = (U[j] * t.nn.functional.normalize(cen, dim=-1)).sum(-1)
        valid = present[j] & ok
        out[j, valid.numpy()] = cos[valid].numpy()
    return out


def boot_mean(values: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    """Mean over the non-NaN entries of ``values`` [C] and its 95% bootstrap CI (``draws`` resample concepts)."""
    v = np.where(np.isnan(values), 0.0, values)
    w = (~np.isnan(values)).astype(float)
    means = v[draws].sum(1) / np.maximum(w[draws].sum(1), 1)
    return float(np.nansum(values) / max(w.sum(), 1)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def hub_retrieval(U, present, langs, hubs=("en", "hi", "zh")) -> dict[str, np.ndarray]:
    """``{hub: [C]}`` mean over source languages (not a hub) of whether the concept's nearest
    neighbour among the hub's concepts is itself."""
    li_ = {l: j for j, l in enumerate(langs)}
    out = {}
    sources = [j for j, l in enumerate(langs) if l not in hubs]
    for hub in hubs:
        if hub not in li_:
            continue
        b = li_[hub]
        sims = t.einsum("acd,ed->ace", U[sources], U[b]).masked_fill(~present[b][None, None, :], float("-inf"))
        correct = (sims.argmax(-1) == t.arange(present.shape[1])[None, :]).float()
        valid = (present[sources] & present[b][None, :]).float()
        out[hub] = ((correct * valid).sum(0) / valid.sum(0).clamp(min=1)).numpy()
    return out


def fair_thresholds(rows_path: Path) -> list[dict]:
    """Band-mean rates: rank within the own script <= 10, <= 0.1% and <= 1% of the script's size."""
    with rows_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        rank, size = int(r["rank_script"]), int(r["script_size"])
        groups[(r["lens"], r["role"])][(r["pair"], r["concept"])].append(
            (rank <= 10, rank <= max(1, 0.001 * size), rank <= max(1, 0.01 * size), rank / size))
    out = []
    for (lens, role), items in sorted(groups.items()):
        vals = np.array([np.mean(v, axis=0) for v in items.values()])
        out.append({"lens": lens, "role": role, "items": len(items), "top10": round(float(vals[:, 0].mean()), 3),
                    "top0.1pct": round(float(vals[:, 1].mean()), 3), "top1pct": round(float(vals[:, 2].mean()), 3),
                    "median_rank_fraction": round(float(np.median(vals[:, 3])), 5)})
    return out


def self_check() -> int:
    """Regression check (CPU, no data): overlapping scripts never partner in the other-script modes."""
    langs = list(languages.XLING) + ["zh"]
    ok = True
    for j, lang in enumerate(langs):
        for k in partners(langs, j, "xscript"):
            if SCRIPTS_OF[langs[k]] & SCRIPTS_OF[lang]:
                print(f"FAIL {lang} partners {langs[k]} (shared script)")
                ok = False
    by = {l: {langs[k] for k in partners(langs, j, "xscript")} for j, l in enumerate(langs)}
    for lang, banned in (("ja", "zh"), ("zh", "ja"), ("hi", "mr"), ("ur", "ar"), ("as", "bn")):
        if banned in by[lang]:
            print(f"FAIL {lang}'s other-script centroid includes {banned}")
            ok = False
    for lang, wanted in (("ja", "en"), ("zh", "hi"), ("en", "zh")):
        if wanted not in by[lang]:
            print(f"FAIL {lang}'s other-script centroid lacks {wanted}")
            ok = False
    print("self-check", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?")
    parser.add_argument("--band")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out", default="geometry_controls_v2.json",
                        help="file name in run_dir (v2 = set-based scripts; geometry_controls.json is the pre-fix v1)")
    parser.add_argument("--self-check", action="store_true", help="run the script-overlap regression check and exit")
    args = parser.parse_args()
    if args.self_check:
        return self_check()
    if not args.run_dir or not args.band:
        parser.error("run_dir and --band are required")
    run = Path(args.run_dir)
    lo, hi = (int(x) for x in args.band.split("-"))
    band = list(range(lo, hi + 1))
    report: dict = {"band": band, "carriers": {}}
    for carrier in ("word", "line"):
        path = run / f"vectors_{carrier}.pt"
        if not path.is_file():
            continue
        saved = t.load(path)
        V, cells = saved["vectors"], [tuple(c) for c in saved["cells"]]
        per_mode = defaultdict(list)  # mode -> per band layer: {lang: (mean, lo, hi)}
        hub_layers = []
        for layer in band:
            langs, concepts, U, present = unit_vectors(V, cells, layer)
            draws = np.random.default_rng(0).integers(0, len(concepts), size=(args.resamples, len(concepts)))
            for mode in ("all", "xscript", "xscript_family"):
                prox = proximity(U, present, langs, mode)
                stats = {l: boot_mean(prox[j], draws) for j, l in enumerate(langs)}
                j_en, j_hi = langs.index("en"), langs.index("hi")
                diff = prox[j_en] - prox[j_hi]
                stats["en_minus_hi"] = boot_mean(diff, draws)
                if "zh" in langs:
                    stats["en_minus_zh"] = boot_mean(prox[j_en] - prox[langs.index("zh")], draws)
                per_mode[mode].append(stats)
            hubs = hub_retrieval(U, present, langs)
            hub_layers.append({h: boot_mean(v, draws) for h, v in hubs.items()} |
                              {"en_minus_hi": boot_mean(hubs["en"] - hubs["hi"], draws)} |
                              ({"en_minus_zh": boot_mean(hubs["en"] - hubs["zh"], draws)} if "zh" in hubs else {}))
        summary = {}
        for mode, layers_stats in per_mode.items():
            langs_ = [k for k in layers_stats[0] if not k.startswith("en_minus")]
            mean_by_lang = {l: float(np.mean([s[l][0] for s in layers_stats])) for l in langs_}
            order = sorted(mean_by_lang, key=lambda l: -mean_by_lang[l])
            summary[mode] = {"order": order, "en_rank": order.index("en") + 1, "n_langs": len(order),
                             "by_lang": {l: round(v, 4) for l, v in mean_by_lang.items()},
                             "en_minus_hi_by_layer": [[round(x, 4) for x in s["en_minus_hi"]] for s in layers_stats],
                             "en_minus_zh_by_layer": [[round(x, 4) for x in s["en_minus_zh"]] for s in layers_stats]
                             if "en_minus_zh" in layers_stats[0] else None}
        summary["hub_retrieval_by_layer"] = [{k: [round(x, 4) for x in v] for k, v in h.items()} for h in hub_layers]
        report["carriers"][carrier] = summary
    if (run / "fair_rows.csv").is_file():
        report["fair_thresholds"] = fair_thresholds(run / "fair_rows.csv")
    report["scripts"] = "sets (ja = kana+han overlaps zh = han); v1 compared script strings"
    (run / args.out).write_text(json.dumps(report, indent=1))
    for carrier, summary in report["carriers"].items():
        for mode in ("all", "xscript", "xscript_family"):
            s = summary[mode]
            print(f"{carrier:4s} {mode:15s} en rank {s['en_rank']:2d}/{s['n_langs']}  top6 {s['order'][:6]}  "
                  f"en-hi by layer {[x[0] for x in s['en_minus_hi_by_layer']]}  CIs {[x[1:] for x in s['en_minus_hi_by_layer']][:2]}...")
        for layer, h in zip(band, summary["hub_retrieval_by_layer"]):
            print(f"{carrier:4s} hub retrieval L{layer}: " + "  ".join(f"{k} {v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]" for k, v in h.items()))
    for r in report.get("fair_thresholds", []):
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
