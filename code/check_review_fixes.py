"""Regression checks for the outside review's findings (2026-10-20), CPU, no model or GPU:

1. geometry: languages that share a script (ja = kana + Han, zh = Han) never partner in the other-script centroid;
2. clustered bootstraps: singleton clusters reproduce the item bootstrap exactly, and rows repeated within a cluster
   widen the interval instead of narrowing it (routing_ablation.boot, domain_experiment._cluster_ci / _pool_samples);
3. neurons: tied activations share their average rank, and a constant neuron is neither script- nor concept-selective;
5. translation: a part from a reordered source is refused by assemble and redone by a resume;
6. J++ held-out guard: a scored item named in the fit-weights stamp is refused;
7. number parsing: grouping spaces (incl. no-break), decimal commas, mixed separators;
8. script shares: digits and symbols don't count as letters of any script.
Finding 4 (the decoder's deadline) is in post1_check_decode.py (tiny models); finding 9 is in code not published here.

    python check_review_fixes.py        # from code/, PYTHONUTF8=1
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

fails: list[str] = []


def check(ok: bool, name: str, detail: str = "") -> None:
    print(("  ok   " if ok else "  FAIL ") + name, detail, flush=True)
    if not ok:
        fails.append(name)


def geometry() -> None:
    import concept_geometry_controls as g

    langs = ["en", "hi", "mr", "zh", "ja", "ur", "ar"]
    by = {l: {langs[k] for k in g.partners(langs, j, "xscript")} for j, l in enumerate(langs)}
    check("zh" not in by["ja"] and "ja" not in by["zh"], "1 geometry: Japanese and Chinese share Han, not partners", str(by["ja"]))
    check("mr" not in by["hi"] and "ar" not in by["ur"] and "en" in by["ja"], "1 geometry: same-script languages excluded, others kept")
    check(g.self_check() == 0, "1 geometry: self_check over every language")


def bootstraps() -> None:
    import domain_experiment as de
    import routing_ablation as ra

    x = np.random.default_rng(1).random(40)
    singles = [np.array([i]) for i in range(len(x))]
    check(ra.boot(x) == ra.boot(x, clusters=singles), "2 routing.boot: singleton clusters == item bootstrap")
    check(de._ci(x)[1:] == de._cluster_ci(x, list(range(len(x))))[1:], "2 domain._cluster_ci: singleton clusters == _ci")
    # 8 clusters, each one value repeated 10 times: 80 rows but 8 independent draws
    y = np.repeat(np.random.default_rng(2).random(8), 10)
    groups = np.repeat(np.arange(8), 10)
    item_w = np.diff(ra.boot(y))[0]
    clus_w = np.diff(ra.boot(y, clusters=ra.cluster_index(groups)))[0]
    check(clus_w > 2 * item_w, "2 routing.boot: repeated rows widen the clustered CI", f"item {item_w:.3f} vs clustered {clus_w:.3f}")
    _, lo, hi = de._cluster_ci(y, list(groups))
    check(hi - lo > 2 * item_w, "2 domain._cluster_ci: same", f"{hi - lo:.3f}")
    # a pool of 5 languages holding the same 8 prompts: resampling prompts, not 40 independent rows
    vals = [np.random.default_rng(3).random(8) + 0.01 * k for k in range(5)]
    ids = [list(range(8))] * 5
    pooled = de._pct(de._pool_samples(vals, ids))
    naive = de._pct(np.nanmean([de._boot_means(v, seed=s) for s, v in enumerate(vals)], axis=0))
    check(pooled[1] - pooled[0] > 1.5 * (naive[1] - naive[0]), "2 domain._pool_samples: shared prompts widen the pool CI",
          f"per-language {naive[1] - naive[0]:.3f} vs by prompt {pooled[1] - pooled[0]:.3f}")


def neurons() -> None:
    import neuron_experiment as ne
    from routing_ablation import rankdata

    x = np.random.default_rng(0).integers(0, 3, size=(25, 4)).astype(float)
    check(all(np.allclose(ne.rank_columns(x)[:, j], rankdata(x[:, j])) for j in range(4)), "3 neurons: tied values share their average rank")
    z = np.zeros((3, 5, 2))
    z[0, 2, 0] = 1.0
    check(ne.unique_argmax(z, 1).tolist() == [[2, -1], [-1, -1], [-1, -1]], "3 neurons: a tied maximum has no winner")
    words = json.loads((Path(__file__).parent.parent / "results/neuron30/neurons/neuron_items.json").read_text(encoding="utf-8"))
    rows, s = ne.analyze_neurons(np.zeros((len(words), 3), np.float32), words, n_permutations=3)
    check(rows[0]["script_auroc"] == 0.5 and rows[0]["xscript_n_scripts"] == 0 and s["concept_selective"]["n"] == 0
          and s["script_selective"]["n"] == 0, "3 neurons: an all-zero neuron is selective for nothing",
          f"AUROC {rows[0]['script_auroc']}, scripts {rows[0]['xscript_n_scripts']}")


def translation() -> None:
    import translate as tr

    en = {"sets": {"a": {"items": [{"id": "a1", "en": "one"}, {"id": "a2", "en": "two"}]},
                   "b": {"items": [{"id": "b1", "en": "three"}]}}}
    swapped = {"sets": {"a": {"items": [{"id": "a2", "en": "two"}, {"id": "a1", "en": "one"}]}, "b": en["sets"]["b"]}}
    fp = tr.source_fingerprint(en)
    check(fp != tr.source_fingerprint(swapped) and fp["n"] == 3, "5 translate: reordering changes the fingerprint")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "parts").mkdir()
        (d / "en_sets.json").write_text(json.dumps(swapped), encoding="utf-8")
        part = {"lang": "hi", "model": "m", "source": fp, "texts": ["ek", "do", "teen"]}
        (d / "parts/hi.json").write_text(json.dumps(part), encoding="utf-8")
        check(not tr.part_matches(d / "parts/hi.json", tr.source_fingerprint(swapped)), "5 translate: resume redoes a part from another order")
        try:
            tr.assemble(str(d / "en_sets.json"), str(d / "parts"), str(d / "out"), ["hi"])
            refused = False
        except ValueError:
            refused = True
        check(refused, "5 translate: assemble refuses a part translated from another order")
        (d / "en_sets.json").write_text(json.dumps(en), encoding="utf-8")
        tr.assemble(str(d / "en_sets.json"), str(d / "parts"), str(d / "out"), ["hi"])
        a = json.loads((d / "out/a.json").read_text(encoding="utf-8"))
        check([i["tr"]["hi"] for i in a["items"]] == ["ek", "do"], "5 translate: the matching source assembles in order")


def heldout_guard() -> None:
    import run_jpp

    with tempfile.TemporaryDirectory() as d:
        lens = Path(d) / "jpp.pt"
        (Path(d) / "jpp_stamp.json").write_text(json.dumps({"stage": "fit-weights", "args": {}, "fitting_item_names": ["typo-x", "poetry-y"]}))
        fake = SimpleNamespace(LOGIT_LENS_SPEC="logit", stamp_path_beside=lambda p: str(Path(p).with_name(Path(p).stem + "_stamp.json")))
        item = lambda n: SimpleNamespace(name=n)  # noqa: E731
        try:
            run_jpp.refuse_in_sample_items(fake, [str(lens)], [item("assoc-a"), item("typo-x")])
            refused = False
        except SystemExit:
            refused = True
        check(refused, "6 held-out guard: a scored fitting item is refused")
        run_jpp.refuse_in_sample_items(fake, ["logit", str(lens)], [item("assoc-a")])
        check(True, "6 held-out guard: disjoint items pass")


def numbers() -> None:
    import domain_experiment as de

    cases = {"70 000": 70000, "70 000": 70000, "70 000 $": 70000, "1,5": 1.5, "16,00 $": 16, "1,500": 1500,
             "1.5": 1.5, "1,234,567": 1234567, "1.234.567": 1234567, "1,234.5": 1234.5, "1.234,5": 1234.5,
             "۷۰٬۰۰۰": 70000, "१२": 12, "1, 2, 3": 1, "-4": -4}
    bad = {k: de.parse_number(k) for k, v in cases.items() if de.parse_number(k) != v}
    check(not bad, "7 numbers: parse_number", str(bad) if bad else f"{len(cases)} cases")
    fa = {"The answer is 70 000.": 70000, "\\boxed{\\$1 430}": 1430, "= **16,00 $**": 16, "3 then 18": 18}
    bad = {k: de.final_answer(k) for k, v in fa.items() if de.final_answer(k) != v}
    check(not bad, "7 numbers: final_answer", str(bad) if bad else f"{len(fa)} cases")


def scripts() -> None:
    import languages as L

    check(L.script_shares("पानी ५६७")["devanagari"] == 1.0 and L.letter_script("५") is None, "8 scripts: Devanagari digits are not letters")
    s = L.script_shares("পানি। ×÷ 12")
    check(s["bengali"] == 1.0 and s["latin"] == 0.0 and s["devanagari"] == 0.0, "8 scripts: danda, × and ÷ count for no script", str({k: v for k, v in s.items() if v}))
    check(L.script_shares("हिन्दी")["devanagari"] == 1.0, "8 scripts: combining marks (virama, matras) still count")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    for fn in (geometry, bootstraps, neurons, translation, heldout_guard, numbers, scripts):
        print(fn.__name__)
        fn()
    print("FAILS:", fails or "none")
    sys.exit(1 if fails else 0)
