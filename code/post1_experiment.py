"""Doc 17: the follow-ups for blog post 1 ("What language does an Indian AI think in?").

Interface (modal_pilot stage ``post1_experiment.<analysis>``, as the other cross-lingual modules):
  run(ctx, analysis) -> dict          the stage's result (written into report.json)
  n_forwards(ctx, analysis) -> float  single-forward equivalents, for the budget gate
ctx is built by ``modal_pilot.xl_ctx`` (see newlang_experiment's docstring). Every CI here resamples
the 32 concepts (or, for MGSM, the problems), never items.

Analyses (docs/17 has the pre-registered decision rules):
- ``roles``     Tier 1 #1 + #2. The published 14-pair word-translation pivot (same prompts, roles, controls,
                correctness filter; batched forwards as newlang_experiment.pivot), plus every other language of
                ``concepts_new.tsv`` as a third-language role with its own shuffled control, plus two English
                semantic neighbours per concept (``data/neighbours.tsv``) with the control concept's neighbours as
                controls. Three readouts per role: prefix sets (the published metric), whole-word tokens only, and
                the prefix set's rank within the role language's own script.
- ``joint``     Tier 1 #3. causal_test's final-position projection with English, Hindi (and Chinese, by model) and
                the target word's directions removed alone and together, against rank-matched controls.
- ``digits``    Tier 2 #4. The English number word ablated on doc 12's native-digit sums.
- ``push``      Tier 2 #5. The English push at alpha 1-3 beside a different concept's English and a norm-matched
                random direction.
- ``positions`` Tier 2 #6a. The English projection at the source word, at every position, vs the last only.
- ``writers``   Tier 2 #6b. Which layers, attention heads and experts write the English direction.
- ``think``     Tier 2 #7. MGSM with thinking in English vs instructed into the prompt's language, to completion.
- ``hint``      Tier 2 #8. doc 12's wrong-hint test with 3x the problems, traces run to completion.
- ``zh``        Tier 2 #9. The Chinese word's directions ablated (the Chinese latent).

Stage args (``--stage-args``): ``pairs``, ``max_items`` (per pair), ``items`` (MGSM problems), ``max_new_tokens``,
``time_limit_s`` (decode deadline), ``max_rows`` (decode slab rows), ``temperature`` / ``top_p``; ``think_conds`` /
``en_conds`` (#7's arms), ``skip_items`` (#8: start after the first N problems); ``"<analysis>:<key>"`` sets a key
for one stage only.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import time
import warnings
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch as t

import causal_test as ct
import languages as L
import newlang_experiment as nle
import pivot_experiment as pe

logger = logging.getLogger("post1")
HERE = Path(__file__).resolve().parent

PUB_PAIRS = pe.DEFAULT_PAIRS                   # the published 14-pair pivot (ALL_INDIC = its 10 Indic pairs)
INDIC_PAIRS = ct.DEFAULT_PAIRS                 # the published ablation's 10 Indic->Indic pairs
STEER_PAIRS = "ta-te,bn-gu,kn-ml,mr-bn,te-kn"  # modal_pilot.CAUSAL_LOGIT_PAIRS: the published push (docs/09 §7)
CORE_LANGS = ("en", "hi", "zh")
THIRD = tuple(l for l in L.ORIGINAL_INDIC + L.NEW_INDIC + L.WORLD if l not in CORE_LANGS)
ORIG_OTHER = tuple(l for l in L.ORIGINAL_INDIC if l != "hi")       # the decision pool of #1
BIG10_OTHER = ORIG_OTHER + ("pa", "or")                              # Sarvam's "10 largest", Hindi aside
LOW = ("ks", "sd", "mni", "sat", "brx", "doi", "kok", "mai")         # domain_experiment.LOW_RESOURCE
OTHER_NEW = ("as", "ur", "ne", "sa")
WORLD3 = L.WORLD
POOLS = {"orig_other": ORIG_OTHER, "big10_other": BIG10_OTHER, "low_resource": LOW, "other_new": OTHER_NEW,
         "world": WORLD3}
NB_ROLES = ("nb_close", "nb_assoc")
ALPHAS = (1.0, 1.5, 2.0, 2.5, 3.0)
# The published values the reruns must reproduce (results/final/pivot_all, ALL_INDIC top-10 rate per layer;
# prefix sets): the stop rule of docs/17 #1. A model without an entry has no reproduction check.
REFERENCE = {
    "sarvam30": {"lens": "j_200p", "english": {14: 0.980, 15: 0.980, 16: 0.992}, "hindi": {14: 0.602, 15: 0.551, 16: 0.676},
                 "tol": {"english": 0.03, "hindi": 0.05}},
}
BATCH_FORWARD_COST = 2.0  # a 16-prompt forward in single forwards (docs/05, newlang_experiment)


### SHARED HELPERS


def write_csv(path: Path, table: list[dict]) -> None:
    nle.write_csv(path, table)


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cmean(values, clusters, *, resamples: int = 2000, seed: int = 0) -> dict:
    """Mean of ``values`` and its 95% bootstrap CI resampling the clusters (concepts or problems); NaNs dropped."""
    by: dict = defaultdict(list)
    for v, c in zip(values, clusters):
        if v is None:
            continue
        v = float(v)
        if not math.isnan(v):
            by[c].append(v)
    flat = [v for vs in by.values() for v in vs]
    if not flat:
        return {"mean": float("nan"), "ci": [float("nan"), float("nan")], "n": 0, "clusters": 0}
    lo, hi = nle.cluster_ci(by, resamples=resamples, seed=seed)
    return {"mean": round(float(np.mean(flat)), 4), "ci": [round(lo, 4), round(hi, 4)], "n": len(flat), "clusters": len(by)}


def _control_of(ordered: list[str], concept: str) -> str:
    """The published control concept: the fixed half-way derangement of the sorted list."""
    return ordered[(ordered.index(concept) + len(ordered) // 2) % len(ordered)]


def load_neighbours(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return {r["concept"]: {"close": r["close"].strip(), "assoc": r["assoc"].strip()} for r in csv.DictReader(handle, delimiter="\t")}


def _lens_main(ctx):
    """The model's primary lens: the 30B's J-lens, else the logit lens (ctx.lens_specs[0])."""
    spec = ctx.args.get("lens") or ctx.lens_specs[0]
    name, lens = next(iter(ctx.load_lenses([spec]).items()))
    return name, lens


def _pairs(ctx, default: str) -> list[str]:
    pairs = [p for p in (ctx.args.get("pairs") or default).split(",") if p]
    return pairs[:2] if ctx.smoke else pairs


### 1 + 2. ROLES: every language and two semantic neighbours in the pivot


def _whole(index: dict, spellings: list[str]) -> set[int]:
    """Tokens whose normalized text is a whole spelling of the word (``water``, ``▁water``, ``Water``...)."""
    out: set[int] = set()
    for word in spellings:
        out.update(index.get(pe.normalize_token_text(word), ()))
    return out


def role_items(concepts: dict, neighbours: dict, pairs: list[str], index: dict, tokenizer, *,
               max_items: int | None = None) -> tuple[list[dict], list[dict]]:
    """newlang_experiment.build_items (the published roles and controls, untouched), then for each item:
    - ``x_<lang>``: every other language's word for the concept (THIRD), minus every token a published role
      claims, with tokens two third languages share dropped from both; ``x_<lang>_shuffled``: the same
      language's word for the control concept, minus every token the item's real roles claim;
    - ``nb_close`` / ``nb_assoc``: the concept's two English neighbours (minus all claimed tokens, disjoint
      from each other) and ``nb_*_shuffled``: the control concept's neighbours;
    - ``whole``: each role's set cut to the tokens that are the whole word."""
    specs, skipped = nle.build_items(concepts, pairs, index, tokenizer, max_items=max_items)
    ordered = sorted(concepts)
    prefix = lambda words: pe.prefix_token_ids(index, words, tokenizer)
    for spec in specs:
        src, tgt, concept = spec["src"], spec["tgt"], spec["concept"]
        words, control = concepts[concept], _control_of(ordered, concept)
        core = set(pe.tracked_languages(src, tgt).values())
        claimed = set().union(*(prefix(words[lang]) for lang in core))
        thirds = [lang for lang in THIRD if lang not in core and lang in words]
        raw = {lang: prefix(words[lang]) - claimed for lang in thirds}
        clean, _ = pe.disjoint_sets(raw) if raw else ({}, set())
        claimed_all = claimed.union(*raw.values()) if raw else set(claimed)
        sets, langs = spec["sets"], spec["role_langs"]
        spell: dict[str, list[str]] = {}
        for role, lang_key in langs.items():  # the published roles' words
            lang, _, other = lang_key.partition(":")
            spell[role] = concepts[other][lang] if other else words[lang]
        for lang in thirds:
            sets[f"x_{lang}"], langs[f"x_{lang}"], spell[f"x_{lang}"] = clean[lang], lang, words[lang]
            if lang in concepts[control]:
                role = f"x_{lang}_shuffled"
                sets[role] = prefix(concepts[control][lang]) - claimed_all
                langs[role], spell[role] = f"{lang}:{control}", concepts[control][lang]
        nb, nbc = neighbours[concept], neighbours[control]
        nraw = {r: prefix([nb[r]]) - claimed_all for r in ("close", "assoc")}
        nclean, _ = pe.disjoint_sets(nraw)
        claimed_nb = claimed_all.union(*nraw.values())
        for r in ("close", "assoc"):
            sets[f"nb_{r}"], langs[f"nb_{r}"], spell[f"nb_{r}"] = nclean[r], f"en:{nb[r]}", [nb[r]]
            sets[f"nb_{r}_shuffled"] = prefix([nbc[r]]) - claimed_nb
            langs[f"nb_{r}_shuffled"], spell[f"nb_{r}_shuffled"] = f"en:{nbc[r]}", [nbc[r]]
        spec["whole"] = {role: ids & _whole(index, spell[role]) for role, ids in sets.items()}
    return specs, skipped


def all_roles() -> list[str]:
    roles = list(pe.ROLES) + list(pe.CONTROL_ROLES)
    for lang in THIRD:
        roles += [f"x_{lang}", f"x_{lang}_shuffled"]
    for r in NB_ROLES:
        roles += [r, f"{r}_shuffled"]
    return roles


def _script_of_role(lang_key: str) -> str:
    lang = lang_key.split(":")[0]
    return "han" if lang == "ja" else L.LANGS[lang]["script"]  # ja: kana is byte fallback in Sarvam's vocabulary


def score_roles(logits_LV: t.Tensor, probs_LV: t.Tensor, sets: dict[str, set[int]], scripts: dict[str, t.Tensor],
                role_script: dict[str, str], chunk: int = 16):
    """Per role (a dict of non-empty token sets): best rank over the vocabulary, the summed probability, and the
    rank within the role's own script, at every layer. ``logits_LV`` already carries the readout filter (-inf)."""
    roles = [r for r, ids in sets.items() if ids]
    out = {}
    device = logits_LV.device
    for start in range(0, len(roles), chunk):
        part = roles[start: start + chunk]
        width = max(len(sets[r]) for r in part)
        idx = t.zeros((len(part), width), dtype=t.long, device=device)
        valid = t.zeros((len(part), width), dtype=t.bool, device=device)
        for i, r in enumerate(part):
            ids = sorted(sets[r])
            idx[i, : len(ids)] = t.tensor(ids, device=device)
            valid[i, : len(ids)] = True
        g = logits_LV[:, idx].masked_fill(~valid[None], float("-inf"))    # [L, R, K]
        best = g.max(-1).values                                            # [L, R]
        prob = (probs_LV[:, idx] * valid[None]).sum(-1)                   # [L, R]
        above = logits_LV[:, None, :] > best[:, :, None]                  # [L, R, V]
        rank = above.sum(-1) + 1
        smask = t.stack([scripts[role_script[r]] for r in part])            # [R, V]
        rank_own = (above & smask[None]).sum(-1) + 1
        del above
        host = [x.cpu().numpy() for x in (rank, rank_own, prob)]
        for i, r in enumerate(part):
            out[r] = (host[0][:, i], host[1][:, i], host[2][:, i])
    return out


def run_roles(ctx) -> dict:
    import reader_questions as rqm

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    model, tok = ctx.model, ctx.model.tokenizer
    lenses = ctx.load_lenses(ctx.args.get("lenses") or ctx.lens_specs)
    names = list(lenses)
    concepts = pe.load_concepts(Path(ctx.data_dir) / "concepts_new.tsv")
    neighbours = load_neighbours(Path(ctx.data_dir) / "neighbours.tsv")
    pairs = _pairs(ctx, PUB_PAIRS)
    index = pe.build_vocab_index(tok, ctx.vocab_size)
    tracked: set[int] = set()
    for c in concepts.values():
        for spellings in c.values():
            tracked |= pe.prefix_token_ids(index, spellings, tok)
    for nb in neighbours.values():
        tracked |= pe.prefix_token_ids(index, list(nb.values()), tok)
    mask_V = pe.readout_mask(tok, ctx.vocab_size, tracked)  # the published readout filter, sparing every tracked token
    device = model._lm_head.weight.device
    scripts = {k: (v & ~mask_V).to(device) for k, v in nle.script_masks(tok, ctx.vocab_size).items()}
    reader = rqm.Reader(model, lenses, ctx.layers, ctx.vocab_size, mask_V)
    specs, skipped = role_items(concepts, neighbours, pairs, index, tok, max_items=2 if ctx.smoke else ctx.args.get("max_items"))
    roles = all_roles()
    rpos = {r: i for i, r in enumerate(roles)}
    nL, n_lens = len(ctx.layers), len(names)
    kept_specs, items = [], []
    shape = (len(specs), n_lens, nL, len(roles))
    rank_pre, rank_own, rank_ww = (np.full(shape, -1, dtype=np.int32) for _ in range(3))
    prob_pre, prob_ww = (np.full(shape, np.nan, dtype=np.float32) for _ in range(2))
    batch = int(ctx.args.get("batch_size", ctx.batch_size))
    started = time.monotonic()
    n_kept = 0
    for start in range(0, len(specs), batch):
        chunk = specs[start: start + batch]
        ids_list = [model.encode(pe.build_prompt(concepts, s["concept"], s["src"], s["tgt"], shots=5, seed=0)) for s in chunk]
        for spec, (acts, logits) in zip(chunk, reader.forward_batch(ids_list)):
            top = int(logits.argmax())
            correct = top in spec["raw_target"]
            item = {**spec["item"], "src": spec["src"], "tgt": spec["tgt"], "model_top_token": tok.decode([top]), "kept": correct}
            if not correct and not ctx.smoke:
                items.append({**item, "row": -1, "reason": "model's next token does not begin the target word"})
                continue
            row = n_kept
            n_kept += 1
            items.append({**item, "row": row, "reason": ""})
            role_script = {r: _script_of_role(spec["role_langs"][r]) for r in spec["sets"]}
            with t.no_grad():
                for li, name in enumerate(names):
                    lens = lenses[name]
                    h = t.cat([lens.transport(acts[layer][None], layer).to(device) for layer in ctx.layers])
                    lg = model.unembed(h).float()[:, : ctx.vocab_size].masked_fill(reader.mask, float("-inf"))
                    pr = t.softmax(lg, -1)
                    for key, (rk, ro, pb) in score_roles(lg, pr, spec["sets"], scripts, role_script).items():
                        rank_pre[row, li, :, rpos[key]], rank_own[row, li, :, rpos[key]], prob_pre[row, li, :, rpos[key]] = rk, ro, pb
                    whole = {r: ids for r, ids in spec["whole"].items() if ids}
                    for key, (rk, _, pb) in score_roles(lg, pr, whole, scripts, role_script).items():
                        rank_ww[row, li, :, rpos[key]], prob_ww[row, li, :, rpos[key]] = rk, pb
        if start // batch % 10 == 0:
            logger.info("roles: read %d / %d items (%.0f s)", min(start + batch, len(specs)), len(specs), time.monotonic() - started)
    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k = n_kept
    np.savez_compressed(out / "roles_metrics.npz", rank_pre=rank_pre[:k], rank_own=rank_own[:k], rank_ww=rank_ww[:k],
                        prob_pre=prob_pre[:k], prob_ww=prob_ww[:k])
    write_csv(out / "roles_items.csv", items + skipped)
    meta = {"model_kind": ctx.model_kind, "lenses": names, "layers": list(ctx.layers), "band": list(ctx.band),
            "roles": roles, "pairs": pairs, "n_kept": k, "n_items": len(specs), "seconds": round(time.monotonic() - started, 1),
            "script_sizes": {s: int(m.sum()) for s, m in scripts.items()}}
    (out / "roles_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    ctx.commit()
    summary = summarize_roles(out, model_kind=ctx.model_kind)
    ctx.commit()
    result = {"n_kept": k, "n_items": len(specs), "seconds": meta["seconds"], **{key: summary[key] for key in ("repro", "decision")}}
    if summary["repro"] and not summary["repro"]["ok"] and not ctx.smoke:  # docs/17 #1's stop rule
        result["stop_rule"] = f"the published pivot did not reproduce: {summary['repro']}"
    return result


def _band_values(rank: np.ndarray, band_idx: list[int], cutoff: int) -> np.ndarray:
    """Per item: the share of band layers whose rank is <= cutoff; NaN where the role is absent."""
    r = rank[:, band_idx]
    present = (r > 0).all(1)
    vals = ((r > 0) & (r <= cutoff)).mean(1).astype(float)
    vals[~present] = np.nan
    return vals


def summarize_roles(out_dir, *, model_kind: str | None = None) -> dict:
    """roles_summary.csv (per group x lens x role x readout: band top-10 and top-100 with concept-cluster CIs, median
    band rank, per-layer top-10 curve), the third-language pools, the reproduction check and the decision lines."""
    out = Path(out_dir)
    meta = json.loads((out / "roles_meta.json").read_text(encoding="utf-8"))
    data = np.load(out / "roles_metrics.npz")
    items = [i for i in read_csv(out / "roles_items.csv") if i.get("row") not in (None, "", "-1")]
    items.sort(key=lambda i: int(i["row"]))
    roles, layers, band = meta["roles"], meta["layers"], meta["band"]
    band_idx = [layers.index(l) for l in band]
    indic = {p for p in meta["pairs"] if all(pe.FAMILY.get(x) in ("IA", "Dr") and x not in ("hi", "hi_latn") for x in p.split("-"))}
    groups = {"ALL_INDIC": [i for i, it in enumerate(items) if it["pair"] in indic]}
    for p in meta["pairs"]:
        groups[p] = [i for i, it in enumerate(items) if it["pair"] == p]
    concepts = np.array([it["concept"] for it in items])
    readouts = {"prefix": data["rank_pre"], "whole_word": data["rank_ww"], "own_script": data["rank_own"]}
    rows, curves, pools = [], [], []
    for li, lens in enumerate(meta["lenses"]):
        for g, idx in groups.items():
            if not idx:
                continue
            for rname, arr in readouts.items():
                a = arr[idx, li]                       # [n, L, R]
                for ri, role in enumerate(roles):
                    rk = a[:, :, ri]
                    if (rk > 0).sum() == 0:
                        continue
                    t10, t100 = _band_values(rk, band_idx, 10), _band_values(rk, band_idx, 100)
                    s10, s100 = cmean(t10, concepts[idx]), cmean(t100, concepts[idx])
                    med = np.nanmedian(np.where(rk[:, band_idx] > 0, rk[:, band_idx], np.nan))
                    rows.append({"group": g, "lens": lens, "readout": rname, "role": role, "n": s10["n"],
                                 "band_top10": s10["mean"], "top10_lo": s10["ci"][0], "top10_hi": s10["ci"][1],
                                 "band_top100": s100["mean"], "top100_lo": s100["ci"][0], "top100_hi": s100["ci"][1],
                                 "median_band_rank": None if math.isnan(med) else float(med)})
                    if g == "ALL_INDIC":
                        present = rk[(rk > 0).all(1)]
                        curves.append({"lens": lens, "readout": rname, "role": role, "n": int(len(present)),
                                       **{f"L{l}": round(float((present[:, j] <= 10).mean()), 4) if len(present) else None
                                          for j, l in enumerate(layers)}})
                # third-language pools: each item's mean over its pool languages (and over their controls)
                for pool, langs in POOLS.items():
                    for suffix in ("", "_shuffled"):
                        cols = [roles.index(f"x_{x}{suffix}") for x in langs if f"x_{x}{suffix}" in roles]
                        per = np.stack([_band_values(a[:, :, c], band_idx, 10) for c in cols], 1)
                        per100 = np.stack([_band_values(a[:, :, c], band_idx, 100) for c in cols], 1)
                        with warnings.catch_warnings():  # an item with none of the pool's languages: NaN, dropped
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            s = cmean(np.nanmean(per, 1), concepts[idx])
                            s100 = cmean(np.nanmean(per100, 1), concepts[idx])
                        pools.append({"group": g, "lens": lens, "readout": rname, "pool": pool + suffix, "n": s["n"],
                                      "band_top10": s["mean"], "top10_lo": s["ci"][0], "top10_hi": s["ci"][1],
                                      "band_top100": s100["mean"]})
    write_csv(out / "roles_summary.csv", rows)
    write_csv(out / "roles_curves.csv", curves)
    write_csv(out / "roles_pools.csv", pools)
    # reproduction of the published ALL_INDIC curve (prefix readout)
    kind = model_kind or meta.get("model_kind")
    repro: dict = {}
    ref = REFERENCE.get(kind)
    if ref and ref["lens"] in meta["lenses"]:
        li = meta["lenses"].index(ref["lens"])
        a = data["rank_pre"][groups["ALL_INDIC"], li]
        repro = {"lens": ref["lens"], "ok": True}
        for role in ("english", "hindi"):
            rk = a[:, :, roles.index(role)]
            for layer, want in ref[role].items():
                if layer not in layers:
                    continue
                col = rk[:, layers.index(layer)]
                got = float(((col > 0) & (col <= 10))[col > 0].mean())
                repro[f"{role}@L{layer}"] = [round(got, 3), want]
                if abs(got - want) > ref["tol"][role]:
                    repro["ok"] = False
        repro["n_all_indic"] = len(groups["ALL_INDIC"])
    lens0 = meta["lenses"][0]
    pick = lambda readout, role: next((r for r in rows if r["group"] == "ALL_INDIC" and r["lens"] == lens0 and r["readout"] == readout
                                       and r["role"] == role), {})
    pool = lambda readout, name: next((r for r in pools if r["group"] == "ALL_INDIC" and r["lens"] == lens0 and r["readout"] == readout
                                       and r["pool"] == name), {})
    cell = lambda d: [d.get("band_top10"), d.get("top10_lo"), d.get("top10_hi"), d.get("n")]
    decision = {"lens": lens0, "band": band}
    for readout in ("prefix", "whole_word", "own_script"):
        decision[readout] = {**{role: cell(pick(readout, role)) for role in ("english", "english_shuffled", "hindi", "hindi_shuffled",
                                                                               "target", "source", "chinese", "chinese_shuffled",
                                                                               "nb_close", "nb_close_shuffled", "nb_assoc",
                                                                               "nb_assoc_shuffled")},
                             **{f"pool:{p}{s}": cell(pool(readout, p + s)) for p in POOLS for s in ("", "_shuffled")}}
    summary = {"repro": repro, "decision": decision}
    (out / "roles_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


### EDIT ENGINE: several interventions as rows of ONE forward (joint, push, positions, digits, zh)


@dataclass
class Edit:
    """At block ``layer``'s output, at positions ``pos``: h <- h - alpha * Q Q^T h (a projection; alpha 2 reflects,
    3 pushes to -2x), or, with ``random_u``, h <- h - alpha * |Q Q^T h| * u: a push along a fixed random unit
    vector that moves h exactly as far as the projection-push would (norm-matched random control)."""
    layer: int
    Q: t.Tensor
    alpha: float = 1.0
    pos: object = "last"          # "last", "all" (every position but BOS), "all_but_last", or a list of positions
    random_u: t.Tensor | None = None
    # with random_u: take |delta| from batch row ``match_row``'s own edit at this layer, in this forward (that row's
    # realized edit size, which compounds over the band), instead of from this row's English component
    match_row: int | None = None


def _positions(pos, seq_len: int):
    if pos == "last":
        return slice(seq_len - 1, seq_len)
    if pos == "all":
        return slice(1, seq_len)
    if pos == "all_but_last":
        return slice(1, seq_len - 1)
    return list(pos)


class RowEdits:
    """Forward hooks applying ``plans[row]`` (a list of :class:`Edit`, or None) to batch row ``row``. With one
    "last" projection per layer this is exactly causal_test.Ablation's arithmetic on that row. ``rel[row]``
    collects |delta| / |h| per edit (mean over edited positions), as tensors (one host sync at the end)."""

    def __init__(self, model, plans: list[list[Edit] | None]) -> None:
        self.rel: dict[int, list[t.Tensor]] = defaultdict(list)
        self.abs: dict[int, list[t.Tensor]] = defaultdict(list)
        by_layer: dict[int, list] = defaultdict(list)
        for row, plan in enumerate(plans):
            for e in plan or ():
                by_layer[e.layer].append((row, e))
        self.handles = [model.layers[layer].register_forward_hook(self._hook(edits)) for layer, edits in by_layer.items()]

    def _hook(self, edits):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            seq = hidden.shape[1]
            sizes: dict[int, t.Tensor] = {}
            for row, e in sorted(edits, key=lambda re: re[1].match_row is not None):  # matched rows after their sources
                sel = _positions(e.pos, seq)
                h = hidden[row, sel, :].float()
                Q = e.Q.to(h.device)
                comp = (h @ Q) @ Q.T
                if e.random_u is None:
                    delta = -e.alpha * comp
                elif e.match_row is not None:
                    delta = -sizes[e.match_row].to(h.device) * e.random_u.to(h.device)[None, :]
                else:
                    delta = -e.alpha * comp.norm(dim=-1, keepdim=True) * e.random_u.to(h.device)[None, :]
                sizes[row] = delta.norm(dim=-1, keepdim=True)
                # before the write: on an fp32 model .float() is a view of ``hidden``
                self.rel[row].append(delta.norm(dim=-1).mean() / h.norm(dim=-1).mean().clamp(min=1e-6))
                self.abs[row].append(delta.norm(dim=-1).mean())
                hidden[row, sel, :] = (h + delta).to(hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def run_rows(model, ids: t.Tensor, plans: list[list[Edit] | None], target_sets: list[set[int]]) -> list[dict]:
    """ONE forward over ``ids`` ([rows, S]; rows share a length), with ``plans`` applied per row. Returns per row:
    P(target set) at the last position, whether the top token is in it, and the mean relative edit norm.
    Through the text module with the LM head on the last position only (docs/12 _generate_lean)."""
    with t.no_grad(), RowEdits(model, plans) as edits:
        out = model._text_module(input_ids=ids.to(model.input_device), use_cache=False)
        head = model._lm_head
        logits = head(out[0][:, -1].to(head.weight.device)).float()
    probs = t.softmax(logits, -1)
    tops = logits.argmax(-1).tolist()
    res = []
    for r, target in enumerate(target_sets):
        idx = t.tensor(sorted(target), dtype=t.long, device=probs.device)
        rel, ab = edits.rel.get(r), edits.abs.get(r)
        res.append({"p": float(probs[r, idx].sum()), "correct": tops[r] in target,
                    "rel_norm": float(t.stack(rel).mean()) if rel else 0.0, "abs_norm": float(t.stack(ab).mean()) if ab else 0.0})
    return res


def _direction_W(model) -> t.Tensor:
    return (model._lm_head.weight.float() * ct.final_norm_gain(model)).detach()


def _subspaces(lens, layers: list[int], tokens: list[int], W: t.Tensor) -> dict[int, t.Tensor]:
    return {l: ct.subspace(lens, l, tokens, W) for l in layers}


def _random_unit(d: int, key: str) -> t.Tensor:
    v = t.randn(d, generator=t.Generator().manual_seed(zlib.crc32(key.encode())))
    return v / v.norm()


def _dedupe(tokens: list[int]) -> list[int]:
    return list(dict.fromkeys(tokens))


class TokenRules:
    """The token lists every causal analysis needs, from one vocabulary index."""

    def __init__(self, model, vocab_size: int) -> None:
        self.tok = model.tokenizer
        self.index = pe.build_vocab_index(self.tok, vocab_size)
        self.text_of = {i: key for key, ids in self.index.items() for i in ids}

    def prefix(self, words: list[str], min_chars: int = 2) -> set[int]:
        return pe.prefix_token_ids(self.index, words, self.tok, min_chars=min_chars)

    def published(self, own: list[str], ctrl: list[str]) -> tuple[list[int], list[int]]:
        """causal_test.run's rule: >= 3-char prefixes sorted by token id, the control minus the item's own tokens,
        both cut to the smaller count (this can drop the whole-word token; kept for reproduction)."""
        a = sorted(self.prefix(own, 3))
        b = sorted(self.prefix(ctrl, 3) - set(a))
        k = min(len(a), len(b))
        return a[:k], b[:k]

    def specific(self, own: list[str], ctrl: list[str], exclude: set[int] = frozenset()) -> tuple[list[int], list[int]]:
        """The same counts, each list most specific first (causal_test.most_specific_first), so a cut keeps the
        whole-word token; ``exclude`` removes tokens the item's other languages already claim."""
        a = ct.most_specific_first(self.prefix(own, 3) - set(exclude), self.text_of)
        b = ct.most_specific_first(self.prefix(ctrl, 3) - set(a) - set(exclude), self.text_of)
        k = min(len(a), len(b))
        return a[:k], b[:k]


def _causal_items(ctx, pairs: list[str], latents: tuple[str, ...]):
    """(pair, src, tgt, concept, control, prompt ids, raw target set) for the published causal prompts
    (concepts.tsv, 5-shot, seed 0), skipping pairs whose endpoints are a latent language."""
    concepts = pe.load_concepts(Path(ctx.data_dir) / "concepts.tsv")
    ordered = sorted(concepts)
    rules = TokenRules(ctx.model, ctx.vocab_size)
    out = []
    for pair in pairs:
        src, tgt = pair.split("-")
        if any(l in (src, tgt) for l in latents):
            continue
        for concept in ordered[: (3 if ctx.smoke else None)]:
            words = concepts[concept]
            if not all(l in words for l in latents + (src, tgt)):
                continue
            ids = ctx.model.encode(pe.build_prompt(concepts, concept, src, tgt, shots=5, seed=0))
            out.append({"pair": pair, "src": src, "tgt": tgt, "concept": concept, "control": _control_of(ordered, concept),
                        "ids": ids, "target": rules.prefix(words[tgt])})
    return concepts, rules, out


def with_matched(conds: dict, *, offset: int, key: str, d: int, only_pushes: bool = False) -> dict:
    """``conds`` plus, for every push condition (an edit with alpha > 1 and no random direction), ``<name>~rand``: a
    push along fixed random unit vectors whose size at every layer (and position) is that condition's own realized
    |delta| in the same forward (:class:`Edit` ``match_row``), so it moves the residual exactly as far. ``offset`` is
    the batch row of this item's base row (its conditions follow in order). ``only_pushes`` keeps only the pushes."""
    names = list(conds)
    out = {n: e for n, e in conds.items() if not only_pushes or any(x.alpha > 1 and x.random_u is None for x in e)}
    rows = {n: offset + 1 + i for i, n in enumerate(out)}
    for name in list(out):
        edits = out[name]
        if any(x.alpha > 1 and x.random_u is None for x in edits):
            out[f"{name}~rand"] = [Edit(x.layer, x.Q, x.alpha, x.pos, random_u=_random_unit(d, f"{key}|{name}|{x.layer}|rand"),
                                        match_row=rows[name]) for x in edits]
    assert names  # (conds is never empty when it is called)
    return out


def _matched_names(conds: list[str]) -> tuple[list[str], tuple]:
    """The ``~rand`` partners of the push conditions in ``conds`` and the (push, partner) contrasts."""
    pushes = [c for c in conds if "@" in c and not c.split("@")[1].startswith("1") and "random" not in c]
    return [f"{c}~rand" for c in pushes], tuple((c, f"{c}~rand") for c in pushes)


def _run_conditions(ctx, items: list[dict], build, out_name: str, *, require_correct: bool | None = None) -> list[dict]:
    """For each item: ``build(item)`` -> {condition: [Edit, ...]}; base + every condition as rows of one forward.
    Items the model gets wrong at base are dropped (unless smoke). Writes ``<out_name>_rows.csv``. Stage args
    ``matched_pushes`` (add each push's size-matched random partner, :func:`with_matched`) and ``only_pushes``."""
    require = (not ctx.smoke) if require_correct is None else require_correct
    rows = []
    started = time.monotonic()
    d = ctx.model._lm_head.weight.shape[1]
    for n, item in enumerate(items):
        conds = build(item)
        if conds and ctx.args.get("matched_pushes"):
            conds = with_matched(conds, offset=0, key=f"{item['pair']}|{item['concept']}", d=d,
                                 only_pushes=bool(ctx.args.get("only_pushes")))
        names = ["base", *conds]
        ids = item["ids"].repeat(len(names), 1)
        res = run_rows(ctx.model, ids, [None, *conds.values()], [item["target"]] * len(names))
        if require and not res[0]["correct"]:
            continue
        row = {k: item[k] for k in ("pair", "concept", "control") if k in item}
        row.update({k: item[k] for k in item.get("extra_keys", ())})
        for name, r in zip(names, res):
            row[f"{name}_p"], row[f"{name}_correct"], row[f"{name}_rel"] = r["p"], r["correct"], r["rel_norm"]
            row[f"{name}_abs"] = r["abs_norm"]
        row.update(item.get("ranks", {}))
        rows.append(row)
        if n % 50 == 0:
            logger.info("%s: %d / %d items (%.0f s)", out_name, n + 1, len(items), time.monotonic() - started)
    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / f"{out_name}_rows.csv", rows)
    ctx.commit()
    return rows


def condition_table(rows: list[dict], conditions: list[str], *, cluster: str = "concept",
                    contrasts: tuple = ()) -> dict:
    """Per condition: mean drop in P(target) in points (base - condition) with a cluster CI, the share still
    correct, the mean relative edit norm; per contrast (a, b): drop(a) - drop(b), paired, cluster CI."""
    if not rows:
        return {"n": 0}
    f = lambda r, k: float(r[k])
    truth = lambda v: str(v) == "True"
    has = lambda r, c: r.get(f"{c}_p") not in (None, "")
    table = {"n": len(rows), "clusters": len({r[cluster] for r in rows}),
             "base_p": round(float(np.mean([f(r, "base_p") for r in rows])), 4),
             "base_correct": round(float(np.mean([truth(r["base_correct"]) for r in rows])), 3)}
    for c in conditions:
        sel = [r for r in rows if has(r, c)]
        if not sel:
            continue
        s = cmean([100 * (f(r, "base_p") - f(r, f"{c}_p")) for r in sel], [r[cluster] for r in sel])
        table[c] = {"drop_pts": s["mean"], "ci": s["ci"], "n": s["n"],
                    "still_correct": round(float(np.mean([truth(r[f"{c}_correct"]) for r in sel])), 3),
                    "rel_norm": round(float(np.mean([f(r, f"{c}_rel") for r in sel])), 4)
                    if all(r.get(f"{c}_rel") not in (None, "") for r in sel) else None,
                    "abs_norm": round(float(np.mean([f(r, f"{c}_abs") for r in sel])), 3)
                    if all(r.get(f"{c}_abs") not in (None, "") for r in sel) else None}
    for spec in contrasts:
        a, b = spec[0], spec[1:]
        sel = [r for r in rows if has(r, a) and all(has(r, x) for x in b)]
        if not sel:
            continue
        # drop(a) - sum of drop(b...): with one b a plain paired contrast; with several, a against their sum
        diff = [100 * ((f(r, "base_p") - f(r, f"{a}_p")) - sum(f(r, "base_p") - f(r, f"{x}_p") for x in b)) for r in sel]
        s = cmean(diff, [r[cluster] for r in sel])
        table[f"{a}_minus_{'+'.join(b)}"] = {"pts": s["mean"], "ci": s["ci"], "n": s["n"]}
    return table


### 3. JOINT: English, Hindi (and Chinese) and the target word removed together


def run_joint(ctx) -> dict:
    name, lens = _lens_main(ctx)
    latents = ("en", "hi", "zh") if ctx.model_kind == "sarvam105" else ("en", "hi")
    concepts, rules, items = _causal_items(ctx, _pairs(ctx, INDIC_PAIRS), latents=("hi",))
    W = _direction_W(ctx.model)
    d = W.shape[1]
    band, early = list(ctx.band), list(ctx.early_band)

    def build(item):
        words, cw = concepts[item["concept"]], concepts[item["control"]]
        en_a, en_b = rules.published(words["en"], cw["en"])
        hi_a, hi_b = rules.published(words["hi"], cw["hi"])
        own, ctrl, claimed = {}, {}, set()
        for lang in latents + ("tgt",):
            key = item["tgt"] if lang == "tgt" else lang
            own[lang], ctrl[lang] = rules.specific(words[key], cw[key], exclude=claimed)
            claimed |= set(own[lang])
        joint = _dedupe([x for lang in own for x in own[lang]])
        cjoint = [x for x in _dedupe([x for lang in ctrl for x in ctrl[lang]]) if x not in set(joint)]
        k = min(len(joint), len(cjoint))
        joint, cjoint = joint[:k], cjoint[:k]
        item["ranks"] = {"k_en_pub": len(en_a), "k_hi_pub": len(hi_a), **{f"k_{l}": len(own[l]) for l in own}, "k_joint": k}
        sub = lambda toks, layers: _subspaces(lens, layers, toks, W)
        edits = lambda bases, alpha=1.0: [Edit(l, Q, alpha) for l, Q in bases.items()]
        conds = {"english_pub": edits(sub(en_a, band)), "control_pub": edits(sub(en_b, band)),
                 "hindi_pub": edits(sub(hi_a, band)), "hindi_control_pub": edits(sub(hi_b, band))}
        for lang in own:
            if own[lang]:
                conds[f"{lang}_ms"] = edits(sub(own[lang], band))
        if own["en"] and own["hi"]:
            conds["en_hi"] = edits(sub(_dedupe(own["en"] + own["hi"]), band))
        if k:
            jb, cb = sub(joint, band), sub(cjoint, band)
            rb = {l: ct.random_basis(d, k, zlib.crc32(f"{item['pair']}|{item['concept']}|{l}|joint".encode())) for l in band}
            conds.update({"joint": edits(jb), "control_joint": edits(cb), "random_joint": edits(rb),
                          "joint_early": edits(sub(joint, early)), "joint@3": edits(jb, 3.0), "control_joint@3": edits(cb, 3.0)})
        return conds

    rows = _run_conditions(ctx, items, build, "joint")
    singles = tuple(f"{l}_ms" for l in latents + ("tgt",))
    conds = ["english_pub", "control_pub", "hindi_pub", "hindi_control_pub", *singles, "en_hi", "joint", "control_joint",
             "random_joint", "joint_early", "joint@3", "control_joint@3"]
    contrasts = (("english_pub", "control_pub"), ("hindi_pub", "hindi_control_pub"), ("joint", "control_joint"),
                 ("joint", "random_joint"), ("joint@3", "control_joint@3"), ("joint", *singles))
    extra, pairs_ = _matched_names(conds)
    table = condition_table(rows, conds + extra, contrasts=contrasts + pairs_)
    table.update({"lens": name, "band": band, "early_band": early, "latents": list(latents)})
    # docs/17 #3's stop rule (the 30B's published J-lens ablation: English -5.7, Hindi -4.4 points)
    if ctx.model_kind == "sarvam30" and not ctx.smoke and table.get("english_pub"):
        en, hi = table["english_pub"]["drop_pts"], table["hindi_pub"]["drop_pts"]
        table["repro"] = {"english_pub": [en, 5.7, [4.0, 7.5]], "hindi_pub": [hi, 4.4, [2.8, 6.0]],
                          "ok": 4.0 <= en <= 7.5 and 2.8 <= hi <= 6.0}
        if not table["repro"]["ok"]:
            table["stop_rule"] = f"the published ablation did not reproduce: {table['repro']}"
    (Path(ctx.out_dir) / "joint_summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    ctx.commit()
    return table


### 4. DIGITS: does native-digit arithmetic need the English number word?


def run_digits(ctx) -> dict:
    import domain_experiment as de

    name, lens = _lens_main(ctx)
    opts = {**de.DEFAULTS["digits"], **(de.SMOKE["digits"] if ctx.smoke else {}), "langs": None}
    jobs = de.digits_jobs(ctx, opts)
    info = de.vocab_info(ctx.model.tokenizer, ctx.vocab_size)
    W = _direction_W(ctx.model)
    d = W.shape[1]
    band, early = list(ctx.band), list(ctx.early_band)
    all_words = _dedupe([i for w in de.NUMBER_WORDS[1:] for i in de._ids_of(info, w)])
    cache: dict = {}

    def sub(tokens: list[int], layers: list[int]):
        key = (tuple(tokens), tuple(layers))
        if key not in cache:
            cache[key] = _subspaces(lens, layers, tokens, W)
        return cache[key]

    def edits(bases, alpha=1.0):
        return [Edit(l, Q, alpha) for l, Q in bases.items()]

    def build(job):
        word, wrong = job["roles"]["word"], [i for i in job["roles"]["word_ctrl"] if i not in job["roles"]["word"]]
        k = min(len(word), len(wrong))
        word, wrong = word[:k], wrong[:k]
        rk = len(all_words)
        rb = {l: ct.random_basis(d, rk, zlib.crc32(f"digits|{rk}|{l}".encode())) for l in band}
        conds = {}
        if k:
            conds.update({"word": edits(sub(word, band)), "word_ctrl": edits(sub(wrong, band)), "word_early": edits(sub(word, early)),
                          "word@3": edits(sub(word, band), 3.0), "word_ctrl@3": edits(sub(wrong, band), 3.0)})
        conds.update({"allwords": edits(sub(all_words, band)), "random_allwords": edits(rb),
                      "allwords@3": edits(sub(all_words, band), 3.0), "random_allwords@3": edits(rb, 3.0)})
        if job["system"] != "ascii" and job["roles"]["ascii"]:
            conds["ascii"] = edits(sub(job["roles"]["ascii"], band))
        return conds

    # jobs whose answer token set is known: the native digit (native systems; Tamil/Telugu digits that are not
    # tokens drop out, as in doc 12's pooled rows) or the ASCII digit
    usable = []
    for j in jobs:
        target = set(j["roles"]["native"]) if j["system"] != "ascii" else set(j["roles"]["ascii"])
        if target:
            usable.append({**j, "target": target, "ids": t.tensor([j["ids"]], dtype=t.long), "concept": f"{j['a']}{j['op']}{j['b']}",
                           "extra_keys": ("system", "a", "op", "b", "answer", "wrong")})
    # rows of one forward: same-length prompts of a system share a forward, several items x their conditions
    rows: list[dict] = []
    started = time.monotonic()
    by_len: dict = defaultdict(list)
    for j in usable:
        by_len[(j["system"], j["ids"].shape[1])].append(j)
    group = int(ctx.args.get("items_per_forward", 4))
    for (system, _), js in by_len.items():
        for s in range(0, len(js), group):
            chunk = js[s: s + group]
            plans, targets, owners, names_all = [], [], [], []
            for j in chunk:
                conds = build(j)
                if ctx.args.get("matched_pushes"):
                    conds = with_matched(conds, offset=len(plans), key=f"{j['system']}|{j['concept']}", d=d,
                                         only_pushes=bool(ctx.args.get("only_pushes")))
                names = ["base", *conds]
                plans += [None, *conds.values()]
                targets += [j["target"]] * len(names)
                owners += [j] * len(names)
                names_all += names
            ids = t.cat([j["ids"].repeat(len([o for o in owners if o is j]), 1) for j in chunk])
            res = run_rows(ctx.model, ids, plans, targets)
            for j in chunk:
                sel = [(n, r) for n, r, o in zip(names_all, res, owners) if o is j]
                if not ctx.smoke and not sel[0][1]["correct"]:
                    continue
                row = {"system": j["system"], "problem": j["concept"], "answer": j["answer"], "wrong": j["wrong"]}
                for n, r in sel:
                    row[f"{n}_p"], row[f"{n}_correct"], row[f"{n}_rel"] = r["p"], r["correct"], r["rel_norm"]
                    row[f"{n}_abs"] = r["abs_norm"]
                rows.append(row)
        logger.info("digits: %s done (%.0f s)", system, time.monotonic() - started)
    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "digits_rows.csv", rows)
    conds = ["word", "word_ctrl", "word_early", "word@3", "word_ctrl@3", "allwords", "random_allwords", "allwords@3",
             "random_allwords@3", "ascii"]
    contrasts = (("word", "word_ctrl"), ("word@3", "word_ctrl@3"), ("allwords", "random_allwords"),
                 ("allwords@3", "random_allwords@3"))
    extra, pairs_ = _matched_names(conds)
    conds, contrasts = conds + extra, contrasts + pairs_
    native = [r for r in rows if r["system"] != "ascii"]
    table = {"lens": name, "band": band, "early_band": early, "n_all_words_tokens": len(all_words),
             "native_systems": condition_table(native, conds, cluster="problem", contrasts=contrasts),
             "ascii": condition_table([r for r in rows if r["system"] == "ascii"], conds, cluster="problem", contrasts=contrasts),
             "per_system": {s: condition_table([r for r in native if r["system"] == s], ["word", "word_ctrl", "word@3", "word_ctrl@3"],
                                               cluster="problem") for s in sorted({r["system"] for r in native})}}
    (out / "digits_ablation_summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    ctx.commit()
    return {k: table[k] for k in ("lens", "band", "native_systems", "ascii")}


### 5. PUSH: a dose-response curve with a norm-matched random control


def run_push(ctx) -> dict:
    name, lens = _lens_main(ctx)
    concepts, rules, items = _causal_items(ctx, _pairs(ctx, STEER_PAIRS), latents=())
    W = _direction_W(ctx.model)
    d = W.shape[1]
    band = list(ctx.band)
    alphas = tuple(ctx.args.get("alphas") or ALPHAS)

    def build(item):
        words, cw = concepts[item["concept"]], concepts[item["control"]]
        en_a, en_b = rules.published(words["en"], cw["en"])
        if not en_a:
            return {}
        qa, qb = _subspaces(lens, band, en_a, W), _subspaces(lens, band, en_b, W)
        u = {l: _random_unit(d, f"{item['pair']}|{item['concept']}|{l}|push") for l in band}
        u2 = {l: _random_unit(d, f"{item['pair']}|{item['concept']}|{l}|push2") for l in band}
        names = [f"{c}@{a:g}" for a in alphas for c in ("english", "control", "random", "random_en", "random_ctrl")]
        row = {n: i + 1 for i, n in enumerate(names)}  # batch row of each condition (row 0 is the base)
        conds = {}
        for a in alphas:
            conds[f"english@{a:g}"] = [Edit(l, qa[l], a) for l in band]
            conds[f"control@{a:g}"] = [Edit(l, qb[l], a) for l in band]
            # random@: scaled to this row's own English component (matches the English edit at the first layer only)
            conds[f"random@{a:g}"] = [Edit(l, qa[l], a, random_u=u[l]) for l in band]
            # random_en@ / random_ctrl@: the English row's / the control row's realized |delta| at every layer
            conds[f"random_en@{a:g}"] = [Edit(l, qa[l], a, random_u=u[l], match_row=row[f"english@{a:g}"]) for l in band]
            conds[f"random_ctrl@{a:g}"] = [Edit(l, qa[l], a, random_u=u2[l], match_row=row[f"control@{a:g}"]) for l in band]
        assert list(conds) == names
        return conds

    rows = _run_conditions(ctx, items, build, "push")
    conds = [f"{c}@{a:g}" for a in alphas for c in ("english", "control", "random", "random_en", "random_ctrl")]
    contrasts = tuple((f"english@{a:g}", f"{c}@{a:g}") for a in alphas for c in ("control", "random", "random_en")) + \
        tuple((f"control@{a:g}", f"random_ctrl@{a:g}") for a in alphas)
    table = condition_table(rows, conds, contrasts=contrasts)
    # docs/17 #5: the largest alpha at which both controls cost <= 5 points, and English's effect there. The random
    # control "moves the residual exactly as far" (the pre-registration) only in random_en@ (the English row's realized
    # |delta| at every layer); random@ matches the first layer only and is kept for comparison.
    decision = {}
    for rand in ("random_en", "random"):
        ok = [a for a in alphas if f"control@{a:g}" in table and f"{rand}@{a:g}" in table and table[f"control@{a:g}"]["drop_pts"] <= 5
              and table[f"{rand}@{a:g}"]["drop_pts"] <= 5]
        a_star = max(ok) if ok else None
        decision[rand] = {"alpha_star": a_star, **({"english_minus_random": table.get(f"english@{a_star:g}_minus_{rand}@{a_star:g}"),
                                                     "english_minus_control": table.get(f"english@{a_star:g}_minus_control@{a_star:g}")}
                                                    if a_star else {})}
    table["decision"] = decision
    table.update({"lens": name, "band": band, "alphas": list(alphas)})
    (Path(ctx.out_dir) / "push_summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    ctx.commit()
    return table


### 6a. POSITIONS: is English held upstream of the final position?


def source_span(tokenizer, prompt: str, src: str, word: str) -> list[int] | None:
    """Positions (in model.encode's ids: BOS + the text) of the query's source word in the prompt's last line."""
    marker = f'{pe.LANG_LABELS[src]}: "'
    start = prompt.rindex(marker) + len(marker)
    end = start + len(word)
    if prompt[start:end] != word:
        return None
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    pos = [i + 1 for i, (a, b) in enumerate(enc["offset_mapping"]) if b > a and a < end and b > start]
    return pos or None


def run_positions(ctx) -> dict:
    name, lens = _lens_main(ctx)
    concepts, rules, items = _causal_items(ctx, _pairs(ctx, INDIC_PAIRS), latents=())
    W = _direction_W(ctx.model)
    band, early = list(ctx.band), list(ctx.early_band)
    tok = ctx.model.tokenizer
    body_ok = 0
    for item in items:
        prompt = pe.build_prompt(concepts, item["concept"], item["src"], item["tgt"], shots=5, seed=0)
        body = tok(prompt, add_special_tokens=False).input_ids
        aligned = item["ids"][0, 1:].tolist() == body
        body_ok += aligned
        item["src_pos"] = source_span(tok, prompt, item["src"], concepts[item["concept"]][item["src"]][0]) if aligned else None

    def build(item):
        words, cw = concepts[item["concept"]], concepts[item["control"]]
        en_a, en_b = rules.published(words["en"], cw["en"])
        if not en_a:
            return {}
        qa, qb, qe = _subspaces(lens, band, en_a, W), _subspaces(lens, band, en_b, W), _subspaces(lens, early, en_a, W)
        e = lambda bases, pos, alpha=1.0: [Edit(l, Q, alpha, pos) for l, Q in bases.items()]
        conds = {"english_last": e(qa, "last"), "control_last": e(qb, "last"), "english_all_but_last": e(qa, "all_but_last"),
                 "english_all": e(qa, "all"), "control_all": e(qb, "all"), "english_all_early": e(qe, "all"),
                 "english_all@3": e(qa, "all", 3.0), "control_all@3": e(qb, "all", 3.0)}
        if item.get("src_pos"):
            conds["english_src"] = e(qa, item["src_pos"])
            conds["control_src"] = e(qb, item["src_pos"])
        return conds

    rows = _run_conditions(ctx, items, build, "positions")
    conds = ["english_last", "control_last", "english_src", "control_src", "english_all_but_last", "english_all", "control_all",
             "english_all_early", "english_all@3", "control_all@3"]
    contrasts = (("english_all", "english_last"), ("english_all", "control_all"), ("english_src", "control_src"),
                 ("english_last", "control_last"), ("english_all@3", "control_all@3"))
    extra, pairs_ = _matched_names(conds)
    table = condition_table(rows, conds + extra, contrasts=contrasts + pairs_)
    table.update({"lens": name, "band": band, "early_band": early, "prompts_aligned": f"{body_ok}/{len(items)}"})
    (Path(ctx.out_dir) / "positions_summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    ctx.commit()
    return table


### 6b. WRITERS: which attention heads and experts write the English direction?


class _WriterHooks:
    """Captures, at the last position of a batch-1 forward: the embedding (block 0's input), each block's attention
    output and its per-head split (through the output projection), each MoE block's selected experts' weighted
    outputs and shared expert (recomputed from the block's input and the router), the dense layer-0 MLP, each block's
    output, and checks that the parts add up."""

    def __init__(self, model) -> None:
        self.model, self.cap, self.handles = model, {}, []
        layers = model.layers
        self.handles.append(layers[0].register_forward_pre_hook(self._embed))
        for l, layer in enumerate(layers):
            self.handles.append(layer.attention.register_forward_hook(self._attn(l)))
            self.handles.append(layer.attention.dense.register_forward_pre_hook(self._dense(l)))
            self.handles.append(layer.mlp.register_forward_hook(self._mlp(l)))
            self.handles.append(layer.register_forward_hook(self._block(l)))

    def _embed(self, module, args):
        self.cap["embed"] = args[0][0, -1].detach().float()

    def _attn(self, l):
        def hook(module, inputs, output):
            self.cap[("attn", l)] = (output[0] if isinstance(output, tuple) else output)[0, -1].detach().float()
        return hook

    def _dense(self, l):
        def hook(module, args):
            z = args[0][0, -1].detach().float()
            Wd = module.weight.float()                                     # [d, H * hd]
            heads = module.weight.shape[1] // self.model._hf_model.config.head_dim
            hd = Wd.shape[1] // heads
            self.cap[("heads", l)] = (Wd.view(Wd.shape[0], heads, hd) * z.view(1, heads, hd)).sum(-1).T   # [H, d]
        return hook

    def _mlp(self, l):
        def hook(module, inputs, output):
            x = inputs[0][0, -1].detach()
            y = (output[0] if isinstance(output, tuple) else output)[0, -1].detach().float()
            self.cap[("mlp", l)] = y
            if hasattr(module, "experts") and hasattr(module, "gate"):
                idx, w, _ = module.gate(x[None, None])
                idx, w = idx.view(-1), w.view(-1)
                outs = t.stack([module.experts[int(e)](x[None])[0].float() * float(wi) for e, wi in zip(idx.tolist(), w.tolist())])
                shared = module.shared_experts(x[None])[0].float() if getattr(module, "shared_experts", None) is not None else t.zeros_like(y)
                self.cap[("experts", l)] = (idx.cpu(), w.float().cpu(), outs)
                self.cap[("shared", l)] = shared
                self.cap[("moe_err", l)] = float((outs.sum(0) + shared - y).norm() / y.norm().clamp(min=1e-6))
        return hook

    def _block(self, l):
        def hook(module, inputs, output):
            self.cap[("out", l)] = (output[0] if isinstance(output, tuple) else output)[0, -1].detach().float()
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def run_writers(ctx) -> dict:
    """docs/17 #6b. For each kept item (the published 10 pairs), every component's score on each role's readout
    direction, two ways: through the J-lens (``(w_R - mean_V w) J_l . c / rms_final``: what the component adds to the
    role's logit once the average downstream Jacobian carries it to the output) and direct (the logit lens: the
    component's own effect on the role's final logit, ``(w_R - mean_V w) . c / rms_final``). ``w_R`` = the mean
    of g * W_U over the role's whole-word tokens (Indic words without one: the tokenizer's first token).
    Roles: the English word, the control concept's English word, Hindi, the target and the source word."""
    model, tok = ctx.model, ctx.model.tokenizer
    name, lens = _lens_main(ctx)
    jac = getattr(lens, "jacobians_L_dict_FN", None) or {}
    concepts, rules, items = _causal_items(ctx, _pairs(ctx, INDIC_PAIRS), latents=("hi",))
    W = _direction_W(model)
    dev = W.device
    w_mean = W.mean(0)
    n = model.n_layers
    eps = float(getattr(model._final_norm, "variance_epsilon", getattr(model._final_norm, "eps", 1e-6)))
    roles = ("english", "english_ctrl", "hindi", "target", "source")
    J = {l: jac[l].float().to(dev) for l in jac} if jac else {}
    dir_cache: dict = {}

    def word_tokens(word_list: list[str]) -> list[int]:
        whole = sorted(_whole(rules.index, word_list))
        if whole:
            return whole
        first = tok(f" {word_list[0]}", add_special_tokens=False).input_ids[:1] or tok(word_list[0], add_special_tokens=False).input_ids[:1]
        return first

    def directions(tokens: list[int]) -> t.Tensor:
        """[n_layers, 2, d]: per layer the J-lens direction (identity at the final block, and wherever no Jacobian)
        and the direct one."""
        key = tuple(tokens)
        if key not in dir_cache:
            base = W[tokens].mean(0) - w_mean
            per = []
            for l in range(n):
                jl = (base @ J[l]) if l in J else base
                per.append(t.stack([jl, base]))
            dir_cache[key] = t.stack(per)
        return dir_cache[key]

    acc = {"n": 0}
    attn = np.zeros((0, len(roles), 2, n), dtype=np.float32)
    rows_attn, rows_mlp, rows_shared, rows_heads, rows_embed, rows_resid = [], [], [], [], [], []
    exp_idx, exp_w, exp_score = [], [], []
    moe_errs, kept_items = [], []
    started = time.monotonic()
    for it_n, item in enumerate(items):
        words, cw = concepts[item["concept"]], concepts[item["control"]]
        toks = {"english": word_tokens(words["en"]), "english_ctrl": word_tokens(cw["en"]), "hindi": word_tokens(words["hi"]),
                "target": word_tokens(words[item["tgt"]]), "source": word_tokens(words[item["src"]])}
        D = t.stack([directions(toks[r]) for r in roles])          # [R, n, 2, d]
        with t.no_grad(), _WriterHooks(model) as hooks:
            out = model._text_module(input_ids=item["ids"].to(model.input_device), use_cache=False)
            logits = model._lm_head(out[0][:, -1].to(model._lm_head.weight.device)).float()[0]
        if not ctx.smoke and int(logits.argmax()) not in item["target"]:
            continue
        cap = hooks.cap
        final = cap[("out", n - 1)].to(dev)
        rms = float(final.pow(2).mean().add(eps).sqrt())
        score = lambda vec, l: (D[:, l] @ vec.to(dev)) / rms                    # [R, 2]
        rows_embed.append(score(cap["embed"], 0).cpu().numpy())
        rows_attn.append(np.stack([score(cap[("attn", l)], l).cpu().numpy() for l in range(n)], -1))       # [R, 2, n]
        rows_mlp.append(np.stack([score(cap[("mlp", l)], l).cpu().numpy() for l in range(n)], -1))
        rows_resid.append(np.stack([score(cap[("out", l)], l).cpu().numpy() for l in range(n)], -1))
        rows_heads.append(np.stack([((D[:, l] @ cap[("heads", l)].to(dev).T) / rms).cpu().numpy() for l in range(n)], -2))  # [R,2,n,H]
        sh, ei, ew, es = [], [], [], []
        for l in range(n):
            if ("experts", l) in cap:
                idx, w, outs = cap[("experts", l)]
                sh.append(score(cap[("shared", l)], l).cpu().numpy())
                ei.append(idx.numpy())
                ew.append(w.numpy())
                es.append(((D[:, l] @ outs.to(dev).T) / rms).cpu().numpy())          # [R, 2, k]
                moe_errs.append(cap[("moe_err", l)])
            else:  # the dense layer 0
                sh.append(np.zeros((len(roles), 2), dtype=np.float32))
                k = len(cap[("experts", 1)][0]) if ("experts", 1) in cap else 1
                ei.append(np.full(k, -1))
                ew.append(np.zeros(k, dtype=np.float32))
                es.append(np.zeros((len(roles), 2, k), dtype=np.float32))
        rows_shared.append(np.stack(sh, -1))
        exp_idx.append(np.stack(ei))
        exp_w.append(np.stack(ew))
        exp_score.append(np.stack(es, 2))                      # [R, 2, n, k]
        kept_items.append({k: item[k] for k in ("pair", "concept", "control", "src", "tgt")})
        if it_n % 50 == 0:
            logger.info("writers: %d / %d items (%.0f s)", it_n + 1, len(items), time.monotonic() - started)
    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"embed": np.stack(rows_embed), "attn": np.stack(rows_attn), "mlp": np.stack(rows_mlp), "resid": np.stack(rows_resid),
              "heads": np.stack(rows_heads).astype(np.float16), "shared": np.stack(rows_shared), "expert_idx": np.stack(exp_idx).astype(np.int16),
              "expert_w": np.stack(exp_w).astype(np.float16), "expert_score": np.stack(exp_score).astype(np.float16)}
    np.savez_compressed(out / "writers.npz", **arrays)
    write_csv(out / "writers_items.csv", kept_items)
    meta = {"lens": name, "roles": list(roles), "modes": ["jlens", "direct"], "n_layers": n, "band": list(ctx.band),
            "moe_recompute_rel_err_max": round(max(moe_errs), 5) if moe_errs else None, "n_items": len(kept_items)}
    (out / "writers_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ctx.commit()
    summary = summarize_writers(out)
    ctx.commit()
    return {**meta, "headline": summary.get("headline")}


def summarize_writers(out_dir) -> dict:
    """Per layer: mean attention / MoE (routed + shared) / dense contribution to each role (J-lens and direct), English
    minus its control with concept-cluster CIs; the top (layer, head) and (layer, expert) English writers; and the
    docs/17 #6b concentration test (top-5 experts' share of positive English attribution vs the target word's)."""
    out = Path(out_dir)
    meta = json.loads((out / "writers_meta.json").read_text(encoding="utf-8"))
    a = np.load(out / "writers.npz")
    items = read_csv(out / "writers_items.csv")
    concepts = [i["concept"] for i in items]
    roles = meta["roles"]
    R = {r: i for i, r in enumerate(roles)}
    n = meta["n_layers"]
    layers = []
    for mode_i, mode in enumerate(meta["modes"]):
        for l in range(n):
            row = {"mode": mode, "layer": l}
            for comp in ("attn", "mlp"):
                for role in ("english", "english_ctrl", "target", "hindi"):
                    row[f"{comp}_{role}"] = round(float(a[comp][:, R[role], mode_i, l].mean()), 4)
                diff = a[comp][:, R["english"], mode_i, l] - a[comp][:, R["english_ctrl"], mode_i, l]
                s = cmean(diff, concepts)
                row[f"{comp}_en_minus_ctrl"], row[f"{comp}_en_minus_ctrl_ci"] = s["mean"], s["ci"]
            row["resid_english"] = round(float(a["resid"][:, R["english"], mode_i, l].mean()), 4)
            row["resid_english_ctrl"] = round(float(a["resid"][:, R["english_ctrl"], mode_i, l].mean()), 4)
            layers.append(row)
    write_csv(out / "writers_layers.csv", layers)
    # heads: mean English minus control score per (layer, head), J-lens mode
    heads = a["heads"].astype(np.float32)                       # [N, R, 2, n, H]
    hd = heads[:, R["english"], 0] - heads[:, R["english_ctrl"], 0]
    mean_h = hd.mean(0)
    top_heads = sorted(((float(mean_h[l, h]), l, h) for l in range(n) for h in range(mean_h.shape[1])), reverse=True)[:10]
    # experts: accumulate scores by (layer, expert) over the items routing to it
    idx, sc = a["expert_idx"], a["expert_score"].astype(np.float32)      # [N, n, k], [N, R, 2, n, k]
    units: dict = defaultdict(lambda: {"en": [], "ctrl": [], "tgt": []})
    for i in range(idx.shape[0]):
        for l in range(n):
            for j in range(idx.shape[2]):
                e = int(idx[i, l, j])
                if e < 0:
                    continue
                u = units[(l, e)]
                u["en"].append(float(sc[i, R["english"], 0, l, j]))
                u["ctrl"].append(float(sc[i, R["english_ctrl"], 0, l, j]))
                u["tgt"].append(float(sc[i, R["target"], 0, l, j]))
    def share_top5(key: str) -> tuple[float, list]:
        pos = {k: sum(max(0.0, v) for v in u[key]) for k, u in units.items()}
        total = sum(pos.values()) or 1.0
        top = sorted(pos.items(), key=lambda kv: -kv[1])[:5]
        return sum(v for _, v in top) / total, top
    en_share, en_top = share_top5("en")
    tgt_share, _ = share_top5("tgt")
    top_units = []
    for (l, e), total in en_top:
        u = units[(l, e)]
        top_units.append({"layer": l, "expert": e, "items_routed": len(u["en"]), "sum_en": round(total, 3),
                          "mean_en": round(float(np.mean(u["en"])), 4), "mean_ctrl": round(float(np.mean(u["ctrl"])), 4),
                          "positive_share": round(float(np.mean([v > 0 for v in u["en"]])), 3)})
    english_experts = en_share >= 2 * tgt_share and all(u["positive_share"] >= 0.5 for u in top_units)
    band = meta["band"]
    j_rows = [r for r in layers if r["mode"] == "jlens"]
    headline = {"band": band,
                "attn_en_minus_ctrl_by_layer": {r["layer"]: r["attn_en_minus_ctrl"] for r in j_rows},
                "mlp_en_minus_ctrl_by_layer": {r["layer"]: r["mlp_en_minus_ctrl"] for r in j_rows},
                "top_heads_en_minus_ctrl": [{"layer": l, "head": h, "mean": round(v, 4)} for v, l, h in top_heads],
                "top5_expert_share_english": round(en_share, 4), "top5_expert_share_target": round(tgt_share, 4),
                "top_english_experts": top_units, "english_writer_experts_criterion": bool(english_experts)}
    (out / "writers_summary.json").write_text(json.dumps({"headline": headline}, indent=1), encoding="utf-8")
    return {"headline": headline}


### 7 + 8. DECODING TO COMPLETION: grouped-query decode, a preallocated KV slab, continuous batching, sampling


def _gqa_decode_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                        output_attentions=False, use_cache=False, position_embeddings=None, **kwargs):
    """Sarvam-30B's attention for ONE new token per row without ``repeat_kv``: the 64 query heads are grouped by
    their 4 KV heads and attend to the cache directly (same arithmetic as the stock path's matmuls, without
    materialising 16 copies of the keys and values: 800 rows x 3,000 cached tokens would need ~50 GB per layer).
    Any other call (a prefill, attention weights requested) goes to the class's own forward."""
    import sys

    bsz, q_len, _ = hidden_states.size()
    if q_len != 1 or past_key_value is None or output_attentions:
        return type(self).forward(self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                                  past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                                  position_embeddings=position_embeddings, **kwargs)
    mod = sys.modules[type(self).__module__]
    qkv = self.query_key_value(hidden_states).view(bsz, 1, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
    q, k, v = qkv.split([self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if self.config.use_qk_norm:
        q, k = self.query_layernorm(q), self.key_layernorm(k)
    cos, sin = position_embeddings
    q, k = mod.apply_rotary_pos_emb(q, k, cos, sin)
    k, v = past_key_value.update(k, v, self.layer_idx, {"sin": sin, "cos": cos})        # [B, kvH, S, D]
    groups = self.num_heads // self.num_key_value_heads
    qg = q.reshape(bsz, self.num_key_value_heads, groups, self.head_dim)                 # head h = kv * groups + g
    scores = t.matmul(qg, k.transpose(-1, -2)) * self.scaling                             # [B, kvH, G, S]
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, -1:, :]
    probs = t.softmax(scores, dim=-1, dtype=t.float32).to(q.dtype)
    out = t.matmul(probs, v).reshape(bsz, 1, self.num_heads * self.head_dim)
    return self.dense(out), None, past_key_value


def _mla_absorbed_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
                          output_attentions=False, use_cache=False, **kwargs):
    """Sarvam-105B's MLA attention for ONE new token per row, attending to the cached compressed latent directly
    ("weight absorption"): the per-head key and value up-projections (``kv_b_proj``) are folded into the query and the
    output, so the cache holds only the normalized 512-dim latent and the 64-dim roped key (37 KB per token over 32
    layers) and nothing is re-expanded per step. Scores = (q_nope W_UK) . latent + q_pe . k_pe, output = W_UV (p . latent):
    the stock arithmetic in another order. A prefill (several tokens) goes to domain_experiment's latent-cache
    forward, which writes the same cache format."""
    import sys

    import domain_experiment as de

    bsz, q_len, _ = hidden_states.size()
    if q_len != 1 or past_key_value is None or output_attentions:
        return de._mla_latent_forward(self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                                      past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
                                      **kwargs)
    mod = sys.modules[type(self).__module__]
    nope, rope, r, vdim, heads = self.qk_nope_head_dim, self.qk_rope_head_dim, self.kv_lora_rank, self.v_head_dim, self.num_heads
    q = self.q_proj(hidden_states) if self.q_lora_rank is None else self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(bsz, 1, heads, self.q_head_dim).transpose(1, 2)                    # [B, H, 1, nope + rope]
    q_nope, q_pe = t.split(q, [nope, rope], dim=-1)
    compressed_kv, k_pe = t.split(self.kv_a_proj_with_mqa(hidden_states), [r, rope], dim=-1)
    k_pe = k_pe.view(bsz, 1, 1, rope).transpose(1, 2)                              # [B, 1, 1, rope]
    latent = self.kv_a_layernorm(compressed_kv).unsqueeze(1)                      # [B, 1, 1, r]
    kv_seq_len = 1 + mod._get_usable_past_kv_length(past_key_value, 1, self.layer_idx)
    cos, sin = self.rotary_emb(latent, seq_len=kv_seq_len)
    q_pe, k_pe = mod.apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
    latent, k_pe = past_key_value.update(latent, k_pe, self.layer_idx, {"sin": sin, "cos": cos})   # [B, 1, S, r], [B, 1, S, rope]
    w = self.kv_b_proj.weight.view(heads, nope + vdim, r)
    lat = latent[:, 0]                                                             # [B, S, r]
    q_lat = t.einsum("bhd,hdr->bhr", q_nope[:, :, 0], w[:, :nope])                 # [B, H, r]
    scores = t.matmul(q_lat, lat.transpose(1, 2)) + t.matmul(q_pe[:, :, 0], k_pe[:, 0].transpose(1, 2))   # [B, H, S]
    scores = scores * self.softmax_scale
    if attention_mask is not None:
        scores = scores + attention_mask[:, 0, -1:, :]
    probs = t.softmax(scores, dim=-1, dtype=t.float32).to(q.dtype)
    out = t.einsum("bhr,hvr->bhv", t.matmul(probs, lat), w[:, nope:])               # [B, H, v]
    return self.o_proj(out.reshape(bsz, 1, heads * vdim)), None, past_key_value


def _attention_of(layer):
    return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)


def _is_mla(model) -> bool:
    return hasattr(_attention_of(model.layers[0]), "kv_a_proj_with_mqa")


class _fast_decode:
    """Context manager: the one-token decode step of every attention module goes through :func:`_gqa_decode_forward`
    (Sarvam-30B) or :func:`_mla_absorbed_forward` (Sarvam-105B); prefill keeps (30B) or gets the latent-cache (105B) path."""

    def __init__(self, model) -> None:
        import types

        self.types = types
        self.fn = _mla_absorbed_forward if _is_mla(model) else _gqa_decode_forward
        self.patched = [_attention_of(layer) for layer in model.layers]

    def __enter__(self):
        for attn in self.patched:
            attn.forward = self.types.MethodType(self.fn, attn)
        return self

    def __exit__(self, *exc):
        for attn in self.patched:
            del attn.forward


_gqa_decode = _fast_decode  # the 30B runs' name


def _cache_layout(model) -> tuple[tuple[int, int], tuple[int, int], list]:
    """(key heads, key dim), (value heads, value dim) and each layer's device, as the attention caches them: the 30B
    its 4 KV heads x 64; the 105B (latent MLA cache) a 512-dim latent as "key" and the 64-dim roped key as "value"."""
    cfg = model._hf_model.config
    if _is_mla(model):
        devices = [_attention_of(l).kv_b_proj.weight.device for l in model.layers]
        return (1, cfg.kv_lora_rank), (1, cfg.qk_rope_head_dim), devices
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    devices = [_attention_of(l).query_key_value.weight.device for l in model.layers]
    return (cfg.num_key_value_heads, head_dim), (cfg.num_key_value_heads, head_dim), devices


try:
    from transformers.cache_utils import Cache as _CacheBase
except ImportError:  # pragma: no cover
    _CacheBase = object


class SlabCache(_CacheBase):
    """A preallocated KV cache for continuous batching: per layer ``K [rows, k_heads, width, k_dim]`` and ``V [rows,
    v_heads, width, v_dim]``, each on its layer's device. Rows 0..B-1 are live; columns c0..c1 are in use, every row's
    tokens right-aligned at c1 (left padding, masked). ``update`` writes the new column(s) at c1 and returns views (no
    per-step copy, unlike DynamicCache's torch.cat). A ``transformers`` Cache subclass, because the 105B's forward wraps
    anything else as a legacy cache; the base initializer is not called (no layer objects are needed)."""

    def __init__(self, n_layers: int, rows: int, width: int, k_shape: tuple[int, int], v_shape: tuple[int, int], dtype,
                 devices: list) -> None:
        self.k = [t.zeros((rows, k_shape[0], width, k_shape[1]), dtype=dtype, device=devices[l]) for l in range(n_layers)]
        self.v = [t.zeros((rows, v_shape[0], width, v_shape[1]), dtype=dtype, device=devices[l]) for l in range(n_layers)]
        self.n_layers, self.rows, self.width = n_layers, rows, width
        self.B, self.c0, self.c1 = 0, 0, 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.c1 - self.c0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def update(self, key, value, layer_idx, cache_kwargs=None):
        q = key.shape[-2]
        self.k[layer_idx][: self.B, :, self.c1: self.c1 + q] = key
        self.v[layer_idx][: self.B, :, self.c1: self.c1 + q] = value
        k = self.k[layer_idx][: self.B, :, self.c0: self.c1 + q]
        v = self.v[layer_idx][: self.B, :, self.c0: self.c1 + q]
        if layer_idx == self.n_layers - 1:
            self.c1 += q
        return k, v


def _sample(logits: t.Tensor, temperature: float, top_p: float, generator, prefilter: int = 2048) -> t.Tensor:
    """Greedy at temperature 0; else temperature + nucleus sampling, over the ``prefilter`` most likely tokens (the
    nucleus at top-p 0.95 essentially never reaches past them)."""
    if temperature <= 0:
        return logits.argmax(-1)
    vals, idx = logits.topk(min(prefilter, logits.shape[-1]), dim=-1)
    probs = t.softmax(vals / temperature, dim=-1)
    keep = (probs.cumsum(-1) - probs) < top_p
    probs = probs * keep
    probs = probs / probs.sum(-1, keepdim=True)
    choice = t.multinomial(probs, 1, generator=generator)
    return idx.gather(-1, choice).squeeze(-1)


def _looping(gen: list[int], window: int = 300, max_period: int = 150) -> bool:
    """The last ``window`` tokens repeat with some period <= ``max_period`` (an exact loop)."""
    if len(gen) < window + max_period:
        return False
    a = np.asarray(gen[-(window + max_period):])
    tail = a[-window:]
    return any(np.array_equal(tail, a[-window - p: len(a) - p]) for p in range(1, max_period + 1))


def generate_cb(model, rows: list[list[int]], *, max_new_tokens: int, end_ids: set[int], temperature: float = 0.0,
                top_p: float = 1.0, seed: int = 0, max_rows: int | None = None, admit_rows: int = 48, pad_id: int = 0,
                think_end: int | None = None, answer_cap: int = 600, loop_check_every: int = 64, deadline: float | None = None,
                mem_reserve_gb: float = 14.0, on_finish=None, log_every: int = 100, force_tail: list[int] | None = None,
                force_new: int = 12, order: list[int] | None = None, on_logits=None) -> tuple[list[list[int]], list[str], dict]:
    """Continuous-batching generation through the text module (LM head on the last position only), with
    :class:`SlabCache` and the fast decode step (:class:`_fast_decode`): up to ``max_rows`` sequences decode together;
    finished rows leave the slab at once and new prompts are prefilled ``admit_rows`` at a time into the freed rows.
    Each row stops at an end id ("eos"), at ``max_new_tokens`` ("cap"), in an exact repetition loop ("loop"),
    ``answer_cap`` tokens after ``think_end`` ("answer_cap"), or at ``deadline`` ("deadline"; prompts never started:
    "not_run"). With ``force_tail`` (token ids), a row that stops by cap, loop or deadline before ``think_end`` is
    instead fed those tokens and then decodes ``force_new`` greedy tokens in place: doc 12's forced answer without
    re-reading the trace (``stats["forced"][row]`` = the tail + the forced tokens). At the deadline every such row is
    forced and the queue is dropped. ``max_rows`` defaults to what fits in each GPU's free memory (less
    ``mem_reserve_gb`` per GPU). ``order``: the admission order (default: by prompt length); pass one that interleaves
    the arms of an experiment, so batch composition can't correlate with arm. ``on_logits(live_rows, logits)`` sees each
    step's next-token logits. Returns (generated ids per prompt, reason per prompt, stats). Greedy at temperature 0."""
    from transformers import DynamicCache

    text, head, device = model._text_module, model._lm_head, model.input_device
    n_layers = len(model.layers)
    k_shape, v_shape, devices = _cache_layout(model)
    p_max = max(len(r) for r in rows)
    width = p_max + max_new_tokens + (len(force_tail) + force_new if force_tail else 0) + 64
    dtype = next(text.parameters()).dtype
    el = t.tensor([], dtype=dtype).element_size()
    per_layer_row = width * (k_shape[0] * k_shape[1] + v_shape[0] * v_shape[1]) * el
    if max_rows is None:
        fits = []
        for dev in sorted(set(devices), key=str):
            layers_here = sum(d == dev for d in devices)
            free = t.cuda.mem_get_info(dev)[0] if t.cuda.is_available() and dev.type == "cuda" else 8e9
            fits.append(int((free - mem_reserve_gb * 1e9) // (layers_here * per_layer_row)))
        max_rows = max(1, min(fits))
    max_rows = max(1, min(max_rows, len(rows)))
    slab = SlabCache(n_layers, max_rows, width, k_shape, v_shape, dtype, devices)
    M = t.zeros((max_rows, width), dtype=t.long, device=device)      # attention mask columns
    P = t.zeros(max_rows, dtype=t.long, device=device)               # each row's last position id
    H = None                                                          # last hidden state per live row
    live: list[int] = []
    gen: list[list[int]] = [[] for _ in rows]
    reason = ["not_run"] * len(rows)
    think_at = [None] * len(rows)
    forcing: dict[int, dict] = {}                                     # row -> {"tail": deque, "left": int, "why": str}
    forced: dict[int, list[int]] = {}
    queue = deque(order if order is not None else sorted(range(len(rows)), key=lambda i: len(rows[i])))
    generator = t.Generator(device=head.weight.device).manual_seed(seed)
    stats = {"max_rows": max_rows, "width": width, "steps": 0, "admissions": 0, "shifts": 0,
             "slab_gb": round(max_rows * n_layers * per_layer_row / 1e9, 1), "devices": sorted({str(d) for d in devices})}
    started, step_times = time.monotonic(), []
    deadline_hit = False
    slab.c0 = slab.c1 = p_max

    def admit(n: int) -> None:
        nonlocal H
        take = [queue.popleft() for _ in range(min(n, len(queue)))]
        wn = max(len(rows[i]) for i in take)
        ids = t.full((len(take), wn), pad_id, dtype=t.long, device=device)
        m = t.zeros_like(ids)
        for r, i in enumerate(take):
            ids[r, wn - len(rows[i]):] = t.tensor(rows[i], dtype=t.long)
            m[r, wn - len(rows[i]):] = 1
        pos = (m.cumsum(-1) - 1).clamp(min=0)
        out = text(input_ids=ids, attention_mask=m, position_ids=pos, past_key_values=DynamicCache(), use_cache=True)
        cache, h_new = out.past_key_values, out[0][:, -1]
        b0, b1 = slab.B, slab.B + len(take)
        if slab.B == 0:
            slab.c0 = slab.c1 = p_max
        if slab.c1 - wn < slab.c0:  # widen to the left: the live rows' columns there may hold a finished row's data
            new_c0 = slab.c1 - wn
            M[:b0, new_c0: slab.c0] = 0
            for l in range(n_layers):
                slab.k[l][:b0, :, new_c0: slab.c0] = 0
                slab.v[l][:b0, :, new_c0: slab.c0] = 0
            slab.c0 = new_c0
        for l in range(n_layers):
            layer = cache.layers[l]
            slab.k[l][b0:b1, :, slab.c0: slab.c1] = 0
            slab.v[l][b0:b1, :, slab.c0: slab.c1] = 0
            slab.k[l][b0:b1, :, slab.c1 - wn: slab.c1] = layer.keys.to(slab.k[l].device)
            slab.v[l][b0:b1, :, slab.c1 - wn: slab.c1] = layer.values.to(slab.v[l].device)
        M[b0:b1] = 0
        M[b0:b1, slab.c1 - wn: slab.c1] = m
        P[b0:b1] = pos[:, -1]
        H = h_new if H is None or slab.B == 0 else t.cat([H, h_new.to(H.device)], 0)
        slab.B = b1
        live.extend(take)
        stats["admissions"] += 1
        del out, cache

    def compact(keep: list[int]) -> None:
        nonlocal H
        kt = t.tensor(keep, dtype=t.long, device=device)
        nb = len(keep)
        for l in range(n_layers):
            for buf in (slab.k[l], slab.v[l]):
                part = buf[: slab.B, :, slab.c0: slab.c1].index_select(0, kt.to(buf.device))
                buf[:nb, :, slab.c0: slab.c1] = part
        Mp = M[: slab.B, slab.c0: slab.c1].index_select(0, kt)
        M[:nb, slab.c0: slab.c1] = Mp
        P[:nb] = P[: slab.B].index_select(0, kt)
        H = H.index_select(0, kt.to(H.device)) if nb else None
        slab.B = nb
        if nb:  # columns that are padding for every live row
            used = M[:nb, slab.c0: slab.c1].any(0).nonzero()
            if len(used):
                slab.c0 += int(used[0])

    def shift() -> None:
        """At the slab's right edge: move the live columns left (c1 - c0 never exceeds the width less p_max)."""
        span = slab.c1 - slab.c0
        new_c1 = max(span, p_max)
        new_c0 = new_c1 - span
        for l in range(n_layers):
            for buf in (slab.k[l], slab.v[l]):
                buf[: slab.B, :, new_c0: new_c1] = buf[: slab.B, :, slab.c0: slab.c1].clone()
        M[: slab.B, new_c0: new_c1] = M[: slab.B, slab.c0: slab.c1].clone()
        slab.c0, slab.c1 = new_c0, new_c1
        stats["shifts"] += 1

    def finish(i: int, why: str) -> None:
        reason[i] = why
        if on_finish:
            on_finish(i, gen[i], why)

    with t.no_grad(), _fast_decode(model):
        while live or queue:
            if deadline is not None and not deadline_hit and time.monotonic() > deadline:
                deadline_hit = True
                queue.clear()
                for i in list(live):
                    if i not in forcing and think_at[i] is None and force_tail:
                        forcing[i] = {"tail": deque(force_tail), "left": force_new, "why": "deadline"}
                if not force_tail:
                    for i in live:
                        finish(i, "deadline")
                    break
            if queue and (slab.B == 0 or max_rows - slab.B >= min(admit_rows, len(queue))):
                admit(min(admit_rows, max_rows - slab.B))
                continue
            if not live:  # nothing to decode: the deadline emptied the queue between admissions (H is None here)
                break
            t0 = time.monotonic()
            logits = head(H.to(head.weight.device)).float()
            if on_logits:
                on_logits(list(live), logits)
            sampled = _sample(logits, temperature, top_p, generator).tolist()
            greedy = logits.argmax(-1).tolist() if forcing else None
            keep, feed = [], []
            for r, i in enumerate(live):
                if i in forcing:  # feeding the answer cue, then greedy answer tokens
                    st = forcing[i]
                    if st["tail"]:
                        tok_id = st["tail"].popleft()
                        forced.setdefault(i, []).append(tok_id)
                    else:
                        tok_id = greedy[r]
                        forced.setdefault(i, []).append(tok_id)
                        st["left"] -= 1
                        if tok_id in end_ids or st["left"] <= 0:
                            finish(i, st["why"])
                            continue
                    keep.append(r)
                    feed.append(tok_id)
                    continue
                if deadline_hit:  # the answer after </think> was being written: keep what there is
                    finish(i, "deadline")
                    continue
                tok_id = sampled[r]
                gen[i].append(tok_id)
                if think_end is not None and think_at[i] is None and tok_id == think_end:
                    think_at[i] = len(gen[i])
                why = None
                if tok_id in end_ids:
                    why = "eos"
                elif len(gen[i]) >= max_new_tokens:
                    why = "cap"
                elif think_at[i] is not None and len(gen[i]) - think_at[i] >= answer_cap:
                    why = "answer_cap"
                elif loop_check_every and len(gen[i]) % loop_check_every == 0 and think_at[i] is None and _looping(gen[i]):
                    why = "loop"
                if why in ("cap", "loop") and force_tail and think_at[i] is None:
                    forcing[i] = {"tail": deque(force_tail), "left": force_new, "why": why}
                    why = None
                if why:
                    finish(i, why)
                    continue
                keep.append(r)
                feed.append(tok_id)
            if len(keep) < len(live):
                live[:] = [live[r] for r in keep]
                compact(keep)
            if not live:
                continue
            if slab.c1 + 1 > slab.width:
                shift()
            nxt = t.tensor(feed, dtype=t.long, device=device)
            M[: slab.B, slab.c1] = 1
            pos = (P[: slab.B] + 1)[:, None]
            out = text(input_ids=nxt[:, None], attention_mask=M[: slab.B, slab.c0: slab.c1 + 1], position_ids=pos,
                       past_key_values=slab, use_cache=True)
            H = out[0][:, -1]
            P[: slab.B] += 1
            stats["steps"] += 1
            step_times.append(time.monotonic() - t0)
            if stats["steps"] % log_every == 0:  # printed too: modal_pilot's root logger writes only to pilot.log
                msg = (f"decode step {stats['steps']}: {slab.B} live, {len(queue)} queued, width {slab.c1 - slab.c0}, "
                       f"{float(np.mean(step_times[-log_every:])):.2f} s/step, {sum(r != 'not_run' for r in reason)} done, "
                       f"{time.monotonic() - started:.0f} s")
                logger.info(msg)
                print(msg, flush=True)
    stats.update({"seconds": round(time.monotonic() - started, 1), "mean_step_s": round(float(np.mean(step_times)), 3) if step_times else None,
                  "reasons": {k: reason.count(k) for k in set(reason)}, "n_forced": len(forced), "forced": {str(k): v for k, v in forced.items()}})
    return gen, reason, stats


### 7. THINK: MGSM with thinking in English vs instructed into the prompt's language


THINK_VARIANTS = ("hi", "ta", "bn_h", "te_h")
THINK_CONDS = ("default", "sys_en", "sys_native", "sys_native_prefill")
THINK_FILE = HERE / "data" / "domains" / "think_lang_translations.json"
LANG_NAMES = {"en": "English", "hi": "Hindi", "ta": "Tamil", "bn": "Bengali", "te": "Telugu"}
OPENER_EN = "Okay, let me solve this problem step by step."


def think_source_texts() -> list[str]:
    """The English sentences machine-translated into hi/ta/bn/te for the instructed arms (``modal_translate.py
    --step list``): one instruction per language name, then the opener."""
    return [f"Think through this problem step by step in {LANG_NAMES[l]}, and write all of your reasoning in {LANG_NAMES[l]}."
            for l in ("en", "hi", "ta", "bn", "te")] + [OPENER_EN]


def think_texts(lang: str) -> dict[str, str]:
    """The prompt language's instruction to reason in English, in itself, and the native opener."""
    src = think_source_texts()
    if lang == "en":
        return {"sys_en": src[0]}
    data = json.loads(THINK_FILE.read_text(encoding="utf-8"))
    tr = data["translations"][lang]
    order = ["en", "hi", "ta", "bn", "te"]
    return {"sys_en": tr[0], "sys_native": tr[order.index(lang)], "opener": tr[-1]}


def mgsm_problems(ctx, n: int, variants: tuple[str, ...]) -> list[dict]:
    """The first ``n`` MGSM test problems (doc 12's chat demos excluded) whose numbers survived machine translation
    for every MT variant given, and that have text in every variant (the hint test's selection rule)."""
    import domain_experiment as de

    items = de.load_sets(ctx.data_dir)["mgsm"]
    test = de._select(items, None, de.MGSM_DEMOS)
    flags = de._number_flags()
    test = [i for i in test if not any(i["id"] in flags.get(v, set()) for v in variants if de.source_of(v) == "mt")]
    test = [i for i in test if all(de.text_of(i, v) for v in variants)]
    return test[:n]


def _chat(tok, system: str | None, user: str) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)


def think_jobs(ctx) -> list[dict]:
    import domain_experiment as de

    tok = ctx.model.tokenizer
    n = 1 if ctx.smoke else int(ctx.args.get("items", 60))
    variants = tuple(ctx.args.get("variants") or THINK_VARIANTS)
    # selected on every think variant plus the hint test's MT set, whatever this launch runs, so #7 and #8 share problems
    problems = mgsm_problems(ctx, n, THINK_VARIANTS + ("hi", "ta", "sat"))
    jobs = []
    for item in problems:
        for variant in ("en",) + variants:
            lang = de.base_lang(variant)
            q = de.text_of(item, variant)
            texts = think_texts(lang) if not ctx.smoke or THINK_FILE.is_file() else {"sys_en": "x", "sys_native": "y", "opener": "z"}
            # stage args ``think_conds`` / ``en_conds`` select arms (the 105B run drops the Indian default arm for budget)
            # (``en_conds: []`` drops English: the 105B's second launch adds only the Indian default arm)
            pick = lambda key, default: tuple(ctx.args[key]) if ctx.args.get(key) is not None else default
            conds = pick("en_conds", ("default", "sys_en")) if lang == "en" else pick("think_conds", THINK_CONDS)
            for cond in conds:
                system = None if cond == "default" else texts["sys_en"] if cond == "sys_en" else texts["sys_native"]
                text = _chat(tok, system, q) if getattr(tok, "chat_template", None) else f"{system or ''}\n{q}\n"
                if cond == "sys_native_prefill":
                    text += "<think>\n" + texts["opener"] + "\n"
                jobs.append({"variant": variant, "lang": lang, "cond": cond, "id": item["id"], "answer": float(item["meta"]["answer"]),
                             "ids": tok(text).input_ids})
    return jobs


def _forced_answers(ctx, jobs: list[dict], gens: list[list[int]], todo: list[int], cue: str = "The final answer is ",
                    max_new: int = 12, deadline: float | None = None) -> dict[int, str]:
    """doc 12's forced-answer probe on the traces that did not close their reasoning: the trace is closed with
    ``</think>`` and the cue, and the model writes ``max_new`` greedy tokens."""
    import batch_gen

    if not todo:
        return {}
    tok = ctx.model.tokenizer
    tail = tok("</think>\n\n" + cue, add_special_tokens=False).input_ids
    rows = [jobs[i]["ids"] + gens[i] + tail for i in todo]
    outs, _, _ = generate_cb(ctx.model, rows, max_new_tokens=max_new, end_ids=batch_gen.end_ids_of(tok), temperature=0.0,
                             # the 105B's eager prefill holds rows x 64 heads x S^2 fp32 scores: 2 rows of a 2,500-token trace = 3 GB
                             admit_rows=int(ctx.args.get("force_admit_rows", 2 if ctx.is_mla else 16)),
                             max_rows=int(ctx.args.get("force_max_rows", 64 if ctx.is_mla else 256)),
                             loop_check_every=0, deadline=deadline)
    return {i: tok.decode(o, skip_special_tokens=True) for i, o in zip(todo, outs)}


def _decode_jobs(ctx, jobs: list[dict], out: Path, label: str):
    """Sampled generation of every job to completion (``generate_cb``), with a decode deadline that leaves time for
    the forced answers; writes the raw generations as they finish."""
    import batch_gen

    tok = ctx.model.tokenizer
    think_end = tok.convert_tokens_to_ids("</think>")
    think_end = think_end if isinstance(think_end, int) and think_end >= 0 and think_end != tok.unk_token_id else None
    budget = float(ctx.args.get("time_limit_s", 1e9))
    started = time.monotonic()
    deadline = started + budget
    max_new = 8 if ctx.smoke else int(ctx.args.get("max_new_tokens", 3072))
    finished: list = []
    partial = out / f"{label}_generations.jsonl"  # every finished trace's ids, as it finishes (survives a later crash)
    partial.write_text("", encoding="utf-8")

    def on_finish(i, g, why):
        finished.append((i, list(g), why))
        if len(finished) % 50 == 0:
            with partial.open("a", encoding="utf-8") as handle:
                for j, ids, reason in finished[-50:]:
                    handle.write(json.dumps({"job": j, "reason": reason, "ids": ids}) + "\n")
            ctx.commit()

    # the forced answer in place (no re-read of the trace): the default on the 105B, whose eager prefill of a
    # 3,000-token trace is quadratic in memory
    in_slab = bool(ctx.args.get("force_in_slab", ctx.is_mla))
    tail = tok("</think>\n\n" + FORCE_CUE, add_special_tokens=False).input_ids if in_slab else None
    gens, reasons, stats = generate_cb(
        ctx.model, [j["ids"] for j in jobs], max_new_tokens=max_new, end_ids=batch_gen.end_ids_of(tok),
        temperature=float(ctx.args.get("temperature", 0.8)), top_p=float(ctx.args.get("top_p", 0.95)), seed=int(ctx.args.get("seed", 0)),
        max_rows=ctx.args.get("max_rows"), admit_rows=int(ctx.args.get("admit_rows", 16 if ctx.is_mla else 48)), think_end=think_end,
        deadline=deadline, on_finish=on_finish, force_tail=tail, force_new=12,
        # every condition of a problem next to each other: each admission chunk mixes the arms (bf16 batch-shape noise
        # can't line up with arm)
        order=sorted(range(len(jobs)), key=lambda i: (jobs[i]["id"], jobs[i]["variant"], jobs[i]["cond"])),
        mem_reserve_gb=float(ctx.args.get("mem_reserve_gb", 8.0 if ctx.is_mla else 14.0)))
    stats["think_end_token"] = think_end
    stats["force_tail"] = tail
    (out / f"{label}_decode_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return gens, reasons, stats, tail


FORCE_CUE = "The final answer is "


def _forced_texts(ctx, jobs: list[dict], gens: list[list[int]], reasons: list[str], stats: dict, tail, unclosed: list[int]) -> dict[int, str]:
    """The forced answer text of every unclosed trace: from the in-slab forcing where it ran, else by re-reading the
    trace (:func:`_forced_answers`; e.g. a trace that ended without closing its reasoning)."""
    tok = ctx.model.tokenizer
    out = {}
    done = stats.get("forced") or {}
    for i in unclosed:
        if tail is not None and str(i) in done:
            out[i] = tok.decode(done[str(i)][len(tail):], skip_special_tokens=True)
    rest = [i for i in unclosed if i not in out]
    out.update(_forced_answers(ctx, jobs, gens, rest))
    return out


def _split_trace(tok, gen: list[int], end_ids: set[int]) -> tuple[str, bool, str]:
    text = tok.decode([i for i in gen if i not in end_ids], skip_special_tokens=False)
    reasoning, closed, after = text.partition("</think>")
    return reasoning.replace("<think>", ""), bool(closed), after


def run_think(ctx) -> dict:
    import batch_gen
    import domain_experiment as de

    tok = ctx.model.tokenizer
    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = think_jobs(ctx)
    gens, reasons, stats, tail = _decode_jobs(ctx, jobs, out, "think")
    end_ids = batch_gen.end_ids_of(tok)
    items, todo = [], []
    for i, (job, gen) in enumerate(zip(jobs, gens)):
        reasoning, closed, after = _split_trace(tok, gen, end_ids)
        shares = L.script_shares(reasoning)
        own = sum(shares[s] for s in de.own_scripts(job["variant"]))
        value = de.final_answer(after) if closed else None
        items.append({"job": i, "variant": job["variant"], "lang": job["lang"], "cond": job["cond"], "id": job["id"],
                      "answer": job["answer"], "n_tokens": len(gen), "stop": reasons[i], "closed_think": closed,
                      "natural_final": value, "natural_correct": value is not None and abs(value - job["answer"]) < 1e-6,
                      "latin_share": round(shares["latin"], 4), "own_share": round(own, 4),
                      "n_letters": sum(1 for ch in reasoning if L.letter_script(ch)), "reasoning": reasoning[:6000], "after_think": after[:400]})
        if not closed and reasons[i] != "not_run":
            todo.append(i)
    forced = _forced_texts(ctx, jobs, gens, reasons, stats, tail, todo)
    for i, text in forced.items():
        value = de.parse_number(text)
        items[i].update({"forced_text": text[:80], "forced_final": value,
                         "forced_correct": value is not None and abs(value - items[i]["answer"]) < 1e-6})
    for it in items:
        it["final"] = it["natural_final"] if it["closed_think"] else it.get("forced_final")
        it["correct"] = it["natural_correct"] if it["closed_think"] else bool(it.get("forced_correct"))
        it["answered"] = it["closed_think"] or "forced_final" in it
    write_csv(out / "think_items.csv", items)
    ctx.commit()
    summary = summarize_think(out)
    ctx.commit()
    return {"n_jobs": len(jobs), "decode": {k: v for k, v in stats.items() if k not in ("forced", "force_tail")}, "headline": summary["headline"]}


def summarize_think(out_dir) -> dict:
    """Per variant x condition: compliance (own-script and Latin share of the reasoning), closure, accuracy (natural or
    forced; and natural-only), with CIs over problems; pooled over the Indian languages; and the docs/17 #7 contrasts:
    each native arm minus the English-instructed arm, paired by (language, problem), CI over problems."""
    out = Path(out_dir)
    items = [i for i in read_csv(out / "think_items.csv") if i["stop"] != "not_run"]
    truth = lambda v: str(v) == "True"
    f = lambda v: float(v) if v not in ("", "None", None) else float("nan")
    rows = []
    cells = defaultdict(list)
    for i in items:
        cells[(i["variant"], i["cond"])].append(i)
        if i["lang"] != "en":
            cells[("indian pooled", i["cond"])].append(i)
    for (variant, cond), xs in cells.items():
        ids = [x["id"] for x in xs]
        acc = cmean([truth(x["correct"]) for x in xs if truth(x["answered"])], [x["id"] for x in xs if truth(x["answered"])])
        nat = [x for x in xs if truth(x["closed_think"])]
        rows.append({"variant": variant, "cond": cond, "n": len(xs), "closed": round(float(np.mean([truth(x["closed_think"]) for x in xs])), 3),
                     "own_share": round(float(np.nanmean([f(x["own_share"]) for x in xs])), 3),
                     "latin_share": round(float(np.nanmean([f(x["latin_share"]) for x in xs])), 3),
                     "accuracy": acc["mean"], "accuracy_ci": acc["ci"],
                     "accuracy_natural": round(float(np.mean([truth(x["natural_correct"]) for x in nat])), 3) if nat else None,
                     "n_natural": len(nat), "mean_tokens": round(float(np.mean([f(x["n_tokens"]) for x in xs])), 1)})
    write_csv(out / "think_summary.csv", rows)
    # paired contrasts against the English-instructed arm, Indian languages pooled
    by = {(i["lang"], i["id"], i["cond"]): i for i in items if truth(i["answered"])}
    contrasts = {}
    for cond in ("default", "sys_native", "sys_native_prefill"):
        keys = [(l, p) for (l, p, c) in by if c == cond and l != "en" and (l, p, "sys_en") in by]
        diff = [100 * (float(truth(by[(l, p, cond)]["correct"])) - float(truth(by[(l, p, "sys_en")]["correct"]))) for l, p in keys]
        s = cmean(diff, [p for _, p in keys])
        contrasts[f"{cond}_minus_sys_en"] = {"pts": s["mean"], "ci": s["ci"], "n": s["n"]}
    compliant = {r["cond"]: r["own_share"] for r in rows if r["variant"] == "indian pooled"}
    headline = {"pooled": {r["cond"]: {k: r[k] for k in ("accuracy", "accuracy_ci", "own_share", "closed", "n")} for r in rows
                           if r["variant"] == "indian pooled"},
                "contrasts": contrasts, "compliant_arms": [c for c, v in compliant.items() if c.startswith("sys_native") and v >= 0.5]}
    (out / "think_summary.json").write_text(json.dumps({"headline": headline}, indent=1), encoding="utf-8")
    return {"headline": headline}


### THE FAST DECODE STEP ON THE REAL MODEL


DECODE_BASELINE_105B = {"final_top1_agree": 0.854, "final_kl_mean": 0.234}  # batch_check_deep, results/xl_ref_105 (lone vs batched)


def run_decode_check(ctx) -> dict:
    """The fast decode step (absorbed MLA on the 105B, grouped-query on the 30B) against the model's own eager path, on
    the real model, before any paid decode stage. A batched greedy ``generate_cb`` run over the first ``check_prompts``
    think prompts records every step's next-token log-probabilities; one eager full forward of each prompt plus its
    continuation (stock attention, no cache, lone) gives the reference at the same positions. Reports top-1 agreement
    and mean KL(eager || fast), against ``batch_check_deep``'s batched-vs-lone baseline on the 105B (top-1 0.854, KL
    0.234), and row 0's continuation text for an eye check of the thinking markers. Stop rule: top-1 < 0.80 or mean
    KL > 0.47 (twice the baseline): every later stage is skipped."""
    model, tok = ctx.model, ctx.model.tokenizer
    jobs = think_jobs(ctx)
    order = sorted(range(len(jobs)), key=lambda i: (jobs[i]["id"], jobs[i]["variant"], jobs[i]["cond"]))
    pick = [jobs[i] for i in order[: 2 if ctx.smoke else int(ctx.args.get("check_prompts", 8))]]
    n_tok = 4 if ctx.smoke else int(ctx.args.get("check_tokens", 96))
    store: dict = defaultdict(list)

    def on_logits(live, logits):
        lp = t.log_softmax(logits, -1).half().cpu()
        for r, i in enumerate(live):
            store[i].append(lp[r])

    started = time.monotonic()
    gens, _, stats = generate_cb(model, [j["ids"] for j in pick], max_new_tokens=n_tok, end_ids=set(), temperature=0.0,
                                 max_rows=len(pick), admit_rows=len(pick), loop_check_every=0, on_logits=on_logits)
    top1, kls = [], []
    head = model._lm_head
    for i, job in enumerate(pick):
        full = t.tensor([job["ids"] + gens[i]], dtype=t.long, device=model.input_device)
        with t.no_grad():
            hidden = model._text_module(input_ids=full, use_cache=False)[0][0]
            lo = len(job["ids"]) - 1
            ref = t.log_softmax(head(hidden[lo: lo + len(gens[i])].to(head.weight.device)).float(), -1)
        fast = t.stack(store[i]).to(ref.device).float()
        top1 += (ref.argmax(-1) == fast.argmax(-1)).tolist()
        kls += (ref.exp() * (ref - fast)).sum(-1).tolist()
    text = tok.decode(gens[0], skip_special_tokens=False)
    result = {"n_prompts": len(pick), "n_steps": len(top1), "top1_agree": round(float(np.mean(top1)), 4),
              "kl_mean": round(float(np.mean(kls)), 5), "kl_max": round(float(np.max(kls)), 5),
              "baseline_105b_batched_vs_lone": DECODE_BASELINE_105B, "decode_stats": {k: v for k, v in stats.items() if k != "forced"},
              "row0": {"variant": pick[0]["variant"], "cond": pick[0]["cond"], "text": text[:900],
                       "think_opened": "<think>" in text or (pick[0]["cond"] == "sys_native_prefill")},
              "seconds": round(time.monotonic() - started, 1)}
    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "decode_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    ctx.commit()
    if not ctx.smoke and (result["top1_agree"] < 0.80 or result["kl_mean"] > 0.47):
        result["stop_rule"] = f"fast decode step disagrees with the eager path: top-1 {result['top1_agree']}, KL {result['kl_mean']}"
    return result


### 8. HINT: doc 12's wrong-hint test, 3x the problems, traces to completion


def run_hint(ctx) -> dict:
    import batch_gen
    import domain_experiment as de

    tok = ctx.model.tokenizer
    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    opts = {**de.DEFAULTS["hint"], **(de.SMOKE["hint"] if ctx.smoke else {}), "items": 2 if ctx.smoke else int(ctx.args.get("items", 60)),
            "langs": None}
    jobs = de.hint_jobs(ctx, opts)
    skip = 0 if ctx.smoke else int(ctx.args.get("skip_items", 0))
    if skip:  # problems skip+1 .. skip+items (the 105B's second half): hint_jobs takes the first N, so drop the first `skip`
        jobs = de.hint_jobs(ctx, {**opts, "items": skip + opts["items"]})
        first = set(list(dict.fromkeys(j["id"] for j in jobs))[:skip])
        jobs = [j for j in jobs if j["id"] not in first]
    gens, reasons, stats, tail = _decode_jobs(ctx, jobs, out, "hint")
    end_ids = batch_gen.end_ids_of(tok)
    words = de.hint_templates()["teacher_words"]
    items, todo = [], []
    for i, (job, gen) in enumerate(zip(jobs, gens)):
        reasoning, closed, after = _split_trace(tok, gen, end_ids)
        lang = de.base_lang(job["variant"])
        value = de.final_answer(after) if closed else None
        items.append({"job": i, "variant": job["variant"], "lang": lang, "source": de.source_of(job["variant"]), "cond": job["cond"],
                      "id": job["id"], "answer": job["answer"], "hint": job["hint"], "n_tokens": len(gen), "stop": reasons[i],
                      "closed_think": closed, "natural_final": value,
                      "natural_correct": value is not None and abs(value - job["answer"]) < 1e-6,
                      "natural_took_hint": value is not None and abs(value - job["hint"]) < 1e-6,
                      "mentions_hint": de.mentions_hint(reasoning, lang, words), "mentions_hint_number": de.mentions_number(reasoning, job["hint"]),
                      "latin_share": L.script_shares(reasoning)["latin"], "reasoning": reasoning[:12000], "after_think": after[:600]})
        if not closed and reasons[i] != "not_run":
            todo.append(i)
    forced = _forced_texts(ctx, jobs, gens, reasons, stats, tail, todo)
    for it in items:
        if it["job"] in forced:
            value = de.parse_number(forced[it["job"]])
            it.update({"forced_text": forced[it["job"]][:80], "forced_final": value})
        value = it["natural_final"] if it["closed_think"] else it.get("forced_final")
        it["final"] = value
        it["answered"] = value is not None
        it["correct"] = value is not None and abs(value - it["answer"]) < 1e-6
        it["took_hint"] = value is not None and abs(value - it["hint"]) < 1e-6
    items = [i for i in items if i["stop"] != "not_run"]
    write_csv(out / "hint_items.csv", items)
    ctx.commit()
    summary = de.summarize_hint(out)          # every row with an answer (natural, else forced): doc 12's table
    natural = out / "natural"
    natural.mkdir(exist_ok=True)
    nat_items = [{**i, "final": i["natural_final"], "correct": i["natural_correct"], "took_hint": i["natural_took_hint"],
                  "answered": i["closed_think"]} for i in items]
    write_csv(natural / "hint_items.csv", nat_items)
    summary_nat = de.summarize_hint(natural)  # natural conclusions only (doc 12's done_key: closed_think)
    ctx.commit()
    return {"n_jobs": len(jobs), "decode": {k: v for k, v in stats.items() if k not in ("forced", "force_tail")},
            "headline": summary.get("headline"), "headline_natural": summary_nat.get("headline")}


### 9. ZH: is the Chinese latent used?


def run_zh(ctx) -> dict:
    """causal_test.run with the Chinese word's directions (alpha 1, 2, 3), as rows of one forward per item, on the
    published 10 Indic pairs; summarised with concept-cluster CIs."""
    name, lens = _lens_main(ctx)
    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = ",".join(_pairs(ctx, INDIC_PAIRS))
    ct.run(ctx.model, lens, concepts_path=str(Path(ctx.data_dir) / "concepts.tsv"), pairs=pairs, mid_layers=list(ctx.band),
           early_layers=list(ctx.early_band), out_dir=str(out), vocab_size=ctx.vocab_size, require_correct=not ctx.smoke,
           latent_lang="zh", alphas=(1.0, 2.0, 3.0), batch_conditions=True)
    rows = read_csv(out / "causal_rows.csv") if (out / "causal_rows.csv").is_file() else []
    conds = ["english_mid", "control_mid", "english_early", "english_mid@2", "control_mid@2", "english_mid@3", "control_mid@3"]
    for r in rows:  # condition_table's column names
        for c in conds:
            r[f"{c}_rel"] = 0.0
    table = condition_table(rows, conds, contrasts=(("english_mid", "control_mid"), ("english_mid@2", "control_mid@2"),
                                                    ("english_mid@3", "control_mid@3")))
    table.update({"lens": name, "band": list(ctx.band), "early_band": list(ctx.early_band), "latent": "zh",
                  "note": "condition names are causal_test's: 'english_*' here means the CHINESE word's directions"})
    (out / "zh_summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    ctx.commit()
    return table


def run_zh2(ctx) -> dict:
    """:func:`run_zh` on this module's edit engine (the same token rule: causal_test.run's for latent "zh"), so the
    pushes get size-matched random partners (stage args ``matched_pushes``, ``only_pushes``)."""
    name, lens = _lens_main(ctx)
    concepts, rules, items = _causal_items(ctx, _pairs(ctx, INDIC_PAIRS), latents=("zh",))
    W = _direction_W(ctx.model)
    band, early = list(ctx.band), list(ctx.early_band)

    def build(item):
        own, ctrl = rules.published(concepts[item["concept"]]["zh"], concepts[item["control"]]["zh"])
        if not own:
            return {}
        qa, qb = _subspaces(lens, band, own, W), _subspaces(lens, band, ctrl, W)
        e = lambda bases, alpha=1.0: [Edit(l, Q, alpha) for l, Q in bases.items()]
        return {"zh": e(qa), "control": e(qb), "zh_early": e(_subspaces(lens, early, own, W)), "zh@2": e(qa, 2.0),
                "control@2": e(qb, 2.0), "zh@3": e(qa, 3.0), "control@3": e(qb, 3.0)}

    rows = _run_conditions(ctx, items, build, "zh2")
    conds = ["zh", "control", "zh_early", "zh@2", "control@2", "zh@3", "control@3"]
    extra, pairs_ = _matched_names(conds)
    table = condition_table(rows, conds + extra, contrasts=(("zh", "control"), ("zh@2", "control@2"), ("zh@3", "control@3")) + pairs_)
    table.update({"lens": name, "band": band, "early_band": early})
    (Path(ctx.out_dir) / "zh2_summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    ctx.commit()
    return table


### MODAL HOOK


RUNNERS = {"roles": run_roles, "joint": run_joint, "digits": run_digits, "push": run_push, "positions": run_positions,
           "writers": run_writers, "think": run_think, "hint": run_hint, "zh": run_zh, "zh2": run_zh2,
           "decode_check": run_decode_check}


def _stage_args(args: dict, analysis: str) -> dict:
    """One ``--stage-args`` dict for several stages: ``"<analysis>:<key>"`` overrides ``<key>`` for that stage only
    (e.g. ``{"hint:items": 30, "think:items": 60}``); other stages' prefixed keys are dropped."""
    plain = {k: v for k, v in args.items() if ":" not in k}
    mine = {k.split(":", 1)[1]: v for k, v in args.items() if k.startswith(analysis + ":")}
    return {**plain, **mine}


def run(ctx, analysis: str) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ctx.args = _stage_args(dict(ctx.args or {}), analysis)
    if analysis == "hello":
        return {"model_kind": ctx.model_kind, "lens_specs": ctx.lens_specs, "band": ctx.band, "early_band": ctx.early_band}
    if analysis not in RUNNERS:
        raise ValueError(f"unknown analysis {analysis!r}")
    return RUNNERS[analysis](ctx)


def n_forwards(ctx, analysis: str) -> float:
    """Single-forward equivalents for modal_pilot's budget gate (25% headroom is added there). Measured anchors:
    the 30B's lone forward 0.12 s; a 16-prompt forward ~2 forwards; the 105B's forward is launch-bound, so a
    batch of rows costs ~1.5 of its forwards; a 30B decode step at batch ~300+ ~0.7 s."""
    ctx.args = _stage_args(dict(ctx.args or {}), analysis)
    spf = max(ctx.seconds_per_forward, 1e-3)
    mla = ctx.is_mla
    setup = 40.0 / spf  # vocab index, readout mask, script masks (262k decodes each)
    if ctx.smoke:
        return 30.0 / spf
    if analysis == "roles":
        n_items = len(PUB_PAIRS.split(",")) * 32
        n_lenses = len(ctx.args.get("lenses") or ctx.lens_specs)
        return math.ceil(n_items / ctx.batch_size) * BATCH_FORWARD_COST + n_items * n_lenses * 0.06 / spf + setup
    # one item = base + every condition as rows of ONE forward. The 105B's forward is launch-bound (rq105c: 8 rows ~0.5
    # forward per item); the 30B's grows with the rows' tokens, ~0.1 forward per extra ~100-token row.
    per_item = (lambda rows: 1.5 + 0.03 * rows) if mla else (lambda rows: 1.0 + 0.1 * rows)
    if ctx.args.get("only_pushes"):
        per_item = (lambda f: (lambda rows: f(min(rows, 6))))(per_item)
    if analysis == "joint":
        return 320 * per_item(13) + setup
    if analysis == "push":
        return 160 * per_item(16) + setup
    if analysis == "positions":
        return 320 * per_item(11) + setup
    if analysis in ("zh", "zh2"):
        return 320 * per_item(8) + setup
    if analysis == "digits":
        return 864 / int(ctx.args.get("items_per_forward", 4)) * (2.0 if mla else 1.5) + setup
    if analysis == "writers":
        return 320 * 2.5 + setup
    if analysis == "decode_check":  # a batched greedy decode (~2 forwards per step on the 105B) + one lone forward per prompt
        return int(ctx.args.get("check_tokens", 96)) * (2.0 if mla else 1.5) + int(ctx.args.get("check_prompts", 8)) * 2 + setup
    if analysis in ("think", "hint"):
        # decoding stops at its deadline, so the stage is bounded by deadline + forced answers + set-up; divided by the
        # gate's 25% headroom, which a deadline-bounded stage doesn't need
        limit = float(ctx.args.get("time_limit_s", 1800)) + float(ctx.args.get("force_reserve_s", 300)) + 90
        return limit / (1.25 * spf)
    return 1.0
