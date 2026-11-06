"""Does removing the English direction change which experts fire downstream? (docs/09 §11)

The published English ablation on Sarvam-30B, unchanged (:func:`causal_test.run`'s items and
conditions: J-lens ``j_200p`` directions of the concept's English tokens, projected out of the final
position after blocks 13-16, ``causal_test.DEFAULT_PAIRS``, items the model translates correctly).
Each kept item runs three forwards: ``clean``, ``english`` (its English subspace removed) and
``control`` (a different concept's English subspace of the same size). Forward hooks on the MoE gates
of layers 13..18 record the routed top-k experts and their weights at every position, and the fp32
router logits at the final position. Hooks registered before the ablation's own record the share of
the final position's residual each edit removes (``||Q Q^T h|| / ||h||``).

Only the final position can change: the edit is applied there alone, and under causal attention no
earlier position sees it (checked per item). Layer 13's router runs before the layer-13 edit, so
its expert set must not change (a built-in check). The 30B is bit-reproducible at batch 1 (every
``sanity`` rerun differs by 0.0), so any changed expert is caused by the edit; a few clean reruns
re-check that here.

Per layer: Jaccard(clean, ablated) of the expert sets, English vs control, with paired bootstrap CIs.
Whether the routing change tracks the item's target-probability drop (Spearman). Whether it beats
the control at an equal removed share (OLS with a condition term). With §11's bare-word and
pivot-prompt routing (``results/neuron30/routing``): whether the English-ablated sets lose
English-selective experts or gain target-language-selective ones.

Stage (modal_pilot's XL hook): ``routing_ablation.english``; ``--stage-args '{"words_run": "neuron30"}'``
names the §11 run on the Volume (default ``neuron30``).
Offline: ``python routing_ablation.py --analyze results/<tag>/routing_ablation_english --words results/neuron30/routing``
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as t

import causal_test as ct
import pivot_experiment as pe

logger = logging.getLogger("routing_ablation")
CONDITIONS = ("clean", "english", "control")
N_RERUNS = 8  # clean forwards repeated, to re-check determinism with every hook in place
KEPT_SHARE = 0.85  # the published run kept 272 / 320 items (sizes the budget gate)


### HOOKS


class GateTap:
    """Forward hooks on the gates of ``layers``: top-k ids and weights ``[batch, seq, k]`` and the
    final position's fp32 router logits ``h @ W_gate^T``."""

    def __init__(self, model, layers: list[int]) -> None:
        self.gates = {l: model.layers[l].mlp.gate for l in layers}
        self.out: dict[int, tuple[t.Tensor, t.Tensor, t.Tensor]] = {}
        self.handles: list = []

    def _hook(self, layer: int):
        def hook(module, args, output):
            batch, seq = args[0].shape[:2]
            last = args[0].reshape(batch, seq, -1)[:, -1].float()
            self.out[layer] = (output[0].reshape(batch, seq, -1).detach(), output[1].reshape(batch, seq, -1).detach().float(),
                               last @ module.weight.float().T.to(last.device))
        return hook

    def __enter__(self) -> "GateTap":
        self.handles = [g.register_forward_hook(self._hook(l)) for l, g in self.gates.items()]
        return self

    def __exit__(self, *exc) -> None:
        for handle in self.handles:
            handle.remove()

    def snapshot(self) -> dict:
        return {l: tuple(x.clone() for x in v) for l, v in self.out.items()}


class EditShare:
    """Forward hooks on the edited blocks, registered BEFORE :class:`causal_test.Ablation`'s so they see
    each block's output as it was: the share ``||Q Q^T h|| / ||h||`` of the final position's residual
    that the edit removes there."""

    def __init__(self, model, bases: dict[int, t.Tensor]) -> None:
        self.share: dict[int, float] = {}
        self.handles = [model.layers[l].register_forward_hook(self._hook(l, Q)) for l, Q in bases.items()]

    def _hook(self, layer: int, Q: t.Tensor):
        def hook(module, inputs, output):
            last = (output[0] if isinstance(output, tuple) else output)[0, -1].float()
            Qd = Q.to(last.device)
            self.share[layer] = float((last @ Qd).norm() / last.norm())
        return hook

    def __enter__(self) -> "EditShare":
        return self

    def __exit__(self, *exc) -> None:
        for handle in self.handles:
            handle.remove()


### RUN


def n_forwards(ctx, analysis: str) -> float:
    """Clean forwards for every candidate item, two ablated ones for the kept share, the reruns."""
    n_items = len(pairs_for(ctx)) * 32
    return n_items + 2 * KEPT_SHARE * n_items + N_RERUNS


def pairs_for(ctx) -> list[str]:
    pairs = ct.DEFAULT_PAIRS.split(",")
    return pairs[:2] if ctx.smoke else pairs


def run(ctx, analysis: str = "english") -> dict:
    if analysis != "english":
        raise ValueError(f"unknown analysis {analysis!r} (only 'english')")
    model, hf = ctx.model, ctx.model._hf_model
    tokenizer = model.tokenizer
    require_correct = not ctx.smoke
    spec = ctx.args.get("lens") or ctx.lens_specs[0]
    lens = next(iter(ctx.load_lenses([spec]).values()))  # keyed by lens name, not path
    mid = list(ctx.band)
    moe = [i for i, layer in enumerate(model.layers) if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "experts")]
    layers = [l for l in moe if l >= mid[0]]
    W = (model._lm_head.weight.float() * ct.final_norm_gain(model)).detach()
    # The Jacobians on W's device once (causal_test.subspace moves each one per call: the same arithmetic)
    jac = getattr(lens, "jacobians_L_dict_FN", None)
    if jac:
        lens = SimpleNamespace(jacobians_L_dict_FN={l: jac[l].float().to(W.device) for l in mid})
    index = pe.build_vocab_index(tokenizer, ctx.vocab_size)
    concepts = pe.load_concepts(Path(ctx.data_dir) / "concepts.tsv")
    ordered = sorted(concepts)
    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    ids_arr, w_arr, logit_arr, share_arr = [], [], [], []
    reruns: list[float] = []
    n_candidates = 0
    with GateTap(model, layers) as tap:
        for pair in pairs_for(ctx):
            src, tgt = pair.split("-")
            for concept in ordered:  # causal_test.run's item selection, step for step
                words = concepts[concept]
                control = ordered[(ordered.index(concept) + len(ordered) // 2) % len(ordered)]
                target_ids = pe.prefix_token_ids(index, words[tgt], tokenizer)
                english = sorted(pe.prefix_token_ids(index, words["en"], tokenizer, min_chars=3))
                english_ctrl = sorted(pe.prefix_token_ids(index, concepts[control]["en"], tokenizer, min_chars=3) - set(english))
                k = min(len(english), len(english_ctrl))
                if k == 0:
                    continue
                english, english_ctrl = english[:k], english_ctrl[:k]
                ids = model.encode(pe.build_prompt(concepts, concept, src, tgt, shots=5, seed=0))
                target_tensor = t.tensor(sorted(target_ids))
                n_candidates += 1

                def measure() -> tuple[float, bool]:
                    with t.no_grad():
                        logits = hf(input_ids=ids, use_cache=False).logits[0, -1].float()
                    probs = t.softmax(logits, -1)
                    return float(probs[target_tensor.to(probs.device)].sum()), int(logits.argmax()) in target_ids

                p_clean, ok = measure()
                clean = tap.snapshot()
                if require_correct and not ok:
                    continue
                if len(items) < N_RERUNS:  # determinism with the hooks in place
                    measure()
                    again = tap.snapshot()
                    reruns.append(float(np.mean([t.equal(clean[l][0], again[l][0]) for l in layers])))
                row = {"pair": pair, "src": src, "tgt": tgt, "concept": concept, "control_concept": control, "k": k,
                       "clean_p": p_clean, "clean_correct": ok}
                snaps, shares = {"clean": clean}, {}
                for name, toks in (("english", english), ("control", english_ctrl)):
                    bases = {l: ct.subspace(lens, l, toks, W) for l in mid}
                    with EditShare(model, bases) as editshare, ct.Ablation(model, bases):
                        row[f"{name}_p"], row[f"{name}_correct"] = measure()
                    snaps[name], shares[name] = tap.snapshot(), [editshare.share[l] for l in mid]
                    # earlier positions must not move (the edit is at the final position only)
                    row[f"{name}_prefix_same"] = all(t.equal(snaps[name][l][0][:, :-1], clean[l][0][:, :-1]) for l in layers)
                ids_arr.append([[snaps[c][l][0][0, -1].cpu().numpy() for l in layers] for c in CONDITIONS])
                w_arr.append([[snaps[c][l][1][0, -1].cpu().numpy() for l in layers] for c in CONDITIONS])
                logit_arr.append([[snaps[c][l][2][0].cpu().numpy() for l in layers] for c in CONDITIONS])
                share_arr.append([shares["english"], shares["control"]])
                items.append(row)
    if not items:
        raise RuntimeError("no item kept")
    arrays = {"ids": np.array(ids_arr, dtype=np.int16), "weights": np.array(w_arr, dtype=np.float32),
              "logits": np.array(logit_arr, dtype=np.float32), "removed_share": np.array(share_arr, dtype=np.float32),
              "layers": np.array(layers), "mid": np.array(mid)}
    np.savez_compressed(out / "routing_ablation.npz", **arrays)
    write_csv(out / "items.csv", items)
    meta = {"lens": Path(spec).stem if spec != "logit" else "logit", "pairs": pairs_for(ctx), "n_candidates": n_candidates,
            "require_correct": require_correct, "clean_rerun_identical": reruns}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    words_dir = Path(ctx.out_dir).parents[1] / ctx.args.get("words_run", "neuron30") / "routing"
    summary = analyze(out, words_dir if (words_dir / "routing_topk.npz").is_file() else None)
    return summary


### ANALYSIS


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def _r(x, nd: int = 4):
    x = float(x)
    return None if np.isnan(x) else round(x, nd)


def cluster_index(labels) -> list[np.ndarray]:
    """Row indices of each cluster (e.g. each concept), clusters in first-seen order."""
    groups: dict = {}
    for i, g in enumerate(labels):
        groups.setdefault(g, []).append(i)
    return [np.array(v) for v in groups.values()]


def cluster_rows(clusters: list[np.ndarray], seed: int = 0, n: int = 2000) -> list[np.ndarray]:
    """``n`` bootstrap row sets: each draws len(clusters) clusters with replacement and keeps all their rows."""
    rng = np.random.default_rng(seed)
    return [np.concatenate([clusters[g] for g in draw]) for draw in rng.integers(0, len(clusters), (n, len(clusters)))]


def boot(values: np.ndarray, seed: int = 0, n: int = 2000, nd: int = 4, clusters: list[np.ndarray] | None = None) -> list:
    """95% bootstrap CI of the mean over items. With ``clusters`` (row indices per concept), whole
    clusters are resampled with all their rows: the 272 items hold only 32 concepts, each in up to 10
    pairs, so items are not independent. Singleton clusters reproduce the item bootstrap exactly."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    if clusters is None:
        means = values[rng.integers(0, len(values), (n, len(values)))].mean(1)
    else:
        if len(clusters) < 2:
            return [None, None]
        sums = np.array([values[c].sum() for c in clusters])
        counts = np.array([len(c) for c in clusters])
        draws = rng.integers(0, len(clusters), (n, len(clusters)))
        means = sums[draws].sum(1) / counts[draws].sum(1)
    return [_r(np.percentile(means, 2.5), nd), _r(np.percentile(means, 97.5), nd)]


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    return (sums / counts)[inverse]


def spearman(x: np.ndarray, y: np.ndarray, seed: int = 0, n: int = 2000, clusters: list[np.ndarray] | None = None) -> dict:
    """Spearman's rho with a 95% bootstrap CI over items (over whole ``clusters`` when given)."""
    def rho(a, b):
        ra, rb = rankdata(a), rankdata(b)
        if ra.std() == 0 or rb.std() == 0:
            return np.nan
        return float(np.corrcoef(ra, rb)[0, 1])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(x), (n, len(x))) if clusters is None else cluster_rows(clusters, seed, n)
    boots = [rho(x[i], y[i]) for i in draws]
    boots = [b for b in boots if not np.isnan(b)]
    return {"rho": _r(rho(x, y), 3), "ci": [_r(np.percentile(boots, 2.5), 3), _r(np.percentile(boots, 97.5), 3)] if boots else None}


def jaccard_sets(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Jaccard of k-sets: ``a, b`` are ``[..., k]`` expert ids."""
    inter = (a[..., :, None] == b[..., None, :]).any(-1).sum(-1)
    k = a.shape[-1]
    return inter / (2 * k - inter)


def analyze(run_dir: Path, words_dir: Path | None = None, *, clustered: bool = True,
            out_name: str = "routing_ablation_summary.json") -> dict:
    """Every number of the summary, from the saved arrays (also the offline entry point). CIs resample
    whole concepts (``clustered``; until 2026-10-20 they resampled the 272 items as if independent)."""
    run_dir = Path(run_dir)
    arrays = np.load(run_dir / "routing_ablation.npz")
    with (run_dir / "items.csv").open(encoding="utf-8") as handle:
        items = list(csv.DictReader(handle))
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    ids, logits, share = arrays["ids"], arrays["logits"], arrays["removed_share"]
    layers, mid = arrays["layers"].tolist(), arrays["mid"].tolist()
    n = len(items)
    clusters = cluster_index([r["concept"] for r in items]) if clustered else None
    cb = lambda v, **kw: boot(v, clusters=clusters, **kw)  # noqa: E731
    p = {c: np.array([float(r[f"{c}_p"]) for r in items]) for c in CONDITIONS}
    correct = {c: np.array([r[f"{c}_correct"] == "True" for r in items]) for c in CONDITIONS}
    jac = {c: jaccard_sets(ids[:, 0], ids[:, ci]) for ci, c in enumerate(CONDITIONS) if c != "clean"}  # [n, L]
    cos = {}
    for ci, c in ((1, "english"), (2, "control")):
        a, b = logits[:, 0], logits[:, ci]
        cos[c] = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
    per_layer = []
    for li, layer in enumerate(layers):
        je, jc = jac["english"][:, li], jac["control"][:, li]
        per_layer.append({
            "layer": layer, "edited_before": layer > mid[0],
            "jaccard_english": _r(je.mean()), "jaccard_english_ci": cb(je),
            "jaccard_control": _r(jc.mean()), "jaccard_control_ci": cb(jc),
            "control_minus_english": _r((jc - je).mean()), "control_minus_english_ci": cb(jc - je),
            "unchanged_english": _r((je == 1).mean(), 3), "unchanged_control": _r((jc == 1).mean(), 3),
            "router_logit_cos_english": _r(cos["english"][:, li].mean(), 5),
            "router_logit_cos_control": _r(cos["control"][:, li].mean(), 5),
        })
    downstream = [li for li, layer in enumerate(layers) if layer > mid[0]]
    change = {c: (1 - jac[c][:, downstream]).mean(1) for c in jac}
    drop = {c: p["clean"] - p[c] for c in ("english", "control")}
    removed = {c: share[:, ci].mean(1) for ci, c in enumerate(("english", "control"))}
    # At an equal removed share, does the English edit move routing more? change ~ a + b*share + c*[english]
    y = np.concatenate([change["english"], change["control"]])
    X = np.column_stack([np.ones(2 * n), np.concatenate([removed["english"], removed["control"]]),
                         np.concatenate([np.ones(n), np.zeros(n)])])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(0)
    boots = []
    draws = cluster_rows(clusters, 0, 2000) if clustered else (rng.integers(0, n, n) for _ in range(2000))
    for i in draws:
        rows = np.concatenate([i, i + n])  # an item's English and control rows stay together
        boots.append(np.linalg.lstsq(X[rows], y[rows], rcond=None)[0])
    boots = np.array(boots)
    summary = {
        "lens": meta["lens"], "n_items": n, "n_concepts": len(set(r["concept"] for r in items)),
        "ci_method": "bootstrap over concepts (all of a concept's items together)" if clustered else "bootstrap over items", "n_candidates": meta["n_candidates"], "mid_layers": mid, "layers": layers,
        "top_k": int(ids.shape[-1]), "n_experts": int(logits.shape[-1]),
        "target_p": {"clean": _r(p["clean"].mean()), "english_drop": _r(drop["english"].mean()),
                     "english_drop_ci": cb(drop["english"]), "control_drop": _r(drop["control"].mean()),
                     "control_drop_ci": cb(drop["control"]), "still_correct_english": _r(correct["english"].mean(), 3),
                     "still_correct_control": _r(correct["control"].mean(), 3)},
        "checks": {"earlier_positions_unchanged": _r(np.mean([r[f"{c}_prefix_same"] == "True" for r in items for c in ("english", "control")]), 3),
                   "first_layer_unchanged": _r(min(per_layer[0]["unchanged_english"], per_layer[0]["unchanged_control"]), 3),
                   "clean_rerun_identical": meta["clean_rerun_identical"]},
        "per_layer": per_layer,
        "downstream_mean": {"layers": [layers[li] for li in downstream],
                            "change_english": _r(change["english"].mean()), "change_english_ci": cb(change["english"]),
                            "change_control": _r(change["control"].mean()), "change_control_ci": cb(change["control"]),
                            "english_minus_control": _r((change["english"] - change["control"]).mean()),
                            "english_minus_control_ci": cb(change["english"] - change["control"])},
        "removed_share": {"english": _r(removed["english"].mean()), "control": _r(removed["control"].mean()),
                          "by_layer_english": [_r(v) for v in share[:, 0].mean(0)],
                          "by_layer_control": [_r(v) for v in share[:, 1].mean(0)]},
        "equal_share_regression": {"english_term": _r(coef[2]), "english_term_ci": [_r(np.percentile(boots[:, 2], 2.5)), _r(np.percentile(boots[:, 2], 97.5))],
                                   "share_slope": _r(coef[1]), "intercept": _r(coef[0])},
        "change_vs_drop": {"english": spearman(change["english"], drop["english"], clusters=clusters),
                           "control": spearman(change["control"], drop["control"], clusters=clusters),
                           "pooled": spearman(np.concatenate([change["english"], change["control"]]),
                                              np.concatenate([drop["english"], drop["control"]]),
                                              clusters=None if clusters is None else [np.concatenate([c, c + n]) for c in clusters])},
    }
    # Which experts were swapped out: their rank among the item's k by clean router weight, and their weight share
    weights, k_ = arrays["weights"], ids.shape[-1]
    swaps: dict = {}
    for ci, c in ((1, "english"), (2, "control")):
        swapped = k_ - (ids[:, 0][..., :, None] == ids[:, ci][..., None, :]).any(-1).sum(-1)  # [n, L] experts replaced
        ranks, shares = [], []
        for i in range(n):
            for li in downstream:
                order = np.argsort(-weights[i, 0, li])
                for pos, e in enumerate(ids[i, 0, li]):
                    if e not in ids[i, ci, li]:
                        ranks.append(int(np.flatnonzero(order == pos)[0]) + 1)
                        shares.append(float(weights[i, 0, li, pos] / weights[i, 0, li].sum()))
        broken = ~correct[c]
        swaps[c] = {"experts_swapped_by_layer": [_r(v, 3) for v in swapped.mean(0)],
                    "items_with_any_downstream_swap": _r((swapped[:, downstream].sum(1) > 0).mean(), 3),
                    "swapped_rank_counts_1_to_k": np.bincount(ranks, minlength=k_ + 1)[1:].tolist(),
                    "swapped_weight_share_mean": _r(np.mean(shares), 3) if shares else None,
                    "average_weight_share": _r(1 / k_, 3),
                    "downstream_swaps_broken_vs_kept": [_r(swapped[broken][:, downstream].sum(1).mean(), 2) if broken.any() else None,
                                                        _r(swapped[~broken][:, downstream].sum(1).mean(), 2)],
                    "n_broken": int(broken.sum())}
    summary["swaps"] = swaps
    if words_dir is not None:
        summary["selectivity"] = selectivity(ids, items, layers, Path(words_dir), clusters=clusters)
    (run_dir / out_name).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if out_name == "routing_ablation_summary.json":  # a re-analysis under another name leaves the run's files alone
        write_csv(run_dir / "per_layer.csv", [{k: v for k, v in r.items() if not k.endswith("_ci")} for r in per_layer])
    logger.info("routing_ablation: %d items; downstream change english %.4f vs control %.4f", n,
                change["english"].mean(), change["control"].mean())
    return summary


def selectivity(ids: np.ndarray, items: list[dict], layers: list[int], words_dir: Path,
                clusters: list[np.ndarray] | None = None) -> dict:
    """Do the English-ablated expert sets lose English-selective experts or gain target-selective ones?

    Scores per layer and expert, from §11's routing (``words_dir``):
      en_word   share of English bare words routed to the expert minus the mean share over the other 9 languages
      tgt_word  the same for the item's target language
      tgt_out   at the pivot prompts' final position: share of prompts with this target minus prompts with
                another target (the item's concept left out, so its own clean routing never scores itself)
      en_out    the same for prompts whose target is English (ta-en) against Indic targets
    For an item, condition and layer: mean score of the ablated set minus that of the clean set (gained
    experts minus lost ones, over k)."""
    meta = json.loads((words_dir / "routing_items.json").read_text(encoding="utf-8"))
    saved = np.load(words_dir / "routing_topk.npz")
    w_layers = meta["layers"]
    n_exp = int(meta["n_experts"])
    col = [w_layers.index(l) for l in layers]
    words, prompts = meta["words"], meta["prompts"]

    def freq(topk: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """``[L, E]`` share of the selected rows routed to each expert, at each of our layers."""
        x = np.zeros((len(col), n_exp))
        for li, c in enumerate(col):
            np.add.at(x[li], topk[rows][:, c].reshape(-1).astype(np.int64), 1)
        return x / max(1, rows.sum() if rows.dtype == bool else len(rows))

    w_lang = np.array([w["lang"] for w in words])
    langs = sorted(set(w_lang))
    f_lang = {lang: freq(saved["words"], w_lang == lang) for lang in langs}
    score_word = {lang: f_lang[lang] - np.mean([f_lang[o] for o in langs if o != lang], axis=0) for lang in langs}
    p_tgt = np.array([q["tgt"] for q in prompts])
    p_concept = np.array([q["concept"] for q in prompts])
    out_cache: dict = {}

    def score_out(target: str, concept: str) -> np.ndarray:
        key = (target, concept)
        if key not in out_cache:
            keep = p_concept != concept
            mine, rest = keep & (p_tgt == target), keep & (p_tgt != target)
            if target == "en":
                rest = keep & (p_tgt != "en") & (p_tgt != "hi_latn")
            out_cache[key] = freq(saved["prompts"], mine) - freq(saved["prompts"], rest)
        return out_cache[key]

    def delta(score: np.ndarray, clean: np.ndarray, ablated: np.ndarray) -> np.ndarray:
        """``[L]``: mean score of the ablated set minus the clean set's."""
        return np.array([score[li, ablated[li]].mean() - score[li, clean[li]].mean() for li in range(len(col))])

    names = ("en_word", "tgt_word", "tgt_out", "en_out")
    deltas = {c: {s: [] for s in names} for c in ("english", "control")}
    lost_gained = {c: {"lost_en_top10": 0, "gained_en_top10": 0, "lost_tgt_top10": 0, "gained_tgt_top10": 0, "changed": 0}
                   for c in ("english", "control")}
    for i, row in enumerate(items):
        scores = {"en_word": score_word["en"], "tgt_word": score_word[row["tgt"]],
                  "tgt_out": score_out(row["tgt"], row["concept"]), "en_out": score_out("en", row["concept"])}
        for ci, c in ((1, "english"), (2, "control")):
            for s in names:
                deltas[c][s].append(delta(scores[s], ids[i, 0].astype(int), ids[i, ci].astype(int)))
            for li in range(len(col)):
                clean, abl = set(ids[i, 0, li].tolist()), set(ids[i, ci, li].tolist())
                lost, gained = clean - abl, abl - clean
                top_en = set(np.argsort(-scores["en_word"][li])[:10].tolist())
                top_tgt = set(np.argsort(-scores["tgt_out"][li])[:10].tolist())
                stats = lost_gained[c]
                stats["changed"] += len(lost)
                stats["lost_en_top10"] += len(lost & top_en)
                stats["gained_en_top10"] += len(gained & top_en)
                stats["lost_tgt_top10"] += len(lost & top_tgt)
                stats["gained_tgt_top10"] += len(gained & top_tgt)
    out: dict = {"scores": {"en_word": "English bare words vs the other 9 languages", "tgt_word": "target-language bare words vs the rest",
                            "tgt_out": "pivot prompts with this target vs other targets (final position, concept left out)",
                            "en_out": "pivot prompts translating INTO English (ta-en) vs Indic targets"},
                 "layers": layers, "lost_gained_counts": lost_gained}
    downstream = list(range(1, len(layers)))  # layers[0] is the first edited block: its router runs before the edit
    for s in names:
        e, c = np.array(deltas["english"][s]), np.array(deltas["control"][s])  # [n, L]
        out[s] = {"by_layer_english": [_r(v, 5) for v in e.mean(0)], "by_layer_control": [_r(v, 5) for v in c.mean(0)],
                  "downstream_english": _r(e[:, downstream].mean(), 5), "downstream_control": _r(c[:, downstream].mean(), 5),
                  "english_minus_control": _r((e - c)[:, downstream].mean(), 5),
                  "english_minus_control_ci": boot((e - c)[:, downstream].mean(1), nd=5, clusters=clusters)}
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recompute a downloaded routing_ablation run's summary on CPU.")
    parser.add_argument("--analyze", required=True, help="the run's routing_ablation_english dir")
    parser.add_argument("--words", default=None, help="§11's routing dir (results/neuron30/routing)")
    parser.add_argument("--out", default="routing_ablation_summary_v2.json",
                        help="summary file name in the run dir (v2 = concept-clustered CIs)")
    parser.add_argument("--item-ci", action="store_true", help="the pre-2026-10-20 item bootstrap (for comparison)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(json.dumps(analyze(Path(args.analyze), Path(args.words) if args.words else None,
                             clustered=not args.item_ci, out_name=args.out), indent=1)[:4000])
