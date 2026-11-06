"""Does bf16 unembedding change the published lens readouts? (docs/07 open item;
write-up: docs/14_bf16_readout_check.md)

Sarvam's logits sit in the hundreds (its outlier final-norm gains, docs/04 §4.5), where bf16
resolves only about whole units. jlens's ``unembed`` runs the final norm and LM head in the head's
dtype (bf16); :func:`pivot_experiment.score` casts to fp32 only afterwards and counts a rank as
(#strictly greater logits) + 1, so a tie gets the better rank. This module re-reads the published
pivot (``pivot_experiment.DEFAULT_PAIRS`` on ``data/concepts.tsv``, correctness filter on: the
items, prompts, token sets, roles and shuffled controls of ``results/final/pivot_all``) and scores
every item x lens x layer x role twice from the SAME transported residual:

  (a) ``bf16``: ``model.unembed(h).float()``, as published, with the published (optimistic) rank
      and a pessimistic one that counts every outside token tied with the best set token as ahead;
  (b) ``fp32``: the final norm and LM head cast to fp32 (:class:`Fp32Head`).

One lone forward per item, as ``pivot_experiment.main``, so (a) should reproduce the published
rows. The correctness filter keeps the published (bf16) decision; ``items.csv`` also records the
fp32 head's top token and whether it would have passed.

Interface (modal_pilot stage ``quant_check.pivot``): ``run(ctx, analysis)`` and
``n_forwards(ctx, analysis)``; ``ctx`` is documented in newlang_experiment's docstring. Stage args:
``pairs`` (default DEFAULT_PAIRS), ``lenses`` (default ``ctx.lens_specs``: the J-lens j_200p, then
logit), ``max_items`` (per pair, on candidates), ``filter`` (default true).

Writes ``rows.csv`` (item x lens x layer x role: P and ranks under both heads), ``logits.csv``
(item x lens x layer: logit magnitudes, top-1/top-10 agreement, |bf16 - fp32|), ``items.csv``,
``summary_bf16.csv`` / ``summary_fp32.csv`` (``pivot_experiment.summarize`` format) and
``quant_summary.csv`` (:func:`analyze`: paired bootstrap CIs of fp32 - bf16).

Local:
  python quant_check.py check <tiny model dir> --lens <tiny j lens> --out <dir>
      (a) against pivot_experiment.main's rows on the same tiny model (bf16), then (a) vs (b)
  python quant_check.py analyze <run dir> [--published ../results/final/pivot_all] [--md out.md]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import warnings
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch as t

import pivot_experiment as pe

logger = logging.getLogger("quant_check")
HERE = Path(__file__).resolve().parent


class Fp32Head:
    """The final norm and LM head in fp32. Sarvam's logits can sit in the hundreds (its outlier
    final-norm gains), where bf16 resolves only whole units, so ties and rounding there are bf16's,
    not the model's."""

    def __init__(self, model, vocab_size: int) -> None:
        import copy

        self.device = model._lm_head.weight.device
        self.norm = copy.deepcopy(model._final_norm).float()
        self.weight = model._lm_head.weight[:vocab_size].float()
        bias = getattr(model._lm_head, "bias", None)
        self.bias = None if bias is None else bias[:vocab_size].float()
        self.softcap = getattr(model, "_logit_softcap", None)

    def __call__(self, residual: t.Tensor) -> t.Tensor:
        logits = self.norm(residual.float().to(self.device)) @ self.weight.T
        if self.bias is not None:
            logits = logits + self.bias
        if self.softcap is not None:
            logits = self.softcap * t.tanh(logits / self.softcap)
        return logits


BAND = [13, 14, 15, 16]
HEADLINE_LAYERS = [11, 13, 14, 15, 16, 17]  # docs/04 §4.1's table
HEADLINE_ROLES = ("english", "english_shuffled", "hindi", "hindi_shuffled", "target", "chinese")
REPORT_ROLES = ("target", "english", "hindi", "english_shuffled", "hindi_shuffled")
HEADS = ("bf16", "fp32")
# Scoring one lens for one item: two unembeds of every layer plus the rank counts (both heads,
# ~8 roles) over 262k logits: ~150 small kernels, a few ms on an H200 (newlang's 10 ms covers one head).
SCORE_SECONDS = 0.015
SETUP_SECONDS = 20.0  # vocab index (262k decodes), readout mask, token sets, J-lens load, fp32 head copy
TOP10_PTS, REL_P = 2.0, 0.10  # the verdict's thresholds: top-10 points, relative P


### SCORING


def score_both(model, lens, layers: list[int], acts: dict[int, t.Tensor], role_ids: dict[str, set[int]],
               mask_V: t.Tensor | None, vocab_size: int, head32) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Every layer of one lens decoded twice from the same transported residual ``h``: (a)
    ``model.unembed(h)`` in the head's dtype, as published, and (b) ``head32(h)`` in fp32.
    Returns ``{role: [8, n_layers]}`` (per head: P, strict rank, pessimistic rank, best set logit)
    and per-layer logit statistics ``{name: [n_layers]}``."""
    device = model._lm_head.weight.device
    with t.no_grad():
        h = t.cat([lens.transport(acts[layer][None], layer).to(device) for layer in layers])
        logits = {"bf16": model.unembed(h).float()[:, :vocab_size], "fp32": head32(h)[:, :vocab_size]}
        if mask_V is not None:
            logits = {k: v.masked_fill(mask_V, float("-inf")) for k, v in logits.items()}
        probs = {k: t.softmax(v, -1) for k, v in logits.items()}
        roles = {}
        for role, ids in role_ids.items():
            if not ids:
                continue
            idx = t.tensor(sorted(ids), dtype=t.long, device=device)
            out = []
            for head in HEADS:
                sub = logits[head][:, idx]
                best = sub.max(1).values[:, None]
                strict = (logits[head] > best).sum(1) + 1  # pivot_experiment.score
                # ties with tokens outside the set count as ahead (set tokens tied with the best don't)
                pess = (logits[head] >= best).sum(1) - (sub >= best).sum(1) + 1
                out += [probs[head][:, idx].sum(1), strict.float(), pess.float(), best[:, 0]]
            roles[role] = t.stack(out).cpu().numpy()
        a, b = logits["bf16"], logits["fp32"]
        live = t.isfinite(b)
        diff = (a - b).abs().masked_fill(~live, 0)
        top_a, top_b = a.topk(11, dim=1), b.topk(11, dim=1)
        overlap = (top_a.indices[:, :10, None] == top_b.indices[:, None, :10]).any(2).sum(1)
        stats = {
            "max_logit_bf16": top_a.values[:, 0], "max_logit_fp32": top_b.values[:, 0],
            "logit10_bf16": top_a.values[:, 9], "logit10_fp32": top_b.values[:, 9],
            # tokens sharing the 10th-largest bf16 logit (1 = no tie at the top-10 boundary)
            "n_at_logit10_bf16": (a == top_a.values[:, 9:10]).sum(1),
            "gap10_11_fp32": top_b.values[:, 9] - top_b.values[:, 10],
            "top1_agree": (top_a.indices[:, 0] == top_b.indices[:, 0]).int(),
            "top10_overlap": overlap,
            "max_abs_dlogit": diff.max(1).values, "mean_abs_dlogit": diff.sum(1) / live.sum(1),
        }
        stats = {k: v.float().cpu().numpy() for k, v in stats.items()}
    return roles, stats


### THE STAGE


def _pairs(ctx) -> list[str]:
    pairs = [p for p in (ctx.args.get("pairs") or pe.DEFAULT_PAIRS).split(",") if p]
    return pairs[:2] if ctx.smoke else pairs


def run(ctx, analysis: str) -> dict:
    if analysis == "hello":
        return {"model_kind": ctx.model_kind, "lens_specs": ctx.lens_specs, "n_layers": ctx.n_layers, "args": ctx.args}
    if analysis != "pivot":
        raise ValueError(f"unknown analysis {analysis!r}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    import newlang_experiment as nle
    import reader_questions as rqm

    model, vocab = ctx.model, ctx.vocab_size
    tokenizer = model.tokenizer
    lenses = ctx.load_lenses(ctx.args.get("lenses") or ctx.lens_specs)
    layers = list(ctx.layers)
    concepts = pe.load_concepts(Path(ctx.data_dir) / "concepts.tsv")
    pairs = _pairs(ctx)
    require_correct = not ctx.smoke and bool(ctx.args.get("filter", True))
    max_items = 2 if ctx.smoke else ctx.args.get("max_items")

    index = pe.build_vocab_index(tokenizer, vocab)
    tracked: set[int] = set()
    for concept in concepts.values():
        for spellings in concept.values():
            tracked |= pe.prefix_token_ids(index, spellings, tokenizer)
    device = model._lm_head.weight.device
    mask_V = pe.readout_mask(tokenizer, vocab, tracked).to(device)
    # pivot_experiment.main's items, in its order (Japanese is in no default pair: no kanji rule)
    specs, skipped = nle.build_items(concepts, pairs, index, tokenizer, max_items=max_items, kanji_rule=False)
    head32 = Fp32Head(model, vocab)
    reader = rqm.Reader(model, {}, layers, vocab, None)
    final, last_layer = model.n_layers - 1, model.n_layers - 1
    logger.info("%d items over %d pairs, lenses %s, head dtype %s", len(specs), len(pairs), list(lenses),
                model._lm_head.weight.dtype)

    rows, logit_rows, items = [], [], []
    for n_done, spec in enumerate(specs):
        ids = model.encode(pe.build_prompt(concepts, spec["concept"], spec["src"], spec["tgt"], shots=5, seed=0))
        acts, logits = reader.forward(ids)  # lone forward: pivot_experiment.main's numerics
        top = int(logits.argmax())
        with t.no_grad():
            top32 = int(head32(acts[final][None])[0].argmax())
        correct = top in spec["raw_target"]
        kept = correct or not require_correct
        items.append({**spec["item"], "model_top_token": tokenizer.decode([top]), "kept": kept,
                      "reason": "" if kept else "model's next token does not begin the target word",
                      "fp32_top_token": tokenizer.decode([top32]), "fp32_top_in_target": top32 in spec["raw_target"],
                      "top_agree": top == top32})
        if not kept:
            continue
        base = {"pair": spec["item"]["pair"], "src": spec["src"], "tgt": spec["tgt"], "concept": spec["concept"]}
        for lens_name, lens in lenses.items():
            role_stats, layer_stats = score_both(model, lens, layers, acts, spec["sets"], mask_V, vocab, head32)
            for i, layer in enumerate(layers):
                logit_rows.append({**base, "lens": lens_name, "layer": layer,
                                   **{k: round(float(v[i]), 4) for k, v in layer_stats.items()}})
                for role, lang in spec["role_langs"].items():
                    row = {**base, "lens": lens_name, "layer": layer, "depth": layer / last_layer, "role": role, "lang": lang}
                    s = role_stats.get(role)
                    for j, head in enumerate(HEADS):
                        if s is None:
                            row.update({f"prob_{head}": float("nan"), f"rank_{head}": -1, f"rank_{head}_pess": -1,
                                        f"best_logit_{head}": float("nan")})
                        else:
                            row.update({f"prob_{head}": float(s[4 * j, i]), f"rank_{head}": int(s[4 * j + 1, i]),
                                        f"rank_{head}_pess": int(s[4 * j + 2, i]), f"best_logit_{head}": float(s[4 * j + 3, i])})
                    rows.append(row)
        if n_done % 50 == 0:
            logger.info("read %d / %d items", n_done + 1, len(specs))
    items += skipped

    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (("rows.csv", rows), ("logits.csv", logit_rows), ("items.csv", items)):
        write_csv(out / name, table)
    indic = indic_pairs(pairs)
    for head in HEADS:
        write_csv(out / f"summary_{head}.csv", pe.summarize(as_published(rows, head), indic))
    ctx.commit()
    result = {"pairs": len(pairs), "lenses": list(lenses), "items_kept": sum(i["kept"] for i in items),
              "items": len(items), "fp32_filter_disagrees": sum(1 for i in items if "top_agree" in i and not i["top_agree"]),
              "out": str(out)}
    try:  # the rows are already committed: an analysis bug must not cost the run
        summary, verdict = analyze(rows, logit_rows, pairs)
        write_csv(out / "quant_summary.csv", summary)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=1, ensure_ascii=False), encoding="utf-8")
        ctx.commit()
        result["verdict"] = {k: verdict[k] for k in ("moves", "n_checked", "max_top10_move_pts", "max_rel_p_move")}
        result["band_all_indic"] = [{k: s[k] for k in ("lens", "role", "top10_bf16", "top10_fp32", "d_top10_pts",
                                                         "prob_bf16", "prob_fp32", "rel_d_prob_pct", "boundary_ties")}
                                    for s in summary if s["group"] == "ALL_INDIC" and s["layers"] == band_name()]
    except Exception as exc:  # noqa: BLE001
        logger.exception("analysis failed")
        result["analysis_error"] = repr(exc)
    return result


def n_forwards(ctx, analysis: str) -> float:
    if analysis != "pivot":
        return 1.0
    import newlang_experiment as nle

    pairs = _pairs(ctx)
    n_items = 2 * len(pairs) if ctx.smoke else nle.count_items(Path(ctx.data_dir) / "concepts.tsv", pairs)
    n_lenses = len(ctx.args.get("lenses") or ctx.lens_specs)
    spf = max(ctx.seconds_per_forward, 1e-3)
    return n_items + (n_items * n_lenses * SCORE_SECONDS + SETUP_SECONDS) / spf


### ANALYSIS


def indic_pairs(pairs) -> set[str]:
    return {p for p in pairs if all(pe.FAMILY.get(l) in ("IA", "Dr") and l not in ("hi", "hi_latn") for l in p.split("-"))}


def band_name() -> str:
    return f"L{BAND[0]}-{BAND[-1]}"


def as_published(rows: list[dict], head: str, pess: bool = False) -> list[dict]:
    """``rows`` with one head's P and rank under ``pivot_experiment``'s column names."""
    rank_key = f"rank_{head}_pess" if pess else f"rank_{head}"
    out = []
    for r in rows:
        rank = int(float(r[rank_key]))
        out.append({k: r[k] for k in ("pair", "src", "tgt", "concept", "lens", "layer", "depth", "role", "lang")} |
                   {"layer": int(r["layer"]), "depth": float(r["depth"]), "prob": float(r[f"prob_{head}"]),
                    "best_rank": rank, "top10": 0 < rank <= 10})
    return out


def _tensor(rows: list[dict]):
    """Rows as arrays ``[item, lens, layer, role]`` per metric (NaN where a role has no row)."""
    items = list(dict.fromkeys((r["pair"], r["concept"]) for r in rows))
    lenses = list(dict.fromkeys(r["lens"] for r in rows))
    layers = sorted({int(r["layer"]) for r in rows})
    roles = list(dict.fromkeys(r["role"] for r in rows))
    pos = [{k: i for i, k in enumerate(seq)} for seq in (items, lenses, layers, roles)]
    metrics = ("prob_bf16", "rank_bf16", "rank_bf16_pess", "prob_fp32", "rank_fp32", "rank_fp32_pess")
    arr = {m: np.full((len(items), len(lenses), len(layers), len(roles)), np.nan) for m in metrics}
    for r in rows:
        at = (pos[0][(r["pair"], r["concept"])], pos[1][r["lens"]], pos[2][int(r["layer"])], pos[3][r["role"]])
        for m in metrics:
            arr[m][at] = float(r[m])
    return items, lenses, layers, roles, arr


def _weights(n: int, resamples: int = 2000, seed: int = 0) -> np.ndarray:
    """Bootstrap resamples of ``n`` items as count weights ``[resamples, n]``: a paired bootstrap of
    any per-item statistic is then one matrix product."""
    draws = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    w = np.zeros((resamples, n))
    np.add.at(w, (np.repeat(np.arange(resamples), n), draws.ravel()), 1.0)
    return w


def _paired(xa: np.ndarray, xb: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float, float]:
    """mean(a), mean(b), mean(b - a) and its 95% paired bootstrap CI over the items valid in both."""
    valid = ~(np.isnan(xa) | np.isnan(xb))
    if not valid.any():
        return (float("nan"),) * 5
    v = valid.astype(float)
    d = np.where(valid, xb - xa, 0.0)
    boots = (w @ d) / np.maximum(w @ v, 1.0)
    return (float(np.nanmean(np.where(valid, xa, np.nan))), float(np.nanmean(np.where(valid, xb, np.nan))),
            float(d.sum() / v.sum()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def analyze(rows: list[dict], logit_rows: list[dict], pairs: list[str]) -> tuple[list[dict], dict]:
    """Per group (ALL_INDIC, each pair) x lens x layer set (each layer, the band) x role: top-10
    rate and mean P under bf16 (published rank), bf16 (pessimistic rank) and fp32, paired bootstrap
    CIs (over items) of fp32 - bf16, and the share of rows whose rank changes, whose top-10 call
    flips, and whose bf16 rank depends on the tie rule (anywhere / at the top-10 boundary). Top-10
    counts an item with an empty set as not in the top 10 and P skips it, as ``pivot_experiment.summarize``."""
    items, lenses, layers, roles, arr = _tensor(rows)
    rank = {h: arr[f"rank_{h}"] for h in HEADS}
    present = ~np.isnan(rank["bf16"])  # the role has a row for this item
    has_set = present & (rank["bf16"] > 0)
    top10 = {
        "bf16": np.where(present, has_set & (rank["bf16"] <= 10), np.nan),
        "bf16_pess": np.where(present, has_set & (arr["rank_bf16_pess"] <= 10), np.nan),
        "fp32": np.where(present, has_set & (rank["fp32"] <= 10), np.nan),
    }
    prob = {h: np.where(has_set, arr[f"prob_{h}"], np.nan) for h in HEADS}
    cube = {  # every per-row quantity, [item, lens, layer, role]
        "top10_bf16": top10["bf16"], "top10_bf16_pess": top10["bf16_pess"], "top10_fp32": top10["fp32"],
        "prob_bf16": prob["bf16"], "prob_fp32": prob["fp32"],
        "changed": np.where(has_set, rank["bf16"] != rank["fp32"], np.nan),
        "flips": np.where(present, top10["bf16"] != top10["fp32"], np.nan),
        "ties": np.where(has_set, rank["bf16"] != arr["rank_bf16_pess"], np.nan),
        "boundary": np.where(present, top10["bf16"] != top10["bf16_pess"], np.nan),
        "fp32_ties": np.where(has_set, rank["fp32"] != arr["rank_fp32_pess"], np.nan),
    }

    groups = {"ALL_INDIC": [i for i, (p, _) in enumerate(items) if p in indic_pairs(pairs)]}
    for pair in pairs:
        groups[pair] = [i for i, (p, _) in enumerate(items) if p == pair]
    layer_sets = [(f"L{l}", [l]) for l in layers] + [(band_name(), [l for l in BAND if l in layers])]
    summary = []
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # nanmean over an item's all-NaN cells
        for g, idx in groups.items():
            if not idx:
                continue
            w = _weights(len(idx))
            sub = {k: v[idx] for k, v in cube.items()}
            for li, lens in enumerate(lenses):
                for name, lset in layer_sets:
                    cols = [layers.index(l) for l in lset]
                    if not cols:
                        continue
                    for ri, role in enumerate(roles):
                        cell = {k: v[:, li, cols, ri] for k, v in sub.items()}  # [items, layers in set]
                        if np.all(np.isnan(cell["top10_bf16"])):
                            continue
                        per_item = {k: np.nanmean(v, axis=1) for k, v in cell.items()}
                        ta, tb, dt, dt_lo, dt_hi = _paired(per_item["top10_bf16"], per_item["top10_fp32"], w)
                        tp = float(np.nanmean(per_item["top10_bf16_pess"]))
                        pa, pb, dp, dp_lo, dp_hi = _paired(per_item["prob_bf16"], per_item["prob_fp32"], w)
                        share = lambda k: float(np.nanmean(cell[k])) if np.any(~np.isnan(cell[k])) else float("nan")
                        summary.append({
                            "group": g, "lens": lens, "layers": name, "role": role, "n_items": len(idx),
                            "n_prob": int(np.sum(~np.isnan(per_item["prob_bf16"]))),
                            "top10_bf16": r4(ta), "top10_bf16_pess": r4(tp), "top10_fp32": r4(tb),
                            "d_top10_pts": r2(100 * dt), "d_top10_ci": f"[{100 * dt_lo:.2f}, {100 * dt_hi:.2f}]",
                            "d_top10_pess_pts": r2(100 * (tp - ta)),
                            "prob_bf16": r6(pa), "prob_fp32": r6(pb), "d_prob": r6(dp), "d_prob_ci": f"[{dp_lo:.5f}, {dp_hi:.5f}]",
                            "rel_d_prob_pct": r2(100 * dp / pa) if pa and not math.isnan(pa) else float("nan"),
                            "rank_changed": r4(share("changed")), "top10_flips": r4(share("flips")),
                            "ties": r4(share("ties")), "boundary_ties": r4(share("boundary")), "fp32_ties": r4(share("fp32_ties")),
                        })
    verdict = headline(summary, pairs)
    verdict["logits"] = logit_summary(logit_rows)
    return summary, verdict


def r2(x: float) -> float:
    return round(float(x), 2)


def r4(x: float) -> float:
    return round(float(x), 4)


def r6(x: float) -> float:
    return round(float(x), 6)


def logit_summary(logit_rows: list[dict]) -> list[dict]:
    """Per lens and layer: median top logit (bf16 and fp32), the bf16 spacing there, median and max
    |bf16 - fp32| over the vocab, top-1 agreement, mean top-10 overlap, and the share of readouts
    whose 10th-largest bf16 logit is shared with another token."""
    groups = defaultdict(list)
    for r in logit_rows:
        groups[(r["lens"], int(r["layer"]))].append(r)
    out = []
    for (lens, layer), rr in sorted(groups.items()):
        col = lambda k: np.array([float(r[k]) for r in rr])
        top = col("max_logit_bf16")
        out.append({"lens": lens, "layer": layer, "n": len(rr),
                    "median_top_logit_bf16": r2(np.median(top)), "median_top_logit_fp32": r2(np.median(col("max_logit_fp32"))),
                    "median_logit10_fp32": r2(np.median(col("logit10_fp32"))),
                    "bf16_spacing_at_median_top": bf16_spacing(float(np.median(top))),
                    "median_max_abs_dlogit": r4(np.median(col("max_abs_dlogit"))),
                    "max_max_abs_dlogit": r4(col("max_abs_dlogit").max()),
                    "median_mean_abs_dlogit": r4(np.median(col("mean_abs_dlogit"))),
                    "top1_agree": r4(col("top1_agree").mean()), "mean_top10_overlap": r2(col("top10_overlap").mean()),
                    "tie_at_logit10_bf16": r4((col("n_at_logit10_bf16") > 1).mean()),
                    "median_gap10_11_fp32": r4(np.median(col("gap10_11_fp32")))})
    return out


def bf16_spacing(x: float) -> float:
    """The gap between adjacent bf16 values at magnitude ``x`` (8 significant bits)."""
    if not x or math.isnan(x) or math.isinf(x):
        return float("nan")
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7)


def _cell(summary: list[dict], group: str, lens: str, layers: str, role: str) -> dict | None:
    return next((s for s in summary if (s["group"], s["lens"], s["layers"], s["role"]) == (group, lens, layers, role)), None)


def headline(summary: list[dict], pairs: list[str], published: dict | None = None) -> dict:
    """The docs/04 §4.1 numbers under bf16 (as published) and fp32: the J-lens top-10 table
    (ALL_INDIC, L11 and L13-17), the same cells under the logit lens, peak P(English) per lens,
    each pair's peak P(English) under the J-lens, peak P(Hindi), and the controls' peak top-10.
    A number "moves" if fp32 - bf16 exceeds 2 points of top-10, or 10% of P relative (P >= 0.01;
    below that, 0.1 point absolute). ``published``: ``{(group, lens, layer, role): row}`` from the
    published summary.csv, added for comparison."""
    lenses = sorted({s["lens"] for s in summary})
    j_lens = next((l for l in lenses if l != "logit"), None)
    checks = []

    def add(label, metric, group, lens, layer_names, role, peak=False):
        cells = [c for c in (_cell(summary, group, lens, n, role) for n in layer_names) if c]
        if not cells:
            return
        key_a, key_b = ("top10_bf16", "top10_fp32") if metric == "top10" else ("prob_bf16", "prob_fp32")
        if peak:  # the layer of the bf16 peak, and the fp32 peak wherever it falls
            ca = max(cells, key=lambda c: c[key_a])
            va, vb, where = ca[key_a], max(c[key_b] for c in cells), ca["layers"]
        else:
            va, vb, where = cells[0][key_a], cells[0][key_b], cells[0]["layers"]
        entry = {"number": label, "metric": metric, "group": group, "lens": lens, "layer": where, "role": role,
                 "bf16": va, "fp32": vb}
        if metric == "top10":
            entry["move_pts"] = r2(100 * (vb - va))
            entry["moves"] = abs(entry["move_pts"]) > TOP10_PTS
        else:
            entry["rel_move"] = r4((vb - va) / va) if va else float("nan")
            entry["moves"] = abs(vb - va) > (REL_P * va if va >= 0.01 else 0.001)
        if metric == "top10":
            entry["bf16_pess"] = cells[0]["top10_bf16_pess"] if not peak else ca["top10_bf16_pess"]
        if published is not None:
            key = "top10_rate" if metric == "top10" else "mean_prob"
            layer_nums = [int(c["layers"][1:]) for c in cells] if peak else [int(where[1:])]
            vals = [float(published[k][key]) for k in ((group, lens, l, role) for l in layer_nums) if k in published]
            if vals:
                entry["published"] = max(vals)
        checks.append(entry)

    for lens in [l for l in (j_lens, "logit") if l in lenses]:
        for role in HEADLINE_ROLES:
            for layer in HEADLINE_LAYERS:
                add(f"{lens} top-10 {role} L{layer}", "top10", "ALL_INDIC", lens, [f"L{layer}"], role)
        all_layers = sorted({s["layers"] for s in summary if s["layers"].startswith("L") and "-" not in s["layers"]},
                            key=lambda x: int(x[1:]))
        add(f"{lens} peak P(english)", "prob", "ALL_INDIC", lens, all_layers, "english", peak=True)
        add(f"{lens} peak P(hindi)", "prob", "ALL_INDIC", lens, all_layers, "hindi", peak=True)
        for control in ("english_shuffled", "hindi_shuffled"):
            add(f"{lens} peak top-10 {control}", "top10", "ALL_INDIC", lens, all_layers, control, peak=True)
        if lens == j_lens:
            for pair in pairs:
                add(f"{lens} peak P(english) {pair}", "prob", pair, lens, all_layers, "english", peak=True)
    moved = [c for c in checks if c["moves"]]
    tops = [abs(c["move_pts"]) for c in checks if "move_pts" in c]
    rels = [abs(c["rel_move"]) for c in checks if "rel_move" in c and c["bf16"] >= 0.01]
    return {"n_checked": len(checks), "moves": len(moved), "moved": moved,
            "max_top10_move_pts": max(tops, default=float("nan")), "max_rel_p_move": max(rels, default=float("nan")),
            "checks": checks}


### LOCAL: CROSS-CHECK, TINY-MODEL CHECK, REPORT


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, table: list[dict]) -> None:
    if not table:
        return
    fields = list(dict.fromkeys(k for row in table for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)


def cross_check(rows: list[dict], reference: list[dict], ref_items: list[dict] | None = None,
                items: list[dict] | None = None) -> dict:
    """(a) (bf16, published rank) against a reference rows.csv (``pivot_experiment.main``'s
    format), joined on (pair, concept, lens, layer, role) over the lenses both have."""
    mine = {(r["pair"], r["concept"], r["lens"], str(r["layer"]), r["role"]): r for r in rows}
    lenses = {r["lens"] for r in rows}
    ref = {(r["pair"], r["concept"], r["lens"], str(r["layer"]), r["role"]): r for r in reference if r["lens"] in lenses}
    both = mine.keys() & ref.keys()
    dprob, rank_diff, top10_diff = [], 0, 0
    for k in both:
        a, b = float(mine[k]["prob_bf16"]), float(ref[k]["prob"])
        if not (math.isnan(a) and math.isnan(b)):
            dprob.append(abs(a - b))
        rank_a = int(float(mine[k]["rank_bf16"]))
        rank_diff += rank_a != int(ref[k]["best_rank"])
        top10_diff += (0 < rank_a <= 10) != (ref[k]["top10"] == "True")
    out = {"lenses": sorted(lenses), "rows_mine": len(mine), "rows_reference": len(ref), "rows_matched": len(both),
           "max_abs_dprob": max(dprob, default=float("nan")), "mean_abs_dprob": float(np.mean(dprob)) if dprob else float("nan"),
           "rank_diffs": rank_diff, "top10_diffs": top10_diff}
    if ref_items is not None and items is not None:
        kept = lambda tbl: {(i["pair"], i["concept"]) for i in tbl if str(i["kept"]) == "True"}
        out["items_kept_mine"], out["items_kept_reference"] = len(kept(items)), len(kept(ref_items))
        out["items_kept_same"] = kept(items) == kept(ref_items)
    return out


def local_ctx(model_dir: str, lens_path: str | None, out_dir: str, args: dict, smoke: bool = False):
    import run_jpp

    jpp_cli = run_jpp._import_jpp_cli()
    run_jpp.install(jpp_cli)
    model = jpp_cli.get_hf_model(model_dir, attn_implementation="sdpa", device_map=None)
    layers = list(range(model.n_layers - 1))
    specs = ([lens_path] if lens_path else []) + ["logit"]
    return SimpleNamespace(
        model=model, model_kind="sarvam30", model_dir=model_dir, is_mla=False, n_layers=model.n_layers, layers=layers,
        band=BAND, early_band=[5, 6, 7, 8], lens_specs=specs,
        load_lenses=lambda s=None: jpp_cli.load_lenses(list(s or specs), model, hf_model_name=model_dir, layers=layers),
        vocab_size=jpp_cli.vocab_size_of(model), out_dir=out_dir, data_dir=str(HERE / "data"), batch_size=1, smoke=smoke,
        args=args, commit=lambda: None, seconds_per_forward=0.1)


def check(model_dir: str, lens_path: str | None, out_dir: str, pairs: str) -> int:
    """On a tiny model in bf16: (a) must reproduce ``pivot_experiment.main``'s rows (same keys,
    P to 1e-5, identical ranks), and (b) may differ from (a) only by rounding."""
    import run_jpp

    run_jpp.install(run_jpp._import_jpp_cli())
    out = Path(out_dir)
    specs = ["logit"] + ([lens_path] if lens_path else [])
    reference = out / "reference"
    code = pe.main(["--hf-model-name", model_dir, *sum((["--lens", s] for s in specs), []), "--pairs", pairs,
                    "--no-correctness-filter", "--out", str(reference)])
    assert code == 0
    ctx = local_ctx(model_dir, lens_path, str(out / "quant"), {"pairs": pairs, "filter": False})
    print("head dtype", ctx.model._lm_head.weight.dtype)
    result = run(ctx, "pivot")
    print(json.dumps({k: v for k, v in result.items() if k != "band_all_indic"}, indent=1, default=str))
    rows = read_csv(out / "quant" / "rows.csv")
    xc = cross_check(rows, read_csv(reference / "rows.csv"))
    print("(a) vs pivot_experiment.main:", json.dumps(xc))
    ok = xc["rows_matched"] == xc["rows_mine"] == xc["rows_reference"] and xc["max_abs_dprob"] < 1e-5 and xc["rank_diffs"] == 0
    probs = [(float(r["prob_bf16"]), float(r["prob_fp32"])) for r in rows if not math.isnan(float(r["prob_bf16"]))]
    dmax = max(abs(a - b) for a, b in probs)
    rel = max(abs(a - b) / max(a, 1e-12) for a, b in probs)
    changed = sum(r["rank_bf16"] != r["rank_fp32"] for r in rows)
    ties = sum(r["rank_bf16"] != r["rank_bf16_pess"] for r in rows)
    logits = read_csv(out / "quant" / "logits.csv")
    print(f"(a) vs (b): {len(rows)} rows; max |dP| {dmax:.2e}; max relative dP {rel:.2e}; ranks changed {changed} "
          f"({changed / len(rows):.1%}); bf16 ties {ties}; median top logit "
          f"{np.median([float(r['max_logit_bf16']) for r in logits]):.3f}; max |dlogit| "
          f"{max(float(r['max_abs_dlogit']) for r in logits):.2e}")
    print("EQUIVALENT" if ok else "MISMATCH")
    return 0 if ok else 1


def markdown(summary: list[dict], verdict: dict, xc: dict | None, pairs: list[str]) -> str:
    """Tables for docs/14_bf16_readout_check.md."""
    lines = []
    lenses = sorted({s["lens"] for s in summary}, key=lambda l: l == "logit")
    if xc:
        lines += ["### Cross-check: (a) against results/final/pivot_all", "", "```", json.dumps(xc, indent=1), "```", ""]
    lines += [f"### Band {band_name()}, ALL_INDIC", "",
              "| lens | role | top-10 bf16 | top-10 bf16 pess. | top-10 fp32 | fp32 − bf16 (pts) [95% CI] | P bf16 | P fp32 | fp32 − bf16 [95% CI] | rel. | ranks changed | top-10 flips | bf16 ties | at top-10 boundary |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for lens in lenses:
        for role in REPORT_ROLES + ("chinese", "source"):
            c = _cell(summary, "ALL_INDIC", lens, band_name(), role)
            if c:
                lines.append(f"| {lens} | {role} | {c['top10_bf16']:.3f} | {c['top10_bf16_pess']:.3f} | {c['top10_fp32']:.3f} | "
                             f"{c['d_top10_pts']:+.2f} {c['d_top10_ci']} | {c['prob_bf16']:.4f} | {c['prob_fp32']:.4f} | "
                             f"{c['d_prob']:+.5f} {c['d_prob_ci']} | {c['rel_d_prob_pct']:+.2f}% | {c['rank_changed']:.1%} | "
                             f"{c['top10_flips']:.2%} | {c['ties']:.1%} | {c['boundary_ties']:.2%} |")
    for lens in lenses:
        lines += ["", f"### Per layer, ALL_INDIC, {lens}: top-10 bf16 → fp32 (P bf16 → fp32)", "",
                  "| layer | " + " | ".join(REPORT_ROLES) + " |", "|---" * (len(REPORT_ROLES) + 1) + "|"]
        layer_names = sorted({s["layers"] for s in summary if s["lens"] == lens and "-" not in s["layers"]}, key=lambda x: int(x[1:]))
        for name in layer_names:
            cells = []
            for role in REPORT_ROLES:
                c = _cell(summary, "ALL_INDIC", lens, name, role)
                cells.append("" if not c else f"{c['top10_bf16']:.3f} → {c['top10_fp32']:.3f} ({c['prob_bf16']:.4f} → {c['prob_fp32']:.4f})")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for lens in lenses:
        lines += ["", f"### Per pair, band {band_name()}, {lens}: top-10 fp32 − bf16 in points (P relative change)", "",
                  "| pair | n | " + " | ".join(REPORT_ROLES) + " |", "|---" * (len(REPORT_ROLES) + 2) + "|"]
        for pair in pairs:
            cells, n = [], ""
            for role in REPORT_ROLES:
                c = _cell(summary, pair, lens, band_name(), role)
                if c:
                    n = c["n_items"]
                rel = "" if not c or math.isnan(c["rel_d_prob_pct"]) else f" ({c['rel_d_prob_pct']:+.1f}%)"
                cells.append("" if not c else f"{c['d_top10_pts']:+.1f}{rel}")
            lines.append(f"| {pair} | {n} | " + " | ".join(cells) + " |")
    lines += ["", "### Logit magnitudes (per lens and layer, medians over items)", "",
              "| lens | layer | top logit bf16 | top logit fp32 | bf16 spacing there | median max |Δlogit| | max |Δlogit| | top-1 agree | top-10 overlap | 10th logit tied (bf16) | gap 10th-11th fp32 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in verdict["logits"]:
        lines.append(f"| {r['lens']} | {r['layer']} | {r['median_top_logit_bf16']} | {r['median_top_logit_fp32']} | "
                     f"{r['bf16_spacing_at_median_top']} | {r['median_max_abs_dlogit']} | {r['max_max_abs_dlogit']} | "
                     f"{r['top1_agree']:.3f} | {r['mean_top10_overlap']} | {r['tie_at_logit10_bf16']:.3f} | {r['median_gap10_11_fp32']} |")
    lines += ["", f"### Headline numbers (docs/04 §4.1): {verdict['moves']} of {verdict['n_checked']} move "
              f"(> {TOP10_PTS:g} pts top-10 or > {REL_P:.0%} relative P)", "",
              "| number | layer | published | bf16 (a) | bf16 pess. | fp32 (b) | move |", "|---|---|---|---|---|---|---|"]
    for c in verdict["checks"]:
        move = f"{c['move_pts']:+.2f} pts" if "move_pts" in c else f"{100 * c['rel_move']:+.2f}%"
        pub = f"{c['published']:.4f}" if "published" in c else ""
        pess = f"{c['bf16_pess']:.4f}" if "bf16_pess" in c else ""
        lines.append(f"| {c['number']} | {c['layer']} | {pub} | {c['bf16']:.4f} | {pess} | {c['fp32']:.4f} | "
                     f"{move}{' **MOVES**' if c['moves'] else ''} |")
    return "\n".join(lines) + "\n"


def report(run_dir: str, published_dir: str | None, md_path: str | None) -> int:
    run_path = Path(run_dir)
    rows = read_csv(run_path / "rows.csv")
    logit_rows = read_csv(run_path / "logits.csv")
    items = read_csv(run_path / "items.csv")
    pairs = list(dict.fromkeys(r["pair"] for r in items))
    summary, verdict = analyze(rows, logit_rows, pairs)
    xc = None
    if published_dir:
        pub = Path(published_dir)
        xc = cross_check(rows, read_csv(pub / "rows.csv"), read_csv(pub / "items.csv"), items)
        published = {(s["pair"], s["lens"], int(s["layer"]), s["role"]): s for s in read_csv(pub / "summary.csv")}
        verdict = {**headline(summary, pairs, published), "logits": verdict["logits"]}
    write_csv(run_path / "quant_summary.csv", summary)
    (run_path / "verdict.json").write_text(json.dumps({**verdict, "cross_check": xc}, indent=1, ensure_ascii=False),
                                           encoding="utf-8")
    text = markdown(summary, verdict, xc, pairs)
    if md_path:
        Path(md_path).write_text(text, encoding="utf-8")
    print(text)
    fp32_filter = [i for i in items if i.get("top_agree") == "False"]
    print(f"items where the fp32 head's top token differs from bf16's: {len(fp32_filter)}; of them the fp32 filter "
          f"would flip: {sum((i['fp32_top_in_target'] == 'True') != (i['kept'] == 'True') for i in fp32_filter)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="(a) vs pivot_experiment.main, then (a) vs (b), on a tiny model")
    c.add_argument("model_dir")
    c.add_argument("--lens", default=None)
    c.add_argument("--pairs", default="ta-te,hi-ta,en-ta")
    c.add_argument("--out", required=True)
    a = sub.add_parser("analyze", help="tables, verdict and the cross-check from a run's rows.csv")
    a.add_argument("run_dir")
    a.add_argument("--published", default=str(HERE.parent / "results/final/pivot_all"))
    a.add_argument("--md", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.cmd == "check":
        return check(args.model_dir, args.lens, args.out, args.pairs)
    return report(args.run_dir, args.published or None, args.md)


if __name__ == "__main__":
    raise SystemExit(main())
