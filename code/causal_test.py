"""Causal test: does the model USE the English representation the lenses show mid-network?

For every Indic->Indic item the model translates correctly, remove from the residual stream
(final position, after each block in a layer window) the subspace spanned by the lens
steering directions of the concept's English tokens:

    v_t(l) = J_l^T (g * W_U[t])     t in E(concept): tokens beginning the English word (>= 3 chars)

and measure the model's probability of the correct target-word start. The conditions are:
  base            no intervention
  english_mid     English subspace removed at the layers where English is readable (13-16 on Sarvam)
  control_mid     same-size subspace for a DIFFERENT concept's English tokens, same layers
  english_early   English subspace removed at early layers (5-8), where no lens reads any language

If English is causally used, english_mid should hurt clearly more than control_mid.

``run_swap`` (J++'s clamp-swap) and ``run_patch`` (activation patching from a concept-B prompt,
whole state vs the English / Hindi / target / random subspace) ask whether those directions
also carry the concept: can editing them change which word the model outputs?
"""

from __future__ import annotations

import csv
import json
import zlib
from pathlib import Path

import numpy as np
import torch as t

import pivot_experiment as pe

DEFAULT_PAIRS = "ta-te,te-kn,kn-ml,ml-ta,bn-gu,gu-mr,mr-bn,bn-ta,ta-bn,te-gu"


def final_norm_gain(model) -> t.Tensor:
    weight = model._final_norm.weight.float().to(model._lm_head.weight.device)
    config = model._hf_model.config
    text = getattr(config, "text_config", None) or config
    kinds = f"{getattr(config, 'model_type', '')}/{getattr(text, 'model_type', '')}".lower()
    return 1.0 + weight if ("gemma" in kinds or "qwen3_5" in kinds) else weight


def readout_directions(lens, layer: int, token_ids: list[int], W: t.Tensor) -> t.Tensor:
    """``k x d``: row t is the lens's steering direction ``J_l^T w_t`` for ``w_t = g * W_U[t]``.
    A lens without Jacobians (the logit lens) transports by the identity, so its directions
    are the rows ``w_t`` themselves."""
    jacobians = getattr(lens, "jacobians_L_dict_FN", None)
    if not jacobians:
        return W[token_ids]
    return W[token_ids] @ jacobians[layer].float().to(W.device)


def subspace(lens, layer: int, token_ids: list[int], W: t.Tensor) -> t.Tensor:
    """Orthonormal basis (d x k) of the steering directions J_l^T w_t for the given tokens."""
    directions = readout_directions(lens, layer, token_ids, W)  # k x d
    Q, _ = t.linalg.qr(directions.T)       # d x k
    return Q


class Ablation:
    """Forward hooks that project a per-layer subspace out of the final position's residual.
    ``alpha`` scales the removed component: 1 projects it out, 2 reflects it, 3 pushes it to -2x
    (negative steering: Bhatia, Blank & Nanda 2026 found steering past zero can move behaviour
    that a plain projection leaves alone)."""

    def __init__(self, model, bases: dict[int, t.Tensor], alpha: float = 1.0) -> None:
        self.handles = []
        for layer, Q in bases.items():
            self.handles.append(model.layers[layer].register_forward_hook(self._hook(Q, alpha)))

    @staticmethod
    def _hook(Q: t.Tensor, alpha: float = 1.0):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            last = hidden[:, -1, :].float()
            Qd = Q.to(last.device)
            last = last - alpha * (last @ Qd) @ Qd.T
            hidden = hidden.clone()
            hidden[:, -1, :] = last.to(hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


class RowAblation(Ablation):
    """:class:`Ablation` per batch row, so ONE forward over copies of a prompt runs several
    conditions: ``plans[row]`` is ``(bases, alpha)``, or None to leave that row untouched. Each
    row gets exactly :class:`Ablation`'s arithmetic (the same ``1 x d`` shapes)."""

    def __init__(self, model, plans: list[tuple[dict[int, t.Tensor], float] | None]) -> None:
        edits: dict[int, list] = {}
        for row, plan in enumerate(plans):
            if plan:
                bases, alpha = plan
                for layer, Q in bases.items():
                    edits.setdefault(layer, []).append((row, Q, alpha))
        self.handles = [model.layers[layer].register_forward_hook(self._row_hook(e)) for layer, e in edits.items()]

    @staticmethod
    def _row_hook(edits: list):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden.clone()
            for row, Q, alpha in edits:
                last = hidden[row: row + 1, -1, :].float()
                Qd = Q.to(last.device)
                hidden[row: row + 1, -1, :] = (last - alpha * (last @ Qd) @ Qd.T).to(hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return hook


def run(model, lens, *, concepts_path: str, pairs: str, mid_layers: list[int], early_layers: list[int],
        out_dir: str, vocab_size: int, shots: int = 5, seed: int = 0, require_correct: bool = True,
        latent_lang: str = "en", alphas: tuple[float, ...] = (1.0,), batch_conditions: bool = False) -> dict:
    """``latent_lang``: whose word directions to remove ("en" English, "hi" Hindi). ``alphas``
    beyond 1 add negative-steering conditions ``english_mid@a`` / ``control_mid@a`` (see
    :class:`Ablation`); the default runs exactly the original three conditions.
    ``batch_conditions`` runs an item's base and every condition as rows of ONE forward
    (:class:`RowAblation`; the 105B's forward is launch-bound, so 8 rows cost about one)."""
    tokenizer, hf = model.tokenizer, model._hf_model
    index = pe.build_vocab_index(tokenizer, vocab_size)
    concepts = pe.load_concepts(Path(concepts_path))
    ordered = sorted(concepts)
    W = (model._lm_head.weight.float() * final_norm_gain(model)).detach()
    rows: list[dict] = []
    for pair in pairs.split(","):
        src, tgt = pair.split("-")
        for concept in ordered:
            words = concepts[concept]
            control = ordered[(ordered.index(concept) + len(ordered) // 2) % len(ordered)]
            target_ids = pe.prefix_token_ids(index, words[tgt], tokenizer)
            if latent_lang in (src, tgt):
                continue
            english = sorted(pe.prefix_token_ids(index, words[latent_lang], tokenizer, min_chars=3))
            english_ctrl = sorted(pe.prefix_token_ids(index, concepts[control][latent_lang], tokenizer, min_chars=3)
                                  - set(english))
            k = min(len(english), len(english_ctrl))
            if k == 0:
                continue
            english, english_ctrl = english[:k], english_ctrl[:k]
            ids = model.encode(pe.build_prompt(concepts, concept, src, tgt, shots=shots, seed=seed))
            target_tensor = t.tensor(sorted(target_ids))

            def measure() -> tuple[float, bool]:
                with t.no_grad():
                    logits = hf(input_ids=ids, use_cache=False).logits[0, -1].float()
                probs = t.softmax(logits, -1)
                # Index where the logits are: under device_map="auto", accelerate returns them on the
                # input's GPU, not the LM head's.
                return float(probs[target_tensor.to(probs.device)].sum()), int(logits.argmax()) in target_ids

            def measure_rows(row_plans: list) -> list[tuple[float, bool]]:
                """:func:`measure` under each plan, as the rows of one forward (:class:`RowAblation`)."""
                with t.no_grad(), RowAblation(model, row_plans):
                    logits = hf(input_ids=ids.repeat(len(row_plans), 1), use_cache=False).logits[:, -1].float()
                probs = t.softmax(logits, -1)
                where = target_tensor.to(probs.device)
                return [(float(probs[r, where].sum()), int(logits[r].argmax()) in target_ids) for r in range(len(row_plans))]

            conditions = {
                "english_mid": {l: subspace(lens, l, english, W) for l in mid_layers},
                "control_mid": {l: subspace(lens, l, english_ctrl, W) for l in mid_layers},
                "english_early": {l: subspace(lens, l, english, W) for l in early_layers},
            }
            plans = {**{name: (bases, 1.0) for name, bases in conditions.items()},
                     **{f"{name}@{a:g}": (conditions[name], a) for a in alphas if a != 1.0 for name in ("english_mid", "control_mid")}}
            if batch_conditions:
                measured = dict(zip(["base", *plans], measure_rows([None, *plans.values()])))
            else:
                measured = {"base": measure()}
            p_base, ok = measured["base"]
            if require_correct and not ok:
                continue  # only items the model gets right
            row = {"pair": pair, "concept": concept, "control_concept": control, "k": k,
                   "base_p": p_base, "base_correct": ok}
            for name, (bases, alpha) in plans.items():
                if name not in measured:
                    with Ablation(model, bases, alpha=alpha):
                        measured[name] = measure()
                row[f"{name}_p"], row[f"{name}_correct"] = measured[name]
            rows.append(row)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out / "causal_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    rng = np.random.default_rng(0)

    def paired(cond: str) -> dict:
        drop = np.array([r["base_p"] - r[f"{cond}_p"] for r in rows])
        boots = rng.choice(drop, size=(2000, len(drop)), replace=True).mean(1) if len(drop) else np.array([np.nan])
        return {"mean_p": float(np.mean([r[f"{cond}_p"] for r in rows])) if rows else float("nan"),
                "mean_drop": float(drop.mean()) if len(drop) else float("nan"),
                "drop_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "still_correct": float(np.mean([r[f"{cond}_correct"] for r in rows])) if rows else float("nan")}

    steer_conds = [f"{name}@{a:g}" for a in alphas if a != 1.0 for name in ("english_mid", "control_mid")]
    summary = {"n_items": len(rows), "mid_layers": mid_layers, "early_layers": early_layers,
               "base_mean_p": float(np.mean([r["base_p"] for r in rows])) if rows else float("nan"),
               **{cond: paired(cond) for cond in ("english_mid", "control_mid", "english_early", *steer_conds)}}
    if rows:
        extra = np.array([(r["base_p"] - r["english_mid_p"]) - (r["base_p"] - r["control_mid_p"]) for r in rows])
        boots = rng.choice(extra, size=(2000, len(extra)), replace=True).mean(1)
        summary["english_minus_control_drop"] = {"mean": float(extra.mean()),
                                                 "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    (out / "causal_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


### CONCEPT SWAP (J++'s probe-swap convention)


def unit_readout_vector(lens, layer: int, token_id: int, W: t.Tensor) -> t.Tensor:
    """``J_l^T (g * W_U[token])``, unit-normalised (equal-magnitude edits across tokens)."""
    v = readout_directions(lens, layer, [token_id], W)[0]
    return v / v.norm()


class ClampSwap:
    """At each band layer and every position but 0 (the attention sink), exchange the
    residual's least-squares coordinates on the two-atom basis [u_src, u_dst], times ``scale``.
    The coordinates come from the incoming residual. (J++ uses a cached clean pass; across
    a 4-layer band the difference is small.)"""

    def __init__(self, model, atoms: dict[int, t.Tensor], scale: float) -> None:
        self.handles = []
        for layer, A in atoms.items():
            self.handles.append(model.layers[layer].register_forward_hook(self._hook(A, scale)))

    @staticmethod
    def _hook(A: t.Tensor, scale: float):
        pinv = t.linalg.pinv(A)  # 2 x d

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            h = hidden[:, 1:, :].float()
            Ad, Pd = A.to(h.device), pinv.to(h.device)
            coords = h @ Pd.T                              # B x P x 2
            swapped = scale * coords.flip(-1)
            h = h + (swapped - coords) @ Ad.T
            hidden = hidden.clone()
            hidden[:, 1:, :] = h.to(hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def best_token(index, tokenizer, spelling: str) -> int:
    """The most specific single token beginning ``spelling``: the longest vocab entry that is a
    >=2-char prefix of it, else the tokenizer's own first token of " spelling"."""
    key = pe.normalize_token_text(spelling)
    for end in range(len(key), 1, -1):
        ids = index.get(key[:end])
        if ids:
            return int(ids[0])
    return int(tokenizer(f" {spelling}", add_special_tokens=False).input_ids[0])


def run_swap(model, lens, *, concepts_path: str, pairs: str, mid_layers: list[int], early_layers: list[int],
             out_dir: str, vocab_size: int, scales=(2.0, 4.0), shots: int = 5, seed: int = 0,
             require_correct: bool = True) -> dict:
    """Swap concept A's latent token for concept B's and score whether the TARGET-language answer
    becomes B's word. Conditions (each at every scale unless noted):
      english_mid    English A <-> English B at the mid band
      hindi_mid      Hindi A <-> Hindi B at the mid band
      answer_mid     target-language A <-> B tokens directly (positive control / ceiling)
      english_early  English A <-> B at early layers (should fail)
      null_mid       English A <-> English C, scored against B (should stay ~0)
    """
    tokenizer, hf = model.tokenizer, model._hf_model
    index = pe.build_vocab_index(tokenizer, vocab_size)
    concepts = pe.load_concepts(Path(concepts_path))
    ordered = sorted(concepts)
    W = (model._lm_head.weight.float() * final_norm_gain(model)).detach()
    rows: list[dict] = []
    for pair in pairs.split(","):
        src, tgt = pair.split("-")
        for concept in ordered:
            i = ordered.index(concept)
            swap = ordered[(i + len(ordered) // 2) % len(ordered)]
            third = ordered[(i + len(ordered) // 4) % len(ordered)]
            words, swap_words = concepts[concept], concepts[swap]
            target_A = pe.prefix_token_ids(index, words[tgt], tokenizer)
            target_B = pe.prefix_token_ids(index, swap_words[tgt], tokenizer) - target_A
            if not target_B:
                continue
            ids = model.encode(pe.build_prompt(concepts, concept, src, tgt, shots=shots, seed=seed))
            tA, tB = t.tensor(sorted(target_A)), t.tensor(sorted(target_B))

            def measure() -> dict:
                with t.no_grad():
                    logits = hf(input_ids=ids, use_cache=False).logits[0, -1].float()
                probs, top = t.softmax(logits, -1), int(logits.argmax())
                return {"pA": float(probs[tA.to(probs.device)].sum()), "pB": float(probs[tB.to(probs.device)].sum()),
                        "top_is_B": top in target_B, "top_is_A": top in target_A}

            base = measure()
            if require_correct and not base["top_is_A"]:
                continue
            tok = lambda lang, c: best_token(index, tokenizer, concepts[c][lang][0])
            conditions = {
                "english_mid": (mid_layers, tok("en", concept), tok("en", swap)),
                "hindi_mid": (mid_layers, tok("hi", concept), tok("hi", swap)),
                "answer_mid": (mid_layers, best_token(index, tokenizer, words[tgt][0]),
                               best_token(index, tokenizer, swap_words[tgt][0])),
                "english_early": (early_layers, tok("en", concept), tok("en", swap)),
                "null_mid": (mid_layers, tok("en", concept), tok("en", third)),
            }
            row = {"pair": pair, "concept": concept, "swap_to": swap, **{f"base_{k}": v for k, v in base.items()}}
            for name, (layers, token_a, token_b) in conditions.items():
                if src == "hi" and name == "hindi_mid":
                    continue
                atoms = {l: t.stack([unit_readout_vector(lens, l, token_a, W),
                                     unit_readout_vector(lens, l, token_b, W)], dim=1) for l in layers}
                for scale in scales:
                    with ClampSwap(model, atoms, scale):
                        result = measure()
                    for k, v in result.items():
                        row[f"{name}@{scale}_{k}"] = v
            rows.append(row)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (out / "swap_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary: dict = {"n_items": len(rows), "mid_layers": mid_layers, "early_layers": early_layers}
    import re

    for key in sorted({m.group(1) for r in rows for k in r if (m := re.match(r"^(.+@[0-9.]+)_", k))}):
        flips = [r[f"{key}_top_is_B"] for r in rows if f"{key}_top_is_B" in r]
        kept = [r[f"{key}_top_is_A"] for r in rows if f"{key}_top_is_A" in r]
        pB = [r[f"{key}_pB"] - r["base_pB"] for r in rows if f"{key}_pB" in r]
        summary[key] = {"n": len(flips), "flip_to_B_rate": float(np.mean(flips)) if flips else float("nan"),
                        "still_A_rate": float(np.mean(kept)) if kept else float("nan"),
                        "mean_pB_gain": float(np.mean(pB)) if pB else float("nan")}
    (out / "swap_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


### ACTIVATION PATCHING (replaces the clamp-swap)

PATCH_CONDITIONS = ("whole", "english", "english_ab", "english_other", "hindi", "target", "random")
# The follow-up that splits english_ab's effect into "remove A" and "insert B", and asks whether
# English is special: english_ab again (the in-run reference), then its Hindi / target / random analogues.
PATCH_SPLIT_CONDITIONS = ("english_ab", "english_a_mean", "hindi_ab", "target_ab", "random_ab")
# Is English special per dimension? The A + B swap with every language's span cut to the same rank
# per item (r = the smallest of the three A + B token counts), plus a random subspace of rank r.
PATCH_RANK_CONDITIONS = ("english_eq", "hindi_eq", "target_eq", "random_eq")
PATCH_CONTRASTS = (
    ("english", "english_other"), ("english", "random"), ("english_ab", "english_other"), ("english_ab", "random"),
    ("hindi", "english_other"), ("hindi", "random"), ("target", "english_other"), ("target", "random"),
    ("english_ab", "random_ab"), ("english_ab", "english_a_mean"), ("english_ab", "hindi_ab"),
    ("english_ab", "target_ab"), ("hindi_ab", "random_ab"), ("target_ab", "random_ab"),
    ("english_eq", "hindi_eq"), ("english_eq", "target_eq"), ("hindi_eq", "target_eq"),
    ("english_eq", "random_eq"), ("hindi_eq", "random_eq"), ("target_eq", "random_eq"),
)


class Capture:
    """Forward hooks that record the final position's residual after each given block."""

    def __init__(self, model, layers: list[int]) -> None:
        self.residuals: dict[int, t.Tensor] = {}
        self.handles = [model.layers[layer].register_forward_hook(self._hook(layer)) for layer in layers]

    def _hook(self, layer: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.residuals[layer] = hidden[0, -1, :].detach().float().clone()
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


class Patch:
    """Forward hooks that write a source run's final-position residual into this run after each
    block in ``sources``: the whole state where the basis is None, else only its component in
    span(Q), replacing this run's component there (``h + Q Q^T (h_src - h)``)."""

    def __init__(self, model, sources: dict[int, t.Tensor], bases: dict[int, t.Tensor | None]) -> None:
        self.handles = [model.layers[layer].register_forward_hook(self._hook(sources[layer], bases[layer]))
                        for layer in bases]

    @staticmethod
    def _hook(source: t.Tensor, Q: t.Tensor | None):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            last = hidden[:, -1, :].float()
            src = source.to(last.device)
            if Q is None:
                new = src.expand_as(last)
            else:
                Qd = Q.to(last.device)
                new = last + ((src - last) @ Qd) @ Qd.T
            hidden = hidden.clone()
            hidden[:, -1, :] = new.to(hidden.dtype)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()


def random_basis(d: int, k: int, seed: int) -> t.Tensor:
    """Orthonormal basis (d x k) of a uniformly random k-dimensional subspace."""
    Q, _ = t.linalg.qr(t.randn(d, k, generator=t.Generator().manual_seed(seed)))
    return Q


def most_specific_first(token_ids, text_of: dict[int, str]) -> list[int]:
    """A word's tokens, longest (most word-specific) normalized prefix first, ties by id."""
    return sorted(token_ids, key=lambda i: (-len(text_of.get(i, "")), i))


def balanced_pick(per_word: list[list[int]], r: int) -> list[int]:
    """Up to ``r`` distinct tokens, taken from the words' lists in turn (A, B, A, B, ...), each list
    in its own order; a word whose list runs out leaves the rest to the other."""
    picked: list[int] = []
    queues = [list(tokens) for tokens in per_word]
    while len(picked) < r and any(queues):
        for queue in queues:
            while queue and queue[0] in picked:
                queue.pop(0)
            if queue and len(picked) < r:
                picked.append(queue.pop(0))
    return picked


def run_patch(model, lens, *, concepts_path: str, pairs: str, layers: list[int], out_dir: str, vocab_size: int,
              shots: int = 5, seed: int = 0, require_correct: bool = True,
              conditions: tuple[str, ...] = PATCH_CONDITIONS) -> dict:
    """Activation patching at the final position. Run concept A's prompt, but after each block in
    ``layers`` write in (part of) the residual from concept B's prompt: the same pair, so it asks
    for B's word in the same target language (B = the half-way derangement, as in :func:`run_swap`).
    Success = the answer becomes B's target word. ``conditions``, one forward each:
      whole          the entire residual: the ceiling. Only the final position reads its own past,
                     so this equals patching the band's last layer alone.
      english        only the component in span{J_l^T w_t : t in English(B)}, replacing A's there
      english_ab     the same in span(English(A) + English(B)), so A's English is reset to B's level too
      english_other  span of a third concept C's English tokens, rank-matched to ``english`` (control)
      hindi          span(Hindi(B))
      target         span(B's target-language tokens): the answer's own readout subspace
      random         a random subspace of ``english``'s rank (control)
    and the follow-up (:data:`PATCH_SPLIT_CONDITIONS`):
      english_a_mean span(English(A)) only, set to the mean of the pair's other concepts' runs
                     (not A, not B): A's English reset to a typical not-A level, nothing of B inserted.
                     (Zeroing it instead is ``run``'s english_mid ablation.)
      hindi_ab       span(Hindi(A) + Hindi(B)), from B's run
      target_ab      span(target(A) + target(B)), from B's run
      random_ab      a random subspace of ``english_ab``'s rank (its rank-matched control)
    and the rank-matched test (:data:`PATCH_RANK_CONDITIONS`), with r = min over English, Hindi and
    target of the A + B token count, so every language gets the same number of directions:
      english_eq / hindi_eq / target_eq   the A + B swap on r of that language's tokens, picked by
                     :func:`balanced_pick` from each word's :func:`most_specific_first` list
      random_eq      a random subspace of rank r
    ``*_share`` is the fraction of the clean (B - A) residual difference, averaged over the band,
    that lies in the condition's span: how much of the ceiling's edit the condition moves."""
    tokenizer, hf = model.tokenizer, model._hf_model
    index = pe.build_vocab_index(tokenizer, vocab_size)
    text_of = {i: key for key, ids in index.items() for i in ids}
    concepts = pe.load_concepts(Path(concepts_path))
    ordered = sorted(concepts)
    n = len(ordered)
    W = (model._lm_head.weight.float() * final_norm_gain(model)).detach()
    latent = {lang: {c: pe.prefix_token_ids(index, concepts[c][lang], tokenizer, min_chars=3) for c in ordered}
              for lang in ("en", "hi")}
    rows: list[dict] = []
    skipped = {"A_wrong": 0, "B_wrong": 0, "no_B_target": 0, "no_B_english": 0}

    def score(probs: t.Tensor, top: int, target_A: set[int], target_B: set[int]) -> dict:
        tA, tB = (t.tensor(sorted(s), dtype=t.long, device=probs.device) for s in (target_A, target_B))
        return {"pA": float(probs[tA].sum()), "pB": float(probs[tB].sum()),
                "top_is_B": top in target_B, "top_is_A": top in target_A}

    for pair in pairs.split(","):
        src, tgt = pair.split("-")
        clean = {}  # concept -> (input ids, final-position residual by layer, probabilities, top-1)
        for concept in ordered:
            ids = model.encode(pe.build_prompt(concepts, concept, src, tgt, shots=shots, seed=seed))
            with Capture(model, layers) as capture, t.no_grad():
                logits = hf(input_ids=ids, use_cache=False).logits[0, -1].float()
            clean[concept] = (ids, capture.residuals, t.softmax(logits, -1), int(logits.argmax()))
        totals = {l: sum(clean[c][1][l] for c in ordered) for l in layers}
        for i, concept in enumerate(ordered):
            swap = ordered[(i + n // 2) % n]
            target_A = pe.prefix_token_ids(index, concepts[concept][tgt], tokenizer)
            target_B_full = pe.prefix_token_ids(index, concepts[swap][tgt], tokenizer)
            target_B = target_B_full - target_A
            if not target_B:
                skipped["no_B_target"] += 1
                continue
            ids, res_A, probs_A, top_A = clean[concept]
            _, res_B, probs_B, top_B = clean[swap]
            if require_correct and top_A not in target_A:
                skipped["A_wrong"] += 1
                continue
            if require_correct and top_B not in target_B_full:  # B's own run must carry B's answer
                skipped["B_wrong"] += 1
                continue
            en_A, en_B = latent["en"][concept], latent["en"][swap] - latent["en"][concept]
            if not en_B:
                skipped["no_B_english"] += 1
                continue
            k = len(en_B)
            # C: the first concept from the quarter-way offset with >= k English tokens of its own.
            others = [ordered[(i + n // 4 + j) % n] for j in range(n)]
            pool = [(sorted(latent["en"][c] - en_A - en_B), c) for c in others if c not in (concept, swap)]
            ctrl_tokens, ctrl = next((p for p in pool if len(p[0]) >= k), max(pool, key=lambda p: len(p[0])))
            hindi_ok = "hi" not in (src, tgt)
            token_sets = {
                "english": sorted(en_B),
                "english_ab": sorted(en_A | en_B),
                "english_other": ctrl_tokens[:k],
                "hindi": sorted(latent["hi"][swap] - latent["hi"][concept]) if hindi_ok else [],
                "target": sorted(target_B),
                "english_a_mean": sorted(en_A),
                "hindi_ab": sorted(latent["hi"][swap] | latent["hi"][concept]) if hindi_ok else [],
                "target_ab": sorted(target_A | target_B),
            }
            # Equal rank across languages: r of each language's A + B tokens, most specific first.
            words = {"english": (en_A, latent["en"][swap]), "target": (target_A, target_B),
                     "hindi": (latent["hi"][concept], latent["hi"][swap]) if hindi_ok else (set(), set())}
            r = min(len(a | b) for a, b in words.values())
            for lang, (a, b) in words.items():
                token_sets[f"{lang}_eq"] = balanced_pick([most_specific_first(a, text_of),
                                                          most_specific_first(b - a, text_of)], r)
            # Mean over the pair's other concepts: a typical "not A" state that carries nothing of B.
            not_ab = {l: (totals[l] - res_A[l] - res_B[l]) / (n - 2) for l in layers}
            row = {"pair": pair, "concept": concept, "swap_to": swap, "english_other_concept": ctrl,
                   **{f"base_{key}": v for key, v in score(probs_A, top_A, target_A, target_B).items()},
                   **{f"B_run_{key}": v for key, v in score(probs_B, top_B, target_A, target_B).items()}}
            for name in conditions:
                if name == "whole":
                    bases = {l: None for l in layers}
                elif name in ("random", "random_ab", "random_eq"):
                    rank, tag = {"random": (k, ""), "random_ab": (len(token_sets["english_ab"]), "|ab"),
                                 "random_eq": (r, "|eq")}[name]
                    if not rank:
                        continue
                    bases = {l: random_basis(W.shape[1], rank, zlib.crc32(f"{pair}|{concept}|{l}{tag}".encode()))
                             for l in layers}
                elif token_sets[name]:
                    bases = {l: subspace(lens, l, token_sets[name], W) for l in layers}
                else:
                    continue  # e.g. no Hindi token of >= 3 characters
                sources = not_ab if name == "english_a_mean" else res_B
                with Patch(model, sources, bases), t.no_grad():
                    logits = hf(input_ids=ids, use_cache=False).logits[0, -1].float()
                result = score(t.softmax(logits, -1), int(logits.argmax()), target_A, target_B)
                shares = []
                for l, Q in bases.items():
                    diff = res_B[l] - res_A[l]
                    shares.append(1.0 if Q is None else float(((diff @ Q.to(diff.device)) ** 2).sum() / (diff ** 2).sum()))
                row[f"{name}_rank"] = W.shape[1] if name == "whole" else next(iter(bases.values())).shape[1]
                row[f"{name}_share"] = float(np.mean(shares))
                for key, v in result.items():
                    row[f"{name}_{key}"] = v
            rows.append(row)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = list(dict.fromkeys(key for r in rows for key in r))
        with (out / "patch_rows.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    rng = np.random.default_rng(0)

    def mean_ci(values) -> tuple[float, list[float]]:
        values = np.asarray(values, dtype=float)
        if not len(values):
            return float("nan"), [float("nan"), float("nan")]
        boots = rng.choice(values, size=(2000, len(values)), replace=True).mean(1)
        return float(values.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    summary: dict = {"n_items": len(rows), "layers": layers, "pairs": pairs, "skipped": skipped, "conditions": list(conditions),
                     "base_pB": mean_ci([r["base_pB"] for r in rows])[0],
                     "B_run_pB": mean_ci([r["B_run_pB"] for r in rows])[0]}
    for name in conditions:
        sel = [r for r in rows if f"{name}_pB" in r]
        flip, flip_ci = mean_ci([r[f"{name}_top_is_B"] for r in sel])
        gain, gain_ci = mean_ci([r[f"{name}_pB"] - r["base_pB"] for r in sel])
        summary[name] = {"n": len(sel), "rank_mean": mean_ci([r[f"{name}_rank"] for r in sel])[0],
                         "share_of_diff": mean_ci([r[f"{name}_share"] for r in sel])[0],
                         "flip_to_B_rate": flip, "flip_ci95": flip_ci,
                         "still_A_rate": mean_ci([r[f"{name}_top_is_A"] for r in sel])[0],
                         "mean_pB_gain": gain, "pB_gain_ci95": gain_ci,
                         "mean_pA_drop": mean_ci([r["base_pA"] - r[f"{name}_pA"] for r in sel])[0]}
    ceiling = summary.get("whole", {}).get("mean_pB_gain")  # absent in the follow-up run
    for name in conditions:
        summary[name]["frac_of_ceiling_pB_gain"] = summary[name]["mean_pB_gain"] / ceiling if ceiling else float("nan")
    contrasts = {}
    for name, ref in PATCH_CONTRASTS:
        if name not in conditions or ref not in conditions:
            continue
        sel = [r for r in rows if f"{name}_pB" in r and f"{ref}_pB" in r]
        flip, flip_ci = mean_ci([float(r[f"{name}_top_is_B"]) - float(r[f"{ref}_top_is_B"]) for r in sel])
        gain, gain_ci = mean_ci([r[f"{name}_pB"] - r[f"{ref}_pB"] for r in sel])
        contrasts[f"{name}_minus_{ref}"] = {"n": len(sel), "flip": flip, "flip_ci95": flip_ci,
                                            "pB_gain": gain, "pB_gain_ci95": gain_ci}
    summary["contrasts"] = contrasts
    (out / "patch_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
