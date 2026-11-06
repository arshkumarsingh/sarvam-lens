"""An expert- and neuron-level view of the English pivot (the ``routing`` and ``neurons`` pilot stages).

``routing``: forward hooks on every MoE gate record the top-k expert ids per token and layer
(30B: ``SarvamMoEGate`` returns ``(topk_idx, topk_weight, logits)``; 105B: ``MoEGate`` returns
``(topk_idx, topk_weight)``; both flatten batch x sequence). Two input sets:
  (a) the 32 concepts x 10 languages of ``data/concepts.tsv`` as bare words (BOS + word), read at
      the word's last token;
  (b) the pivot experiment's 5-shot prompts (``pivot_experiment.build_prompt`` over
      ``DEFAULT_PAIRS``), read at the final position (the target word's opening quote).
Per MoE layer:
  - language and concept decodability: nearest-centroid balanced accuracy, cross-validated so the
    test word's concept (for language) or script (for concept) never appears in training;
  - concept sharing: mean Jaccard of the expert sets, same concept vs different concept, counting
    only pairs of words in different scripts (Hindi and Marathi share Devanagari, often tokens);
  - English alignment: for each Indic word, the Jaccard with its English translation's experts
    minus that with the other English words ("excess"), against the same excess for its
    translations in other-script Indic languages, and for its Chinese translation (a second
    non-Indic language, the control for "English is special");
  - for (b): the share of routing variance explained by the language pair vs the concept (eta^2,
    beside its chance level (g-1)/(N-1)), and decoding of the concept across held-out pairs and
    of the source / target language across held-out pairs.
``neurons``: layer 0 is a dense MLP in both models. Its neurons are ``act_fn(gate_proj(x)) *
up_proj(x)`` (the input of ``down_proj``, caught by a pre-hook that then stops the forward), read
at each word's last token. A neuron is concept-selective when the same concept has the highest
within-language z-score in >= 5 different scripts (a permutation null shuffles concepts within
each language); script-selective when one script's words separate from the rest at AUROC >= 0.95.

Cognate control (both analyses). Over a third of the cross-script Indic translation pairs are one word
in two scripts (river: nadī in 7 of 8). :func:`cognate_matrix` finds them from script-neutral spellings
(:data:`INDIC_BLOCKS`). Routing then reports the Indic-partner excess split into cognate and
non-cognate partners; a neuron's cross-script agreement is also counted in independent word forms
(cognate clusters), against the same permutation null.

    python neuron_experiment.py --reanalyze results/neuron30   # recompute every summary on CPU

Batching (the 105B): prompts are right-padded and read at each row's last real token. Under the
causal mask no real token sees the padding after it and the router scores each token on its own,
so a row matches a lone forward up to kernel numerics (``batch_consistency`` measures it).
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch as t

import pivot_experiment as pe

logger = logging.getLogger("neurons")
HERE = Path(__file__).resolve().parent
WORD_LANGS = ("en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "zh")
INDIC = ("hi", "bn", "ta", "te", "mr", "gu", "kn", "ml")
SCRIPT = {"en": "latin", "hi": "devanagari", "mr": "devanagari", "bn": "bengali", "gu": "gujarati", "ta": "tamil",
          "te": "telugu", "kn": "kannada", "ml": "malayalam", "zh": "han", "hi_latn": "latin"}
# Each model's English band (where the logit lens reads English for most Indic->Indic items).
BANDS = {19: [13, 14, 15, 16]}  # other depths: the generic mid band (modal_pilot.bands)
MIN_SCRIPTS = 5
AUROC_MIN = 0.95
N_PERMUTATIONS = 200
# Cognates. Unicode's Indic blocks share one ISCII-derived layout (the same offset in each 128-code-point
# block is the same letter), so subtracting the block base gives a script-neutral spelling. Letters Tamil
# merges (k/kh/g/gh, ...) are folded and script-specific marks dropped. Two Indic words are cognates when
# their neutral spellings are >= COGNATE_SIMILARITY alike (1 - Levenshtein / longer length). 36% of the
# cross-script Indic translation pairs are (river: nadī in 7 scripts; sun: sūrya in all 8).
INDIC_BLOCKS = {"devanagari": 0x900, "bengali": 0x980, "gujarati": 0xA80, "tamil": 0xB80, "telugu": 0xC00,
                "kannada": 0xC80, "malayalam": 0xD00}
_FOLD = {ch: group[0] for group in ("कखगघ", "चछजझ", "टठडढ", "तथदध", "पफबभ", "नणऩ", "लळऴ", "रऱ", "शषस") for ch in group}
_DROP = set("ँंः़्॒॑")  # candrabindu, anusvara, visarga, nukta, virama, accents
COGNATE_SIMILARITY = 0.5
MIN_FORMS = (4, 5)  # independent word forms (cognate clusters; English and Chinese are their own) a neuron spans


class _StopForward(Exception):
    """Raised by the layer-0 hook once the neurons are caught: nothing after layer 0 is needed."""


### HOOKS AND FORWARDS


def moe_gates(model) -> dict[int, t.nn.Module]:
    """``{layer: gate}`` for every sparse-MoE block (both models: ``layers[i].mlp.gate`` beside ``.experts``)."""
    gates = {}
    for i, layer in enumerate(model.layers):
        mlp = getattr(layer, "mlp", None)
        if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
            gates[i] = mlp.gate
    return gates


class RoutingRecorder:
    """Forward hooks on the MoE gates; ``topk[layer]`` is the last forward's ``[batch, seq, k]`` expert ids."""

    def __init__(self, model) -> None:
        self.gates = moe_gates(model)
        self.topk: dict[int, t.Tensor] = {}
        self.handles: list = []

    def _hook(self, layer: int):
        def hook(module, args, output):
            batch, seq = args[0].shape[:2]
            self.topk[layer] = output[0].reshape(batch, seq, -1).detach()
        return hook

    def __enter__(self) -> "RoutingRecorder":
        self.handles = [gate.register_forward_hook(self._hook(i)) for i, gate in self.gates.items()]
        return self

    def __exit__(self, *exc) -> None:
        for handle in self.handles:
            handle.remove()


class DenseNeuronRecorder:
    """A pre-hook on layer 0's ``down_proj``: its input is ``act_fn(gate_proj(x)) * up_proj(x)``.
    It stores that ``[batch, seq, d_ff]`` tensor and stops the forward (see :class:`_StopForward`)."""

    def __init__(self, model) -> None:
        mlp = model.layers[0].mlp
        if hasattr(mlp, "experts") or not hasattr(mlp, "down_proj"):
            raise RuntimeError(f"layer 0 is not a dense MLP: {type(mlp).__name__}")
        self.down_proj = mlp.down_proj
        self.acts: t.Tensor | None = None

    def _hook(self, module, args):
        self.acts = args[0].detach()
        raise _StopForward

    def __enter__(self) -> "DenseNeuronRecorder":
        self.handle = self.down_proj.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *exc) -> None:
        self.handle.remove()


def forward_rows(model, ids_list: list[t.Tensor], batch_size: int, read) -> None:
    """Forward every prompt, ``batch_size`` at a time (right-padded with id 0, sorted by length so
    rows in a batch pad little), then ``read(items, lengths)`` with the batch's item indices."""
    order = sorted(range(len(ids_list)), key=lambda i: int(ids_list[i].shape[1]))
    for start in range(0, len(order), batch_size):
        items = order[start:start + batch_size]
        lengths = [int(ids_list[i].shape[1]) for i in items]
        batch = ids_list[items[0]].new_zeros((len(items), max(lengths)))
        for row, i in enumerate(items):
            batch[row, : lengths[row]] = ids_list[i][0]
        try:
            with t.no_grad():
                model.forward(batch)
        except _StopForward:
            pass
        read(items, lengths)


def record_routing(model, ids_list: list[t.Tensor], batch_size: int) -> tuple[np.ndarray, list[int]]:
    """``[n, n_moe_layers, k]`` top-k expert ids at each prompt's last real token, and the MoE layers."""
    with RoutingRecorder(model) as recorder:
        layers = sorted(recorder.gates)
        out: np.ndarray | None = None

        def read(items, lengths):
            nonlocal out
            missing = set(layers) - set(recorder.topk)
            if missing:
                raise RuntimeError(f"gate hooks did not fire at layers {sorted(missing)}")
            rows = t.arange(len(items))
            pos = t.tensor([n - 1 for n in lengths])
            per_layer = [recorder.topk[l][rows.to(recorder.topk[l].device), pos.to(recorder.topk[l].device)].cpu()
                         for l in layers]
            ids = t.stack(per_layer, dim=1).numpy()  # [batch, n_layers, k]
            if out is None:
                out = np.zeros((len(ids_list), len(layers), ids.shape[-1]), dtype=np.int16)
            out[items] = ids
            recorder.topk.clear()

        forward_rows(model, ids_list, batch_size, read)
    return out, layers


def record_dense_neurons(model, ids_list: list[t.Tensor], batch_size: int) -> np.ndarray:
    """``[n, d_ff]`` layer-0 MLP neurons (fp32) at each prompt's last real token."""
    with DenseNeuronRecorder(model) as recorder:
        out: np.ndarray | None = None

        def read(items, lengths):
            nonlocal out
            acts = recorder.acts
            rows = t.arange(len(items), device=acts.device)
            pos = t.tensor([n - 1 for n in lengths], device=acts.device)
            values = acts[rows, pos].float().cpu().numpy()
            if out is None:
                out = np.zeros((len(ids_list), values.shape[-1]), dtype=np.float32)
            out[items] = values

        forward_rows(model, ids_list, batch_size, read)
    return out


### INPUTS


def word_items(concepts: dict, names: list[str]) -> list[dict]:
    """One item per (language, concept): the word's first spelling."""
    return [{"lang": lang, "concept": c, "script": SCRIPT[lang], "text": concepts[c][lang][0]}
            for lang in WORD_LANGS for c in names if lang in concepts[c]]


def prompt_items(concepts: dict, names: list[str], pairs: list[str]) -> list[dict]:
    items = []
    for pair in pairs:
        src, tgt = pair.split("-")
        for c in names:
            if src in concepts[c] and tgt in concepts[c]:
                items.append({"pair": pair, "src": src, "tgt": tgt, "concept": c,
                              "text": pe.build_prompt(concepts, c, src, tgt, shots=5, seed=0)})
    return items


### STATISTICS


def multihot(topk: np.ndarray, n_experts: int) -> np.ndarray:
    """``[n, L, k]`` expert ids -> ``[L, n, n_experts]`` 0/1 float32."""
    x = np.zeros((*topk.shape[:2], n_experts), dtype=np.float32)
    np.put_along_axis(x, topk.astype(np.int64), 1.0, axis=2)
    return x.transpose(1, 0, 2).copy()


def jaccard(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Jaccard of k-sets coded as multi-hot rows: |A n B| / (2k - |A n B|)."""
    inter = a @ b.T
    return inter / (2 * k - inter)


def centroid_cv(x: np.ndarray, labels: np.ndarray, folds: np.ndarray, seed: int = 0) -> float:
    """Balanced accuracy of nearest-centroid (Euclidean) classification, one fold held out at a
    time. Ties are broken at random (a seeded jitter far below any real distance gap)."""
    rng = np.random.default_rng(seed)
    pred = np.empty(len(labels), dtype=labels.dtype)
    for fold in np.unique(folds):
        test = folds == fold
        train_labels = labels[~test]
        classes = np.unique(train_labels)
        cents = np.stack([x[~test][train_labels == c].mean(0) for c in classes])
        dist = ((x[test][:, None, :] - cents[None]) ** 2).sum(-1) + rng.uniform(0, 1e-6, (int(test.sum()), len(classes)))
        pred[test] = classes[dist.argmin(1)]
    return float(np.mean([np.mean(pred[labels == c] == c) for c in np.unique(labels)]))


def eta2(x: np.ndarray, labels: np.ndarray) -> float:
    """Share of the total sum of squares (summed over features) between the label groups' means."""
    mu = x.mean(0)
    total = float(((x - mu) ** 2).sum())
    if total == 0:
        return float("nan")
    between = sum(float((labels == g).sum() * ((x[labels == g].mean(0) - mu) ** 2).sum()) for g in np.unique(labels))
    return between / total


def eta2_null(labels: np.ndarray) -> float:
    """eta^2's expectation under random labels: (groups - 1) / (n - 1)."""
    return (len(np.unique(labels)) - 1) / (len(labels) - 1)


def neutral_spelling(word: str) -> str:
    """An Indic word in script-neutral form (see :data:`INDIC_BLOCKS`); other characters pass through."""
    out = []
    for ch in word:
        if 0x900 <= ord(ch) < 0xD80:
            ch = chr(0x900 + (ord(ch) - 0x900) % 0x80)
            if ch in _DROP:
                continue
            ch = _FOLD.get(ch, ch)
        out.append(ch)
    return "".join(out)


def spelling_similarity(a: str, b: str) -> float:
    """1 - Levenshtein distance / the longer length."""
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1 - prev[-1] / max(len(a), len(b))


def cognate_matrix(items: list[dict]) -> np.ndarray:
    """``[n, n]`` bool: both words Indic and cognate (English and Chinese words are never cognates here)."""
    forms = [neutral_spelling(i["text"]) if i["script"] in INDIC_BLOCKS else None for i in items]
    out = np.zeros((len(items), len(items)), dtype=bool)
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if forms[a] and forms[b] and spelling_similarity(forms[a], forms[b]) >= COGNATE_SIMILARITY:
                out[a, b] = out[b, a] = True
    return out


def n_word_forms(word_ids: list[int], cognate: np.ndarray) -> int:
    """Independent word forms among these words: connected components of the cognate graph."""
    parent = list(range(len(word_ids)))

    def root(a: int) -> int:
        while parent[a] != a:
            a = parent[a]
        return a

    for a in range(len(word_ids)):
        for b in range(a + 1, len(word_ids)):
            if cognate[word_ids[a], word_ids[b]]:
                parent[root(a)] = root(b)
    return len({root(a) for a in range(len(word_ids))})


def ci(values) -> list:
    lo, hi = pe.bootstrap_ci(np.asarray(values, dtype=float))
    return [_r(lo), _r(hi)]


def _r(x, nd: int = 4):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, nd)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


### ROUTING ANALYSES


def analyze_words(topk: np.ndarray, items: list[dict], layers: list[int], n_experts: int, band: list[int]):
    """Per-layer rows, per-(layer, Indic language) alignment rows, and the band summary."""
    k = topk.shape[-1]
    x = multihot(topk, n_experts)
    langs = np.array([i["lang"] for i in items])
    concepts = np.array([i["concept"] for i in items])
    scripts = np.array([i["script"] for i in items])
    names = sorted(set(concepts))
    index = {(i["lang"], i["concept"]): n for n, i in enumerate(items)}
    same_c = concepts[:, None] == concepts[None, :]
    cross_s = scripts[:, None] != scripts[None, :]
    same_l_diff_c = (langs[:, None] == langs[None, :]) & ~same_c
    # Indic words with an English and a Chinese translation (every concept has them)
    words = [n for n, i in enumerate(items) if i["lang"] in INDIC and ("en", i["concept"]) in index and ("zh", i["concept"]) in index]
    w_concept = np.array([names.index(items[n]["concept"]) for n in words])
    w_lang = np.array([items[n]["lang"] for n in words])
    col = {lang: np.array([index.get((lang, c), -1) for c in names]) for lang in WORD_LANGS}
    cognate = cognate_matrix(items)
    # per word, per Indic partner language: other script, and is the translation a cognate?
    partner_use = {ref: np.array([SCRIPT[lang] != SCRIPT[ref] for lang in w_lang]) & (col[ref] >= 0).all() for ref in INDIC}
    partner_cog = {ref: cognate[np.array(words), col[ref][w_concept]] for ref in INDIC}
    n_partners = sum(int(u.sum()) for u in partner_use.values())
    share_cognate = sum(int((partner_use[r] & partner_cog[r]).sum()) for r in INDIC) / max(1, n_partners)
    rows, lang_rows = [], []
    per_word: dict[str, list[np.ndarray]] = defaultdict(list)  # metric -> [layer] arrays over ``words``
    per_concept_excess = []  # [layer] arrays over concepts
    for li, layer in enumerate(layers):
        xl = x[li]
        jac = jaccard(xl, xl, k)
        excess_c = []
        for c in names:
            ic = concepts == c
            within = jac[np.ix_(ic, ic)][cross_s[np.ix_(ic, ic)]]
            between = jac[ic][(~same_c[ic]) & cross_s[ic]]
            excess_c.append(within.mean() - between.mean() if within.size and between.size else np.nan)
        per_concept_excess.append(np.array(excess_c))
        # English / Chinese / other-script Indic alignment of each Indic word
        w_idx = np.array(words)
        metrics = {}
        for ref in ("en", "zh"):
            jr = jac[np.ix_(w_idx, col[ref])]  # [n_words, n_concepts]
            same = jr[np.arange(len(words)), w_concept]
            other = (jr.sum(1) - same) / (len(names) - 1)
            metrics[f"{ref}_same"], metrics[f"{ref}_excess"] = same, same - other
        ind_same_sum = np.zeros(len(words))
        ind_other_sum = np.zeros(len(words))
        ind_n = np.zeros(len(words))
        split = {True: [np.zeros(len(words)), np.zeros(len(words))], False: [np.zeros(len(words)), np.zeros(len(words))]}
        for ref in INDIC:
            use = partner_use[ref]
            if not use.any():
                continue
            jr = jac[np.ix_(w_idx, col[ref])]
            same = jr[np.arange(len(words)), w_concept]
            other = (jr.sum(1) - same) / (len(names) - 1)
            ind_same_sum += np.where(use, same, 0)
            ind_other_sum += np.where(use, other, 0)
            ind_n += use
            for is_cog, (total, count) in split.items():
                sel = use & (partner_cog[ref] == is_cog)
                total += np.where(sel, same - other, 0)
                count += sel
        metrics["indic_same"] = ind_same_sum / np.maximum(ind_n, 1)
        metrics["indic_excess"] = (ind_same_sum - ind_other_sum) / np.maximum(ind_n, 1)
        # the same excess, partners split by cognate status (NaN for a word with no partner of that kind)
        for is_cog, name in ((True, "indic_cognate_excess"), (False, "indic_noncognate_excess")):
            total, count = split[is_cog]
            metrics[name] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        for key, value in metrics.items():
            per_word[key].append(value)
        rows.append({
            "layer": layer, "lang_acc": _r(centroid_cv(xl, langs, concepts)),
            "concept_acc_xscript": _r(centroid_cv(xl, concepts, scripts)),
            "jac_same_concept_xscript": _r(jac[same_c & cross_s].mean()),
            "jac_diff_concept_xscript": _r(jac[~same_c & cross_s].mean()),
            "jac_same_lang_diff_concept": _r(jac[same_l_diff_c].mean()),
            "concept_excess": _r(np.nanmean(excess_c)),
            **{f"mean_{key}": _r(np.nanmean(value)) for key, value in metrics.items()},
            "frac_en_closer_raw": _r(np.mean(metrics["en_same"] > metrics["indic_same"])),
            "frac_en_excess_gt_indic": _r(np.mean(metrics["en_excess"] > metrics["indic_excess"])),
            "frac_en_excess_gt_zh": _r(np.mean(metrics["en_excess"] > metrics["zh_excess"])),
        })
        for lang in INDIC:
            sel = w_lang == lang
            if sel.any():
                lang_rows.append({"layer": layer, "lang": lang, **{key: _r(np.nanmean(value[sel])) if (~np.isnan(value[sel])).any()
                                                                   else None for key, value in metrics.items()}})
    band_li = [li for li, layer in enumerate(layers) if layer in band]

    def band_word(key: str) -> np.ndarray:
        return np.mean([per_word[key][li] for li in band_li], axis=0)

    def by_concept(values: np.ndarray) -> np.ndarray:
        """Per-concept means (NaN words skipped; concepts with none left out)."""
        means = [np.nanmean(values[w_concept == c]) for c in range(len(names))
                 if (~np.isnan(values[w_concept == c])).any()]
        return np.array(means)

    summary: dict = {"layers": [layers[li] for li in band_li], "share_cognate_indic_partners": _r(share_cognate, 3)}
    if band_li:
        en_x, ind_x, zh_x = band_word("en_excess"), band_word("indic_excess"), band_word("zh_excess")
        en_s, ind_s = band_word("en_same"), band_word("indic_same")
        cog_x, noncog_x = band_word("indic_cognate_excess"), band_word("indic_noncognate_excess")
        summary.update({
            "indic_cognate_excess": _r(np.nanmean(cog_x)), "indic_noncognate_excess": _r(np.nanmean(noncog_x)),
            "en_minus_indic_noncognate_excess": _r(np.nanmean(en_x - noncog_x)),
            "en_minus_indic_noncognate_excess_ci": ci(by_concept(en_x - noncog_x)),
        })
        concept_ex = np.nanmean([per_concept_excess[li] for li in band_li], axis=0)
        summary.update({
            **{key: _r(np.mean([rows[li][key] for li in band_li]))
               for key in ("lang_acc", "concept_acc_xscript", "jac_same_concept_xscript", "jac_diff_concept_xscript",
                           "jac_same_lang_diff_concept")},
            "concept_excess": _r(np.nanmean(concept_ex)), "concept_excess_ci": ci(concept_ex[~np.isnan(concept_ex)]),
            "en_same": _r(en_s.mean()), "indic_same": _r(ind_s.mean()),
            "en_excess": _r(en_x.mean()), "indic_excess": _r(ind_x.mean()), "zh_excess": _r(zh_x.mean()),
            "en_minus_indic_excess": _r((en_x - ind_x).mean()), "en_minus_indic_excess_ci": ci(by_concept(en_x - ind_x)),
            "en_minus_zh_excess": _r((en_x - zh_x).mean()), "en_minus_zh_excess_ci": ci(by_concept(en_x - zh_x)),
            "frac_en_closer_raw": _r(np.mean(en_s > ind_s)), "frac_en_excess_gt_indic": _r(np.mean(en_x > ind_x)),
            "by_lang": {lang: {"en_excess": _r(en_x[w_lang == lang].mean()), "indic_excess": _r(ind_x[w_lang == lang].mean()),
                               "zh_excess": _r(zh_x[w_lang == lang].mean())} for lang in INDIC if (w_lang == lang).any()},
        })
    return rows, lang_rows, summary


def analyze_prompts(topk: np.ndarray, items: list[dict], word_topk: np.ndarray, word_items_: list[dict],
                    layers: list[int], n_experts: int, band: list[int]):
    """Per-layer rows and the band summary for the pivot prompts' final position."""
    k = topk.shape[-1]
    xp, xw = multihot(topk, n_experts), multihot(word_topk, n_experts)
    pairs = np.array([i["pair"] for i in items])
    srcs = np.array([i["src"] for i in items])
    tgts = np.array([i["tgt"] for i in items])
    concepts = np.array([i["concept"] for i in items])
    names = sorted(set(i["concept"] for i in word_items_))
    w_index = {(i["lang"], i["concept"]): n for n, i in enumerate(word_items_)}
    same_c = concepts[:, None] == concepts[None, :]
    same_p = pairs[:, None] == pairs[None, :]
    # Source / target decoding across held-out PAIRS: only languages that are the target (source) of >= 2 pairs.
    def multi(langs: np.ndarray) -> np.ndarray:
        n_pairs = {lang: len(set(pairs[langs == lang])) for lang in set(langs)}
        return np.array([n_pairs[lang] >= 2 for lang in langs])

    tgt_sel, src_sel = multi(tgts), multi(srcs)
    indic_pairs = np.array([all(pe.FAMILY.get(l) in ("IA", "Dr") and l not in ("hi", "hi_latn") for l in (s, g))
                            for s, g in zip(srcs, tgts)])
    roles = {"en": lambda i: "en", "hi": lambda i: "hi", "zh": lambda i: "zh", "src": lambda i: items[i]["src"],
             "tgt": lambda i: items[i]["tgt"]}
    rows, align_layers = [], []
    for li, layer in enumerate(layers):
        xl = xp[li]
        jac = jaccard(xl, xl, k)
        off = ~np.eye(len(items), dtype=bool)
        row = {"layer": layer}
        for name, labels in (("pair", pairs), ("src", srcs), ("tgt", tgts), ("concept", concepts)):
            row[f"eta2_{name}"] = _r(eta2(xl, labels))
        row["acc_concept_xpair"] = _r(centroid_cv(xl, concepts, pairs))
        row["acc_pair_xconcept"] = _r(centroid_cv(xl, pairs, concepts))
        row["acc_tgt_xpair"] = _r(centroid_cv(xl[tgt_sel], tgts[tgt_sel], pairs[tgt_sel])) \
            if len(set(tgts[tgt_sel])) >= 2 else None
        row["acc_src_xpair"] = _r(centroid_cv(xl[src_sel], srcs[src_sel], pairs[src_sel])) \
            if len(set(srcs[src_sel])) >= 2 else None
        row["jac_same_concept_diff_pair"] = _r(jac[same_c & ~same_p].mean())
        row["jac_same_pair_diff_concept"] = _r(jac[same_p & ~same_c & off].mean())
        row["jac_diff_both"] = _r(jac[~same_p & ~same_c].mean())
        # Does the final position's routing carry a WORD's expert signature? Excess Jaccard with the
        # concept's bare word in each role's language vs that language's other words (Indic pairs).
        cross = jaccard(xl, xw[li], k)
        align = {}
        for role, lang_of in roles.items():
            vals = []
            for i in np.flatnonzero(indic_pairs):
                lang = lang_of(i)
                cols = [w_index.get((lang, c), -1) for c in names]
                if min(cols) < 0 or concepts[i] not in names:
                    continue
                jr = cross[i, cols]
                same = jr[names.index(concepts[i])]
                vals.append(same - (jr.sum() - same) / (len(names) - 1))
            align[role] = np.array(vals)
            row[f"align_{role}_excess"] = _r(np.mean(vals)) if vals else None
        align_layers.append(align)
        rows.append(row)
    band_li = [li for li, layer in enumerate(layers) if layer in band]
    nulls = {f"eta2_{name}": _r(eta2_null(labels), 4) for name, labels in
             (("pair", pairs), ("src", srcs), ("tgt", tgts), ("concept", concepts))}
    chance = {"acc_concept_xpair": _r(1 / len(set(concepts))), "acc_pair_xconcept": _r(1 / len(set(pairs))),
              "acc_tgt_xpair": _r(1 / max(1, len(set(tgts[tgt_sel])))), "acc_src_xpair": _r(1 / max(1, len(set(srcs[src_sel]))))}
    summary: dict = {"layers": [layers[li] for li in band_li], "eta2_null": nulls, "chance": chance,
                     "tgt_xpair_langs": sorted(map(str, set(tgts[tgt_sel]))), "src_xpair_langs": sorted(map(str, set(srcs[src_sel])))}
    if band_li:
        keys = [k_ for k_ in rows[0] if k_ != "layer"]
        summary.update({key: _r(np.mean([rows[li][key] for li in band_li if rows[li][key] is not None]))
                        if any(rows[li][key] is not None for li in band_li) else None for key in keys})
        for role in roles:
            vals = np.mean([align_layers[li][role] for li in band_li], axis=0) if len(align_layers[band_li[0]][role]) else []
            summary[f"align_{role}_excess_ci"] = ci(vals) if len(vals) else None
    return rows, summary


### NEURON ANALYSES


def rank_columns(x: np.ndarray) -> np.ndarray:
    """Per-column ranks 1..n, tied values sharing their average rank (the Mann-Whitney convention).
    The saved activations are bf16, so every neuron has ties; ``argsort().argsort()`` broke them by row
    order, which follows the language, so a constant neuron looked script-selective."""
    n, d = x.shape
    order = np.argsort(x, axis=0, kind="stable")
    xs = np.take_along_axis(x, order, 0)
    run = np.cumsum(np.vstack([np.ones((1, d), bool), xs[1:] != xs[:-1]]), axis=0) - 1  # tie group per column
    flat = run + np.arange(d)[None, :] * n
    pos = np.broadcast_to(np.arange(1, n + 1, dtype=np.float64)[:, None], (n, d))
    avg = np.bincount(flat.ravel(), weights=pos.ravel(), minlength=n * d) / np.maximum(np.bincount(flat.ravel(), minlength=n * d), 1)
    ranks = np.empty((n, d))
    np.put_along_axis(ranks, order, avg[flat], 0)
    return ranks


def unique_argmax(z: np.ndarray, axis: int) -> np.ndarray:
    """``argmax`` along ``axis``, or -1 where the maximum is shared (ties, constant activations): no
    winner, instead of always the first concept."""
    win = z.argmax(axis)
    top = np.take_along_axis(z, np.expand_dims(win, axis), axis)
    return np.where((z == top).sum(axis) == 1, win, -1)


def analyze_neurons(acts: np.ndarray, items: list[dict], *, n_permutations: int = N_PERMUTATIONS, seed: int = 0):
    """Concept- vs script-selective layer-0 neurons. Returns (per-neuron rows, summary)."""
    rng = np.random.default_rng(seed)
    n_words, d_ff = acts.shape
    langs = [lang for lang in WORD_LANGS if any(i["lang"] == lang for i in items)]
    index = {(i["lang"], i["concept"]): n for n, i in enumerate(items)}
    names = sorted({i["concept"] for i in items if all((lang, i["concept"]) in index for lang in langs)})
    script_names = sorted({SCRIPT[lang] for lang in langs})
    lang_script = np.array([script_names.index(SCRIPT[lang]) for lang in langs])
    v = np.stack([acts[[index[(lang, c)] for c in names]] for lang in langs])  # [langs, concepts, d_ff]

    def n_scripts_of(win: np.ndarray) -> np.ndarray:
        """``win[lang, neuron]`` = winning concept -> ``[concepts, neurons]`` distinct scripts it wins."""
        won = np.zeros((len(script_names), len(names), d_ff), dtype=bool)
        hits = win[:, None, :] == np.arange(len(names))[None, :, None]
        for li, s in enumerate(lang_script):
            won[s] |= hits[li]
        return won.sum(0)

    best: dict[str, np.ndarray] = {}
    per_sign = []
    for sign in (1.0, -1.0):
        vs = sign * v
        z = (vs - vs.mean(1, keepdims=True)) / (vs.std(1, keepdims=True) + 1e-8)
        win = unique_argmax(z, 1)  # [langs, d_ff]; -1 = tied maximum, no winner (2026-10-20)
        n_sc = n_scripts_of(win)
        c_best = n_sc.argmax(0)
        n_best = n_sc.max(0)
        mean_z = z.mean(0)[c_best, np.arange(d_ff)]
        per_sign.append({"win": win, "c": c_best, "n": n_best, "mean_z": mean_z, "z": z})
    # per neuron: the sign with more scripts (ties: higher mean z)
    use_neg = (per_sign[1]["n"] > per_sign[0]["n"]) | ((per_sign[1]["n"] == per_sign[0]["n"]) & (per_sign[1]["mean_z"] > per_sign[0]["mean_z"]))
    for key in ("c", "n", "mean_z"):
        best[key] = np.where(use_neg, per_sign[1][key], per_sign[0][key])
    concept_sel = best["n"] >= MIN_SCRIPTS
    cognate = cognate_matrix(items)

    def forms_flags(labels: np.ndarray, real: np.ndarray, min_forms: int) -> np.ndarray:
        """Neurons where one label wins in languages whose words span >= ``min_forms`` independent forms
        (cognate clusters). ``labels[lang, neuron]`` is the winning label, ``real`` the word's own concept."""
        hits = labels[:, None, :] == np.arange(len(names))[None, :, None]
        flags = np.zeros(d_ff, dtype=bool)
        for c, n in zip(*np.nonzero(hits.sum(0) >= min_forms)):
            if not flags[n]:
                won = np.flatnonzero(hits[:, c, n])
                flags[n] = n_word_forms([index[(langs[li], names[real[li, n]])] for li in won], cognate) >= min_forms
        return flags

    forms_sel = {m: np.logical_or.reduce([forms_flags(s["win"], s["win"], m) for s in per_sign]) for m in MIN_FORMS}
    # null: concepts shuffled within each language (keeps every language's activations, breaks the
    # cross-language alignment of concepts); the winner's label moves with its word
    null_counts = []
    forms_null: dict[int, list[int]] = {m: [] for m in MIN_FORMS}
    for _ in range(n_permutations):
        perms = [rng.permutation(len(names)) for _ in langs]
        counts = np.zeros(d_ff, dtype=bool)
        forms_counts = {m: np.zeros(d_ff, dtype=bool) for m in MIN_FORMS}
        for s in per_sign:
            # a tied language stays without a winner in the null too
            win_null = np.stack([np.where(s["win"][li] >= 0, perms[li][s["win"][li]], -1) for li in range(len(langs))])
            counts |= n_scripts_of(win_null).max(0) >= MIN_SCRIPTS
            for m in MIN_FORMS:
                forms_counts[m] |= forms_flags(win_null, s["win"], m)
        null_counts.append(int(counts.sum()))
        for m in MIN_FORMS:
            forms_null[m].append(int(forms_counts[m].sum()))

    # AUROC of each script's (and each concept's) words vs the rest, sign-free: max(A, 1 - A)
    word_script = np.array([i["script"] for i in items])
    word_concept = np.array([i["concept"] for i in items])
    ranks = rank_columns(acts)  # ties share their average rank (2026-10-20)

    def auroc(labels: np.ndarray, groups: list[str]) -> np.ndarray:
        out = np.zeros((len(groups), d_ff))
        for gi, g in enumerate(groups):
            pos = labels == g
            n_p, n_n = int(pos.sum()), int((~pos).sum())
            a = (ranks[pos].sum(0) - n_p * (n_p + 1) / 2) / (n_p * n_n)
            out[gi] = np.maximum(a, 1 - a)
        return out

    s_auc = auroc(word_script, script_names)
    c_auc = auroc(word_concept, names)
    script_sel = s_auc.max(0) >= AUROC_MIN
    concept_auc_sel = c_auc.max(0) >= AUROC_MIN
    script_null, concept_auc_null = [], []
    for _ in range(20):
        perm = rng.permutation(n_words)
        script_null.append(int((auroc(word_script[perm], script_names).max(0) >= AUROC_MIN).sum()))
        concept_auc_null.append(int((auroc(word_concept[perm], names).max(0) >= AUROC_MIN).sum()))

    def eta2_cols(labels: np.ndarray) -> np.ndarray:
        mu = acts.mean(0)
        total = ((acts - mu) ** 2).sum(0) + 1e-12
        between = sum((labels == g).sum() * (acts[labels == g].mean(0) - mu) ** 2 for g in np.unique(labels))
        return between / total

    e_script, e_concept = eta2_cols(word_script), eta2_cols(word_concept)
    signs = ["-" if neg else "+" for neg in use_neg]

    def won_langs(n: int) -> list[str]:
        """Languages whose word for the neuron's best concept wins (its better sign)."""
        win = (per_sign[1] if use_neg[n] else per_sign[0])["win"]
        return [langs[li] for li in range(len(langs)) if win[li, n] == best["c"][n]]

    def forms_of(n: int) -> int:
        return n_word_forms([index[(lang, names[int(best["c"][n])])] for lang in won_langs(n)], cognate)

    rows = [{"neuron": n, "eta2_script": _r(e_script[n]), "eta2_concept": _r(e_concept[n]),
             "best_script": script_names[int(s_auc[:, n].argmax())], "script_auroc": _r(s_auc[:, n].max()),
             "best_concept_auroc": names[int(c_auc[:, n].argmax())], "concept_auroc": _r(c_auc[:, n].max()),
             "xscript_concept": names[int(best["c"][n])], "xscript_sign": signs[n], "xscript_n_scripts": int(best["n"][n]),
             "xscript_n_word_forms": forms_of(n), "xscript_mean_z": _r(best["mean_z"][n])} for n in range(d_ff)]

    def concept_example(n: int) -> dict:
        c = int(best["c"][n])
        return {"neuron": n, "sign": signs[n], "concept": names[c], "n_scripts": int(best["n"][n]),
                "n_word_forms": forms_of(n), "won_in": won_langs(n), "mean_z": _r(best["mean_z"][n], 2),
                "concept_auroc": _r(c_auc[c, n], 3),
                "words": {langs[li]: [items[index[(langs[li], names[c])]]["text"], _r(v[li, c, n], 3),
                                      _r(np.median(v[li, :, n]), 3)] for li in range(len(langs))}}

    def script_example(n: int) -> dict:
        s = int(s_auc[:, n].argmax())
        inside = word_script == script_names[s]
        return {"neuron": n, "script": script_names[s], "auroc": _r(s_auc[s, n], 4),
                "mean_in": _r(acts[inside, n].mean(), 3), "mean_out": _r(acts[~inside, n].mean(), 3),
                "eta2_script": _r(e_script[n], 3)}

    sel_idx = np.flatnonzero(concept_sel)
    top_c = sorted(sel_idx, key=lambda n: (-best["n"][n], -best["mean_z"][n]))[:10]
    if not len(top_c):  # still show the nearest misses
        top_c = sorted(range(d_ff), key=lambda n: (-best["n"][n], -best["mean_z"][n]))[:5]
    top_s = sorted(np.flatnonzero(script_sel), key=lambda n: (-s_auc[:, n].max(), -e_script[n]))[:10]
    non_indic = {script_names.index(s) for s in ("latin", "han") if s in script_names}
    with_en_zh = sum(any(lang_script[langs.index(lang)] in non_indic for lang in won_langs(n)) for n in sel_idx)
    hist = np.bincount(best["n"], minlength=len(script_names) + 1)
    summary = {
        "layer": 0, "d_ff": d_ff, "n_words": n_words, "n_concepts": len(names), "langs": langs, "scripts": script_names,
        "concept_selective": {"rule": f"same concept top-1 (within-language z) in >= {MIN_SCRIPTS} scripts, either sign",
                              "n": int(concept_sel.sum()), "with_latin_or_han": int(with_en_zh),
                              "null_mean": _r(np.mean(null_counts), 2), "null_p95": _r(np.percentile(null_counts, 95), 1),
                              "null_max": int(max(null_counts)), "n_permutations": n_permutations,
                              "n_scripts_histogram": hist.tolist(),
                              "by_concept": {names[c]: int(((best["c"] == c) & concept_sel).sum()) for c in range(len(names))
                                             if ((best["c"] == c) & concept_sel).any()}},
        # the cognate control: cross-script agreement on a concept counted in independent word forms
        "independent_forms": {f">={m}": {"n": int(forms_sel[m].sum()), "of_which_5_scripts": int((forms_sel[m] & concept_sel).sum()),
                                         "null_mean": _r(np.mean(forms_null[m]), 2), "null_p95": _r(np.percentile(forms_null[m], 95), 1),
                                         "null_max": int(max(forms_null[m]))} for m in MIN_FORMS},
        "script_selective": {"rule": f"one script's words vs the rest, AUROC >= {AUROC_MIN} (either sign)",
                             "n": int(script_sel.sum()), "null_max": int(max(script_null)),
                             "by_script": {script_names[s]: int(((s_auc.argmax(0) == s) & script_sel).sum()) for s in range(len(script_names))}},
        "concept_auroc_selective": {"rule": f"one concept's {len(langs)} words vs the rest, AUROC >= {AUROC_MIN}",
                                    "n": int(concept_auc_sel.sum()), "null_max": int(max(concept_auc_null))},
        "both_concept_and_script": int((concept_sel & script_sel).sum()),
        "eta2": {"script_median": _r(np.median(e_script), 3), "concept_median": _r(np.median(e_concept), 3),
                 "script_null": _r((len(script_names) - 1) / (n_words - 1), 3), "concept_null": _r((len(names) - 1) / (n_words - 1), 3),
                 "n_script_gt_0.5": int((e_script > 0.5).sum()), "n_concept_gt_0.5": int((e_concept > 0.5).sum()),
                 # eta^2 over its chance level, concept vs script: which factor a neuron tracks more, per neuron
                 "n_concept_over_script_rel_null": int((e_concept / ((len(names) - 1) / (n_words - 1))
                                                        > e_script / ((len(script_names) - 1) / (n_words - 1))).sum())},
        "top_concept": [concept_example(int(n)) for n in top_c],
        "top_script": [script_example(int(n)) for n in top_s],
    }
    return rows, summary


### DRIVER ENTRY


def batch_consistency(model, ids_list: list[t.Tensor], batch_size: int, batched: np.ndarray) -> dict:
    """Mean Jaccard (over layers) between the batched routing and lone forwards for a few prompts."""
    pick = list(range(0, len(ids_list), max(1, len(ids_list) // 4)))[:4]
    lone, _ = record_routing(model, [ids_list[i] for i in pick], 1)
    k = lone.shape[-1]
    agree = [len(set(lone[j, l]) & set(batched[i, l])) / (2 * k - len(set(lone[j, l]) & set(batched[i, l])))
             for j, i in enumerate(pick) for l in range(lone.shape[1])]
    return {"n_prompts": len(pick), "mean_jaccard": _r(np.mean(agree)), "min_jaccard": _r(np.min(agree)),
            "identical_share": _r(np.mean(np.array(agree) == 1.0))}


def run(model, *, out_dir: str, analyses=("routing", "neurons"), vocab_size: int | None = None, batch_size: int = 1,
        smoke: bool = False) -> dict:
    """``vocab_size`` is accepted for the driver's uniform call and unused (nothing here reads logits).
    ``smoke`` (a random tiny model) keeps 8 concepts and 6 pairs (two targets and two sources shared by
    two pairs each, for the cross-pair decoders), so every code path runs in seconds."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    concepts = pe.load_concepts(HERE / "data/concepts.tsv")
    names = sorted(c for c in concepts if all(lang in concepts[c] for lang in WORD_LANGS))
    pairs = [p for p in pe.DEFAULT_PAIRS.split(",") if p]
    if smoke:
        names, pairs = names[:8], ["ml-ta", "bn-ta", "mr-bn", "ta-bn", "ta-te", "bn-gu"]
    words = word_items(concepts, names)
    word_ids = [model.encode(w["text"]) for w in words]
    for w, ids in zip(words, word_ids):
        w["n_tokens"] = int(ids.shape[1]) - 1  # without BOS
    n_layers = model.n_layers
    band = BANDS.get(n_layers) or [l for l in range(int(0.68 * n_layers), int(0.89 * n_layers)) if l < n_layers - 1][:6]
    result: dict = {"bos_mode": getattr(model, "bos_mode", None), "batch_size": batch_size, "n_concepts": len(names)}

    if "routing" in analyses:
        gates = moe_gates(model)
        first_gate = next(iter(gates.values()))
        n_experts = int(first_gate.weight.shape[0])
        w_topk, layers = record_routing(model, word_ids, batch_size)
        if band is None:  # a tiny smoke model: the MoE layers at 0.68-0.89 depth, else the deepest
            band = [l for l in layers if 0.68 * model.n_layers <= l < 0.89 * model.n_layers] or layers[-1:]
        prompts = prompt_items(concepts, names, pairs)
        prompt_ids = [model.encode(p["text"]) for p in prompts]
        p_topk, _ = record_routing(model, prompt_ids, batch_size)
        consistency = batch_consistency(model, prompt_ids, batch_size, p_topk) if batch_size > 1 else None
        np.savez_compressed(out / "routing_topk.npz", words=w_topk, prompts=p_topk, layers=np.array(layers))
        (out / "routing_items.json").write_text(json.dumps(
            {"words": words, "prompts": [{k: v for k, v in p.items() if k != "text"} for p in prompts],
             "layers": layers, "n_experts": n_experts, "top_k": int(w_topk.shape[-1]), "band": band,
             "batch_consistency": consistency}, ensure_ascii=False, indent=0), encoding="utf-8")
        result["routing"] = write_routing(out, words, prompts, w_topk, p_topk, layers, n_experts, band, consistency)

    if "neurons" in analyses:
        acts = record_dense_neurons(model, word_ids, batch_size)
        # bf16 activations, stored exactly as their 16 bits (fp16 would round the smallest ones)
        np.savez_compressed(out / "layer0_neurons.npz",
                            acts_bf16_bits=t.from_numpy(acts).to(t.bfloat16).view(t.int16).numpy())
        (out / "neuron_items.json").write_text(json.dumps(words, ensure_ascii=False, indent=0), encoding="utf-8")
        result["neurons"] = write_neurons(out, words, acts)
    return result


def write_routing(out: Path, words: list[dict], prompts: list[dict], w_topk: np.ndarray, p_topk: np.ndarray,
                  layers: list[int], n_experts: int, band: list[int], consistency: dict | None) -> dict:
    """Run the routing analyses, write their CSVs and ``routing_summary.json``; return the summary."""
    w_rows, lang_rows, w_band = analyze_words(w_topk, words, layers, n_experts, band)
    p_rows, p_band = analyze_prompts(p_topk, prompts, w_topk, words, layers, n_experts, band)
    write_csv(out / "routing_words_by_layer.csv", w_rows)
    write_csv(out / "routing_alignment_by_lang.csv", lang_rows)
    write_csv(out / "routing_prompts_by_layer.csv", p_rows)
    n_tok = defaultdict(list)
    for w in words:
        n_tok[w["lang"]].append(w["n_tokens"])

    def curve(rows: list[dict], key: str) -> list:
        return [r[key] for r in rows]

    summary = {
        "n_words": len(words), "n_prompts": len(prompts), "n_experts": n_experts, "top_k": int(w_topk.shape[-1]),
        "moe_layers": layers, "band": band, "batch_consistency": consistency,
        "word_tokens_mean": {lang: round(float(np.mean(v)), 2) for lang, v in n_tok.items()},
        "words": {"chance": {"lang_acc": _r(1 / len(WORD_LANGS)), "concept_acc_xscript": _r(1 / len({w["concept"] for w in words}))},
                  "by_layer": {key: curve(w_rows, key) for key in
                               ("lang_acc", "concept_acc_xscript", "concept_excess", "mean_en_excess",
                                "mean_indic_excess", "mean_indic_noncognate_excess", "mean_zh_excess")},
                  "band": w_band},
        "prompts": {"by_layer": {key: curve(p_rows, key) for key in
                                 ("eta2_pair", "eta2_concept", "acc_concept_xpair", "acc_tgt_xpair", "acc_src_xpair",
                                  "align_en_excess", "align_tgt_excess")},
                    "band": p_band},
    }
    (out / "routing_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    logger.info("routing: %d words, %d prompts, %d MoE layers -> %s", len(words), len(prompts), len(layers), out)
    return summary


def write_neurons(out: Path, words: list[dict], acts: np.ndarray, suffix: str = "") -> dict:
    """Run the layer-0 neuron analyses, write ``layer0_neurons{suffix}.csv`` and
    ``neurons_summary{suffix}.json``; return a brief summary (top examples without their word lists)."""
    rows, summary = analyze_neurons(acts, words)
    write_csv(out / f"layer0_neurons{suffix}.csv", rows)
    (out / f"neurons_summary{suffix}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    brief = {k: v for k, v in summary.items() if k not in ("top_concept", "top_script")}
    brief["top_concept"] = [{k: v for k, v in e.items() if k != "words"} for e in summary["top_concept"][:5]]
    brief["top_script"] = summary["top_script"][:5]
    logger.info("neurons: d_ff %d, concept-selective %d, script-selective %d", acts.shape[1],
                summary["concept_selective"]["n"], summary["script_selective"]["n"])
    return brief


def load_neuron_acts(path: Path) -> np.ndarray:
    """The saved layer-0 activations as fp32 (bf16 bits, or the first runs' fp16 copy)."""
    saved = np.load(path)
    if "acts_bf16_bits" in saved:
        return t.from_numpy(saved["acts_bf16_bits"]).view(t.bfloat16).float().numpy()
    return saved["acts"].astype(np.float32)


def reanalyze(run_dir: str, suffix: str = "") -> None:
    """Recompute every summary of a downloaded run (``<run>/routing``, ``<run>/neurons``) from its saved
    expert ids and activations, on CPU. Runs ``neuron30`` / ``neuron105`` predate the cognate controls.
    With ``suffix`` only the neuron files are written, under new names (routing has no ties to fix)."""
    run = Path(run_dir)
    if not suffix and (run / "routing/routing_topk.npz").is_file():
        meta = json.loads((run / "routing/routing_items.json").read_text(encoding="utf-8"))
        saved = np.load(run / "routing/routing_topk.npz")
        old = run / "routing/routing_summary.json"
        old_summary = json.loads(old.read_text(encoding="utf-8")) if old.is_file() else {}
        band = meta.get("band") or old_summary.get("band")
        consistency = meta.get("batch_consistency", old_summary.get("batch_consistency"))
        prompts = meta["prompts"]
        write_routing(run / "routing", meta["words"], prompts, saved["words"], saved["prompts"], meta["layers"],
                      meta["n_experts"], band, consistency)
    if (run / "neurons/layer0_neurons.npz").is_file():
        words = json.loads((run / "neurons/neuron_items.json").read_text(encoding="utf-8"))
        write_neurons(run / "neurons", words, load_neuron_acts(run / "neurons/layer0_neurons.npz"), suffix)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Recompute a downloaded run's routing / neuron summaries on CPU.")
    parser.add_argument("--reanalyze", required=True, help="a run dir, e.g. results/neuron30")
    parser.add_argument("--suffix", default="", help="e.g. _v2: write layer0_neurons_v2.csv / neurons_summary_v2.json only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    reanalyze(args.reanalyze, args.suffix)
