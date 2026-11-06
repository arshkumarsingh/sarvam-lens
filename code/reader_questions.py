"""Experiments that answer the questions a reader of the write-up would ask (docs/09_reader_questions.md).

- ``run_items(kind="chat_word" | "chat_sentence" | "cloze")``:
  - The English-pivot readout in REAL CHAT USE. Sarvam's own chat template (thinking off, as the
    template renders a reply without reasoning) wraps a translation request.
  - ``chat_word`` uses a word request, with the instruction in English (``instr="en"``) or in the
    source language (``instr="src"``: no English anywhere in the prompt).
  - ``chat_sentence`` translates a whole sentence (``data/sentences_parallel.tsv``). The reply is
    teacher-forced up to the concept word.
  - ``cloze`` completes a plain sentence with no instruction (reader question 4). The concept word
    is located in the model's own greedy continuation (up to ``max_steps`` tokens), so a model that
    first writes an adjective is no longer dropped.
- ``run_collapse``: English-collapse errors (reader question 3).
  - Homonyms (``data/homonyms.tsv``) are senses that share one English word but are different
    words in Indian languages (finger-nail / metal nail). The only bridge between the senses is
    English.
  - Kinship terms (``data/kinship.tsv``) are distinctions English lumps together (mother's brother /
    father's younger brother, both "uncle").
  - Each item records whether the model's answer is the right word, the wrong sense of the same
    English word (a "sibling"), a matched control word, or something else, plus lens readouts of all
    four.
- ``run_think``: thinking mode on. Greedy reasoning, KV-cached, records the script mix of the
  visible reasoning and whether the English word appears in it.

Every readout reads all lenses off ONE forward pass and scores on the GPU. Rows have the same
schema as :mod:`pivot_experiment`, so its ``summarize`` and ``plot_lens`` apply unchanged.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch as t

import pivot_experiment as pe

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SENTENCE_LANGS = ("hi", "ta", "te", "bn")
HOMONYM_LANGS = ("hi", "bn", "ta", "te", "kn", "ml")
KIN_LANGS = ("ta", "te", "kn", "ml")
# What Sarvam's chat template renders for one user turn with thinking off (used only when a
# tokenizer carries no template, e.g. the tiny smoke-test checkpoints).
FALLBACK_TEMPLATE = "[@BOS@]\n<|start_of_turn|><|user|>\n{msg}<|nothink|><|end_of_turn|>\n<|start_of_turn|><|assistant|>\n"
NO_THINK_REPLY = "<think></think>\n"  # how the template renders a reply without reasoning
SCRIPTS = {"latin": (0x41, 0x24F), "devanagari": (0x900, 0x97F), "bengali": (0x980, 0x9FF), "gujarati": (0xA80, 0xAFF),
           "tamil": (0xB80, 0xBFF), "telugu": (0xC00, 0xC7F), "kannada": (0xC80, 0xCFF), "malayalam": (0xD00, 0xD7F),
           "cjk": (0x4E00, 0x9FFF)}


### DATA


def load_prompts() -> dict:
    return json.loads((DATA / "chat_prompts.json").read_text(encoding="utf-8"))


def _split(cell: str) -> list[str]:
    return [unicodedata.normalize("NFC", s.strip()) for s in (cell or "").split("|") if s.strip()]


def load_sentences() -> dict[tuple[str, str], dict]:
    with (DATA / "sentences_parallel.tsv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {(r["lang"], r["concept"]): {"prompt": unicodedata.normalize("NFC", r["prompt"]), "target": _split(r["target"]),
                                        "rest": unicodedata.normalize("NFC", r["rest"])} for r in rows}


def load_table(name: str, key: str, langs: tuple[str, ...]) -> dict[str, dict]:
    with (DATA / name).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {r[key]: {"group": r["group"], "en": _split(r["en"]), **{l: _split(r.get(l, "")) for l in langs},
                     "zh": _split(r.get("zh", ""))} for r in rows}


### PROMPTS


def chat_text(tokenizer, msg: str, *, thinking: bool = False) -> str:
    """Sarvam's chat template around one user message, ready for the assistant's reply."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=thinking)
    return FALLBACK_TEMPLATE.format(msg=msg).replace("<|nothink|>", "" if thinking else "<|nothink|>")


def chat_ids(model, text: str) -> t.Tensor:
    """Tokenize as a deployment does: ``tokenizer(templated_text)`` with its default special
    tokens (the model card's own example), i.e. ``<bos>`` + the template's ``[@BOS@]``."""
    ids = model.tokenizer(text).input_ids
    return t.tensor([ids], dtype=t.long, device=model.input_device)


def word_request(prompts: dict, word: str, src: str, tgt: str, instr: str) -> str:
    lang = "en" if instr == "en" or src not in prompts["word"] else src
    names = prompts["names"][lang]
    return prompts["word"][lang].format(w=word, src=names[src], tgt=names[tgt])


def fewshot_prompt(demo_concepts: dict, query_word: str, src: str, tgt: str, *, key: str, shots: int = 5, seed: int = 0) -> str:
    """:func:`pivot_experiment.build_prompt`'s format, with demos from the main concept list and a
    query word from another table (so the demos never contain the tested word)."""
    others = sorted(c for c in demo_concepts if src in demo_concepts[c] and tgt in demo_concepts[c])
    rng = np.random.default_rng([seed, zlib.crc32(f"{src}|{tgt}|{key}".encode())])
    demos = rng.choice(others, size=min(shots, len(others)), replace=False)
    lines = [f'{pe.LANG_LABELS[src]}: "{demo_concepts[c][src][0]}" - {pe.LANG_LABELS[tgt]}: "{demo_concepts[c][tgt][0]}"' for c in demos]
    lines.append(f'{pe.LANG_LABELS[src]}: "{query_word}" - {pe.LANG_LABELS[tgt]}: "')
    return "\n".join(lines)


### READOUT


class Reader:
    """All lenses off one forward pass; scores on the LM head's device."""

    def __init__(self, model, lenses: dict, layers: list[int], vocab_size: int, mask_V: t.Tensor | None) -> None:
        self.model, self.lenses, self.layers = model, lenses, list(layers)
        self.final = model.n_layers - 1
        self.device = model._lm_head.weight.device
        self.mask = mask_V.to(self.device) if mask_V is not None else None
        self.vocab_size = vocab_size

    def forward(self, ids: t.Tensor) -> tuple[dict[int, t.Tensor], t.Tensor]:
        from jlens.hooks import ActivationRecorder

        at = sorted(set(self.layers) | {self.final})
        with t.no_grad(), ActivationRecorder(self.model.layers, at=at) as recorder:
            self.model.forward(ids)
            acts = {l: recorder.activations[l][0, -1].float() for l in at}
            logits = self.model.unembed(acts[self.final][None]).float()[0]
        return acts, logits

    def forward_batch(self, ids_list: list[t.Tensor]) -> list[tuple[dict[int, t.Tensor], t.Tensor]]:
        """:meth:`forward` for several prompts in ONE forward pass, read at each row's last real
        token. Rows are right-padded: under the causal mask no real token attends to the padding
        after it, positions start at 0 in every row, and the MoE routes each token on its own, so a
        row matches a lone forward up to kernel numerics. It pays where a forward is launch-bound
        (the 105B's per-expert Python loop costs about the same for 1 prompt as for 16)."""
        from jlens.hooks import ActivationRecorder

        if len(ids_list) == 1:
            return [self.forward(ids_list[0])]
        lengths = [int(ids.shape[1]) for ids in ids_list]
        batch = ids_list[0].new_zeros((len(ids_list), max(lengths)))  # pad id 0: only padding attends to it
        for row, ids in enumerate(ids_list):
            batch[row, : lengths[row]] = ids[0]
        at = sorted(set(self.layers) | {self.final})
        out = []
        with t.no_grad(), ActivationRecorder(self.model.layers, at=at) as recorder:
            self.model.forward(batch)
            for row, n in enumerate(lengths):
                acts = {l: recorder.activations[l][row, n - 1].float() for l in at}
                out.append((acts, self.model.unembed(acts[self.final][None]).float()[0]))
        return out

    def locate(self, ids: t.Tensor, target: set[int], max_steps: int):
        """Greedy-decode up to ``max_steps`` tokens until the model's next token begins the target
        word. Returns (activations at that position, its top token, the step or None, the ids)."""
        for step in range(max_steps + 1):
            acts, logits = self.forward(ids)
            top = int(logits.argmax())
            if top in target or step == max_steps:
                return acts, top, (step if top in target else None), ids
            ids = t.cat([ids, t.tensor([[top]], device=ids.device)], dim=1)

    def locate_batch(self, ids_list: list[t.Tensor], targets: list[set[int]], max_steps: int, batch_size: int = 1):
        """:meth:`locate` for many prompts, ``batch_size`` greedy steps per forward
        (:meth:`forward_batch`). Each forward takes the lowest-numbered unresolved prompts, so a
        prompt that needs another step rejoins the next forward beside fresh ones. Batch 1 is
        :meth:`locate` on each prompt in turn."""
        ids_now, steps, out = list(ids_list), [0] * len(ids_list), [None] * len(ids_list)
        todo = list(range(len(ids_list)))
        while todo:
            batch = todo[:batch_size]
            for i, (acts, logits) in zip(batch, self.forward_batch([ids_now[i] for i in batch])):
                top = int(logits.argmax())
                if top in targets[i] or steps[i] == max_steps:
                    out[i] = (acts, top, steps[i] if top in targets[i] else None, ids_now[i])
                else:
                    ids_now[i] = t.cat([ids_now[i], t.tensor([[top]], device=ids_now[i].device)], dim=1)
                    steps[i] += 1
            todo = [i for i in todo if out[i] is None]
        return out

    def score(self, acts: dict[int, t.Tensor], role_ids: dict[str, set[int]], top_k: int = 0):
        """Yields (lens, layer, role, prob, best_rank, top10) and, with ``top_k``, collects the
        top-k token ids per (lens, layer) in ``self.last_top``."""
        index = {role: t.tensor(sorted(ids), dtype=t.long, device=self.device) for role, ids in role_ids.items() if ids}
        self.last_top = {}
        with t.no_grad():
            for name, lens in self.lenses.items():
                for layer in self.layers:
                    logits = self.model.unembed(lens.transport(acts[layer][None], layer)).float()[0][: self.vocab_size]
                    if self.mask is not None:
                        logits = logits.masked_fill(self.mask, float("-inf"))
                    probs = t.softmax(logits, -1)
                    if top_k:
                        self.last_top[(name, layer)] = logits.topk(top_k).indices.tolist()
                    for role in role_ids:
                        if role not in index:
                            yield name, layer, role, float("nan"), -1, False
                            continue
                        ids = index[role]
                        best = logits[ids].max()
                        rank = int((logits > best).sum()) + 1
                        yield name, layer, role, float(probs[ids].sum()), rank, rank <= 10


def roles_for(index, tokenizer, concepts: dict, concept: str, src: str, tgt: str, *,
              target_words: list[str] | None = None, source_words: list[str] | None = None):
    """Pivot-experiment roles (target, source, English, Hindi, Chinese, disjoint) plus the three
    shuffled-concept controls. ``target_words``/``source_words`` override the concept list's
    spellings (sentence forms). Returns (scored sets, raw target set, role -> language)."""
    words = concepts[concept]
    roles = pe.tracked_languages(src, tgt)
    raw = {}
    for role, lang in roles.items():
        spellings = (target_words if role == "target" and target_words else
                     source_words if role == "source" and source_words else words.get(lang, []))
        raw[role] = pe.prefix_token_ids(index, spellings, tokenizer) if spellings else set()
    sets, _ = pe.disjoint_sets(raw)
    ordered = sorted(concepts)
    control = ordered[(ordered.index(concept) + len(ordered) // 2) % len(ordered)]
    claimed = set().union(*raw.values())
    role_langs = dict(roles)
    for role, lang in pe.CONTROL_ROLES.items():
        if lang in concepts[control]:
            sets[role] = pe.prefix_token_ids(index, concepts[control][lang], tokenizer) - claimed
            role_langs[role] = f"{lang}:{control}"
    return sets, raw["target"], role_langs


def _write(out: Path, tables: dict[str, list[dict]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if table:
            fields = list(dict.fromkeys(k for row in table for k in row))
            with (out / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(table)


def _group(items: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for i in items:
        out[i["group"]].append(i)
    return out


def _curves(summary: list[dict], lens: str, pair: str = "ALL_INDIC") -> dict:
    out: dict = {}
    for s in sorted((s for s in summary if s["pair"] == pair and s["lens"] == lens), key=lambda s: s["layer"]):
        out.setdefault(s["role"], {})[s["layer"]] = [round(s["top10_rate"], 3), round(s["mean_prob"], 4)]
    return out


### 1. CHAT AND CLOZE PIVOT


def run_items(model, lenses: dict, *, kind: str, out_dir: str, vocab_size: int, layers: list[int],
              instr: str = "src", pairs: str = pe.DEFAULT_PAIRS, require_correct: bool = True,
              max_steps: int = 2, max_items: int | None = None, meta_band: list[int] | None = None,
              meta_k: int = 20, batch_size: int = 1) -> dict:
    """``batch_size`` > 1 runs that many greedy steps per forward (:meth:`Reader.locate_batch`)."""
    tokenizer = model.tokenizer
    index = pe.build_vocab_index(tokenizer, vocab_size)
    concepts = pe.load_concepts(DATA / "concepts.tsv")
    prompts = load_prompts()
    sentences = load_sentences()
    all_tracked: set[int] = set()
    for c in concepts.values():
        for spellings in c.values():
            all_tracked |= pe.prefix_token_ids(index, spellings, tokenizer)
    for s in sentences.values():
        all_tracked |= pe.prefix_token_ids(index, s["target"], tokenizer)
    reader = Reader(model, lenses, layers, vocab_size, pe.readout_mask(tokenizer, vocab_size, all_tracked))
    last_layer = model.n_layers - 1
    meta_counts: Counter = Counter()
    n_meta = 0

    jobs = []  # (pair label, src, tgt, concept, ids builder, target words, source words)
    if kind == "chat_word":
        for pair in pairs.split(","):
            src, tgt = pair.split("-")
            for concept in sorted(concepts):
                if src in concepts[concept] and tgt in concepts[concept]:
                    msg = word_request(prompts, concepts[concept][src][0], src, tgt, instr)
                    jobs.append((pair, src, tgt, concept, chat_text(tokenizer, msg) + NO_THINK_REPLY, None, None))
    elif kind == "chat_sentence":
        for src in SENTENCE_LANGS:
            for tgt in SENTENCE_LANGS:
                if src == tgt:
                    continue
                names = prompts["names"][src]
                for concept in sorted(concepts):
                    s_src, s_tgt = sentences.get((src, concept)), sentences.get((tgt, concept))
                    if not (s_src and s_tgt):
                        continue
                    sentence = f"{s_src['prompt']} {s_src['target'][0]}{s_src['rest']}"
                    msg = prompts["sentence"][src].format(tgt=names[tgt], s=sentence)
                    jobs.append((f"{src}-{tgt}", src, tgt, concept, chat_text(tokenizer, msg) + NO_THINK_REPLY + s_tgt["prompt"],
                                 s_tgt["target"], s_src["target"] + concepts[concept].get(src, [])))
    elif kind == "cloze":
        for (lang, concept), s in sorted(sentences.items()):
            # no source language: src = tgt, so tracked_languages drops the source role and keeps
            # English, Hindi (unless the sentence is Hindi) and Chinese
            jobs.append((lang, lang, lang, concept, s["prompt"], s["target"], None))
    else:
        raise ValueError(kind)

    rows, items = [], []
    per_pair_kept: Counter = Counter()
    position = 0
    while position < len(jobs):
        # The next chunk of jobs, located together. With ``max_items`` a chunk holds at most
        # ``batch_size`` prompts and stops before a job whose pair could fill up inside it (that job
        # runs only if one there is dropped), so the kept items match one-at-a-time processing.
        chunk, pending = [], Counter()
        while position < len(jobs) and (max_items is None or sum(pending.values()) < batch_size):
            pair, src, tgt, concept, text, target_words, source_words = jobs[position]
            if max_items is not None and per_pair_kept[pair] >= max_items:
                position += 1
                continue
            if max_items is not None and per_pair_kept[pair] + pending[pair] >= max_items:
                break
            position += 1
            if kind == "cloze":
                sets, raw_target, role_langs = roles_for(index, tokenizer, concepts, concept, src, tgt, target_words=target_words)
                ids = model.encode(text)
            else:
                sets, raw_target, role_langs = roles_for(index, tokenizer, concepts, concept, src, tgt,
                                                         target_words=target_words, source_words=source_words)
                ids = chat_ids(model, text)
            chunk.append((pair, src, tgt, concept, sets, raw_target, role_langs, ids))
            pending[pair] += bool(sets.get("target"))
        located = iter(reader.locate_batch([c[-1] for c in chunk if c[4].get("target")],
                                           [c[5] for c in chunk if c[4].get("target")], max_steps, batch_size))
        for pair, src, tgt, concept, sets, raw_target, role_langs, ids in chunk:
            item = {"pair": pair, "concept": concept, "kind": kind, "instr": instr if kind == "chat_word" else ""}
            if not sets.get("target"):
                items.append({**item, "kept": False, "reason": "target tokens all shared with another role"})
                continue
            acts, top, step, final_ids = next(located)
            found = step is not None
            prefix = tokenizer.decode(final_ids[0, ids.shape[1]:]) if final_ids.shape[1] > ids.shape[1] else ""
            item.update(model_top=tokenizer.decode([top]), found_at_step=step, generated_prefix=prefix, n_prompt_tokens=int(ids.shape[1]))
            if require_correct and not found:
                items.append({**item, "kept": False, "reason": f"target word not begun within {max_steps} greedy tokens"})
                continue
            items.append({**item, "kept": True, "reason": ""})
            per_pair_kept[pair] += 1
            for lens_name, layer, role, prob, rank, top10 in reader.score(acts, sets, top_k=meta_k if meta_band else 0):
                rows.append({"pair": pair, "src": src, "tgt": tgt, "concept": concept, "lens": lens_name, "layer": layer,
                             "depth": layer / last_layer, "role": role, "lang": role_langs.get(role, ""), "prob": prob,
                             "best_rank": rank, "top10": top10})
            if meta_band:
                seen = set()
                for (lens_name, layer), top_ids in reader.last_top.items():
                    if layer in meta_band and lens_name == next(iter(lenses)):
                        seen |= set(top_ids)
                meta_counts.update(seen - all_tracked)
                n_meta += 1

    indic = ({p for p in per_pair_kept if all(l in ("ta", "te", "bn") for l in p.split("-"))} if kind == "chat_sentence"
             else {"ta", "te", "bn"} if kind == "cloze"
             else {p for p in per_pair_kept if all(pe.FAMILY.get(l) in ("IA", "Dr") and l not in ("hi", "hi_latn") for l in p.split("-"))})
    summary = pe.summarize(rows, indic)
    out = Path(out_dir)
    _write(out, {"rows.csv": rows, "items.csv": items, "summary.csv": summary})
    for lens_name in lenses:
        for metric in ("mean_prob", "top10_rate"):
            try:
                pe.plot_lens(summary, lens_name, metric, out / f"{lens_name}_{metric}.png", sorted(per_pair_kept))
            except Exception:  # plots are a convenience, never a reason to lose the data
                pass
    result = {"kind": kind, "instr": instr if kind == "chat_word" else "", "n_items": len(items), "kept": sum(i["kept"] for i in items),
              "kept_by_pair": dict(per_pair_kept),
              "found_at_step": dict(Counter(i.get("found_at_step") for i in items if i.get("kept"))),
              "all_indic": {name: _curves(summary, name) for name in lenses}}
    if meta_band and n_meta:
        top = [{"token": tokenizer.decode([tid]), "id": tid, "prevalence": round(n / n_meta, 3)}
               for tid, n in meta_counts.most_common(40)]
        (out / "meta_tokens.json").write_text(json.dumps({"n_items": n_meta, "band": meta_band, "top": top},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
        result["meta_tokens_top10"] = top[:10]
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


### 2. ENGLISH-COLLAPSE ERRORS


def run_collapse(model, lenses: dict, *, out_dir: str, vocab_size: int, layers: list[int], table: str = "homonyms",
                 max_pairs: int | None = None, batch_size: int = 1) -> dict:
    """``batch_size`` > 1 reads that many prompts off one forward (:meth:`Reader.forward_batch`)."""
    tokenizer = model.tokenizer
    index = pe.build_vocab_index(tokenizer, vocab_size)
    concepts = pe.load_concepts(DATA / "concepts.tsv")
    if table == "homonyms":
        entries, langs = load_table("homonyms.tsv", "sense", HOMONYM_LANGS), HOMONYM_LANGS
    else:
        entries, langs = load_table("kinship.tsv", "kin", KIN_LANGS), KIN_LANGS
    keys = sorted(entries)
    groups = defaultdict(list)
    for k in keys:
        groups[entries[k]["group"]].append(k)
    tracked = set()
    for e in entries.values():
        for lang in (*langs, "en", "zh"):
            tracked |= pe.prefix_token_ids(index, e.get(lang, []), tokenizer)
    reader = Reader(model, lenses, layers, vocab_size, pe.readout_mask(tokenizer, vocab_size, tracked))
    last_layer = model.n_layers - 1
    pairs = [(s, g) for s in langs for g in langs if s != g][: max_pairs or None]
    specs = []  # (src, tgt, key, siblings, control, sets, prompt ids), in the original item order
    for src, tgt in pairs:
        for key in keys:
            e = entries[key]
            if not (e.get(src) and e.get(tgt)):
                continue
            siblings = [k for k in groups[e["group"]] if k != key and entries[k].get(tgt)]
            # matched control: a word from ANOTHER English group (fixed derangement), same languages
            others = [k for k in keys if entries[k]["group"] != e["group"] and entries[k].get(tgt)]
            control = others[(keys.index(key) * 7) % len(others)] if others else None
            raw = {"target": pe.prefix_token_ids(index, e[tgt], tokenizer),
                   "sibling": set().union(*(pe.prefix_token_ids(index, entries[k][tgt], tokenizer) for k in siblings)) if siblings else set(),
                   "control": pe.prefix_token_ids(index, entries[control][tgt], tokenizer) if control else set(),
                   "english": pe.prefix_token_ids(index, e["en"], tokenizer),
                   "source": pe.prefix_token_ids(index, e[src], tokenizer)}
            sets, _ = pe.disjoint_sets(raw)
            prompt = fewshot_prompt(concepts, e[src][0], src, tgt, key=key)
            specs.append((src, tgt, key, siblings, control, sets, model.encode(prompt)))
    rows, items = [], []
    for start in range(0, len(specs), batch_size):
        chunk = specs[start: start + batch_size]
        for (src, tgt, key, siblings, control, sets, _), (acts, logits) in zip(chunk, reader.forward_batch([s[-1] for s in chunk])):
            e = entries[key]
            top = int(logits.argmax())
            outcome = next((role for role in ("target", "sibling", "control") if top in sets[role]), "other")
            probs = t.softmax(logits[:vocab_size], -1)
            p_final = {role: float(probs[t.tensor(sorted(ids), device=probs.device)].sum()) if ids else float("nan")
                       for role, ids in sets.items()}
            items.append({"table": table, "pair": f"{src}-{tgt}", "key": key, "group": e["group"], "control": control,
                          "n_siblings": len(siblings), "model_top": tokenizer.decode([top]), "outcome": outcome,
                          **{f"p_final_{r}": v for r, v in p_final.items()}})
            for lens_name, layer, role, prob, rank, top10 in reader.score(acts, sets):
                rows.append({"table": table, "pair": f"{src}-{tgt}", "key": key, "group": e["group"], "outcome": outcome,
                             "lens": lens_name, "layer": layer, "depth": layer / last_layer, "role": role,
                             "prob": prob, "best_rank": rank, "top10": top10})
    out = Path(out_dir)
    _write(out, {f"{table}_rows.csv": rows, f"{table}_items.csv": items})
    has_sib = [i for i in items if i["n_siblings"]]
    no_sib = [i for i in items if not i["n_siblings"]]  # kinship words English keeps distinct (father, son...)
    result = {"table": table, "n_items": len(items), "n_with_sibling": len(has_sib),
              "accuracy_with_sibling": round(float(np.mean([i["outcome"] == "target" for i in has_sib])), 3) if has_sib else None,
              "accuracy_without_sibling": round(float(np.mean([i["outcome"] == "target" for i in no_sib])), 3) if no_sib else None,
              "by_group": {g: {"n": len(v), "accuracy": round(float(np.mean([i["outcome"] == "target" for i in v])), 3),
                               "sibling_rate": round(float(np.mean([i["outcome"] == "sibling" for i in v])), 3)}
                           for g, v in sorted(_group(items).items())},
              "outcome_rate": {o: round(float(np.mean([i["outcome"] == o for i in has_sib])), 3) if has_sib else None
                               for o in ("target", "sibling", "control", "other")},
              "mean_p_final": {r: round(float(np.nanmean([i[f"p_final_{r}"] for i in has_sib])), 4) if has_sib else None
                               for r in ("target", "sibling", "control")}}
    # top-10 rates by layer for the sibling vs control competitors (the collapse signature)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["lens"], r["layer"], r["role"])].append(r["top10"])
    result["top10_by_layer"] = {f"{lens}|{role}": {l: round(float(np.mean(agg[(lens, l, role)])), 3) for l in layers
                                                   if agg.get((lens, l, role))}
                                for lens in lenses for role in ("target", "english", "sibling", "control")}
    (out / f"{table}_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


### 3. THINKING MODE


def script_shares(text: str) -> dict[str, float]:
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        for name, (lo, hi) in SCRIPTS.items():
            if lo <= cp <= hi and (name != "latin" or ch.isalpha()):
                counts[name] += 1
                break
    total = sum(counts.values()) or 1
    return {name: round(counts[name] / total, 3) for name in SCRIPTS}


def run_think(model, *, out_dir: str, pairs: str = "ta-te,bn-gu,kn-ml,mr-bn,te-kn,ml-ta", per_pair: int = 4,
              max_new_tokens: int = 200) -> dict:
    """Thinking mode ON (the model card's default), word requests in the source language. Greedy,
    KV-cached decoding. Records the visible reasoning's script mix, whether the English word appears
    in it, and whether the final answer contains the target word."""
    from transformers import DynamicCache

    tokenizer, hf = model.tokenizer, model._hf_model
    concepts = pe.load_concepts(DATA / "concepts.tsv")
    prompts = load_prompts()
    end_ids = {i for i in (tokenizer.convert_tokens_to_ids(x) for x in ("<|end_of_turn|>",))
               if isinstance(i, int) and i >= 0 and i != tokenizer.unk_token_id}
    if tokenizer.eos_token_id is not None:
        end_ids.add(tokenizer.eos_token_id)
    ordered = sorted(concepts)
    items = []
    for pair in pairs.split(","):
        src, tgt = pair.split("-")
        chosen = [ordered[(i * 5 + 3) % len(ordered)] for i in range(per_pair)]
        for concept in chosen:
            words = concepts[concept]
            if src not in words or tgt not in words:
                continue
            ids = chat_ids(model, chat_text(tokenizer, word_request(prompts, words[src][0], src, tgt, "src"), thinking=True))
            generated: list[int] = []
            with t.no_grad():
                cache = DynamicCache()
                out = hf(input_ids=ids, past_key_values=cache, use_cache=True,
                         attention_mask=t.ones_like(ids))
                total = ids.shape[1]
                for _ in range(max_new_tokens):
                    nxt = int(out.logits[0, -1].argmax())
                    generated.append(nxt)
                    if nxt in end_ids:
                        break
                    total += 1
                    step = t.tensor([[nxt]], device=ids.device)
                    out = hf(input_ids=step, past_key_values=out.past_key_values, use_cache=True,
                             attention_mask=t.ones(1, total, dtype=t.long, device=ids.device))
            text = tokenizer.decode(generated, skip_special_tokens=False)
            reasoning, _, answer = text.partition("</think>")
            reasoning = reasoning.replace("<think>", "")
            norm = lambda s: unicodedata.normalize("NFC", s).casefold()
            en_words = [norm(w) for w in words["en"]]
            items.append({"pair": pair, "concept": concept, "n_tokens": len(generated), "closed_think": "</think>" in text,
                          "english_in_reasoning": any(re.search(rf"\b{re.escape(w)}\b", norm(reasoning)) for w in en_words),
                          "target_in_answer": any(norm(w) in norm(answer) for w in words[tgt]),
                          "target_in_reasoning": any(norm(w) in norm(reasoning) for w in words[tgt]),
                          **{f"script_{k}": v for k, v in script_shares(reasoning).items()},
                          "reasoning": reasoning[:1500], "answer": answer[:300]})
    out = Path(out_dir)
    _write(out, {"think_items.csv": items})
    mean = lambda key: round(float(np.mean([i[key] for i in items])), 3) if items else None
    result = {"n_items": len(items), "closed_think": mean("closed_think"), "english_in_reasoning": mean("english_in_reasoning"),
              "target_in_reasoning": mean("target_in_reasoning"), "target_in_answer": mean("target_in_answer"),
              "reasoning_script_share": {k: mean(f"script_{k}") for k in SCRIPTS},
              "mean_tokens": mean("n_tokens")}
    (out / "think_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
