"""Batched greedy decoding with a KV cache, for the cross-lingual stages (docs/11).

:func:`generate` sorts the prompts by length, left-pads each batch, and passes the attention mask
and explicit position ids (each row counts from 0 at its first real token), so a row sees exactly
what it would see alone. In fp32 a row matches decoding that prompt alone token for token (checked
on the tiny Sarvam-30B and Sarvam-105B models by ``python batch_gen.py <tiny dir>``). In bf16 on a
GPU, batch shape changes kernel tiling, so rows can drift from lone decoding by numerics; the
``batch_check`` stage in modal_pilot measures that on the real model before any results are used.

(:meth:`reader_questions.Reader.forward_batch` is the readout counterpart: right-padded, one forward,
read at each row's last real token.)
"""

from __future__ import annotations

import torch as t


def end_ids_of(tokenizer, extra: tuple[str, ...] = ("<|end_of_turn|>",)) -> set[int]:
    """EOS plus the chat template's end-of-turn token, where the tokenizer has them."""
    ids = {tokenizer.convert_tokens_to_ids(x) for x in extra}
    ids = {i for i in ids if isinstance(i, int) and i >= 0 and i != tokenizer.unk_token_id}
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def generate(model, ids_list: list[t.Tensor], *, max_new_tokens: int, end_ids: set[int] | None = None,
             batch_size: int = 16, pad_id: int = 0) -> list[list[int]]:
    """Greedy continuations of ``ids_list`` (each ``[1, n]``) on ``model`` (a jlens HFLensModel).
    Returns each prompt's generated ids, in input order, up to and including the first end id."""
    from transformers import DynamicCache

    hf = model._hf_model
    end_ids = set(end_ids or ())
    out: list[list[int] | None] = [None] * len(ids_list)
    order = sorted(range(len(ids_list)), key=lambda i: int(ids_list[i].shape[1]))  # less padding per batch
    for start in range(0, len(order), batch_size):
        rows = order[start: start + batch_size]
        seqs = [ids_list[i][0] for i in rows]
        width, device = max(len(s) for s in seqs), seqs[0].device
        input_ids = t.full((len(rows), width), pad_id, dtype=t.long, device=device)
        mask = t.zeros((len(rows), width), dtype=t.long, device=device)
        for r, s in enumerate(seqs):
            input_ids[r, width - len(s):] = s
            mask[r, width - len(s):] = 1
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        generated: list[list[int]] = [[] for _ in rows]
        done = [False] * len(rows)
        with t.no_grad():
            step_out = hf(input_ids=input_ids, attention_mask=mask, position_ids=positions,
                          past_key_values=DynamicCache(config=hf.config), use_cache=True)
            for step in range(max_new_tokens):
                nxt = step_out.logits[:, -1].argmax(-1).to(device)  # logits may sit on the last GPU
                for r in range(len(rows)):
                    if not done[r]:
                        generated[r].append(int(nxt[r]))
                        done[r] = int(nxt[r]) in end_ids
                if all(done) or step == max_new_tokens - 1:
                    break
                mask = t.cat([mask, mask.new_ones((len(rows), 1))], dim=1)
                positions = positions[:, -1:] + 1
                step_out = hf(input_ids=nxt[:, None], attention_mask=mask, position_ids=positions,
                              past_key_values=step_out.past_key_values, use_cache=True)
        for r, i in enumerate(rows):
            out[i] = generated[r]
    return out  # type: ignore[return-value]


def generate_one(model, ids: t.Tensor, *, max_new_tokens: int, end_ids: set[int] | None = None) -> list[int]:
    """The unbatched reference: :func:`generate`'s answer for one prompt, without padding."""
    from transformers import DynamicCache

    hf, end_ids, generated = model._hf_model, set(end_ids or ()), []
    with t.no_grad():
        step_out = hf(input_ids=ids, attention_mask=t.ones_like(ids), past_key_values=DynamicCache(config=hf.config), use_cache=True)
        total = ids.shape[1]
        for _ in range(max_new_tokens):
            nxt = int(step_out.logits[0, -1].argmax())
            generated.append(nxt)
            if nxt in end_ids or len(generated) == max_new_tokens:
                break
            total += 1
            step_out = hf(input_ids=t.tensor([[nxt]], device=ids.device), past_key_values=step_out.past_key_values,
                          use_cache=True, attention_mask=t.ones(1, total, dtype=t.long, device=ids.device))
    return generated


def check(model, prompts: list[str], *, max_new_tokens: int = 12, batch_size: int = 4) -> dict:
    """Batched vs lone decoding on ``prompts``: the share of rows whose tokens match exactly, and
    the first mismatch. The ``batch_check`` stage and ``python batch_gen.py`` both call this."""
    ids_list = [model.tokenizer(p, return_tensors="pt").input_ids.to(model.input_device) for p in prompts]
    end_ids = end_ids_of(model.tokenizer)
    batched = generate(model, ids_list, max_new_tokens=max_new_tokens, end_ids=end_ids, batch_size=batch_size)
    alone = [generate_one(model, ids, max_new_tokens=max_new_tokens, end_ids=end_ids) for ids in ids_list]
    same = [a == b for a, b in zip(batched, alone)]
    first_bad = next((i for i, ok in enumerate(same) if not ok), None)
    return {"n": len(prompts), "exact_match_rate": round(sum(same) / len(same), 3),
            "first_mismatch": None if first_bad is None else
            {"prompt": prompts[first_bad][:80], "batched": batched[first_bad], "alone": alone[first_bad]}}


CHECK_PROMPTS = [
    "भारत की राजधानी",
    "The capital of Tamil Nadu is",
    "தமிழ்நாட்டின் தலைநகரம் சென்னை. கேரளாவின் தலைநகரம்",
    "ଓଡ଼ିଶାର ରାଜଧାନୀ",
    "Q: 3 + 4 = ?\nA:",
    "ꯃꯅꯤꯄꯨꯔꯒꯤ ꯔꯖꯙꯥꯅꯤ",
    "اردو ایک زبان ہے اور",
    "Write one sentence about rivers in India, then stop.",
]


if __name__ == "__main__":  # python batch_gen.py <tiny sarvam dir> (fp32 on CPU: rows must match exactly)
    import json
    import sys

    import sarvam_support

    m = sarvam_support.load_sarvam_lens_model(sys.argv[1], dtype=t.float32, device="cpu")
    print(json.dumps(check(m, CHECK_PROMPTS), ensure_ascii=False, indent=1))
