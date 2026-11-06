"""Is a batched-vs-lone mismatch a padding bug or bf16 batch-variance? (docs/03 Phase 13)

``batch_check`` warned on Sarvam-105B (batched and lone outputs differed) but not on the 30B. Run on 2 GPUs (``modal run modal_app.py --step batch-probe``), this probe takes the
tiny random 105B-shaped model, split over the GPUs as device_map="auto" splits the real one, and the
tiny 30B on one GPU, in fp32 and in bf16, and compares batched with lone runs four ways:
  padded     Reader.forward_batch (right-padded) vs Reader.forward, on prompts of different lengths
  generate   batch_gen.generate (left-padded, masked) vs lone greedy decoding, 12 tokens
  no_pad     a batch of prompts cut to one common length: no padding anywhere
  replicated one prompt 8 times in a batch: no padding, identical rows
Reading: in fp32 every comparison must be exact if the batching logic is right. In bf16, if
no_pad/replicated also differ from lone, the mismatch comes from batch shape (GEMM tiling, amplified
by MoE routing flips), not from padding.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch as t

import batch_gen
import reader_questions as rqm
import sarvam_support


def _readout_compare(reader, ids_list) -> dict:
    lone = [reader.forward(ids)[1] for ids in ids_list]
    batched = [logits for _, logits in reader.forward_batch(ids_list)]
    top1 = [int(a.argmax()) == int(b.argmax()) for a, b in zip(lone, batched)]
    dlp = [float((t.log_softmax(a, -1) - t.log_softmax(b, -1)).abs().max()) for a, b in zip(lone, batched)]
    # KL(lone || batched): the size of the difference where the probability actually is
    kl = [float(t.sum(t.softmax(a, -1) * (t.log_softmax(a, -1) - t.log_softmax(b, -1)))) for a, b in zip(lone, batched)]
    return {"top1_agree": round(sum(top1) / len(top1), 3), "max_abs_dlogprob": round(max(dlp), 5),
            "max_kl": round(max(kl), 6), "rows_identical": None}


def probe(model, label: str) -> dict:
    tok = model.tokenizer
    reader = rqm.Reader(model, {}, [], model._lm_head.weight.shape[0], None)
    ids_list = [tok(p, return_tensors="pt").input_ids.to(model.input_device) for p in batch_gen.CHECK_PROMPTS]
    out = {"padded": _readout_compare(reader, ids_list)}
    gen = batch_gen.check(model, batch_gen.CHECK_PROMPTS, max_new_tokens=12, batch_size=8)
    out["generate"] = {"exact_match_rate": gen["exact_match_rate"]}
    n = min(int(i.shape[1]) for i in ids_list)
    out["no_pad"] = _readout_compare(reader, [i[:, :n] for i in ids_list])
    same = [ids_list[0]] * 8
    rows = [logits for _, logits in reader.forward_batch(same)]
    rep = _readout_compare(reader, same)
    rep["rows_identical"] = all(bool(t.equal(rows[0], r)) for r in rows[1:])
    out["replicated"] = rep
    return {"model": label, **out}


def main(workdir: str = "/tmp/batch_probe") -> dict:
    import run_jpp

    run_jpp._import_jpp_cli()  # puts jpp's src/ (jlens) on sys.path inside the Modal image
    import smoke_test
    import smoke_test_mla

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    results = []
    two_gpu = t.cuda.device_count() >= 2
    mla_dir, *_ = smoke_test_mla.build_tiny_mla(work)
    if two_gpu:
        n_layers = smoke_test_mla.N_LAYERS
        split = {"model.embed_tokens": 0, **{f"model.layers.{i}": int(i >= n_layers // 2) for i in range(n_layers)},
                 "model.norm": 1, "lm_head": 1}
        os.environ[sarvam_support.DEVICE_MAP_ENV] = json.dumps(split)
    for dtype in (t.float32, t.bfloat16):
        m = sarvam_support.load_sarvam_lens_model(str(mla_dir), dtype=dtype, device_map="auto" if two_gpu else None,
                                                  device=None if two_gpu else ("cuda" if t.cuda.is_available() else "cpu"))
        results.append(probe(m, f"tiny-105B {'2 GPUs' if two_gpu else '1 device'} {str(dtype)[6:]}"))
        del m
        t.cuda.empty_cache()
    os.environ.pop(sarvam_support.DEVICE_MAP_ENV, None)
    moe_dir = smoke_test.build_tiny_sarvam(work)
    for dtype in (t.float32, t.bfloat16):
        m = sarvam_support.load_sarvam_lens_model(str(moe_dir), dtype=dtype, device="cuda" if t.cuda.is_available() else "cpu")
        results.append(probe(m, f"tiny-30B 1 GPU {str(dtype)[6:]}"))
        del m
        t.cuda.empty_cache()
    return {"results": results}


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:]), indent=1))
