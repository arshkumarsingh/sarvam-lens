"""Decode checks for post1_experiment.generate_cb on the tiny 30B and tiny 105B (CPU, fp32, or 2 GPUs with --two-gpu).

1. greedy generate_cb == lone greedy decoding (batch_gen.generate_one), token for token, under several batch sizes;
2. in-slab forced answers == re-reading the trace (prompt + trace + cue, greedy), token for token;
3. deadline forcing: every unclosed row gets a forced answer, equal to the re-read one.

    python post1_check_decode.py <code dir> <tiny 30B dir or -> <tiny 105B dir or -> [--two-gpu]
"""
import sys
import time

code, tiny30, tiny105 = sys.argv[1:4]
two_gpu = "--two-gpu" in sys.argv
sys.path.insert(0, code)

import torch as t

import run_jpp

run_jpp._import_jpp_cli()  # jpp's src/ (jlens) on sys.path, as in the Modal image

import batch_gen
import post1_experiment as p1
import sarvam_support

fails = []


def check(name, ok, detail=""):
    print(("  ok   " if ok else "  FAIL ") + name, detail, flush=True)
    if not ok:
        fails.append(name)


def load(path, mla):
    if two_gpu and mla:  # split as smoke_test_mla --two-gpu does (device_map="auto" with an explicit map); fp32 for exactness
        import json
        import os
        n = json.load(open(f"{path}/config.json"))["num_hidden_layers"]
        dm = {"model.embed_tokens": 0, **{f"model.layers.{i}": int(i >= n // 2) for i in range(n)}, "model.norm": 1, "lm_head": 1}
        os.environ[sarvam_support.DEVICE_MAP_ENV] = json.dumps(dm)
        return sarvam_support.load_sarvam_lens_model(path, dtype=t.float32, device_map="auto")
    return sarvam_support.load_sarvam_lens_model(path, dtype=t.float32, device="cuda" if two_gpu else "cpu")


for label, path, mla in [x for x in (("tiny 30B", tiny30, False), ("tiny 105B", tiny105, True)) if x[1] != "-"]:
    model = load(path, mla)
    tok = model.tokenizer
    print(label, type(model._hf_model).__name__, "MLA" if p1._is_mla(model) else "GQA", "devices", sorted({str(d) for d in p1._cache_layout(model)[2]}))
    prompts = ["भारत की राजधानी", "The capital of Tamil Nadu is", 'தமிழ்: "நீர்" - తెలుగు: "', "a b c d e f g h i j k l m n o p",
               "Hello", "पानी", "x" * 40, "The river flows to the sea and then"]
    rows = [tok(p).input_ids for p in prompts]
    end_ids = batch_gen.end_ids_of(tok)
    dev = model.input_device
    lone = [batch_gen.generate_one(model, t.tensor([r], device=dev), max_new_tokens=20, end_ids=end_ids) for r in rows]
    for max_rows, admit in ((3, 2), (8, 8), (1, 1)):
        cb, _, stats = p1.generate_cb(model, rows, max_new_tokens=20, end_ids=end_ids, temperature=0.0, max_rows=max_rows,
                                      admit_rows=admit, loop_check_every=0)
        same = [a == b for a, b in zip(lone, cb)]
        check(f"{label}: generate_cb == lone greedy (rows {max_rows}, admit {admit})", all(same), f"{sum(same)}/{len(same)}")
    # in-slab forcing at the cap vs re-reading the trace
    tail = tok("</think>\n\n" + p1.FORCE_CUE, add_special_tokens=False).input_ids
    for max_rows in (3, 8):
        gens, reasons, stats = p1.generate_cb(model, rows, max_new_tokens=10, end_ids=end_ids, temperature=0.0, max_rows=max_rows,
                                              admit_rows=2, loop_check_every=0, force_tail=tail, force_new=6, think_end=None)
        ok, n = 0, 0
        for i, (r, g) in enumerate(zip(rows, gens)):
            if str(i) not in stats["forced"]:
                continue
            n += 1
            ref = batch_gen.generate_one(model, t.tensor([r + g + tail], device=dev), max_new_tokens=6, end_ids=end_ids)
            ok += stats["forced"][str(i)][len(tail):] == ref
        check(f"{label}: in-slab forced answers == re-read (rows {max_rows})", n > 0 and ok == n, f"{ok}/{n} forced; reasons {stats['reasons']}")
    # deadline forcing: the deadline falls mid-decode
    rows2 = rows * 3
    gens, reasons, stats = p1.generate_cb(model, rows2, max_new_tokens=60, end_ids=end_ids, temperature=0.0, max_rows=4, admit_rows=2,
                                          loop_check_every=0, force_tail=tail, force_new=6, deadline=time.monotonic() + 0.4)
    forced_rows = [i for i in range(len(rows2)) if str(i) in stats["forced"]]
    ok = 0
    for i in forced_rows:
        ref = batch_gen.generate_one(model, t.tensor([rows2[i] + gens[i] + tail], device=dev), max_new_tokens=6, end_ids=end_ids)
        ok += stats["forced"][str(i)][len(tail):] == ref
    check(f"{label}: deadline forcing", forced_rows and ok == len(forced_rows) and "not_run" in reasons,
          f"{ok}/{len(forced_rows)} forced equal; reasons {stats['reasons']}")
    # deadline already past at a moment with no live rows (forced answers on): it used to call H.to() with H None
    try:
        gens, reasons, stats = p1.generate_cb(model, rows, max_new_tokens=10, end_ids=end_ids, temperature=0.0, max_rows=2,
                                              admit_rows=2, loop_check_every=0, force_tail=tail, force_new=4,
                                              deadline=time.monotonic() - 1)
        check(f"{label}: deadline with no live rows", reasons == ["not_run"] * len(rows) and not stats["forced"],
              f"reasons {stats['reasons']}")
    except Exception as e:  # noqa: BLE001
        check(f"{label}: deadline with no live rows", False, f"{type(e).__name__}: {e}")
    # sampling runs
    cb, _, stats = p1.generate_cb(model, rows, max_new_tokens=20, end_ids=end_ids, temperature=0.8, top_p=0.95, max_rows=3,
                                  force_tail=tail)
    check(f"{label}: sampling + forcing runs", all(len(g) > 0 for g in cb), str(stats["reasons"]))
    del model
print("FAILS:", fails or "none")
sys.exit(1 if fails else 0)
