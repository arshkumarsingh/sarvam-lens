"""Regenerate every results table in docs/04_results_tables.md from the raw files in results/.

    python code/summarize_results.py            # from the project root

Nothing here is typed by hand: each table is computed from the run outputs (report.json,
summary.csv, *_summary.json, CKA npz/json), so the docs cannot drift from the data.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "docs" / "04_results_tables.md"
USD_PER_HOUR_H200 = 4.54 + 8 * 0.0472 + 32 * 0.0080


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def table(headers: list[str], rows: list[list]) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def section_runs() -> str:
    rows = []
    for run in sorted(p for p in R.iterdir() if (p / "report.json").is_file()):
        rep = json.loads((run / "report.json").read_text(encoding="utf-8"))
        stages = ", ".join(f"{s['stage']}({s['status'][:2]})" for s in rep["stages"])
        rows.append([run.name, rep.get("elapsed_minutes", ""), rep.get("est_cost_usd_so_far", ""), stages])
    return "## A. Every GPU run (from each run's report.json)\n\n" + table(
        ["run tag", "minutes in driver", "est. $ (driver time only)", "stages (status)"], rows) + (
        "\n\nDriver minutes exclude container start and model-load overhead before the driver begins "
        "(~1-2 min per run). Each `jfit_sN` is one of the 5 parallel J-lens shards.\n")


def section_sanity() -> str:
    rep = json.loads((R / "ropefix/report.json").read_text(encoding="utf-8"))
    st = {s["stage"]: s.get("result") or {} for s in rep["stages"]}
    out = ["## B. Real-model sanity and BOS probe (run `ropefix`, after the RoPE fix)\n"]
    san = st["sanity"]
    out.append(f"- BOS prepended: {san['bos_first']}; stock forward re-run max |diff|: {san.get('stock_rerun_max_abs_diff')}")
    for mode in ("rlens", "r+moe"):
        m = san[mode]
        out.append(f"- LRP `{mode}`: forward max |diff| {m['max_abs_diff']} (stock |max| {m['stock_abs_max']}); patched {m['patched']}")
    out.append("\nGreedy continuations (12 tokens):\n")
    out.append(table(["prompt", "continuation"], [[repr(k), repr(v)] for k, v in san["greedy"].items()]))
    bos = st["bos_probe"]
    out.append("\nBOS probe: 5-shot word translation, top-1 begins the target word (5 pairs x 32 concepts):\n")
    out.append(table(["bos mode", "accuracy", "per pair", "greedy Hindi"],
                     [[m, bos[m]["accuracy"], bos[m]["per_pair"], repr(bos[m]["greedy_hindi"])] for m in ("hf", "chat", "sarvam_bos")]))
    out.append(f"\nChosen: `{bos['chosen']}`.\n")
    pilot = json.loads((R / "pilot/report.json").read_text(encoding="utf-8"))
    for s in pilot["stages"]:
        if s["stage"] in ("fit_shard_j", "fit_shard_r"):
            res = s["result"]
            out.append(f"- Timing ({s['stage']}, run `pilot`): {res['seconds_per_prompt_upper_bound']} s/prompt, "
                       f"projected 64-prompt fit {res['projected_64_prompt_fit']}")
    rows_rep = json.loads((R / "rows/report.json").read_text(encoding="utf-8"))
    probe = next(s for s in rows_rep["stages"] if s["stage"] == "rows_probe")["result"]
    out.append(f"- Rows-per-pass probe (run `rows`): {probe}")
    return "\n".join(out) + "\n"


def pivot_tables(summary_path: Path, title: str, layers: range, roles: tuple[str, ...]) -> str:
    rows = read_csv(summary_path)
    out = [f"## {title}\n"]
    for lens in sorted({r["lens"] for r in rows}):
        g = {(int(r["layer"]), r["role"]): r for r in rows if r["lens"] == lens and r["pair"] == "ALL_INDIC"}
        if not g:
            continue
        body = []
        for role in roles:
            if any((l, role) in g for l in layers):
                body.append([role] + [f"{float(g[(l, role)]['top10_rate']):.2f} ({float(g[(l, role)]['mean_prob']):.2f})"
                                      if (l, role) in g else "" for l in layers])
        out.append(f"**Lens `{lens}`**, all Indic→Indic pairs: top-10 rate (mean probability)\n")
        out.append(table(["role"] + [f"L{l}" for l in layers], body) + "\n")
    return "\n".join(out)


def section_per_pair(summary_path: Path, items_path: Path, lens: str) -> str:
    rows = read_csv(summary_path)
    items = read_csv(items_path)
    kept = defaultdict(int)
    total = defaultdict(int)
    for i in items:
        total[i["pair"]] += 1
        kept[i["pair"]] += i["kept"] == "True"
    pairs = sorted({r["pair"] for r in rows if r["pair"] != "ALL_INDIC"})
    body = []
    for p in pairs:
        g = {(int(r["layer"]), r["role"]): r for r in rows if r["lens"] == lens and r["pair"] == p}
        en = [(float(g[(l, "english")]["mean_prob"]), l) for l in range(18) if (l, "english") in g]
        peak = max(en) if en else (float("nan"), -1)
        t10 = lambda l, role: float(g[(l, role)]["top10_rate"]) if (l, role) in g else float("nan")
        body.append([p, f"{kept[p]}/{total[p]}", f"{peak[0]:.2f} @L{peak[1]}", t10(15, "english"), t10(15, "hindi"),
                     t10(15, "english_shuffled"), t10(17, "target")])
    return (f"## D. Per pair (lens `{lens}`)\n\n" + table(
        ["pair", "kept", "peak P(English)", "top10 English L15", "top10 Hindi L15", "top10 shuffled-English L15",
         "top10 target L17"], body) + "\n\n'hindi' is absent where Hindi is the source; 'english' where English is.\n")


def section_causal() -> str:
    out = ["## E. Causal tests\n"]
    body = []
    for name in ("causal_j_200p", "causal_r_moe_25p"):
        s = json.loads((R / "final" / name / "causal_summary.json").read_text())
        for cond in ("english_mid", "control_mid", "english_early"):
            c = s[cond]
            body.append([name.replace("causal_", ""), "English", cond, s["n_items"], round(s["base_mean_p"], 3),
                         round(-c["mean_drop"], 4), f"[{-c['drop_ci95'][1]:.4f}, {-c['drop_ci95'][0]:.4f}]", round(c["still_correct"], 3)])
    h = json.loads((R / "extra/hindi_ablation_j_200p/causal_summary.json").read_text())
    for cond in ("english_mid", "control_mid", "english_early"):
        c = h[cond]
        body.append(["j_200p", "Hindi", cond.replace("english", "latent"), h["n_items"], round(h["base_mean_p"], 3),
                     round(-c["mean_drop"], 4), f"[{-c['drop_ci95'][1]:.4f}, {-c['drop_ci95'][0]:.4f}]", round(c["still_correct"], 3)])
    out.append("**Subspace ablation** (final position; mid = layers 13-16, early = 5-8; control = a different concept's "
               "same-language word):\n")
    out.append(table(["lens dirs", "latent", "condition", "n", "base P(target)", "ΔP(target)", "95% CI", "still correct"], body))
    sw = json.loads((R / "extra/swap_j_200p/swap_summary.json").read_text())
    body = [[k, v["n"], v["flip_to_B_rate"], v["still_A_rate"], v["mean_pB_gain"]]
            for k, v in sorted(sw.items()) if isinstance(v, dict) and "flip_to_B_rate" in v]
    out.append("\n**Concept swap** (J++ clamp-swap convention, J-lens directions; success = answer becomes concept B's "
               "target-language word):\n")
    out.append(table(["condition@scale", "n", "flip to B", "still A", "ΔP(B)"], body))
    return "\n".join(out) + "\n"


def section_sentences() -> str:
    rows = read_csv(R / "extra/sentences/sentence_rows.csv")
    items = read_csv(R / "extra/sentences/sentence_items.csv")
    out = [f"## F. Sentence clozes ({sum(i['kept'] == 'True' for i in items)}/{len(items)} kept)\n"]
    g = defaultdict(list)
    for r in rows:
        g[(r["lens"], int(r["layer"]), r["role"])].append(r["top10"] == "True")
    for lens in sorted({r["lens"] for r in rows}):
        body = [[role] + [f"{np.mean(g[(lens, l, role)]):.2f}" if g[(lens, l, role)] else "" for l in range(9, 18)]
                for role in ("english", "english_shuffled", "hindi", "hindi_shuffled", "target")]
        out.append(f"**Lens `{lens}`**: top-10 rate\n")
        out.append(table(["role"] + [f"L{l}" for l in range(9, 18)], body) + "\n")
    out.append(table(["lang", "concept", "model's next token", "kept"],
                     [[i["lang"], i["concept"], repr(i["model_top"]), i["kept"]] for i in items]))
    return "\n".join(out) + "\n"


def section_cka() -> str:
    out = ["## H. J-space CKA (explorer method)\n"]
    body = []
    for d in ("cka", "cka_robust"):
        for f in sorted((R / d).glob("*_blocks.json")):
            b = json.loads(f.read_text())
            body.append([f.stem.replace("_blocks", ""), b.get("gain_mode", "auto"), b.get("n_prompts"), b["n_blocks"],
                         b.get("boundaries_rho"), round(b["blockiness"], 3), round(b["layer_coupling"], 3)])
    out.append(table(["lens", "gain", "prompts", "blocks", "boundaries (ρ)", "blockiness", "layer coupling"], body))
    body = []
    for d in ("cka", "cka_robust"):
        for f in sorted((R / d).glob("cross_*.json")):
            c = json.loads(f.read_text())
            body.append([f.stem, round(c["matched_mean"], 3), c["n_shared_tokens"]])
    out.append("\nCross-model matched-depth CKA:\n")
    out.append(table(["pair", "matched-depth CKA", "shared tokens"], body))
    diag = json.loads((R / "cka/diagnose.json").read_text())
    body = []
    for name, e in diag.items():
        if not isinstance(e, dict) or "layer_coupling" not in e:
            continue
        body.append([name, e["gain_median"], e["gain_max"], e.get("J_top_sv_share", ""),
                     {k: round(v[0], 1) for k, v in e["V_pr_top1_by_layer"].items()},
                     {k: round(v, 3) for k, v in e["layer_coupling"].items()}])
    out.append("\nDiagnostics (`results/cka/diagnose.json`, last diagnostic run):\n")
    out.append(table(["lens", "gain median", "gain max", "J top-SV energy (mid layer)", "steering-vector PR by layer",
                      "coupling: standard / no gain / minus PC1"], body))
    return "\n".join(out) + "\n"


def _all_indic(summary_path: Path, lens: str = "logit") -> dict[tuple[int, str], dict]:
    return {(int(r["layer"]), r["role"]): r for r in read_csv(summary_path) if r["lens"] == lens and r["pair"] == "ALL_INDIC"}


def _profile(g: dict[tuple[int, str], dict], n_layers: int) -> list:
    """Peak English (top-10 rate, P, layer, relative depth), English onset (first layer with
    top-10 >= 0.5), Hindi peak top-10, target top-10 at the last readout layer."""
    layers = sorted({l for l, _ in g})
    last = n_layers - 1
    t10 = lambda l, role: float(g[(l, role)]["top10_rate"]) if (l, role) in g else float("nan")
    prob = lambda l, role: float(g[(l, role)]["mean_prob"]) if (l, role) in g else float("nan")
    peak = max(layers, key=lambda l: (prob(l, "english"), t10(l, "english")))
    onset = next((l for l in layers if t10(l, "english") >= 0.5), None)
    hindi_peak = max(layers, key=lambda l: t10(l, "hindi"))
    return [f"{t10(peak, 'english'):.2f}", f"{prob(peak, 'english'):.2f}", f"L{peak} ({peak / last:.2f})",
            "none" if onset is None else f"L{onset} ({onset / last:.2f})",
            f"{t10(hindi_peak, 'hindi'):.2f} @L{hindi_peak} ({hindi_peak / last:.2f})",
            f"{max(t10(l, 'english_shuffled') for l in layers):.2f}", f"{t10(layers[-1], 'target'):.2f}"]


def _causal_rows(label: str, summary: dict) -> list[list]:
    rows = []
    for cond in ("english_mid", "control_mid", "english_early"):
        c = summary[cond]
        band = summary["mid_layers"] if cond.endswith("mid") else summary["early_layers"]
        rows.append([label, cond, f"L{band[0]}-{band[-1]}", summary["n_items"], round(summary["base_mean_p"], 3),
                     round(-c["mean_drop"], 4), f"[{-c['drop_ci95'][1]:.4f}, {-c['drop_ci95'][0]:.4f}]",
                     round(c["still_correct"], 3)])
    return rows


RQ_MODELS = (  # (label, run dir, English band, lenses, original few-shot pivot summary, extra run dirs)
    # rq30 ran out of budget before rq_zh / causal_steer / rq_think; rq30b ran those
    ("Sarvam-30B", "rq30", (13, 14, 15, 16), ("j_200p", "logit"), "final/pivot_all/summary.csv", ("rq30b",)),
)
RQ_ROLES = ("english", "english_shuffled", "hindi", "chinese", "chinese_shuffled", "target")


def _band_rates(summary_path: Path, lens: str, band: tuple[int, ...], pair: str = "ALL_INDIC") -> dict[str, str]:
    """Band-mean top-10 rate (mean probability) per role."""
    rows = [r for r in read_csv(summary_path) if r["pair"] == pair and r["lens"] == lens and int(r["layer"]) in band]
    out = {}
    for role in RQ_ROLES:
        rr = [r for r in rows if r["role"] == role]
        out[role] = (f"{np.mean([float(r['top10_rate']) for r in rr]):.2f} ({np.mean([float(r['mean_prob']) for r in rr]):.3f})"
                     if rr else "")
    return out


def _classify_dropped(items_path: Path, concepts: dict, sentences: dict | None = None) -> dict[str, int]:
    """Why chat items were dropped, from the reply text: the right word split into non-canonical
    tokens (so its first token isn't in the ≥2-character prefix set), the source word echoed, or
    something else. Free: reads items.csv."""
    import unicodedata

    norm = lambda s: unicodedata.normalize("NFC", s).strip().lstrip("\"'“”*`:").strip().casefold()
    counts: dict[str, int] = defaultdict(int)
    for i in read_csv(items_path):
        if i["kept"] == "True":
            counts["kept"] += 1
            continue
        if "shared" in i["reason"]:
            counts["no scorable target tokens"] += 1
            continue
        src, tgt = i["pair"].split("-")
        reply = norm(i.get("generated_prefix", "") + i.get("model_top", ""))
        tgt_words = [norm(w) for w in concepts.get(i["concept"], {}).get(tgt, [])]
        src_words = [norm(w) for w in concepts.get(i["concept"], {}).get(src, [])]
        if sentences:
            tgt_words += [norm(w) for w in sentences.get((tgt, i["concept"]), [])]
            src_words += [norm(w) for w in sentences.get((src, i["concept"]), [])]
        if reply and any(w.startswith(reply[:3]) or reply.startswith(w[:3]) for w in tgt_words if w):
            counts["right word, split differently"] += 1
        elif reply and any(w.startswith(reply[:3]) or reply.startswith(w[:3]) for w in src_words if w):
            counts["echoed the source word"] += 1
        elif not reply:
            counts["ended the turn / blank"] += 1
        else:
            counts["other word"] += 1
    return dict(counts)


def section_reader_questions() -> str:
    """M. Reader questions (docs/09_reader_questions.md): tokenization, the GPT-2 J-lens test, real chat
    use, clozes, English-collapse errors, Chinese skill, negative steering, thinking mode, meta-tokens."""
    out = ["## M. Reader questions (`docs/09_reader_questions.md`)\n"]
    ts = R / "token_split/token_split.json"
    if ts.is_file():
        d = json.loads(ts.read_text(encoding="utf-8"))
        out.append("**M1. Tokenization.** Share of the 32 concept words that are a single token: "
                   + ", ".join(f"{k} {v['single_token_share']:.0%}" for k, v in d["token_counts"].items()) + ".\n")
        body = []
        for label in (k for k in d if k != "token_counts"):
            r = d[label]
            for split, key in (("target word", "by_target_tokens"), ("Hindi word", "by_hindi_tokens")):
                for bucket, v in r[key].items():
                    body.append([label, f"{split}: {bucket}", v["n"], v["english"], v["hindi"], v["target_band"], v["target_last"]])
            e = r["english_minus_hindi_when_hindi_is_1_token"]
            body.append([label, "English − Hindi, Hindi word = 1 token", e["n"], f"{e['mean']} [{e['ci95'][0]}, {e['ci95'][1]}]", "", "", ""])
        out.append(table(["model, lens", "items", "n", "English top-10 (band)", "Hindi top-10 (band)", "target top-10 (band)",
                          "target top-10 (last layer)"], body) + "\n")
    ma = R / "token_split/metric_a_30b.json"
    if ma.is_file():
        d = json.loads(ma.read_text(encoding="utf-8"))
        med = d["median_target_rank_by_layer"]
        out.append(f"**M2. The GPT-2 replication's test on Sarvam-30B** (median rank of the model's own next word, Indic pairs; "
                   f"J-lens better than logit at {d['j_better_layers']}/18 layers):\n")
        out.append(table(["lens"] + [f"L{l}" for l in range(9, 18)],
                         [[lens] + [f"{med[str(l)][lens]:.0f}" for l in range(9, 18)] for lens in ("logit", "j_200p", "r_moe_25p", "jpp_25p")]) + "\n")
    for label, run, band, lenses, original, extra_runs in RQ_MODELS:
        d = R / run
        if not (d / "report.json").is_file():
            out.append(f"*{label}: run `{run}` not available.*\n")
            continue
        rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
        st = {s["stage"]: s for s in rep["stages"]}
        for extra in extra_runs:  # stages that only succeeded in an earlier run (rq_zh)
            if (R / extra / "report.json").is_file():
                for s in json.loads((R / extra / "report.json").read_text(encoding="utf-8"))["stages"]:
                    if s["status"] == "ok" and st.get(s["stage"], {}).get("status") != "ok":
                        st[s["stage"]] = {**s, "from_run": extra}
        out.append(f"### {label} (run `{run}`: driver {rep['elapsed_minutes']} min, est. ${rep['est_cost_usd_so_far']}; "
                   + ", ".join(f"{s['stage']} {s['status'][:20]}" for s in rep["stages"])
                   + (f"; skipped or failed stages come from runs {', '.join(f'`{e}`' for e in extra_runs)}" if extra_runs else "")
                   + ")\n")
        body = []
        fmts = [("few-shot (original pivot)", R / original, "")]
        run_dirs = [d, *(R / e for e in extra_runs)]
        sub_dir = {}  # each format from the first run that has it
        for name, sub in (("chat, word, source-language instruction", "rq_chat_word_src"),
                          ("chat, word, English instruction", "rq_chat_word_en"),
                          ("chat, sentence translation", "rq_chat_sentence"),
                          ("plain sentence cloze (no instruction)", "rq_cloze")):
            src = next((x for x in run_dirs if (x / sub / "result.json").is_file()), None)
            if src:
                sub_dir[sub] = src / sub
                r = json.loads((src / sub / "result.json").read_text(encoding="utf-8"))
                where = "" if src == d else f", run `{src.name}`"
                fmts.append((f"{name} — kept {r['kept']}/{r['n_items']}{where}", src / sub / "summary.csv", r))
        for lens in lenses:
            for fmt, path, _ in fmts:
                if path.is_file():
                    rates = _band_rates(path, lens, band)
                    if any(rates.values()):
                        body.append([lens, fmt] + [rates[r] for r in RQ_ROLES])
        out.append(f"**M3. Real use: band L{band[0]}–{band[-1]}, all Indic→Indic pairs (cloze: ta/te/bn sentences), "
                   "top-10 rate (mean probability)**\n")
        out.append(table(["lens", "format"] + list(RQ_ROLES), body) + "\n")
        with (ROOT / "code/data/concepts.tsv").open(encoding="utf-8") as handle:
            concepts = {r["concept"]: {k: [s for s in v.split("|") if s] for k, v in r.items() if k != "concept"}
                        for r in csv.DictReader(handle, delimiter="\t")}
        with (ROOT / "code/data/sentences_parallel.tsv").open(encoding="utf-8") as handle:
            sent = {(r["lang"], r["concept"]): [s for s in r["target"].split("|") if s] for r in csv.DictReader(handle, delimiter="\t")}
        why = []
        for name, sub in (("chat word, source-language instruction", "rq_chat_word_src"), ("chat word, English instruction", "rq_chat_word_en"),
                          ("chat sentence", "rq_chat_sentence")):
            p = sub_dir.get(sub, d / sub) / "items.csv"
            if p.is_file():
                c = _classify_dropped(p, concepts, sent if sub == "rq_chat_sentence" else None)
                why.append([name] + [c.get(k, 0) for k in ("kept", "right word, split differently", "echoed the source word",
                                                           "other word", "ended the turn / blank", "no scorable target tokens")])
        if why:
            out.append("Chat replies the matcher dropped, classified from their text (the matcher requires the reply's first "
                       "token to be a ≥2-character prefix of the word, as in the few-shot experiment):\n")
            out.append(table(["format", "kept", "right word, split differently", "echoed the source word", "other word",
                              "ended turn / blank", "no scorable tokens"], why) + "\n")
        for _, _, r in fmts[1:]:
            if r and r.get("meta_tokens_top10"):
                out.append(f"Most frequent other tokens in the band ({r['kind']} {r['instr']}, top-20, excluding every concept "
                           "word): " + ", ".join(f"`{m['token'].strip() or repr(m['token'])}` {m['prevalence']:.0%}"
                                                  for m in r["meta_tokens_top10"]) + "\n")
                break
        dirs = [d, *(R / e for e in extra_runs)]  # the main run first, then runs that filled in its gaps
        body, collapse_run = [], None
        for tbl in ("homonyms", "kinship"):
            p = next((x / "rq_collapse" / f"{tbl}_result.json" for x in dirs
                      if (x / "rq_collapse" / f"{tbl}_result.json").is_file()), None)
            if p:
                collapse_run = p.parent.parent.name
                r = json.loads(p.read_text(encoding="utf-8"))
                o = r["outcome_rate"]
                body.append([tbl, r["n_with_sibling"], o["target"], o["sibling"], o["control"], o["other"],
                             r["mean_p_final"]["sibling"], r["mean_p_final"]["control"], r.get("accuracy_without_sibling", "")])
        if body:
            out.append(f"**M4. English-collapse errors** (few-shot, run `{collapse_run}`; items whose word shares its English "
                       "word with another sense/kin term): share of answers that are the right word, the other sense of the "
                       "same English word (sibling), a matched other word (control), or else:\n")
            out.append(table(["set", "n", "right", "sibling", "control", "other", "P(sibling)", "P(control)",
                              "accuracy, words English keeps distinct"], body) + "\n")
        if st.get("rq_zh", {}).get("result"):
            z = st["rq_zh"]["result"]
            src_run = st["rq_zh"].get("from_run", run)
            out.append(f"**M5. Translating into Chinese** (5-shot, 32 concepts; run `{src_run}`): accuracy {z['accuracy']}; "
                       f"{z['accuracy_by_pair']}\n")
        steer = next(iter(sorted(p for x in dirs for p in x.glob("causal_steer_*/causal_summary.json"))), None)
        if steer:
            s = json.loads(steer.read_text())
            body = []
            for cond in ("english_mid", "control_mid", "english_mid@2", "control_mid@2", "english_mid@3", "control_mid@3", "english_early"):
                if cond in s:
                    c = s[cond]
                    body.append([cond, round(-c["mean_drop"], 4), f"[{-c['drop_ci95'][1]:.4f}, {-c['drop_ci95'][0]:.4f}]", round(c["still_correct"], 3)])
            out.append(f"**M6. Negative steering** ({steer.parent.name}, n={s['n_items']}, mid L{s['mid_layers'][0]}–{s['mid_layers'][-1]}; "
                       "@2 reflects the component, @3 pushes it to −2×):\n")
            out.append(table(["condition", "ΔP(target)", "95% CI", "still correct"], body) + "\n")
        th = next((x / "rq_think" / "think_result.json" for x in dirs if (x / "rq_think" / "think_result.json").is_file()), None)
        if th:
            r = json.loads(th.read_text(encoding="utf-8"))
            shares = {k: v for k, v in r["reasoning_script_share"].items() if v}
            out.append(f"**M7. Thinking mode** (n={r['n_items']}, ≤{r['mean_tokens']} tokens): reasoning closed {r['closed_think']}; "
                       f"English word in the reasoning {r['english_in_reasoning']}; target word in the reasoning "
                       f"{r['target_in_reasoning']}, in the answer {r['target_in_answer']}; script share of the reasoning {shares}.\n")
    return "\n".join(out) + "\n"


def _lens_rows(summary_path: Path, layers: range, roles: tuple[str, ...]) -> list[list]:
    """One row per (lens, role): ALL_INDIC top-10 rate per layer, plus the peak probability."""
    g: dict = defaultdict(dict)
    for r in read_csv(summary_path):
        if r["pair"] == "ALL_INDIC":
            g[(r["lens"], r["role"])][int(r["layer"])] = (float(r["top10_rate"]), float(r["mean_prob"]))
    rows = []
    for lens in sorted({k[0] for k in g}):
        for role in roles:
            d = g.get((lens, role))
            if d:
                p, l = max((v[1], l) for l, v in d.items())
                rows.append([lens, role] + [f"{d[l][0]:.2f}" if l in d else "" for l in layers] + [f"{p:.3f} @L{l}"])
    return rows


def section_followups() -> str:
    """N. Follow-ups (doc 07): an Indic-corpus J-lens, Qwen's published J-lens, expert routing and layer-0 neurons."""
    out = ["## N. Follow-ups: Indic-fit lens, Qwen J-lens, expert routing, layer-0 neurons\n"]
    roles = ("english", "english_shuffled", "hindi", "chinese", "chinese_shuffled", "target")
    p = R / "pivot_indic_lens/pivot_ext/summary.csv"
    if p.is_file():
        out.append("**N1. Sarvam-30B pivot under a J-lens fitted on Indian-language Wikipedia** (`j_indic_27p`: 27 prompts, "
                   "3 per language, en/hi/bn/ta/te/mr/gu/kn/ml) vs WikiText J-lenses (`j_wiki_40p`: the 200-prompt fit's "
                   "first shard; `j_200p`) and the logit lens. Run `pivot_indic_lens`; top-10 rate by layer, peak P:\n")
        out.append(table(["lens", "role"] + [f"L{l}" for l in range(9, 18)] + ["peak P"], _lens_rows(p, range(9, 18), roles)) + "\n")
    p = R / "qwen_j/pivot_ext/summary.csv"
    if p.is_file():
        out.append("**N2. Qwen3.6-35B-A3B under its published J-lens** (camilablank/workspace-lenses; run `qwen_j`) vs the "
                   "logit lens; top-10 rate by layer (of 40), peak P:\n")
        out.append(table(["lens", "role"] + [f"L{l}" for l in range(26, 39)] + ["peak P"], _lens_rows(p, range(26, 39), roles)) + "\n")
    rows, prompt_rows, neuron_rows = [], [], []
    for label, tag in (("Sarvam-30B", "neuron30"),):
        rp = R / tag / "report.json"
        if not rp.is_file():
            continue
        st = {s["stage"]: s.get("result") or {} for s in json.loads(rp.read_text(encoding="utf-8"))["stages"]}
        rt = st.get("routing", {}).get("routing")
        if rt:
            b = rt["words"]["band"]
            rows.append([label, f"L{b['layers'][0]}–{b['layers'][-1]}", f"{b['lang_acc']:.2f} (0.10)",
                         f"{b['concept_acc_xscript']:.2f} (0.03)", f"{b['jac_same_concept_xscript']:.3f} / {b['jac_diff_concept_xscript']:.3f}",
                         f"{b['en_excess']:.3f}", f"{b['indic_excess']:.3f}", f"{b['zh_excess']:.3f}",
                         f"{b['en_minus_indic_excess']:+.3f} [{b['en_minus_indic_excess_ci'][0]:+.3f}, {b['en_minus_indic_excess_ci'][1]:+.3f}]"])
            q = rt["prompts"]["band"]
            prompt_rows.append([label, f"{q['eta2_concept']:.2f}", f"{q['eta2_pair']:.2f}", f"{q['acc_tgt_xpair']:.2f} (0.33)",
                                f"{q['acc_src_xpair']:.2f} (0.33)", f"{q['acc_concept_xpair']:.2f} (0.03)",
                                f"{q['align_en_excess']:.3f}", f"{q['align_hi_excess']:.3f}", f"{q['align_zh_excess']:.3f}",
                                f"{q['align_tgt_excess']:.3f}"])
        nr = st.get("neurons", {}).get("neurons")
        v2 = R / tag / "neurons" / "neurons_summary_v2.json"  # tied ranks averaged, tied winners dropped (2026-10-20)
        if nr and v2.is_file():
            nr = json.loads(v2.read_text(encoding="utf-8"))
        if nr:
            cs, ss = nr["concept_selective"], nr["script_selective"]
            top = nr["top_concept"][0] if nr.get("top_concept") else {}
            # The win-count above counts scripts, and cognates/transliterations (nadī in 7 scripts) share a word
            # FORM. neurons_summary.json's independent_forms counts distinct word forms instead (routing session, doc 09 §11).
            ns_path = v2 if v2.is_file() else R / tag / "neurons" / "neurons_summary.json"
            ind = json.loads(ns_path.read_text(encoding="utf-8")).get("independent_forms", {}).get(">=5") if ns_path.is_file() else None
            neuron_rows.append([label, nr["d_ff"], f"{cs['n']} (null mean {cs['null_mean']}, max {cs['null_max']})",
                                f"{ind['n']} (null mean {ind['null_mean']}, p95 {ind['null_p95']})" if ind else "–",
                                f"{ss['n']} ({', '.join(f'{k} {v}' for k, v in ss['by_script'].items() if v)})",
                                f"#{top.get('neuron')} '{top.get('concept')}' in {top.get('n_scripts')} scripts" if top else ""])
    if rows:
        out.append("**N3. Expert routing, bare words** (32 concepts × 10 languages; in each model's English band; chance in "
                   "brackets). Jaccard = overlap of top-k expert sets. *Excess* = overlap with the same concept in that "
                   "language minus with other concepts in it:\n")
        out.append(table(["model", "band", "language from routing", "concept from routing (across scripts)",
                          "Jaccard same / different concept", "excess vs English word", "vs other Indic", "vs Chinese",
                          "English − Indic [95% CI]"], rows) + "\n")
        out.append("**N4. Expert routing, the 5-shot translation prompts' final position** (η² = share of routing variance):\n")
        out.append(table(["model", "η² concept", "η² language pair", "target lang from routing", "source lang",
                          "concept across pairs", "align: English word", "Hindi word", "Chinese word", "target word"], prompt_rows) + "\n")
    if neuron_rows:
        out.append("**N5. Layer-0 (dense MLP) neurons** (320 words): concept-selective = the same concept wins in ≥ 5 scripts "
                   "(null: shuffled concept labels, 200 permutations); script-selective = AUROC ≥ 0.95 for one script. "
                   "**The script count mostly tracks word FORM** (cognates and transliterations, e.g. nadī in 7 scripts): "
                   "counted over ≥ 5 independent word forms, the 30B is at chance (doc 09 §11). "
                   "From `neurons_summary_v2.json` where present:\n")
        out.append(table(["model", "d_ff", "concept-selective across ≥5 scripts", "… across ≥5 independent word forms",
                          "script-selective", "strongest concept neuron"], neuron_rows) + "\n")
    return "\n".join(out) + "\n"


def section_per_pair_at(summary_path: Path, items_path: Path, lens: str, *, probe: int, last: int, title: str) -> str:
    """Per-pair table read at layer ``probe`` (English/Hindi) and ``last`` (target)."""
    rows = read_csv(summary_path)
    items = read_csv(items_path)
    kept, total = defaultdict(int), defaultdict(int)
    for i in items:
        total[i["pair"]] += 1
        kept[i["pair"]] += i["kept"] == "True"
    body = []
    for p in sorted({r["pair"] for r in rows if r["pair"] != "ALL_INDIC"}):
        g = {(int(r["layer"]), r["role"]): r for r in rows if r["lens"] == lens and r["pair"] == p}
        en = [(float(g[(l, "english")]["mean_prob"]), l) for l in range(last + 1) if (l, "english") in g]
        peak = max(en) if en else (float("nan"), -1)
        t10 = lambda l, role: float(g[(l, role)]["top10_rate"]) if (l, role) in g else float("nan")
        body.append([p, f"{kept[p]}/{total[p]}", f"{peak[0]:.2f} @L{peak[1]}", t10(probe, "english"), t10(probe, "hindi"),
                     t10(probe, "english_shuffled"), t10(last, "target")])
    return (f"**{title}**\n\n" + table(
        ["pair", "kept", "peak P(English)", f"top10 English L{probe}", f"top10 Hindi L{probe}",
         f"top10 shuffled-English L{probe}", f"top10 target L{last}"], body) + "\n")


PATCH_LABELS = {
    "whole": "whole residual (ceiling)",
    "english": "English(B) subspace",
    "english_ab": "English(A) + English(B) subspace",
    "english_other": "English(C) subspace, rank-matched (control)",
    "hindi": "Hindi(B) subspace",
    "target": "B's target-language tokens' subspace",
    "random": "random subspace, English rank (control)",
    "english_a_mean": "English(A) only, reset to the other concepts' mean (nothing of B)",
    "hindi_ab": "Hindi(A) + Hindi(B) subspace",
    "target_ab": "target(A) + target(B) subspace",
    "random_ab": "random subspace, English(A) + English(B) rank (control)",
    "english_eq": "English(A) + English(B), cut to the item's common rank r",
    "hindi_eq": "Hindi(A) + Hindi(B), rank r",
    "target_eq": "target(A) + target(B), rank r",
    "random_eq": "random subspace, rank r (control)",
}


def section_patch(runs: tuple[Path, ...] = (R / "patch", R / "patch2", R / "patch3")) -> str:
    """K. Activation patching on Sarvam-30B: concept B's final-position residual (or part of it)
    written into concept A's run at the mid band. ``patch`` has the ceiling and the B-only
    subspaces; ``patch2`` (stage ``patch_split``) the A + B spans and the A-only reset; ``patch3``
    (stage ``patch_rank``) the A + B swap at equal rank across languages."""
    out = ["## K. Activation patching on Sarvam-30B (runs `patch`, `patch2`, `patch3`)\n"]
    ci = lambda v, lo_hi, f: f"{v:{f}} [{lo_hi[0]:{f}}, {lo_hi[1]:{f}}]"
    ceiling, by_name = None, {}  # the first run's whole-state ΔP(B); (run, condition) -> stats
    for idx, run in enumerate(runs, 1):
        paths = sorted(run.glob("patch*/patch_summary.json")) if run.is_dir() else []
        if not paths:
            continue
        out.append(f"**K{idx}. Run `{run.name}`**\n")
        if (run / "report.json").is_file():
            rep = json.loads((run / "report.json").read_text(encoding="utf-8"))
            out.append(f"- Driver {rep['elapsed_minutes']} min, est. ${rep['est_cost_usd_so_far']}; stages: "
                       + ", ".join(f"{s['stage']} ({s['status'][:40]}, {s.get('minutes', '-')} min)" for s in rep["stages"]))
            san = next((s.get("result") or {} for s in rep["stages"] if s["stage"] == "sanity"), {})
            if san:
                out.append(f"- Sanity: {san.get('seconds_per_forward')} s/forward; greedy → water: {san.get('greedy_water_ok')}")
        for path in paths:
            s = json.loads(path.read_text())
            if "whole" in s:
                ceiling = s["whole"]["mean_pB_gain"]
            lens = re.sub(r"^patch(?:_(?:split|rank))?_", "", path.parent.name)
            out.append(f"\n**Lens `{lens}`**, final position, layers "
                       f"L{s['layers'][0]}-{s['layers'][-1]}; {s['n_items']} items with A and B both translated correctly "
                       f"(skipped: {s['skipped']}). Base P(B's word) in A's run {s['base_pB']:.4f}; in B's own run "
                       f"{s['B_run_pB']:.3f}. *Share* = fraction of the clean (B − A) residual difference inside the span "
                       "(mean over the band)." + ("" if "whole" in s else " ΔP(B) / ceiling uses run `patch`'s ceiling.") + "\n")
            body = []
            for name, label in PATCH_LABELS.items():
                c = s.get(name)
                if not isinstance(c, dict) or not c.get("n"):
                    continue
                by_name[(run.name, name)] = c
                frac = c["mean_pB_gain"] / ceiling if ceiling else float("nan")
                body.append([label, c["n"], f"{c['rank_mean']:.1f}", f"{c['share_of_diff']:.3f}",
                             ci(c["flip_to_B_rate"], c["flip_ci95"], ".3f"), f"{c['still_A_rate']:.3f}",
                             ci(c["mean_pB_gain"], c["pB_gain_ci95"], "+.4f"), f"{frac:.3f}", f"{c['mean_pA_drop']:+.4f}"])
            out.append(table(["patched", "n", "rank", "share of B − A", "flip to B [95% CI]", "still A",
                              "ΔP(B) [95% CI]", "ΔP(B) / ceiling", "P(A) drop"], body))
            body = [[k.replace("_minus_", " − "), c["n"], ci(c["flip"], c["flip_ci95"], "+.3f"),
                     ci(c["pB_gain"], c["pB_gain_ci95"], "+.4f")] for k, c in s.get("contrasts", {}).items()]
            out.append("\nPaired contrasts (same items):\n")
            out.append(table(["contrast", "n", "Δ flip rate [95% CI]", "Δ P(B) [95% CI]"], body))
            rows_path = path.parent / "patch_rows.csv"
            if rows_path.is_file():
                rows = read_csv(rows_path)
                names = [n for n in PATCH_LABELS if any(r.get(f"{n}_top_is_B") for r in rows)]
                body = []
                for pair in sorted({r["pair"] for r in rows}):
                    sel = [r for r in rows if r["pair"] == pair]
                    body.append([pair, len(sel)] + [
                        f"{np.mean([r[f'{n}_top_is_B'] == 'True' for r in sel if r.get(f'{n}_top_is_B')]):.2f}"
                        if any(r.get(f"{n}_top_is_B") for r in sel) else "" for n in names])
                out.append("\nFlip-to-B rate per pair:\n")
                out.append(table(["pair", "n"] + names, body) + "\n")
    if not by_name:
        return "\n".join(out) + "\nNot run yet.\n"
    split = [("patch", "english", "insert B's English only"),
             ("patch2", "english_a_mean", "reset A's English only (other concepts' mean)"),
             ("patch", "english_ab", "both: B's state on English(A) + English(B)"),
             ("patch2", "english_ab", "both, re-run in `patch2`")]
    body = [[label, f"`{run}`", c["n"], ci(c["flip_to_B_rate"], c["flip_ci95"], ".3f"), f"{c['still_A_rate']:.3f}",
             ci(c["mean_pB_gain"], c["pB_gain_ci95"], "+.4f")]
            for run, name, label in split if (c := by_name.get((run, name)))]
    if len(body) > 1:
        out.append("**K4. English: remove A × insert B** (across runs; same items up to GPU noise)\n")
        out.append(table(["English edit", "run", "n", "flip to B [95% CI]", "still A", "ΔP(B) [95% CI]"], body) + "\n")
    body = []
    for lang in ("english", "hindi", "target", "random"):
        full, eq = by_name.get(("patch2", f"{lang}_ab")), by_name.get(("patch3", f"{lang}_eq"))
        if full or eq:
            cell = lambda c: [f"{c['rank_mean']:.1f}", ci(c["flip_to_B_rate"], c["flip_ci95"], ".3f"),
                              f"{c['mean_pB_gain']:+.4f}"] if c else ["", "", ""]
            body.append([lang, *cell(full), *cell(eq)])
    if any(by_name.get(("patch3", f"{lang}_eq")) for lang in ("english", "hindi", "target")):
        out.append("**K5. The A + B swap per language: all its tokens (`patch2`) vs equal rank r (`patch3`)**\n")
        out.append(table(["span", "all tokens: rank", "flip to B", "ΔP(B)", "equal rank: rank", "flip to B", "ΔP(B)"], body))
    return "\n".join(out) + "\n"


def section_correctness() -> str:
    rows = read_csv(R / "correctness.csv")
    col = [c for c in rows[0] if c not in ("eval", "item")][0]
    counts = defaultdict(lambda: [0, 0])
    for r in rows:
        if r[col] in ("True", "False"):
            counts[r["eval"]][1] += 1
            counts[r["eval"]][0] += r[col] == "True"
    return ("## I. Sarvam-30B on J++'s readout eval items (greedy next token; `results/correctness.csv`)\n\n" +
            table(["eval", "correct", "graded"], [[k, v[0], v[1]] for k, v in sorted(counts.items())]) + "\n")


def section_post1() -> str:
    """docs/17: the follow-ups for blog post 1 (runs p1_*). CIs resample concepts (problems for MGSM)."""
    out = ["## P. Follow-ups for blog post 1 (`docs/17_post1_followups.md`, runs `p1_*`)\n"]
    f2 = lambda x: "" if x is None else f"{x:.2f}"
    cell = lambda c: f"{c[0]:.2f} [{c[1]:.2f}, {c[2]:.2f}]" if c and c[0] is not None else "–"
    # P1: every language and the semantic neighbours as roles (band top-10, ALL_INDIC)
    rows = []
    for run, label in (("p1_t1_30b", "30B"),):
        path = R / run / "post1_experiment_roles/roles_summary.json"
        if not path.is_file():
            continue
        dec = json.loads(path.read_text(encoding="utf-8"))["decision"]
        for role in ("english", "english_shuffled", "hindi", "target", "source", "chinese", "pool:orig_other",
                     "pool:orig_other_shuffled", "pool:big10_other", "pool:low_resource", "pool:world", "nb_close",
                     "nb_close_shuffled", "nb_assoc"):
            rows.append([f"{label} {dec['lens']}", role] + [cell(dec[ro].get(role)) for ro in ("prefix", "whole_word", "own_script")])
    if rows:
        out.append("### P1. Every language and two English neighbours as roles (band top-10 [CI], ALL_INDIC)\n\n" + table(
            ["model, lens", "role", "whole vocabulary (prefix sets)", "whole-word tokens", "own script"], rows) +
            "\n\n`pool:` = each item's mean over its third languages (orig_other: the original-8 Indian languages outside "
            "the prompt and Hindi; big10_other adds pa, or; low_resource: ks sd mni sat brx doi kok mai; world: ja ru ar). "
            "nb_close / nb_assoc: the concept's English near-synonym / associate (`code/data/neighbours.tsv`).\n")
    # P2: causal tables (joint, digits, positions, zh) with the size-matched partners
    def ctab(run, folder, name, conds, key=None):
        path = R / run / folder / name
        if not path.is_file():
            return []
        t = json.loads(path.read_text(encoding="utf-8"))
        t = t[key] if key else t
        res = []
        for c in conds:
            if c in t and isinstance(t[c], dict) and "drop_pts" in t[c]:
                res.append([run, c, f"{t[c]['drop_pts']:.1f} [{t[c]['ci'][0]:.1f}, {t[c]['ci'][1]:.1f}]", f"{t[c]['still_correct']:.3f}",
                            f2(t[c].get("rel_norm"))])
            elif c in t and isinstance(t[c], dict) and "pts" in t[c]:
                res.append([run, c, f"{t[c]['pts']:+.1f} [{t[c]['ci'][0]:.1f}, {t[c]['ci'][1]:.1f}]", "", ""])
        return res
    joint = ["english_pub", "control_pub", "hindi_pub", "en_ms", "hi_ms", "zh_ms", "tgt_ms", "en_hi", "joint", "control_joint",
             "random_joint", "joint_early", "joint@3", "control_joint@3", "joint_minus_control_joint", "joint_minus_en_ms+hi_ms+tgt_ms",
             "joint_minus_en_ms+hi_ms+zh_ms+tgt_ms"]
    matched = ["joint@3", "joint@3~rand", "control_joint@3", "control_joint@3~rand", "joint@3_minus_joint@3~rand",
               "english_all@3", "english_all@3~rand", "english_all@3_minus_english_all@3~rand", "zh@3", "zh@3~rand", "zh@3_minus_zh@3~rand"]
    digits = ["word", "word_ctrl", "word_minus_word_ctrl", "allwords", "random_allwords", "allwords_minus_random_allwords", "ascii",
              "word@3", "word_ctrl@3", "word@3~rand", "word@3_minus_word@3~rand", "allwords@3", "allwords@3~rand",
              "allwords@3_minus_allwords@3~rand"]
    positions = ["english_last", "control_last", "english_src", "english_all_but_last", "english_all", "control_all", "english_all_early",
                 "english_all_minus_english_last"]
    zh = ["english_mid", "control_mid", "english_mid_minus_control_mid", "english_mid@3", "control_mid@3", "english_mid@3_minus_control_mid@3"]
    rows = (ctab("p1_t1_30b", "post1_experiment_joint", "joint_summary.json", joint)
            + ctab("p1_t2_30b", "post1_experiment_positions", "positions_summary.json", positions)
            + ctab("p1_t2_30b", "post1_experiment_zh", "zh_summary.json", zh))
    for run in ("p1_matched_30b",):
        for folder, name in (("post1_experiment_joint", "joint_summary.json"), ("post1_experiment_positions", "positions_summary.json"),
                             ("post1_experiment_zh2", "zh2_summary.json")):
            rows += ctab(run, folder, name, matched)
    for run in ("p1_t2_30b", "p1_matched_30b"):
        rows += ctab(run, "post1_experiment_digits", "digits_ablation_summary.json", digits, key="native_systems")
    if rows:
        out.append("### P2. Ablations: joint, positions, Chinese, native-digit sums (drop in pts [CI over concepts / problems])\n\n"
                   + table(["run", "condition", "drop or contrast", "still correct", "|Δh| / |h|"], rows) +
                   "\n\n`~rand`: a random direction pushed exactly as far as the named push at every layer (same forward). "
                   "In `p1_*_zh`, causal_test's names: `english_*` = the Chinese word's directions.\n")
    # P3: the push curve with the size-matched random control
    rows = []
    for run in ("p1_push2_30b",):
        path = R / run / "post1_experiment_push/push_summary.json"
        if not path.is_file():
            continue
        t = json.loads(path.read_text(encoding="utf-8"))
        for a in t["alphas"]:
            a = f"{a:g}"
            g = lambda k: f"{t[k]['drop_pts']:.1f}" if k in t else ""
            d = t.get(f"english@{a}_minus_random_en@{a}")
            rows.append([run, a, g(f"english@{a}"), g(f"control@{a}"), g(f"random_en@{a}"), g(f"random_ctrl@{a}"), g(f"random@{a}"),
                         f"{d['pts']:+.1f} [{d['ci'][0]:.1f}, {d['ci'][1]:.1f}]" if d else ""])
    if rows:
        out.append("### P3. The push curve (drop in pts)\n\n" + table(
            ["run", "α", "English", "other concept's English", "random, as far as English", "random, as far as the control",
             "random, first-layer match only", "English − random as far [CI]"], rows) + "\n")
    # P4: MGSM thinking language and the hint rerun (pooled rows)
    rows = []
    for run in ("p1_think3k_30b", "p1_think_30b"):
        path = R / run / "post1_experiment_think/think_summary.csv"
        if path.is_file():
            for r in read_csv(path):
                if r["variant"] in ("indian pooled", "en"):
                    rows.append([run, r["variant"], r["cond"], r["n"], r["own_share"], r["closed"], r["accuracy"], r["accuracy_ci"],
                                 r["accuracy_natural"], r["mean_tokens"]])
    if rows:
        out.append("### P4. MGSM, thinking in English vs the prompt's language (30B)\n\n" + table(
            ["run", "prompts", "condition", "n", "own-script share", "closed", "accuracy", "95% CI (problems)", "accuracy, natural",
             "mean tokens"], rows) + "\n")
    rows = []
    for run, sub in (("p1_hint_30b", ""), ("p1_hint_30b", "natural/")):
        path = R / run / "post1_experiment_hint" / sub / "hint_summary_v2.csv"  # pooled CIs over problems (2026-10-20)
        if not path.is_file():
            path = path.with_name("hint_summary.csv")
        if path.is_file():
            for r in read_csv(path):
                rows.append([run, "natural only" if sub else "natural, else forced", r["variant"], r["cond"], r.get("accuracy", ""),
                             r.get("n_base", ""), r.get("uptake", ""), f"[{r.get('uptake_lo', '')}, {r.get('uptake_hi', '')}]" if r.get("uptake") else "",
                             r.get("n_followed", ""), r.get("verbalised", ""), r.get("mentions_hint", "")])
    if rows:
        out.append("### P5. Wrong-hint test, 60 problems (30B `p1_hint_30b`)\n\n" + table(
            ["run", "answers","prompts", "condition", "accuracy", "n (right without hint)", "uptake", "95% CI", "followers", "verbalised",
             "mentions the hint"], rows) + "\n")
    # P6: writers by layer (J-lens scores, English minus the control concept's English)
    path = R / "p1_t2_30b/post1_experiment_writers/writers_layers.csv"
    if path.is_file():
        rows = [[r["layer"], r["attn_en_minus_ctrl"], r["attn_en_minus_ctrl_ci"], r["mlp_en_minus_ctrl"], r["mlp_en_minus_ctrl_ci"],
                 r["attn_target"], r["mlp_target"], r["resid_english"]] for r in read_csv(path) if r["mode"] == "jlens"]
        out.append("### P6. Who writes the English direction (30B, J-lens scores at the final position)\n\n" + table(
            ["layer", "attention: English − control", "CI", "MoE: English − control", "CI", "attention: target", "MoE: target",
             "residual: English"], rows) + "\n")
    return "\n".join(out)


def main() -> None:
    parts = [
        "# Results tables (auto-generated)\n\nGenerated by `code/summarize_results.py` from `results/`. "
        "Do not edit by hand; rerun the script.\n"
        "This copy leaves out the Sarvam-105B runs and the refusal runs, which aren't part of this release.\n",
        section_runs(),
        section_sanity(),
        pivot_tables(R / "final/pivot_all/summary.csv", "C. Pivot experiment, all lenses + shuffled controls (run `final`)",
                     range(9, 18), ("english", "english_shuffled", "hindi", "hindi_shuffled", "target", "source", "chinese")),
        section_per_pair(R / "final/pivot_all/summary.csv", R / "final/pivot_all/items.csv", "j_200p"),
        section_causal(),
        section_sentences(),
        pivot_tables(R / "qwen_zh/pivot_logit/summary.csv", "G. Qwen3.6-35B-A3B baseline, logit lens (run `qwen_zh`)",
                     range(26, 39), ("english", "chinese", "hindi", "target", "english_shuffled")),
        section_cka(),
        section_correctness(),
        section_reader_questions(),
        section_followups(),
        section_patch(),
        section_post1(),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
