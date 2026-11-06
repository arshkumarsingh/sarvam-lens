# 3. Engineering log: how everything was done, in order, and why

All dates are 2026-10-18. Run tags (in **bold**) match folders in `results/` and logs in `logs/`.
Items about Sarvam-105B are left out of this release, so the numbering has gaps.

## Phase 0: scoping

1. **Question.** Can the "J lens" (Anthropic's Jacobian lens, July 2026) show how Sarvam processes
   Indian languages? We confirmed the reference code (`anthropics/jacobian-lens`) works on HF
   decoder models, and that Sarvam-30B/105B are open-weights (Apache-2.0).
2. **Model choice:** Sarvam-30B. 19 layers, 4,096-wide, 128 experts top-6 + shared, 262K vocab.
   Its architecture is custom Hub code, and the checkpoint is stored in fp32.
3. **Novelty check.** The English-pivot phenomenon is known (Wendler 2024; Zhong 2025;
   Indic-TunedLens on Sarvam-1). New here: Sarvam-30B, workspace lenses, controlled token scoring,
   causal tests, and the Qwen comparison.

## Phase 1: choosing the toolchain

4. The user pointed to the **R-lens** post (LRP backward pass, cleaner early layers) and the **J++**
   post (Jacobian filtering + LRP + readout filtering). We read both. J++'s release
   (`safety-research/jpp_lens` @ `d7efc0d`) implements J, R and J++, the evals and the fitting
   pipeline, so **we built on it rather than reimplementing**.
5. Reading J++'s eval data revealed the **multilingual-eval confound**: its English-target items
   would train J++ to find English in non-English prompts. Decision: fit J++ weights with that
   eval excluded.
6. The Goodfire essay (*Neural Networks Are Already Clean*) was noted as a neuron-level phase 2
   (cross-script neurons).

## Phase 2: making Sarvam work with J++ (local, CPU, $0)

7. **Gap analysis.** Stock J++ fails on Sarvam in three ways:
   - (a) Custom Hub code, with the embedding named `word_embeddings` → jlens' layout
     auto-detection fails.
   - (b) jpp's LRP surgery raises on unknown classes.
   - (c) jpp excludes q/k norms by name (`q_norm`/`k_norm`, or under `self_attn`), but Sarvam's are
     `attention.query_layernorm`/`key_layernorm`, so a naive registration would apply the LN-rule
     inside attention.
8. **`code/sarvam_support.py`:**
   - Imports Sarvam's code from a **pinned revision**, so `trust_remote_code` is never needed.
   - Passes an explicit layout.
   - Implements Sarvam-specific LRP rules mirroring every jpp flag.
   - Hooks into jpp's dispatcher (`install_jpp_hooks`).
9. **`code/run_jpp.py`** wraps `jpp_cli.py`. It adds the Sarvam loader, a JSON fit-corpus override,
   and two stages: `prepare-evals` (empties named evals) and `grade` (jpp ships a correctness CSV
   only for Qwen3.6-27B).
10. **Smoke test** (`code/smoke_test.py`). It uses a tiny random-weight Sarvam (4 layers, d=64, 8
    experts) with the **real** tokenizer and runs the whole pipeline on CPU. It caught several bugs
    before any money was spent:
    - **transformers-5 incompatibility #1:** Sarvam declares `_tied_weights_keys` as a 4.x list,
      which crashes `save_pretrained` on transformers 5. Fixed with a shim, since Sarvam doesn't tie
      weights anyway.
    - jpp's vendored jlens has no `fitting` module, so the test uses its own vector-Jacobian check.
    - `fit-weights`' `--top-n` must be ≤ the number of clusters (a test-config issue only).
11. **Token scoring problems** found by inspecting Sarvam's tokenizer:
    - Indic words are ~2 tokens each (only 2–12 of 32 words are single tokens), so first-piece
      scoring is necessary.
    - Single-letter prefixes (`w`, `प`) begin countless words, so a ≥2-character floor was added.
      The tokenizer's own first token is always kept.
    - Hindi/Marathi share Devanagari, so tokens shared between roles are disjointed.
    - Plot labels overlapped and lens probabilities are small, so the plots use a data-scaled y-axis
      and nudged labels. The palette was validated with the dataviz skill's validator.
12. **Data-leak guard.** jpp splits fit/held-out items with one RNG stream across evals, so emptying
    "multilingual" shifts the later evals' permutations. **31 of the 76 fit items (poetry, typo)
    would be scored as "held-out"** if `evaluate` used the full list. `run_jpp.py evaluate` now
    refuses that combination, and the smoke test checks the refusal.
13. A Qwen-baseline check showed Qwen3.6-35B-A3B needs transformers 5. Sarvam's custom code imports
    fine on 5.14.1, so **one environment (J++'s pins) runs everything**.

## Phase 3: Modal pilot and the bug hunt (~$2)

14. **Modal setup** (`code/modal_app.py`).
    - The token is read from the user's key file into environment variables for each command. It
      is never echoed and never written to `~/.modal.toml`.
    - Cost controls: every function has a hard `timeout`, `retries=0`, and the pilot driver
      (`code/modal_pilot.py`) only starts a stage if its estimate fits the remaining budget.
      `report.json` is saved after every stage.
15. **`prepare`** (CPU, 8.3 min, ~$0.07): fp32 Hub shards → bf16, one shard at a time, onto the
    Volume `sarvam-lens`. This avoids reading 129 GB on GPU time at every load.
16. **First pilot, failed at import (~$0.08).** `modal_pilot.py` imported `workspace_lens` before
    J++'s path was set. Locally jpp was pip-installed, which masked it.
    → **New `smoke` step: the CPU smoke test plus a dry pilot inside the exact image (~$0.04)**,
    run before any GPU spend.
17. **`pilot`** (9.1 min, ~$0.95):
    - Timing: J fit **75 s/prompt**, R fit **72 s/prompt**, so ~$6.60–6.90 per 64-prompt lens.
    - LRP patch counts were exactly as predicted (39 / 19 / 2,304 / 18).
    - But the pivot run **kept only 2/448 items**.
18. **Bug-hunt step 1 (wrong hypothesis): BOS.** The model predicted line ends and the next line's
    language label, as if the word were already finished. The first suspect was BOS: the tokenizer
    prepends `<bos>`, while jlens' `force_bos` had rewritten it to `[@BOS@]`.
    → Added `bos_mode` (hf / chat / sarvam_bos).
    → **`bosfix`** run (~$0.44): every mode stayed at ≈0%, and greedy Hindi decoding **looped**
    ("राजधानी राजधानी…"). So BOS was not the cause.
19. **Bug-hunt step 2 (found it): RoPE.** A free local test on the tiny model showed
    **`inv_freq` = all zeros after `from_pretrained`** (later corrected: it is uninitialized memory,
    `torch.empty_like`, zeros in 10 of the first 12 tiny-model runs and arbitrary huge values in 2).
    transformers 5 builds the model on the meta
    device and re-derives non-persistent buffers only for its own classes, so Sarvam's rotary
    frequencies are never computed. Result: **no position information at all**. The causal mask
    was fine. → `_rebuild_rotary_embedding` after loading, plus a smoke-test check.
    *This silently affects any transformers-5 user of Sarvam-30B.*
20. **LRP drift.** On the real model in bf16, jpp-style LRP patches changed the forward by up to
    11–20 (on values ≤135); tiny fp32 tests had shown ~1e-6. The cause: the identity rule's
    `z*sigmoid(z)` rounds differently from `F.silu`, and in an MoE that flips top-k routing.
    → **Bit-exact straight-through rewrite.** The smoke test now asserts bit-identity in bf16 and
    gradient equality with jpp's rule.
21. **`ropefix`** (~$0.44): with RoPE fixed, everything worked. BOS probe: `[@BOS@]` 96.3% (made
    the default). LRP forward difference 0.0. The **first real result**: under the logit lens,
    English holds ~45% probability at layers 14–16. Grading gave a valid correctness CSV.

## Phase 4: R-lens, J++ and the Qwen baseline (~$5)

22. **`lensfit`** (43 min, ~$3.86): refit the router (the old one was fit on the broken-RoPE model);
    a 25-prompt R+MoE fit at 77 s/prompt; merge; J++ weights (multilingual excluded); a 3-lens pivot.
23. **`prepare-qwen`** (CPU, 2.8 min) and **`qwen`** (~$0.52): Qwen logit lens, English peak 0.46,
    about the same as Sarvam's 0.45.
24. Qwen is Chinese-centric too, so a **Chinese** column and role were added (with a validated
    5-colour palette). **`qwen_zh`** (~$0.47): Chinese is in Qwen's top 10 for ~92% of items, as
    often as English.

## Phase 5: the CKA explorer and the J-lens (~$22)

25. Reproduced the explorer with its own code (`code/cka_compare.py`, CPU). Found:
    - The published `cka_lib` only applies the `1+w` norm offset for Gemma, which matters for
      Qwen3.5/3.6. We applied it correctly, and later measured that it hardly matters.
    - **Sarvam is degenerate:** layer coupling 0.97, Sarvam × Qwen only 0.09.
26. **Diagnose v1/v2** (CPU): not the gain (0.96 without it); the R-lens `J_l` is near rank-1
    (58% of energy); masking massive output dims didn't help. Hypothesis: an R-recipe artifact,
    or a real Sarvam property. → Fit a **plain J-lens**.
27. **`rows`** (~$0.72): 32/64 Jacobian rows per backward pass gave 80/70 s/prompt (16: 75), so the
    pass is compute-bound. We used 64 rows.
28. **`jfit_s0`–`s4`**: a 200-prompt J-lens (the explorer's guest-fit recipe) as **5 parallel H200
    shards** of 40 prompts. ~42 min each, ~$18.80 total. Keep-awake was requested because
    `modal run` stops if the local client disconnects.
29. **`merge-j`** (CPU): sum the shards → `j_200p.pt`.
30. **CKA with the J-lens** (CPU): **coupling 0.991, 1 block, Sarvam × Qwen 0.115.** Not an R-lens or
    prompt-count artifact. **Diagnose v3:** the J-lens `J_l` is *not* rank-1 (6.8%). Removing the
    final-norm gain drops coupling to 0.85. → Sarvam's **outlier final-norm gains** (up to 9.2×
    against a median of 1.3) dominate the explorer's probe.
31. **`final`** (~$1.40): the 4-lens pivot (logit, J-200, R-25, J++-25) with shuffled controls, plus
    the causal English ablation with J and R directions.

## Phase 6: follow-ups (~$1.30)

32. **CKA robustness** (CPU, `cka_robust`):
    - Without the gain, Sarvam shows **2 blocks with the boundary at layer 11**, exactly where
      English becomes readable.
    - The `1+w` issue changes Qwen3.6's map by nothing measurable (CKA 0.995), so it was
      **withdrawn as a bug**.
33. **`extra`**:
    - Concept swap: inconclusive, because even the positive control flips only 4–10%.
    - Hindi ablation: −4.4 pts, ≈ English's effect.
    - Sentence clozes: the same pivot, but only 11/40 sentences usable.

## Phase 7: paused

34. Planned (≤ $8): **activation patching** on Sarvam-30B (patch the B-prompt residual, whole state
    vs English-lens component only), then a first look at Sarvam-105B. No code for these was
    written; the user paused.

## Phase 8: Sarvam-105B (not published yet)

Items 35–38, 40 and 41 covered a first look at Sarvam-105B and aren't part of this release (see the
README's last section). One item from this phase applies to the 30B:

39. **Windows:** jpp's `read_text()` uses the locale codepage, so the local smoke tests need
    `PYTHONUTF8=1`. Two pre-existing dry-run-only failures remain in the 30B in-image dry pilot:
    `rows_probe` (CUDA-only) and `fit_jpp` (before `grade` writes the CSV).

## Phase 9: activation patching on Sarvam-30B (~$1.54)

Plan: item 1 of `docs/07_limitations_and_next_steps.md`. It replaces the clamp-swap, whose positive
control was too weak to read.

42. **Design** (`causal_test.run_patch`, `Capture`, `Patch`):
    - One clean pass per (pair, concept) records the final-position residual after layers 13–16.
      Both roles (A and B) reuse it, so an item costs one forward per condition.
    - `Patch` writes B's residual into A's run: the whole state, or `h + QQᵀ(h_B − h)` for an
      orthonormal basis Q. Q is J-lens steering directions (`subspace`) or a random basis.
    - Two conditions beyond the plan. `english_ab` (A's + B's English) also resets A's English.
      `english_other` (a third concept's English, rank-matched) is a stronger control than random,
      because Sarvam's lens directions share a dominant common component.
    - Each condition records the share of the clean B − A difference inside its span.
    - Items need both A and B translated correctly, so B's run really carries B's answer.
43. **Checks** (`smoke_test.py` step 6, tiny model, CPU):
    - A whole-state patch after the last block reproduces B's logits.
    - Patching a run's own residual is a bit-exact no-op.
    - A full-rank basis equals the whole-state patch.
    - A whole-state patch over a band equals patching its last layer alone. Only the final position
      reads its own past, so the "ceiling" is really "patch layer 16".
    - In-image `smoke`: all 42 checks pass, and `patch` is ok in the dry pilot (16.9 min CPU, ~$0.09).
44. **`patch`** (H200, 6.1 min app wall, ~$0.53):
    - Load 1.2 min; 0.11 s/forward; the `patch` stage took 4.1 min against a 6.85 min estimate.
    - 230/320 items kept: A wrong 47, B wrong 42, and 1 with no B-only target token.
    - Ceiling 98.7%. English(A) + English(B) flips 54.8%. English(B) alone 0.4%, Hindi(B) 0.4%,
      target(B) 0%, both controls 0%.

    See §4.2 of the results and §K of the tables. Missing: A + B versions of the Hindi and target
    subspaces, a rank-matched control for the ~21-dim A + B subspace, and "reset A's English only".
    The user approved a follow-up for those (≤ $1).
45. **The split** (`PATCH_SPLIT_CONDITIONS`, stage `patch_split`):
    - New conditions: `hindi_ab`, `target_ab`, `random_ab` (a random subspace of `english_ab`'s
      rank), and `english_a_mean`. The last resets A's English span to the mean of the pair's other
      30 concepts' runs. That's a typical "not A" state, and it inserts nothing of B. Zeroing is the
      existing ablation, already measured.
    - `english_ab` runs again as the in-run reference. It reproduced bit-exactly on the same 230
      items. `run_patch(conditions=...)` picks the set; `PATCH_CONTRASTS` lists the paired contrasts.
    - **`--timeout-margin`** (modal_app, default 6): under a $1 cap, budget + 6 min left too little
      room for the stage's estimate. This run used budget 7.6 + margin 2 = 9.6 min, so the worst case
      was $0.83.
    - The in-image `smoke` hit its 25-min timeout (~$0.13). The previous one took 16.9 min. The dry
      pilot only starts after `smoke_test.py` passes, so the smoke test, with the new split checks,
      passed in the image. The timeout came in the dry pilot's last stages. A local dry `run_pilot`
      on the tiny model covered the new stage's wiring for $0.
46. **`patch2`** (H200, 4.4 min app wall, ~$0.38): load 0.7 min; 0.096 s/forward; `patch_split`
    took 3.0 min. English(A) + English(B) flipped 54.8% (again). Hindi A + B flipped 32.2% and target
    A + B 31.7% (~8 dims each). The rank-matched random flipped 0%, and the A-only reset 0.4% (A kept
    94%). See §4.2.
47. **Rank-matched test** (`PATCH_RANK_CONDITIONS`, stage `patch_rank`). English had ~2.5× Hindi's
    rank, so the user asked whether English is special per dimension.
    - Per item, r = min(English, Hindi, target A + B token counts), which came out at 4–14 (mean 6.8).
    - Each language keeps r tokens by the same rule (`most_specific_first`, `balanced_pick`):
      longest normalized prefix first, ties by id, taking A's word and B's word in turn.
    - A random subspace of rank r is the control. Hindi was cut in 96 items and the target in 114.
    - The driver's patch factory now takes a kind ("", split, rank).
    - **`smoke --stages`**: the full smoke had hit its 25-min timeout, so `smoke` can now dry-run only
      the named stages. The smoke test's tiny J-lens stands in for `j_200p`, and the timeout is 15
      min. This one: 48 checks and `patch_rank` ok in the image, 4.4 min, ~$0.02.
48. **`patch3`** (H200, 4.5 min app wall, ~$0.38; cap $0.90): load 0.5 min; 0.12 s/forward;
    `patch_rank` took 3.3 min.
    - At equal rank: English flipped 52.6%, Hindi 29.6%, target 30.0%, random 0%.
    - English minus Hindi: +23.0 pts [17.0, 29.6]. English minus target: +22.6 [15.7, 30.0].
    - Cutting English from ~21 dims to ~7 cost only 2 pts (54.8 → 52.6), so **English is special per
      dimension**. That overturns the per-share reading in item 46's write-up. See §4.2.

## Phase 10: Sarvam-105B (not published yet)

49. **Shuffled-Chinese control.** `pivot_experiment.CONTROL_ROLES` gained `chinese_shuffled`: a
    different concept's Chinese tokens, minus any token a real role claims. Sarvam's Chinese sets are
    1–3 tokens per concept, mostly the single character itself. It was tested on the tiny model
    (CPU). Its run was on the 105B and isn't part of this release.

## Phase 11: the reader questions (~$6.87)

Plan: answer what a blog reader would ask (`docs/09_reader_questions.md`); the approval went to ≤ $12.

50. **Free analyses first.**
    - `code/token_split.py` re-slices the existing pivots by token count (question 1).
    - The GPT-2 J-lens replication's metric, run on the 30B's existing rows: the J-lens wins at
      15/18 layers.
    - Hindi by pair family.
    - Sarvam's published training-data facts (question 5).
51. **New data**, hand-written and not native-checked:
    - `sentences_parallel.tsv`: 128 sentences, 32 concepts × hi/ta/te/bn, with sentence endings.
    - `chat_prompts.json`: native instructions and language names for 9 source languages.
    - `homonyms.tsv`: 23 senses over 11 English words.
    - `kinship.tsv`: 16 kin terms, ta/te/kn/ml.
52. **`code/reader_questions.py`.**
    - A `Reader` reads every lens off one forward pass and scores on the GPU.
    - `locate` greedy-decodes up to N tokens until the target word begins.
    - Experiments: chat word/sentence in the real chat template (thinking off, the template's
      `<think></think>` reply prefix), cloze, homonym/kinship collapse, and thinking mode (KV-cached
      greedy).
    - New driver stages: `rq_*` and `causal_steer`. `Ablation` gained `alpha` for negative steering,
      after Bhatia, Blank & Nanda 2026.
55. **The chat matcher's strictness.** It requires the reply's first token to be a ≥2-character
    prefix of the word, as in the few-shot pivot.
    - In chat, the 30B often *echoes the source word* instead of translating, and sometimes splits
      a word non-canonically ("ప" + "ాలు").
    - A looser matcher plus a rerun would have broken the $12 cap. So the drops are classified from
      their text instead: 4–6 per format are correct-but-split.
56. **Budget gate.** `rq30` skipped `rq_zh`, `causal_steer` and `rq_think`: `rq_collapse` took
    5.4 min against a 3.0 estimate (two lenses, 882 items).
    - `rq30b` ran the three skipped 30B stages.
57. **Results:** see doc 09. In brief:
    - The pivot holds in real chat use, and the 30B's visible reasoning is 92% English.
    - English-collapse errors are real: homonyms 4.6% wrong-sense vs 0% control, and kin words
      English lumps together are 38% correct vs 78%.
    - Negative steering at −2× costs 41 pts against 6 for projection.
    - The 30B translates into Chinese at ≥ 98%.

## Phase 12: doc 07's next steps, in five parallel sessions (~$6.10)

59. **Plumbing first, then one session per piece, each owning its own files.**
    - `modal_app.py` gained `--lenses`, `--fit-name`, `--prompts-json`, `--shard-glob`/`--out-name`
      for `merge-j`, and a `build-corpus` step.
    - `modal_pilot.py` gained the `pivot_ext` stage (any lens files beside the logit lens), the
      `routing`/`neurons` stages (calling `neuron_experiment.py`), and a named J-lens fit.
    - The sessions then wrote only their own files: `reader_questions.py` batching,
      `neuron_experiment.py`, and the upstream bug-report drafts.
60. **Git Bash path conversion** ($0.57 lost in two launches): `--lenses /vol/...` and
    `--prompts-json /vol/...` reached Modal as `C:/Program Files/Git/vol/...`.
    - `pivot_ext` silently ran logit-only.
    - The Indic fit failed loudly (`FileNotFoundError`), not silently.
    - Fixes: `_volume_path` in `modal_app.py` un-mangles both arguments; use `MSYS_NO_PATHCONV=1`;
      and `pivot_ext` reports a `missing` field.
61. **Free first.** The "matched Qwen J-lens (~$11)" needed no fit: the published lens was already on
    the Volume from the CKA work. The size-matched WikiText lens (`j_wiki_40p`) is the 200-prompt
    fit's first shard, merged on CPU.
62. **Results** (doc 09 §9–11, tables §N):
    - **Indic-fit J-lens:** English holds (L14–16 0.89–0.99, P 0.61).
    - **Qwen J-lens:** P(English) 0.61 vs Sarvam's 0.74.
    - **Routing:** organized by concept and target language, not an English hub.
    - **Layer 0:** neurons that win for one concept across ≥5 scripts (36, vs null ≤ 19).
      *Corrected later (Phase 13): these track word form (cognates, transliterations). Over
      independent word forms the 30B is at chance (doc 09 §11).*

## Phase 13: refusal, math/technical prompts, all 22 languages (docs 11–13; budget $11 → $15 → ≈$20)

The user asked for three more analyses: refusal/alignment, technical and math prompts, and more
languages (all 22 scheduled Indian languages plus Japanese, Russian and Arabic). A coordinator
session built the shared pieces first. Three parallel sessions then each owned one module.

63. **Shared groundwork** (coordinator, CPU, $0):
    - `code/languages.py`: one registry for 28 languages (self-name label, family, script ranges,
      native digits, translator codes). `pivot_experiment`'s labels and families now come from it,
      and the original 11 are unchanged.
    - **Tokenizer coverage**, measured on the 262k vocabulary. Meitei Mayek (0.21 tokens/char) and
      Ol Chiki (0.43) are native. Japanese is not: 15 of 18 tokens for a short sentence are byte
      fallbacks. Qwen's tokenizer is the reverse: it shreds Meitei Mayek and Ol Chiki.
    - **Stage hooks**: a stage named `<module>.<analysis>` calls `module.run(ctx, analysis)`, and
      `--stage-args` JSON reaches `ctx.args`. The sessions never edited `modal_pilot.py`.
    - `code/batch_gen.py`: batched greedy decoding with left padding, an attention mask, explicit
      positions and a KV cache. Exact against lone decoding in fp32 on both tiny models. A negative
      control (attending to padding) drops the match from 8/8 to 3/8, so the check is sensitive.
    - The new stages batch on the 30B too. The published 30B stages keep batch 1.
64. **Prompt sets** (`code/build_xling_sets.py`):
    - XSTest (450, CC-BY-4.0), MGSM (250, with the human bn/te/ja/ru/zh translations, CC-BY-SA-4.0),
      ARC-Challenge stems (150, CC-BY-SA-4.0), MMLU CS/EE/security/ML stems (82: only those that
      stand without their options, MIT), Dolly everyday questions (150, CC-BY-SA-3.0).
65. **Translation** (`code/translate.py`, `code/modal_translate.py`, L40S, $0.95):
    - IndicTrans2 for the 22 scheduled languages, NLLB-200 3.3B for ja/ru/ar.
    - IndicTrans2 is gated. The user accepted its terms and created a Modal secret; the dashboard
      template named it `huggingface-secret`. It runs in a separate Modal app (transformers 4.51.3,
      IndicTransToolkit 1.1.1), so a missing secret can't break the main app.
    - The smoke test ($0.13) found three issues:
      - NLLB emitted some translations twice; exact repeats of ≥15 characters are now collapsed, for
        NLLB only, because Indic reduplication like "धीरे धीरे" is real.
      - IndicTrans2 put spaces around the Meitei Mayek apun iyek (꯭); these are removed.
      - A remote-code update was pulled silently. Both models are now pinned to their revisions.
    - The first full run went out of memory after Hindi ($0.11): 96 long MGSM problems × beam 4.
      Fixed by batching on padded characters (≤12k), splitting a batch in half on OOM, and setting
      `expandable_segments`. Every language is saved as it finishes, so the rerun resumed (17.8 min).
66. **Translation QA** (`code/check_translations.py`):
    - 0 empty outputs; 98–100% of letters in the expected script in every language.
    - The first number check over-flagged (Arabic 67% "kept"). It compared strings, and so missed
      "80 000", "1,5" and "1:00 PM" → "1 बजे". It now compares numeric values: values above 10
      survive in 92–100% of MGSM problems. The strict misses are mostly small numbers written as
      words ("четыре раза").
    - A shell heredoc patch wrote control characters into three regexes (`\b` became a backspace,
      `\1` became `\x01`). They were rewritten from a Python file, with an assert that no control
      characters remain.
    - Trigger words in XSTest's safe prompts are often softened (Hindi "kill a Python process" →
      समाप्त, "end"). Doc 13 measures what that does to refusal.
67. **Batched vs lone.** On the real 30B, batched vs lone gave readout top-1 8/8 and batched greedy
    decoding 7/8 exact.
    - `code/batch_probe.py` ran on 2×L4 ($0.14) with the tiny models, split over both GPUs as
      `device_map="auto"` splits a real one, in fp32 and bf16. Batched == lone **bitwise** for padded
      readout, left-padded decoding, unpadded batches and replicated rows.
    - So a real-model gap is bf16 kernel choice by batch shape, amplified by MoE routing flips. It is
      not a bug. Per-item greedy text can depend on batch shape; aggregates hold.
68. **Fixes in the sessions:**
    - Qwen3.5's linear-attention layers need `DynamicCache(config=...)`; `batch_gen` now builds it so.
    - Sarvam's forward keeps fp32 logits for every position: 32 MGSM prompts of ~1,700 tokens need
      53 GB. A lean generator takes the LM head at the last position only.
    - The same-model English refusal judge scored compliant Kannada replies as refusals (96% fake
      over-refusal). It was replaced by translate-then-classify (doc 13).
    - One launch spent $0.50 with no result. The budget gate's estimate left out driver
      start-up, and the per-item rate was 5× too high.
69. **Results:** doc 11 (the 22 languages), doc 12 (math and technical prompts), doc 13 (refusal,
    published with the second post).
70. **bf16 readout check** (`code/quant_check.py`, ≈ $0.33, doc 14). The refusal work found that
    `jlens.unembed` runs in bf16 and that `pivot_experiment.score` gives tied tokens the better
    rank. The published 30B pivot was re-scored with an fp32 head on the same transported residuals:
    - none of the 92 headline numbers moves (largest top-10 change 1.6 pts; P(English) ≤ 0.35%);
    - the bf16 path reproduces `results/final/pivot_all` row for row (mean |ΔP| 5e-7);
    - the readout logits are mostly small (median top logit 11), so bf16 resolution rarely binds.
    The R-lens, J++ and the `Reader.score` users were not re-checked.

## Lessons that generalise

- Test the pipeline end-to-end on a tiny random model with the **real tokenizer**, and **in the
  exact runtime image**, before paying for GPUs. That caught 6 bugs for cents.
- Random-weight models can't reveal semantic breakage such as missing positions. Add direct checks
  (`inv_freq` vs the formula) and one cheap real-model sanity stage (greedy text, few-shot accuracy).
- "Kernel dust" in value-preserving rewrites isn't harmless in MoEs: rounding flips routing.
  Prefer straight-through forms that are bit-exact.
- Before reporting someone else's bug, measure whether it matters. The `1+w` issue didn't.
- A check that compares a loaded model with a fresh one built from the same config can't catch a
  config-processing bug. Both sides share it. Compare against the raw `config.json`.
- Multi-GPU code needs a multi-GPU test. A tiny model split over two cheap GPUs costs cents.
  Don't assume where a dispatched model's outputs land. Move indices to the tensor they index.
- Shared storage leaks state between models. "Load the lens if the file exists" picked up another
  model's lens. Key every artifact by the model it was fitted on, and check that before use.
- Time estimates must count everything a stage does (here: two lenses × every item), or the budget
  gate silently drops the stages at the end of the plan.
- On Windows, Git Bash rewrites absolute POSIX paths in command arguments. Normalize them where
  they're consumed, and make a missing input fail loudly.
- Parallel sessions in one folder work if shared files change *before* they start, and each session
  then owns disjoint files.
- When batched and lone runs disagree, separate logic from numerics. A tiny model in fp32, on the
  same device split, must match bitwise. If it does, the real-model gap is rounding, and only
  aggregate results should be reported.
- A model judging its own replies in another language can be badly biased. Translate the reply to
  the judge's language first, and validate the judge against an independent rule.
- Machine translation can quietly change what a prompt tests: dropped numbers in math, softened
  trigger words in safety prompts. Check the units that carry the test's meaning, by value, not
  by string.
- Never write regex escapes through a shell heredoc. Patch from a file, and assert that no control
  characters remain.
