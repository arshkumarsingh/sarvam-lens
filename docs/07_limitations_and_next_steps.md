# 7. Limitations and next steps

## Limitations (read before citing anything)

1. **Single-word tasks dominate.** The main evidence is 5-shot word translation over 32 concrete
   nouns. Sentence-level checks are smaller: 43 plain sentence completions and 366 chat-template
   sentence translations (doc 09 §2, §4). The first sentence check had only 11 usable items.
2. **Readouts are correlational.** The ablations show concept- and depth-specific effects, but
   they're modest (−4 to −7 pts), and the ablation touches only a small subspace at one position.
   A bigger intervention might show a larger (or smaller) role. *Tested in doc 17:* removing
   English, Hindi and the target word together still leaves 94% correct; removing English at every
   position costs only 1.9 pts more than at the last; and the large "push to −2×" effect (−41)
   turned out to be mostly edit size: a random direction pushed exactly as far costs −30. The
   English-specific role is the projection's 3–6 pts, plus ~10 under the push.
3. **The swap test was uninformative:** its positive control reached only 10%. Activation patching
   (§4.2) replaces it, and its ceiling reached 98.7%.
4. **Fit budgets are modest:** J-lens 200 prompts, R-lens / J++ 25. The explorer's reference fits
   use 460–1,000. Others report 25-prompt lenses can be under-converged. (The 200-prompt J-lens
   agrees with the 25-prompt R-lens on the pivot readouts.)
   In the 25-prompt R fit, 16/144 layer-expert slots had < 50 token positions and fell back to the pooled
   Jacobian, so J++ is partly built on pooled copies. The 200-prompt J experts have 0 fallbacks.
5. **One small Indic-corpus lens.** The main fits used WikiText (English). Since then, a J-lens
   fitted on 27 Indian-language Wikipedia passages was tested, and it still shows the English step
   (top-10 0.89–0.99 at L14–16, P 0.61; doc 09 §9). The R-lens and J++ weren't refit, and 27 prompts
   is small.
6. **Qwen's Indic target sets are coarse** (often one character), so compare the models on the
   English role, which is word-specific in both. The pivot comparison was logit-lens only at first.
   It now also runs under Qwen's published J-lens (doc 09 §10), which its authors fitted, not us, so
   the two J-lenses aren't fitted identically.
7. **Token scoring is heuristic:** prefix sets with a 2-character floor plus the tokenizer's first
   token. Some legitimate continuations are missed (e.g. a bare `ப` for பால்).
8. **Concept translations were written by hand** and not checked by native speakers. The correctness
   filter removes items the model doesn't produce, but alternates may be incomplete.
9. **19 layers** means "mid-network" is a handful of layers, and depth resolution is coarse.
10. **CKA explanations are partial.** The gain explains most of Sarvam's flat map. Even without it,
    Sarvam stays unusually unlike Qwen (0.27), and why is open.

## Approved plan (≤ $8)

1. **Activation patching on Sarvam-30B: done** (≈ $1.54 over three runs; results in
   [`04_results.md`](04_results.md) §4.2, tables in §K). B's final-position residual was patched
   into A's run at layers 13–16. The whole state flips 98.7% of answers to B. B's state on the span
   of both words' English directions (~21 dims) flips 54.8%, and neither half alone does anything:
   removing A's English flips 0.4%, inserting B's English 0.4%. A rank-matched random flips 0%.
   At **equal rank per item** (mean 6.8), English flips 52.6%, Hindi 29.6% and the target word
   30.0%. English minus Hindi is +23.0 pts [17.0, 29.6], so English is special per dimension, but
   not the only carrier. Its own limits:
   - The equal-rank test picks tokens by one rule (longest prefix first, A and B in turn). The rule
     treats the languages differently in practice. English's top r are case/space variants of the
     full word (`▁water`, `water`, `▁Water`, `Water`, `▁WATER`). Hindi has no case, so its top r
     include shorter prefixes and synonyms (`▁पानी`, `पानी`, `पान`, `▁जल`). Another rule, such as
     tokens ranked by the lens at that layer, could move the gap. Item (4) of the request, a fixed k
     per word, wasn't run: it didn't fit that step's cap.
   - The A-only reset uses the other concepts' mean; zeroing is the separate ablation (−5.7 pts).
   - One position (the final one), one band (13–16) and one lens (J-200).
2. **Sarvam-105B: a first look.** Not part of this release; see the README's last section. The 30B
   has no shuffled-Chinese control in the published pivot (its Chinese peak is 25%); doc 17's reruns
   have one (0.00).

3. **Reader questions: done** ([`09_reader_questions.md`](09_reader_questions.md), tables §M).
   Their own limits:
   - The homonym, kinship, sentence and chat-instruction lists are hand-written, not native-checked.
   - The chat matcher is as strict as the few-shot one (first token ≥ 2 characters of the word).
     Chat replies that echo the source word dominate the drops, so the chat items kept are those the
     model actually translated.
   - The thinking probe was cut at 200 tokens: it measures the reasoning, not the final answer.
   - The Chinese latent's *cause* is still unknown (no published language mix).

## Other worthwhile next steps

Done since (doc 09 questions 9–11, tables §N, ≈ $5.26):
- **A lens fitted on the Indic corpus:** the pivot holds (doc 09 §9).
  - Limits: 27 prompts, a J-lens only (no R / J++), and the 30B only.
- **A matched Qwen J-lens:** Qwen's *published* J-lens (no fit needed), doc 09 §10.
  - Limits: it was fitted elsewhere with its own recipe (25 prompts, per this project's `qwen_J25`
    label), not Sarvam's 200-prompt WikiText fit.
- **Neuron level:** expert routing by language and layer-0 cross-script neurons (doc 09 §11).
  *Corrected:* the layer-0 "concept" neurons track word form, not meaning (at chance on the 30B
  over independent word forms). The 30B's English-vs-Indic routing difference comes from cognates.
  Removing the English direction moves later experts ~2× the control, mostly marginal ones, with
  no shift toward English- or target-language experts.
  - Limits: bare words plus the translation prompts' final position; top-k expert *sets* only (no
    router weights); the shared expert and mid-layer neurons are not analyzed.

Done since (Phase 13 of the engineering log and later):
- **All 22 scheduled languages + Japanese, Russian, Arabic** in the pivot, with a causal test (doc 11).
  Limits:
  - Word lists come from Wiktionary/Wikidata, cited per cell, but they are not native-checked.
  - Dogri was untestable (6/32 words). Maithili (10 words) and Bodo (13) are thin.
  - Translation into the low-resource languages often fails, so their "as target" rows are small.
  - The causal test covers 6 pairs (or, ur, ks, mni, sat, ja sources), and its "more than the
    original languages" compares separate runs on different items.
- **Math, science, technical and everyday prompts; native-digit arithmetic; reasoning language** (doc 12).
  Limits:
  - Prompts are machine-translated (IndicTrans2/NLLB). Their QA is in `code/data/xling/translation_qa.json`.
  - The whole-text "English share" has a high floor (≈70% Latin at layer 0), so only differences
    between domains mean anything.
  - Direct-answer MGSM is at floor (0–19%), so machine vs human translation is unresolved on accuracy.
  - Thinking traces were cut at 400 tokens, over 8 problems per language.
  - Hint-test cells are small (11–19 problems, 1–10 hint-followers per cell).
- **Refusal across 26 languages** (doc 13, which will be published with the second post, with its
  limits). Doc 04 §4.9 has the part the first post reports.
- **Language-neutral vs English-centric** (doc 15): shared and only weakly skewed. English and
  Hindi are co-anchors on the 30B; Qwen has a real English hub (line carrier).
  Most of the lens's English lead is vocabulary competition. Limits: 32 concepts, one word per
  cell, and retrieval is near ceiling on Sarvam (0.87–0.95), which compresses hub gaps.
- **Is the English reasoning faithful?** (doc 12, hint test): it verbalizes hints in every language
  (63/63), but when cut off and made to answer, it gives the hint's number even where the
  reasoning had found the right answer (86%).
  Limits: 20 problems, traces cut at 1,638 tokens and answers forced, greedy repetition loops,
  machine-translated hint, 30B only. *Rerun in doc 17 #8* with 60 problems, 3,072 tokens and sampling:
  the result holds (verbalised 0.99–1.00; uptake 0.36 / 0.33; on natural conclusions 0.43 / 0.46).
  Still: machine-translated hints, 9–21 followers per cell, and 65–69% of hinted traces still don't
  close within 3,072 tokens.
- **Readouts on failed low-resource translations:** English is still present (0.88 vs 0.96), so
  the failure is in producing the language (doc 11 §4c).

- **bf16 quantization in the lens readouts: checked, the published pivot holds** ([doc 14](14_bf16_readout_check.md), ≈ $0.33).
  `jlens`'s `unembed` runs the final norm and LM head in bf16, and `pivot_experiment.score` gives tied
  tokens the better rank. The same 30B pivot items were scored with a bf16 and an fp32 head:
  - **None of the 92 headline numbers in doc 04 §4.1 moves.** The largest top-10 change is 1.6 pts;
    P(English) changes by ≤ 0.35%.
  - The bf16 path reproduces `results/final/pivot_all` row for row.
  - Ties nudge top-10 up very slightly (0.11% of rows flip in fp32), and P is unbiased.
  - The readouts' logits are mostly not in the hundreds (median top logit 11, p99 122). The refusal
    work's whole-unit problem was the model's own final output on chat prompts.
  - Part of the small pre-L11 English bump is a tie artefact (about 40% at L8, 12% at L9), so
    the J-lens reads even less before L11 than the bf16 numbers show.
  - Not covered: the R-lens and J++, and the `Reader.score` users (docs 09, 11 and 12).

- **Doc 17 (2026-10-20, ≈ $6.64 on the 30B runs): the post-1 follow-ups.** Every language as a pivot role (English
  the most readable, every well-known language readable within its script), a semantic-neighbour
  control (lexical), joint English + Hindi + target ablation (additive, not a bottleneck), the
  digits ablation, a push curve with a size-matched random control (the push is mostly edit size),
  all-position ablation and writer attribution (English at the final position, written by attention
  then experts), MGSM thinking in the prompt's language, and the hint test at 3×. Limits:
  - Third-language token sets lose tokens to same-script tracked languages (Devanagari languages
    behind Hindi, Assamese behind Bengali), so those languages read low partly by construction.
  - The neighbour list is hand-made, 2 words per concept; the 30B result sits on its pre-set boundary.
  - The size-matched random control matches |Δh| per layer, not the direction's fit to the model's
    own variation; the first version matched the first layer only (doc 17 deviation 5).
  - #7: machine-translated instructions; native traces rarely conclude (15%), so their accuracy is
    mostly the forced answer's. #7 and #8 ran on the 30B only.
  - The writer scores use the J-lens's average Jacobian; they say what a component adds to a word's
    readout, not what the model downstream does with it.
- **Doc 17's next steps** (not run): whether native-script reasoning concludes with more room (≈ 6k
  tokens, ≈ $3–4); a size-matched control for doc 04 §4.2's activation patching (its rank-matched
  random subspaces move the state less than a word subspace does: 0.2–0.5% of the B − A difference
  against 6–8%).

Still open:
- A larger, native-speaker-checked concept list and sentence set, including abstract concepts; and
  native checks of the homonym and kinship lists.
- Native-speaker checks of the new-language word lists and of a sample of the machine-translated
  prompts. Dogri needs a source.
- The machine-translated MGSM accuracy on the 30B (the stage is fixed; it was out of memory) and
  thinking-on accuracy with a long enough budget to finish the reasoning.
