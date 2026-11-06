# 4. Results

Every number below comes from the auto-generated [`04_results_tables.md`](04_results_tables.md)
(rebuilt from `results/` by `code/summarize_results.py`). Figures are in `results/`.

## 4.1 Sarvam-30B passes Indic→Indic translation through English

10 Indic→Indic pairs, 256 items Sarvam translated correctly (`results/final/pivot_all`):

| Top-10 rate, J-lens (200p) | L11 | L13 | L14 | L15 | L16 | L17 |
|---|---|---|---|---|---|---|
| English translation | 20% | 80% | **98%** | **98%** | **99%** | 86% |
| English, *different* concept (control) | 0% | 0% | 0% | 0% | 0% | 0% |
| Hindi translation | 0% | 42% | 60% | 55% | 68% | 62% |
| Hindi, *different* concept (control) | 0% | 0% | 0% | 0% | 0% | 0% |
| Target word (the answer) | 0% | 11% | 22% | 27% | 42% | 84% |
| Chinese | 1% | 9% | 5% | 4% | 6% | 11% |

- **All four lenses agree** (logit, J-200, R-25, J++-25). P(English) peaks at 0.45 (logit), 0.74
  (J), 0.75 (R), 0.56 (J++).
- **Every pair** shows it: peak P(English) runs from 0.63 (bn→gu) to 0.85 (te→kn, romanized-hi→ta)
  under the J-lens. The prompts contain no English.
- **Nothing is readable before layer 11** under any lens, in any language.
- **Hindi** is a real, concept-specific secondary latent: high rank, low probability (≤3% at
  L13–16; the J-lens reaches 6.7% at L17).
- **English is the most readable language, not the only one present** ([doc 17](17_post1_followups.md)
  #1, run `p1_t1_30b`, which reproduces this table to within 0.02). Every other language of the word
  list was tracked as a role too. The Indian languages that are neither in the prompt nor Hindi reach
  the band's top 10 for 0.12 [0.08, 0.17] of items over the whole vocabulary (English 0.94, Hindi
  0.58; their shuffled controls 0.00), but **within their own script for 0.74** [0.66, 0.82] (control
  0.03), near the prompt's own two languages (0.85). The big-10 Indian languages read 0.73 there, the
  eight low-resource ones 0.33. Whole-word tokens only
  change nothing (English 0.94, other Indian 0.14), so English's lead is not a prefix-set artefact.
- **The readout is lexical**: an English *neighbour* of the concept (water → liquid) is in the top
  10 for 0.10, an associate (water → ice) for 0.01; both are elevated (median rank 69 and
  258 of 262k, controls 45,454) but the top of the readout is the translation itself (doc 17 #2).
- First-pass logit-lens figure: `results/ropefix/pivot_logit/logit_mean_prob.png`. R-lens figure:
  `results/lensfit/pivot_lenses/r_moe_25p_top10_rate.png`. 4-lens figures: `results/final/pivot_all/*.png`.

## 4.2 The pivot is causally used, but it isn't a bottleneck

Subspace ablation at the final position, 272 correctly translated items:

| Removed | ΔP(target) | 95% CI | still correct |
|---|---|---|---|
| English word, layers 13–16 (J-lens dirs) | **−5.7 pts** | [−7.1, −4.3] | 94.9% |
| different concept's English, 13–16 | −0.4 | [−0.81, +0.05] | 98.2% |
| English word, layers 5–8 | −0.3 | [−0.78, +0.07] | 97.8% |
| English word, 13–16 (R-lens dirs) | −6.7 | [−8.5, −5.1] | 94.5% |
| **Hindi word, layers 13–16** (J-lens dirs) | **−4.4** | [−5.8, −3.1] | 94.1% |
| different concept's Hindi, 13–16 | −0.4 | [−0.69, −0.03] | 98.9% |

The effect is concept-specific and depth-specific, so the latents are *used*. But ~95% of
translations survive, so neither latent is a bottleneck.

**Update (reader questions, §M6):** projection understates the English route. On 143 items,
*negative steering* the English component to −2× costs −41.2 pts (control −8.9), and only 59% of
translations stay correct. Reflecting it (α = 2) costs −10.0. "Not a bottleneck" holds for removing
English, not for actively pushing against it.

**Correction (doc 17 #5, 2026-10-20): most of that push is edit size, not English.** A random
direction pushed exactly as far as the English edit at every layer of the band costs −29.9 pts
(English −39.9 in the rerun, 77% vs 59% still correct), so the English-specific part at α = 3 is +9.9
[2.8, 17.9], not +32. The different-concept control looked mild because a different concept's English
component is smaller, so pushing it moves the residual less (|Δh| / |h| 0.28 vs 0.42); against a
random push of its own size it costs nothing extra. The projection itself is English-specific
against a same-size random edit (+5.7 [3.7, 7.8]).

**Further tests (doc 17):**
- **English, Hindi and the target word together** (#3): removing English + Hindi costs −11.3 pts
  [8.7, 13.7] (91% still correct), about the sum of the two; adding the target word's own directions
  *raises* P(target), and all three together leave 94% correct. Additive, and not a bottleneck even
  jointly.
- **Where English is read** (#6a): projecting it out at every position costs −7.6, against −5.8 at
  the last position only (+1.9 [1.1, 2.7]); at the source word's tokens alone, 0.0. English sits at
  the final position.
- **Who writes it** (#6b): attention at L11 and L13–14 (a few heads: the top 10 of 1,216 carry 44%),
  then the MoE at L15–17; the last block removes it while writing the target. The same components
  write the Hindi and target-word directions at L13–14 almost as strongly. No expert set writes
  English more concentratedly than the target word (top-5 share 24% vs 28%).

**Concept swap: inconclusive (superseded by the patching below).** Swapping English "water"↔"fire"
flipped the answer to "fire" in 2.9% of items at scale 2 (null 0%, early layers 0%, Hindi 0.7%). But
the positive control (swapping the answer tokens themselves) reached only 4% / 10%, so the
intervention is too weak on Sarvam to interpret. At scale 4, every condition destroys the output.

### Activation patching: the words' readout subspaces carry the concept, English's most per dimension

Run concept A's prompt (e.g. Tamil→Telugu "water"). At the final position, after each of layers
13–16, write in part of the residual from concept B's prompt (same pair, "fire"; B is the swap's
half-way derangement). Success = the Telugu answer becomes B's word. Subspaces are spans of J-lens
(200p) steering directions. 230 items in which Sarvam translated both A and B correctly. Three runs
on the same 230 items: `results/patch`, `results/patch2` and `results/patch3`. The condition shared
by runs 1 and 2 reproduced exactly. Tables are in §K of the tables file.

| Patched into A's run, layers 13–16 | dims | share of (B − A) | flips to B [95% CI] | still A | ΔP(B) |
|---|---|---|---|---|---|
| whole residual from B (ceiling) | 4096 | 100% | **98.7%** [97.0, 100] | 0.9% | +0.89 |
| B's state on English(A) + English(B) | ~21 | 8.2% | **54.8%** [48.3, 61.3] | 20.9% | +0.43 |
| B's state on Hindi(A) + Hindi(B) | ~8 | 6.2% | **32.2%** [26.1, 38.3] | 46.1% | +0.26 |
| B's state on target(A) + target(B) | ~8 | 4.5% | **31.7%** [25.7, 37.8] | 41.3% | +0.27 |
| *equal rank r per item (mean 6.8):* English(A) + English(B), cut to r | 6.8 | 7.6% | **52.6%** [46.1, 58.7] | 22.6% | +0.42 |
| *equal rank:* Hindi(A) + Hindi(B), cut to r | 6.8 | 6.1% | **29.6%** [23.9, 35.7] | 49.1% | +0.24 |
| *equal rank:* target(A) + target(B), cut to r | 6.8 | 4.4% | **30.0%** [24.3, 36.1] | 44.3% | +0.26 |
| *equal rank:* random subspace of rank r (control) | 6.8 | 0.2% | 0% | 98.7% | −0.000 |
| B's state on English(B) only (insert B) | ~10 | 1.8% | 0.4% [0, 1.3] | 99.1% | +0.006 |
| B's state on Hindi(B) only | ~4 | 1.1% | 0.4% [0, 1.3] | 99.1% | +0.004 |
| B's state on target(B) only | ~4 | 1.0% | 0% | 97.0% | +0.001 |
| English(A) reset to the other concepts' mean (remove A, nothing of B) | ~10 | 1.8% | 0.4% [0, 1.3] | 94.3% | +0.003 |
| random subspace, English(A)+(B)'s rank (control) | ~21 | 0.5% | 0% | 98.7% | +0.000 |
| a third concept's English, English(B)'s rank (control) | ~10 | 0.4% | 0% | 98.7% | +0.000 |
| random subspace, English(B)'s rank (control) | ~10 | 0.3% | 0% | 100% | +0.000 |

*Share* is the fraction of the clean B − A residual difference that lies inside the subspace.
The A-only reset uses the mean of the pair's other 30 concepts, not zero: zeroing is the ablation
above, and the mean is a typical "not A" state. It costs A −7.6 pts, close to the ablation's −5.7.
For equal rank, r = the smallest of the three A + B token counts for that item (4–14; usually
Hindi's or the target's). Each language keeps r of its tokens by the same rule: longest
(most word-specific) prefix first, ties by token id, alternating between A's word and B's word.
In practice, English's r are case/space variants of the whole word (`▁water`, `Water`, …). Hindi,
which has no case, also gets shorter prefixes and synonyms (`पान`, `जल`).

- **The test is readable.** The whole-state ceiling flips 98.7% of answers, where the swap's
  positive control reached 4–10%.
- **Neither half alone does anything; together they swap the answer.** Removing A's English leaves A
  the answer in 94% of items. Inserting B's English leaves A in 99%. Each flips 0.4%. Doing both, by
  writing B's state onto the ~21 dims spanned by both words' English directions, flips **54.8%**:
  half the full-state effect (ΔP(B) 49% of the ceiling), in every one of the 10 pairs (38–69%). A
  random subspace of the same rank flips 0%. So what matters is *which word occupies the
  subspace*, not the subspace's size. A survives the removal of its English (other routes), and
  loses only when B takes its place.
- **English is special per dimension, but not the only carrier.** With all their tokens, English's
  A + B span (~21 dims) flips 54.8%, Hindi's 32.2% and the target word's 31.7% (~8 dims each).
  Cut to the same rank per item (mean 6.8), English barely changes (**52.6%**), and Hindi (29.6%)
  and target (30.0%) barely change either. English's extra tokens were nearly redundant. At equal
  rank, English beats Hindi by **+23.0 pts [17.0, 29.6]** and the target word by **+22.6
  [15.7, 30.0]** (paired). It leads in 9 of the 10 pairs and ties the target word in mr→bn.
  Hindi and target are indistinguishable
  (−0.4 [−6.5, +6.5]). A random subspace of rank r flips 0%. English's few most specific directions
  also cover more of how B's state differs from A's (7.6% vs 6.1% and 4.4%).
- So the concept's identity at layers 13–16 is readable from, and editable through, all three
  languages' word subspaces. English's is the strongest, as its dominance in the lens readouts
  (§4.1) suggests. Hindi and the target word each carry a substantial, roughly equal part.
- The swap's reading, that Sarvam's lens directions read well but steer poorly, applies to fixed-size
  clamps. Natural activations patched into the same subspaces do steer the answer.

## 4.3 Not just translation: sentence clozes

**Superseded by the larger test in doc 09 §4 (tables §M3).** Locating the concept word anywhere in
the model's own continuation keeps 43 of 128 sentences on the 30B: English is in the top 10 for
0.85 (J-lens; logit 0.73), against 0.02 / 0.00 for a different concept's English. The first,
11-item version is kept below for the record.

On 11 natural Hindi/Tamil/Telugu/Bengali sentences that Sarvam completed with the expected noun
(no translation instruction), the English word is in the J-lens top 10 for **100%** of items at
layers 14–16 (shuffled control 0–9%), and Hindi for 50–67% (control 0%). The pattern is the same as
in translation, but n is small (29/40 sentences were dropped, mostly because Sarvam inserts an
adjective first).

## 4.4 Qwen3.6-35B-A3B: English and Chinese

Logit lens, the same 10 Indic→Indic pairs (`results/qwen_zh`), layer 35 of 40:
- English in the top 10 for 93% of items (P 0.46).
- **Chinese for 92%** (P 0.15).
- Hindi for 1%.

Sarvam's logit-lens English peak (0.45) is the same as Qwen's. **Each model's hidden languages track
its main training languages**, and Sarvam's Indic-first training didn't lower its English reliance.

**With matched J-lenses** (Qwen's published J-lens, run `qwen_j`, 363/448 items; tables §N2), English
is in Qwen's top 10 for 0.91–0.96 of items across L26–37, peaking at P 0.61 (L35). Chinese is up to
0.88 (P 0.19), and Hindi ≤ 0.19. Sarvam-30B's J-lens peaks at P(English) **0.74**, so under matched
lenses the Indic-first model shows *more* English, not less. Its second language is Hindi, where
Qwen's is Chinese. All shuffled controls are 0.00.

## 4.5 J-space CKA: Sarvam is an outlier, explained by its final-norm gains

| Lens | Gain | Blocks | Boundary | Blockiness | Layer coupling |
|---|---|---|---|---|---|
| Sarvam J-200 | explorer recipe | **1** | none | **0.00** | **0.991** |
| Sarvam J-200 | **none** | 2 | **ρ 0.61 (layer 11)** | 0.21 | 0.846 |
| Sarvam R-25 | recipe | 2 | ρ 0.61 | 0.06 | 0.969 |
| Qwen J-25 | recipe | 2 | ρ 0.51 | 0.32 | 0.694 |
| Explorer, 41 models | recipe | ~3 | ~0.48 / 0.64 | ~0.31 | ~0.76 |

- Sarvam × Qwen matched-depth CKA: 0.115 (recipe), 0.272 (no gain). Explorer cross-family pairs: 0.44–0.70.
- **Diagnosis:** the J-lens `J_l` is not rank-1 (top direction 6.8%). Sarvam's final RMSNorm has
  outlier gains (max 9.19, median 1.34), and folding them into the probe makes every layer's steering
  geometry nearly identical. Sarvam's steering vectors span ~2 dimensions at layer 0, against
  Qwen's 63.
- With the gain removed, Sarvam's **input-side block ends at layer 11, the depth where English
  first becomes readable**.
- The explorer code's `1+w` handling for Qwen3.5/3.6 changes Qwen3.6's map by nothing measurable
  (CKA 0.995 between the two maps). **Not a meaningful bug.**
- Figures: `results/cka/sarvam_J200_cka.png` (uniform square), `results/cka/qwen_J25_cka.png`,
  `results/cka_robust/sarvam_J200_nogain_cka.png`.

## 4.6 Sarvam-105B (not published yet)

A first look at Sarvam-105B isn't part of this release; see the README's last section.

## 4.7 Reader questions (full answers: [`09_reader_questions.md`](09_reader_questions.md), tables §M)

| Question | Short answer |
|---|---|
| Is English just easier to read (single tokens)? | No. English leads Hindi by 28–50 pts even when the Hindi word is one token. Single-token targets don't surface earlier. Whole-word tokens only: English 0.94, other Indian languages 0.14 (doc 17 #1). But within each script, every well-known language's word is readable (0.74), so English is the most readable language, not the only one present. |
| Does it hold in real chat use? | Yes. English top-10 is 0.64–0.88 in the chat template (0.88 for sentence translation), and the control is 0. The visible reasoning is 92% English. |
| Does English routing cause mistakes? | Yes. Errors go to the other sense of the same English word: 4.6% vs 0% for a control, 2.1% for native words only. Kin words English lumps together are 38% correct vs 78%. |
| Only one-word translation? | No. Sentence translation (366 items) and 43 plain sentence completions show the same pivot. |
| Why Chinese? | Unknown. Sarvam-30B translates into Chinese at 98%, and Sarvam publishes no language mix. Chinese words are single tokens, hence readable. The 30B's Chinese latent is weak (top-10 peak 0.25) and isn't used: removing it costs nothing (doc 17 #9). |
| J-lens failed on GPT-2? | On Sarvam-30B it beats the logit lens at 15/18 layers on that test, and every claim holds under the fit-free logit lens. |
| Lenses fitted on English text: biased? | No. A J-lens fitted only on Indian-language Wikipedia shows English top-10 0.89–0.99 at L14–16 (P 0.61, vs 0.74 WikiText). |
| Qwen with matched J-lenses? | Sarvam P(English) 0.74 vs Qwen 0.61 under J-lenses. Second languages: Sarvam Hindi, Qwen Chinese (0.88 top-10). |
| An "English expert"? | No. Routing partly groups words by concept and target language. Indic words' experts overlap less with English's than with other Indic words', but on the 30B that comes from cognates: non-cognate pairs give −0.010 [−0.032, +0.010], n.s. Removing the English direction moves later-layer experts ~2× the control, mostly marginal low-weight ones, and not toward English- or target-selective experts. The earlier "layer-0 cross-script concept neurons" (36 of them) track word *form*: counted over independent word forms, the 30B is at chance (2 vs 2.3). |
| Is −6 pts all English is worth? | Mostly, yes (corrected by doc 17 #5). Pushing the English component to −2× costs −41 pts, but a random direction pushed exactly as far costs −30. The English-specific part is +10 [3, 18] at α = 3. The earlier "+32 beyond control" compared pushes of different sizes (a different concept's English component is smaller). |

## 4.8 Side findings

| Finding | Status |
|---|---|
| transformers 5 loads Sarvam-30B with **uninitialized RoPE** (`inv_freq` from `torch.empty_like`: zeros in 10 of the first 12 tiny-model runs, arbitrary huge values in 2; no position info either way) | Confirmed, fixed in `sarvam_support.py`. The bad buffer was printed on the tiny model; on the full checkpoint only its consequences were measured (`bosfix` vs `ropefix`, not a controlled A/B). Reported to Sarvam ([Hugging Face discussion](TODO-ROPE-LINK)). |
| Sarvam's best raw-text prefix is `[@BOS@]` (96%), not the tokenizer's `<bos>` (61%) | Measured |
| jpp's identity rule (`z·sigmoid(z)`) is not bit-exact in bf16, and flips MoE routing | Confirmed on Sarvam; bit-exact fix in `sarvam_support.py` |
| jpp's fit/held-out split leaks 31/76 fit items if evals are emptied | Guarded in `run_jpp.py evaluate` |
| Sarvam-30B on J++'s readout evals (greedy): multihop 17/93, multilingual 28/107, order-ops 26/55, poetry 1/98 | Measured (`results/correctness.csv`) |
| The lens readouts unembed in **bf16** and give tied tokens the better rank | Checked ([doc 14](14_bf16_readout_check.md)): scoring the same 30B pivot with an fp32 head moves none of the 92 §4.1 headline numbers (largest top-10 change 1.6 pts, P(English) ≤ 0.35%). Part of the small pre-L11 English bump is a tie artefact (about 40% at L8, 12% at L9). |
| Sarvam's 262k tokenizer has **no tokens for 5 Tamil and 3 Telugu digits** (௧ ௨ ௪ ௬ ௯, ౫ ౬ ౭); they fall back to bytes. It also has no Japanese kana. | Measured on the tokenizer. Native-digit arithmetic in those two scripts is near the floor (Tamil 1%, Telugu 8%). |

## 4.9 All 22 languages, prompt domains, and a first look at refusal (docs 11 and 12)

The full refusal study (doc 13 onward) will be published with the second blog post. The rows on refusal
below are the part the first post reports.

| Question | Answer |
|---|---|
| Do the other 14 scheduled languages pivot through English too? ([doc 11](11_new_languages.md)) | Yes for the 13 that could be tested (Dogri has 6 of the 32 words, so 21 of the 22 scheduled languages are covered), as source and as target. That includes the Perso-Arabic scripts, Meitei Mayek and Ol Chiki. English top-10 at the band is 0.84–1.00 (J-lens) in every group with more than 3 kept items (Kashmiri and Bodo as targets rest on 1 item each); the shuffled control is at most 0.03. Japanese, Russian and Arabic do the same. Hindi is again a second latent. |
| Is the English step causally used for them? | More than for the original languages, on six added source languages (or, ur, ks, mni, sat and ja, one pair each). Removing English costs 13.2 pts [10.6, 16.2] (published original pairs −5.7). The comparison is across separate runs with different pairs and items. Controls are 0.1–0.2; the early band is null. Per-pair effects (n = 22–31) overlap, so the pairs can't be ranked. |
| What breaks for low-resource languages? | Translating *into* them, not the English step. 30B: Kashmiri 1/32, Bodo 1/13, Konkani 3/18, Sindhi 7/31, Santali 12/64. Misses write the source word or a neighbour language. |
| Is English there when that translation fails? | Usually. On 178 failed 30B translations into ks/sat/mni/brx/kok/sd, English is in the J-lens top 10 at L13–16 for 0.88 [0.83, 0.94], against 0.96 on 44 successes (controls ≤ 0.01). It's highest when the model writes a neighbour language instead (0.97) and lowest for unlisted target-script words (0.81). The source word isn't held mid-network even when it's copied (0.11 on source-copy misses), so the copy is decided late. The failure is in production, not comprehension (doc 11 §4c). |
| Are math and technical prompts more "English inside"? ([doc 12](12_prompt_domains.md)) | Not much. At equal prompt length, math reads 1.7–2.9 pts more English than science or technical prompts (J-lens; the logit gaps of 0.5–1 pt are within noise). Science and technical read no more English than everyday prompts. The naive +8 pts is mostly prompt length. CIs resample prompts with all their translations (corrected 2026-10-20). |
| How is arithmetic done? | Through English number words. In native-digit sums, the English answer word is in the band's top 10 for 0.76 (J-lens) and 0.54 (logit lens) of problems; a wrong answer's word scores 0.00. The native digit appears only at layer 17 of 19, after the word peaks at layers 15–16. Deleting the answer's word ("seven") doesn't break the sum (−0.2 pts vs the wrong answer's word); the nine number words together cost 3.9 beyond a random subspace of the same size. Pushed against, they matter no more than a same-size random push (doc 17 #4). |
| What language is the visible reasoning in? | English, in all 25 languages tested: 93–97% Latin (8 problems per language, the first 400 tokens). The 8 low-resource languages pool at 0.958. The rest is mostly quotation of the prompt, with no fallback to Hindi. It calls Arabic "Urdu" (7 of 8 prompts). Told, in the prompt's own language, to reason in Hindi/Tamil/Bengali/Telugu, the 30B still reasons in English (own-script share 0.05). Made to (the think block opened with a native sentence), it reasons natively to the end at the same accuracy (0.72 vs 0.71 for English-instructed, +0.8 [−6.3, +7.9], 60 MGSM problems × 4 languages), but only 15% of those traces conclude within 3,072 tokens (doc 17 #7). |
| Is the middle space language-neutral or English-centric? ([doc 15](15_concept_geometry.md)) | Shared, only weakly skewed, not English-centric. Across 27 languages, concepts retrieve each other at 0.83–0.84 at L13–16 (30B, line carrier; bare words 0.45–0.48; chance 0.03), 0.45–0.51 on Qwen. As retrieval hubs, English and Hindi nearly tie on the 30B (0.925 vs 0.910 at L15, CI includes 0; +0.016 [0.001, 0.030] at L13). Qwen has a clear English edge (+0.14, line carrier). Centroid rankings depend on the definition; none gives English a large lead. Ranked within its own script, Hindi reaches top-10 0.92 and the target 0.85, against English 0.94 (whole vocabulary: 0.58 and 0.27), so most of English's readout lead is vocabulary competition. |
| Is the English reasoning faithful? ([doc 12](12_prompt_domains.md)) | Partly. 20 MGSM problems with a wrong "a teacher says the answer is N" hint, in en/hi/ta/bn/sat/ru, thinking on. The 30B follows the hint 33% [24, 44] of the time with a native-language hint and 40% with an English one (English prompts 42%). Every hint-following trace mentions the hint (65/65, Wilson [94, 100]; per-language cells of 1–10 followers), so a native-language hint is not hidden in the English reasoning. But 86% of hint-following traces contain the correct answer, note the mismatch, and still answer the teacher's number (forced-answer probe). The answer is unfaithful to the reasoning, not the reasoning to the hint. Caveats: traces cut at 1,638 tokens and forced, greedy loops, small cells. **Rerun at 3× (doc 17 #8):** 60 problems, 3,072 tokens, sampling: uptake 0.36 [0.29, 0.44] native vs 0.33 [0.25, 0.40] English hint (+4 [−6, +13]), English prompts 0.33; verbalised 0.99 / 1.00 (every cell ≥ 0.92); on natural conclusions only, uptake 0.43 / 0.46; 98% of hint-following traces contain the right answer. The result holds. (Answers re-read 2026-10-20 with a parser that keeps "70 000" and "16,00" whole; CIs over problems.) |
| Is refusal decided in English? | Yes. In Indian-language prompts the 30B refuses in the prompt's script (n = 2,207), English refusal openers are in the J-lens top 10 at layers 13–16 for 56% [54, 57] (complied replies 18%). The language's own openers are at 0.0% until layer 17. The pattern holds in 21 of 22 languages. Qwen's band holds *Chinese* refusal phrases (不可以 73%, 我无法 71%; 10 Indian languages). |
| How often does it refuse outside English? | Less. On unsafe XSTest prompts the 30B refuses 82% [76, 86] in English and 42% [40, 43] pooled over the 22 Indian languages (Sanskrit 28%, Hindi and Bengali 57%). Over-refusal is 14% and 8% by the judge, likely overstated (only 4 of 16 hand-checked judge-flagged safe replies were real refusals, 25% [10, 49]; judge negatives were not checked). Translation softening explains about 2 of the ~25 points in Hindi and Bengali. |
| Recognized or not? | Mostly recognized. At the user message's last token (layer 13, before the reply), an English-trained logistic harm probe has AUROC 0.91 pooled over 22 Indian languages (English 1.00). At English's 10%-FPR threshold it flags 74% [72, 76] of the harmful Indian-language prompts the model complied with (n = 1,234), 87% of those it refused, and 13% of safe prompts. |
| Could an English harm probe monitor other languages? | Not as calibrated. Harmful prompts caught at 1% English FPR: English XSTest held-out 97% (93/96; FPR 2%, 2/120); 22 Indian languages 58% (FPR 4%); original 8 languages 69%, other 14 52%. False alarms transfer; recall doesn't. |
| Measurement caveat | The same model as an English judge mislabels non-English replies (96% fake over-refusal in Kannada). Labels use translate-then-classify (80% agreement with a manual check on 55 replies). |
