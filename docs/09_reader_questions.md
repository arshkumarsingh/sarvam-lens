# 9. Questions a reader would ask

The headline claim is that Sarvam translates between Indian languages through English mid-network
(docs 04 §4.1–4.5). These are the objections a careful reader raises, and what the data says. Every
number comes from `docs/04_results_tables.md` (§M for this doc, §K for earlier ones), and is
regenerated from `results/` by `code/summarize_results.py`.

Runs:
- `rq30` / `rq30b`: Sarvam-30B, 1× H200.
- Analyses marked "free" reuse earlier results.

---

## 1. "Isn't English just easier to read out? English words are one token; Indic words are split."

**No. Tokenization doesn't explain English's lead.** (free; §M1)

- All 32 English concept words are single tokens. Indic words mostly aren't (single-token share
  6–38%), and neither are Chinese words (97% single).
- **Single-token target words don't surface earlier.** In the English band, the target is in the
  top 10 for 23% of items when it's one token, against 32% when it's 2+ (J-lens).
- **English still leads when Hindi has no handicap.** On items whose Hindi word is a single token,
  English's top-10 rate beats Hindi's by:
  - **+28 pts** [18, 39] under the J-lens;
  - +50 [41, 59] under the logit lens.
- **Chinese words are single tokens too, and still far behind English.** Chinese peaks at top-10
  0.25 under the logit lens and 0.11 under the J-lens, against English's 0.93–0.99 (§9's table).
- **The equal-size swap test agrees.** Swapping word A for word B along equal-size spans of
  directions (~7 dims), English flips 52.6% of answers, Hindi 29.6% and the target language 30.0%
  (§K3, activation patching).
- **Whole-word tokens only (doc 17 #1):** English 0.94 in the band, Hindi 0.60, the other
  Indian languages outside the prompt 0.14. But ranked within its own script, each of those languages'
  words is in the top 10 for 0.74 of items (Hindi 0.92, English 0.94). Tokenization doesn't explain
  English's lead over the whole vocabulary; vocabulary size explains much of it (doc 15 §4).

## 2. "That's a few-shot test prompt. Does it hold in real chat use?"

**Yes, in the model's real chat template.** Bigger samples: 448 word requests per instruction
language and 384 sentence translations. (§M3)

The prompt is Sarvam's own template with thinking off, tokenized the way the model card does. The
instruction is either in the **source language** (no English anywhere in the prompt) or in English.
`sentence` asks for a whole sentence (`code/data/sentences_parallel.tsv`, 128 sentences:
32 concepts × hi/ta/te/bn).

English top-10 rate in the English band (L13–16, J-lens), all Indic→Indic pairs:

| Format | Sarvam-30B |
|---|---|
| Few-shot (original) | 0.94 |
| Chat, word request, source-language instruction | 0.77 (n=166) |
| Chat, word request, English instruction | 0.64 (n=252) |
| Chat, sentence translation | **0.88 (n=366)** |
| Different concept's English (control), every format | 0.00 |

- The pivot is a little weaker in chat than in the few-shot format, but clearly present in every
  format. Sentence translation, the most realistic use, keeps it strong.
- **Thinking mode, the model card's default.** The visible reasoning is **92% Latin script**,
  and it names the English word in **58%** of items. A typical trace: *"This Tamil query asks to
  translate the word 'cow' (பசு) into Telugu."* (n=24, capped at 200 tokens, before any trace
  reached its answer; §M7). So in real use the model reasons *out loud* in English, too. Doc 12
  confirms it on 25 languages; doc 17 #7 adds that it keeps reasoning in English even when told, in the
  prompt's own language, to reason in Hindi, Tamil, Bengali or Telugu (own-script share 0.05).
- **What else sits in the workspace.** These are the most frequent tokens in the English band, other
  than concept words (the meta-token idea from Bhatia, Blank & Nanda 2026): " transl" (48%),
  " Malayalam" (61%), " Marathi" (48%). The *task* and the language names are represented in
  English, even when the request is written in Tamil or Telugu.
- **Real-life accuracy, thinking off** (single-word requests): the model gets the word right in about
  39% of requests with a source-language instruction, and 59% with an English one. It often just
  repeats the source word (42% of requests). Sentence translation is 97%. This counts each reply's
  text, so it includes 4–6 correct replies per format whose first token the matcher missed (§M3,
  classification table).

## 3. "Does routing through English actually cause mistakes?"

**Yes, measurably: errors land on the other meaning of the same English word.** (few-shot; §M4;
run `rq30`)

**Homonyms** (`code/data/homonyms.tsv`): 23 senses that share an English word but are different words
in Indian languages, such as finger-nail நகம் vs metal nail ஆணி. Across hi/bn/ta/te/kn/ml, 670 items:

| | Right word | Other sense of the same English word | Matched other word (control) | P(other sense) | P(control) |
|---|---|---|---|---|---|
| All homonyms | 49% | **4.6%** | 0.0% | **0.042** | 0.003 |
| Native words only (no loanword sense) | 45% | **2.1%** | 0.0% | **0.025** | 0.003 |
| Groups with a loanword sense (mouse, crane, bat, bank) | 58% | **9.2%** | 0.0% | 0.070 | 0.001 |

- The other sense carries several times the probability of a matched other word: 8× for native
  words. The difference's CI excludes 0 in every row; for native words it's 0.013–0.031.
- The clearest cases are Hindi माउस (computer mouse) → *rodent*: Tamil/Telugu/Malayalam எலி, ఎలుక,
  എലി. The other is क्रेन (machine) → *bird*, in 5 of 30 crane items. There the source is itself a
  loan of the English word.

**Kinship** (`code/data/kinship.tsv`, Dravidian ta/te/kn/ml): English lumps mother's brother and
father's younger/elder brother into "uncle", and elder/younger brother into "brother".

| | Sarvam-30B |
|---|---|
| Correct, kin words English collapses (n=120) | **38%** |
| Correct, kin words English keeps distinct (n=72) | **78%** |
| Same-English-word sibling answers | 12% |
| Control answers | 0% |
| P(sibling) vs P(control) | 0.136 vs 0.024 |

- The model loses 40 points on the words English collapses. The distinct words are father, son,
  daughter, husband, wife and grandson.
- The errors are mostly elder↔younger swaps: brother and sister get 21% sibling answers; "uncle"
  gets 3%.
- Caveat: these siblings are also close in *meaning*, so kinship is weaker evidence than the
  homonyms, whose senses are unrelated except through English.

Caveats for both lists:
- I wrote them by hand; a native speaker hasn't checked them.
- "Other" answers (25–50%) include synonyms and loanwords outside the lists.
- The sibling and control sets are matched by construction, so these list issues hit both columns
  equally.

## 4. "Is it only one-word translation?"

**No.** The original sentence test had 11 usable sentences. Two bigger tests (§M3):

- **Chat sentence translation** (above): 366 items. English is in the top 10 for 0.88.
- **Plain sentence completion** (no instruction): 128 sentences. The concept word is now located in
  the model's own continuation, so a model that adds an adjective first is no longer dropped. That
  kept **43** sentences (was 11). English is in the top 10 for **0.85** (J-lens; logit 0.73), against
  0.02 / 0.00 for a different concept's English.

## 5. "Why would an Indian model think in Chinese?"

**Unknown, and on Sarvam-30B it's a weak signal.** (§M5) Chinese reaches top-10 0.25 under the logit
lens (0.68 under the Indic-fit J-lens, question 9).

- **What Sarvam has published** (blog, 2026-03-06):
  - The 30B was trained from scratch on 16T tokens.
  - The data is code, web, math, knowledge corpora and multilingual text, with "a substantial
    portion" in the 10 largest Indian languages.
  - Heavy synthetic data, and SFT prompts sourced from open datasets.
  - No language percentages, and no mention of Chinese.
- **It translates into Chinese almost perfectly** (5-shot, 7 source languages × 32 words): 98.2%.
- **A partial explanation, from the J-space meta-token work:** a lens can only read concepts that
  are single tokens, and Chinese words nearly always are. Chinese is a *readable* name for the
  concept where Indic words aren't. That explains why Chinese is visible at all.
- **It isn't used (doc 17 §9).** Removing the Chinese-word directions costs nothing.

## 6. "J-lens failed to replicate on GPT-2. Can you trust it?"

**The headline doesn't depend on it, and on Sarvam the J-lens works.** (free; §M2)

- Every claim holds under the **logit lens**, which has no fitted parameters.
- The failed replication found the J-lens ranked GPT-2's own next token worse than the logit lens. On
  Sarvam-30B, the same metric favors the **J-lens at 15 of 18 layers**. At layer 13 it ranks the
  model's next word at a median of 167, against 1,210 for the logit lens.
- That post warns that rank metrics can be gamed by token frequency. The **shuffled-concept controls**
  rule that out here: a different concept's equally common English word scores 0%.

## 7. "Removing a direction is a weak test. Does English matter more than −6 points?"

**Mostly no (corrected 2026-10-20, doc 17 #5): against a random push of the same size, the push adds
~10 English-specific points.** The table below is the original run. (§M6. J-lens directions, layers
13–16, n=143.)

| Change to the English component | ΔP(right answer) | Still correct |
|---|---|---|
| Removed (α = 1, the original ablation) | −6.3 pts | 94% |
| Reflected (α = 2) | −10.0 | 94% |
| Pushed to −2× (α = 3) | **−41.2** | **59%** |
| Same push on a different concept's English | −8.9 | 90% |
| **English minus control at α = 3** (paired, 95% CI) | **+32.3** [26.0, 38.8] | |

**Correction (doc 17 #5, 2026-10-20).** The control above is a different concept's English, whose
component is smaller, so its push moves the residual less (|Δh| / |h| 0.28 vs 0.42). Against a random
direction pushed exactly as far at every layer, most of the push is edit size:

| | English | Random, as far | English − random |
|---|---|---|---|
| removed (α = 1) | −5.9 | −0.2 | **+5.7** [3.7, 7.8] |
| pushed to −2× (α = 3) | −39.9 | −29.9 | **+9.9** [2.8, 17.9] |

(rerun `p1_push2_30b`; CIs over concepts). The projection is English-specific; the push adds about
10 English-specific points. The text below is the original reading; its "+32 beyond control"
compares pushes of different sizes.

Bhatia, Blank & Nanda (2026) found that projection often does nothing where negative steering
works, and the same holds here: pushing against the English word breaks 4 in 10 translations,
32 points beyond the matched control (34% of items are broken by the English push alone, with the
control push leaving them correct).

## 8. "Is Hindi just the nearest big language for Indo-Aryan pairs?"

**No.** (free) Hindi's top-10 rate in the English band, by pair type:

| | Dravidian ↔ Dravidian | Indo-Aryan ↔ Indo-Aryan | Across families |
|---|---|---|---|
| 30B, J-lens | 66% | 48% | 67% |

Its shuffled control is 0% in every case.

## 9. "Your lenses were fitted on English text. Doesn't that bias them toward English?"

**No. A J-lens fitted only on Indian-language text still shows the English step.** (§N1; Sarvam-30B)

The new lens was fitted on 27 Wikipedia passages, 3 each in en/hi/bn/ta/te/mr/gu/kn/ml (run
`pivot_indic_lens`, same 448 items):

| Lens (fit corpus, prompts) | English top-10, L14 / L15 / L16 | Peak P(English) | Hindi top-10, L16 | Chinese top-10 peak | Controls |
|---|---|---|---|---|---|
| J, Indian-language Wikipedia, 27 | 0.89 / 0.96 / 0.99 | 0.61 | 0.61 | 0.68 | 0.00 |
| J, WikiText, 40 | 0.98 / 0.98 / 0.99 | 0.73 | 0.66 | 0.14 | 0.00 |
| J, WikiText, 200 | 0.98 / 0.98 / 0.99 | 0.74 | 0.68 | 0.11 | 0.00 |
| logit (no fit) | 0.93 / 0.95 / 0.93 | 0.45 | 0.57 | 0.25 | 0.00 |

- The Indic-fit lens reads English slightly later (L13: 0.42 vs 0.80) and a little weaker (P 0.61 vs
  0.74). By L15–16 it agrees with the WikiText lenses.
- It shows *more* Chinese. That's another hint that Chinese is the readable single-token name for
  a concept (question 5).
- 40 and 200 WikiText prompts give the same lens, so the comparison isn't a prompt-count effect.

## 10. "Qwen was only compared with the logit lens. With matched J-lenses?"

**Same conclusion, sharper.** (§N2; Qwen's published J-lens, run `qwen_j`, 363/448 items)

| | Peak P(English) | English top-10 in the workspace band | Second hidden language | Hindi top-10 peak |
|---|---|---|---|---|
| Sarvam-30B, J-lens | 0.74 (L16 of 19) | 0.98–0.99 (L14–16) | Hindi (0.68) | 0.68 |
| Qwen3.6-35B-A3B, J-lens | 0.61 (L35 of 40) | 0.91–0.96 (L26–37) | **Chinese** (top-10 up to 0.88, P 0.19) | 0.19 |

Controls are 0.00 for both. Sarvam's Indic-first training didn't lower its English reliance: under
matched J-lenses Sarvam's P(English) is *higher* than Qwen's. Each model's second hidden language is
its own big training language.

## 11. "Is there an 'English expert' the model routes through?"

**No. The English step is in the representation, not in which experts get picked.** (§N3–N5)

- **Bare words** (32 concepts × 10 languages):
  - In the English band, the router groups words by *concept* across scripts. A word's concept can be
    read from its experts 35% of the time, against 3% chance.
  - It still separates languages: 53%, against 10%.
  - An Indian-language word's experts overlap *less* with its English translation's than with the
    same word in other Indian languages: −0.038 [−0.054, −0.023]. That comes from cognates: 36% of the
    Indian-language translation pairs are one word in two scripts (river is *nadī* in 7 of 8).
    Against non-cognate translations only: −0.010 [−0.032, +0.010] (no difference). English and
    Chinese translations get the same overlap (+0.000 [−0.012, +0.012]).
- **Translation prompts:**
  - Routing depends about equally on the concept (η² 0.30–0.33) and the language pair (0.31–0.32).
  - It strongly encodes the **target** language: 67%, against 33% chance.
  - Overlap with the English word's routing is the *lowest* of English, Hindi, Chinese and the
    target word.
- **Layer-0 neurons:** the first MLP has neurons that pick one concept in ≥ 5 scripts far above
  chance, but they follow the word's *form*, not its meaning.
  - 36 such neurons, against a shuffled average of 2.5 (max 9).
  - Counted in independent word forms (cognates merged), the excess is gone: 2 neurons span
    ≥ 5 forms, against 2.3 shuffled (95th percentile 5).
    The 10 "river" neurons fire on *nadī*, one word in 7 scripts.
  - Script-only neurons are few (20, mostly Latin and Han).
  - Recomputed 2026-10-20 (`neurons_summary_v2.json`). The activations are bf16, so every neuron has
    tied values. The AUROC now gives ties their average rank, and a concept counts as a language's
    winner only when its maximum is not shared (in the shuffled control too). Before, ties were broken
    by row order, so an all-zero neuron scored as a perfect Han neuron and as a "bird" neuron in all
    9 scripts. The counts above are unchanged; individual AUROCs move by up to 0.001, and 61 neurons'
    script counts change by 1–2.

So the lens-readable English word lives in the shared residual stream. The experts are organized by
concept and target language, not by an English hub.

**Does removing the English direction change which experts fire? A little, and not toward English
or the target.** (run `routing_abl30`, ≈ $0.26 with its smoke, under a ≤ $0.60 approval;
`code/routing_ablation.py`)

The §4.2 ablation was re-run with hooks on the routers: J-lens directions, final position, layers
13–16, the same 272 items. It reproduces bit for bit (−5.7 pts, control −0.4). Experts swapped (of
6) at the final position:

| MoE layer | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|
| English removed | 0 | 0.18 | **0.42** | 0.31 | 0.18 | 0.23 |
| other concept's English removed (control) | 0 | 0.16 | 0.18 | 0.13 | 0.08 | 0.11 |

- **About twice the control.** Over layers 14–18, 1 − Jaccard(clean, ablated) is 0.074 [0.064,
  0.085] against 0.038 [0.031, 0.044]. 74% of items swap at least one expert (control 48%).
- **Mostly the size of the edit.** Removing English takes 4.9% of the residual's norm, the control
  3.0%. At an equal removed share, English adds only +0.015 [−0.0002, 0.029], a CI that now includes
  0 (it was [0.001, 0.028] before concept clustering).
- **The swapped experts are marginal and not language-specific.** Two thirds were the lowest-weighted
  of the item's 6 experts (87% one of the bottom two); each carried ~11% of the router weight,
  against 17% on average. By §11's bare-word and prompt routing, they are neither English-selective
  nor target-selective: the shift in English-word selectivity, English minus control, is +0.0003
  [−0.0012, +0.0017], and every other selectivity score's CI spans 0.
- **Routing change barely tracks the damage.** Spearman ρ with the item's probability drop is 0.13
  [−0.01, 0.26], a CI that now includes 0 (was [0.01, 0.25]; control 0.02). The 14 translations the ablation broke swapped as many experts as
  the ones that survived (1.3 each over layers 14–18).
- **Checks:** layer 13's router (it runs before its block's edit) never changed. Clean reruns are
  identical. Earlier positions kept their experts in 541 of 544 item-conditions. The 3 exceptions
  are bf16 numerics, not information flow: the final token's new experts change the shapes of the
  other tokens' expert matmuls.

CIs resample the 32 concepts with all their items (`routing_ablation_summary_v2.json`, 2026-10-20).
The run's own `routing_ablation_summary.json` resampled the 272 items as if independent, though each
concept appears in up to 10 pairs; that made every interval here slightly too narrow.

The ablation acts through the residual stream, not by re-routing. The experts that change were
nearly tied, and which ones change doesn't explain what breaks.

## 12. "Can I reproduce it or try the lenses?"

- **Code and commands:** `docs/06_reproduce.md`. This doc's runs are the `rq_*`, `causal_steer` and
  `token_split.py` steps.
- **Lenses:** the fitted J-, R- and J++ lenses and the Indic-fit J-lens (`j_indic_27p`) are on
  Hugging Face as `arshkumarsingh/sarvam-lens` (linked from the README).
- **Upstream bugs:** the Sarvam-30B RoPE bug is reported on [Hugging Face](TODO-ROPE-LINK).

---

**Cost of this doc's runs:**
- Questions 1–8 (`rq30`, `rq30b` and a smoke test): ≈ $2.22.
- Questions 9–11: ≈ $5.26. That's the Indic lens $3.37, Qwen $1.03, and routing and neurons $0.86
  (that row also includes a 105B launch).
- The routing ablation (`routing_abl30` + smoke): ≈ $0.26.

Details are in `docs/05_costs.md`.
