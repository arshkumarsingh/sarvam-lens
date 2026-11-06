# 12. Prompt domains: are math and technical prompts more "English inside"?

Run 2026-10-19. Code: `code/domain_experiment.py` (analyses `profile`,
`digits`, `mgsm`, `mgsm_think`, `hint`, `hint_force`; `python domain_experiment.py tables ../results/<run>` regenerates the
tables). Runs: `xl_dom_30a`, `xl_dom_30b`, `xl_dom_30think`, `xl_hint_30`, `xl_hint_30_force` (Sarvam-30B).

## Question

The headline (docs 04, 09) is that Sarvam routes Indic→Indic word translation through English at
layers 14–16 of 19. Does the amount of English inside depend on what the prompt is *about*? If the
model learned math and technical material mostly in English, a math or computer-science question
asked in Tamil might be read more "in English" than an everyday question in Tamil. We test this in
26 languages (English, the 22 scheduled Indian languages, Japanese, Russian, Arabic) on four
domains, plus two math-specific probes: native-digit arithmetic and MGSM word problems.

## Answer

- **No, not in any way that survives the controls.** The absolute English share in the middle layers
  (~0.76–0.85) is mostly a lens floor (0.71–0.83 already at layer 0), so only same-layer differences
  between domains are interpreted. Math reads 1.7–2.9 points more English than science or
  technical questions *of the same length* under the J-lens; the logit-lens gaps (0.5–1 point)
  can't be told from 0. Science and technical read no more English than everyday questions (−0.01
  to −0.02, clearly below 0 only for science under the logit lens). The naive gap, J-lens MGSM vs
  everyday +0.08 in 22/22 scheduled languages, is mostly prompt length (MGSM questions are 5×
  longer). It is absent under the logit lens.
- **Arithmetic is done through English number words.** In arithmetic written entirely in native
  digits (Devanagari, Bengali, Meitei Mayek, Ol Chiki, Arabic…, no words at all), the English word
  for the answer ("seven") is in the top 10 in the middle layers for 0.76 (J-lens) and 0.54 (logit
  lens) of problems. A wrong answer's word scores 0.00. The native digit the model then writes is
  barely visible there (0.05–0.09), and the ASCII digit is lens-dependent (J-lens 0.32, logit 0.00).
- **With thinking on, the reasoning is English in every language.** 93–97% of the visible
  reasoning's letters are Latin for MGSM problems (8 per language, the first 400 tokens of each
  trace) in all 22 scheduled languages (pooled 0.957), the eight low-resource ones included
  (Kashmiri, Sindhi, Manipuri, Santali, Bodo, Dogri, Konkani, Maithili: 0.958), and in Japanese,
  Russian and Arabic. The rest is mostly quotation of the prompt. No language falls back to Hindi or
  to its own script. The model names the prompt's language in all 176 Indian-language traces, but
  calls Arabic "Urdu" (7/8). That's slightly above the 92% found for translation requests.
- **The English reasoning is faithful about hints in every language; the final answer isn't always
  faithful to the reasoning.** Given a wrong hint in the prompt's own language ("A teacher says the
  answer is N"), the 30B's reasoning mentions the hint in **all 65** traces whose answer followed it (Wilson
  [0.94, 1.00]). That holds for Hindi, Tamil, Bengali, Santali, Russian and English prompts alike, and
  for the hint in the prompt's language or in English, though each language-by-hint cell holds only
  1–10 such traces. The reasoning translates the hint into English in its first
  lines. The answer follows the hint for 33% (native hint) and 40% (English hint) of the problems the
  model otherwise solves (English prompts: 42%). So nothing suggests a native-language hint is
  acknowledged less. But in 86% of those traces the reasoning had found the right answer and said the
  teacher's number doesn't match. It then looped trying to reconcile the two until the token budget
  ran out; made to answer, it deferred to the teacher. The influence is out in the open; it's the
  answer that departs from the reasoning.
- **Without reasoning, the model mostly guesses MGSM answers** (0–10% direct-answer accuracy), so
  machine vs human translation can't be separated on accuracy. On the lens metric the two
  translations agree within 0.01 (80 items).
- **Follow-ups (doc 17, 2026-10-20).** Deleting the English answer word ("seven") doesn't break
  native-digit sums (−0.2 pts vs a wrong answer's word); the nine number words together cost about 4 pts.
  Told to reason in the prompt's language, the 30B still reasons in English; made to (a pre-filled
  native opener), it reasons natively as accurately (+0.8 [−6.3, +7.9]) but rarely concludes (15%).
  The hint test at 3× the problems, run to completion: uptake 0.36 native vs 0.33 English hint,
  verbalised 0.99–1.00, uptake on natural conclusions 0.43 / 0.46. Details in the sections below.

## Method

### Prompts

| Set | Source (license) | Items used |
|---|---|---|
| mgsm | MGSM test (CC-BY-SA-4.0), with its human translations for bn/te/ja/ru/zh | profile 80; accuracy 60 en+human |
| science | ARC-Challenge stems that stand alone (CC-BY-SA-4.0) | 80 |
| technical | MMLU CS / security / EE / ML stems (MIT) | 80 |
| everyday | Dolly open/general QA and brainstorming (CC-BY-SA-3.0) | 80 |

The same item ids are used in every language (parallel prompts). Translations are the
coordinator's machine translations: IndicTrans2 for the 22 scheduled languages, NLLB-200 for ja/ru/ar
(`code/data/xling/<set>.json`). MGSM's human translations are the check on machine translation. The
two MGSM chat demos (`mgsm-084`, `mgsm-134`) are excluded everywhere.

### 1. Profile: the script of what each layer predicts

- **Format.** Sarvam's chat template with thinking off, exactly as deployed: `chat_text(msg)` +
  `"<think></think>\n"` (Sarvam ignores `<|nothink|>`, so the empty think block is what turns thinking
  off). Under the causal mask, a position inside the user message sees only the template's fixed
  prefix before it, so the question span reads the same as plain text would. The template adds one
  extra read: the reply position (the last prompt token).
- **Readout.** One batched forward per prompt (right-padded; rows match lone forwards to 1e-6 in fp32
  on the tiny models). At every layer, under both lenses (J-lens and logit), at every
  position of the question span, the lens's **full softmax** (no top-k, no filter) is summed by the
  script of each vocabulary token.
- **Vocab → script map** (built once from the 262k vocabulary). A token's column is the single
  script of its letters (`languages.SCRIPTS`: 16 scripts), "otherscript", "mixed", **neutral** (no
  letters: digits of any script, punctuation, whitespace, operators), byte-fallback pieces, or
  special/unused tokens. Sizes: Latin 148,467; Han 18,907; Devanagari 13,676; Cyrillic 13,395;
  Bengali 9,821; Arabic 8,364; neutral 8,302; special 6,414; Telugu 4,306; Kannada 4,199; Malayalam
  4,018; Gujarati 3,974; Tamil 3,847; Gurmukhi 3,828; Oriya 3,449; Meitei 2,355; Ol Chiki 1,351;
  **kana 0** (Japanese kana are byte fallback); bytes 262.
- **Metric: English share of letter mass**, `E = P(Latin) / P(any letter token)`. Normalizing by
  letter mass removes the neutral class, which MGSM inflates mechanically (numbers). Raw P(Latin) is
  in the CSVs.
- **Positions.** Three aggregates per prompt: the mean over all question positions; over **native
  positions** (the input token itself is in the prompt's own script; the primary read, because a
  Latin input token, e.g. an untranslated term, invites a Latin next token); and the reply position.
- **Contrasts.** The band (layers 13–16) against the early band (5–8) and the final layer (the model's actual next-token prediction). Domains are compared within
  each language: domain − everyday, 95% bootstrap CIs (1,000 resamples over items, the two domains
  resampled independently). Pools weigh each language equally and resample items within each
  language. **Length-matched** contrasts compare items in the same language and question-length bin
  (edges 15/20/25/30/40/50/65/80 tokens), weighted by the smaller cell, with CIs by resampling cells.
- **Uniform-vocabulary level.** A uniform distribution over the vocabulary gives an English share of
  0.60 (Latin's share of letter tokens); the tiny random model reads 0.60, and 0.595 on the top-10
  metric.

### 2. Digits: does the model compute in ASCII when the prompt has none?

- Few-shot single-digit arithmetic, `a+b=` or `a-b=`, answers 1–9, written **entirely in one digit
  system**, six demos of the same operation per query, no words at all:
  `௨+௫=௭ / ௬+௨=௮ / … / ௧+௧=`. 72 problems (36 additions, 36 subtractions) × 14 systems: ASCII
  (en/ja/ru have no native digits) and 13 native sets covering 21 Indic languages plus Arabic
  (Devanagari is shared by 8 languages, Bengali by 2, Extended Arabic-Indic by ur/sd/ks). Plain text
  with Sarvam's `[@BOS@]` (the few-shot best, doc 02 §2.7). Prompts are built in code
  (`digit_problems`, seeded demos); one example per system is in `digits_prompts.json`.
- At the answer position, every layer, every lens: probability (jpp readout filter, as in the
  pivot) and top-10 rate of the ASCII answer digit, the native answer digit, the English number word
  ("seven", any case, with or without a leading space), and the same three for a wrong answer (a
  fixed shift of the answer, never an operand or 0): the shuffled controls. The band's top-10 rate is,
  per item, the share of band layers with the token in the top 10.
- The final layer's top token is the model's answer: native digit, ASCII digit, word, another
  script's digit, or other.

### 3. MGSM: direct answers and the answer position

- **Format.** No instruction (an instruction would need its own translation and would put English
  in the prompt). Instead, two fixed MGSM problems in the same language (`mgsm-084`, `mgsm-134`, the
  shortest; answers 15 and 720) are earlier chat turns whose assistant replies are just the number.
  The test question is the third user turn; the reply starts after `<think></think>\n`. Demo answers
  use native digits when the question's own numbers do, else ASCII (IndicTrans2 kept ASCII digits
  throughout, so all demos were ASCII).
- Greedy batched decoding (`batch_gen.generate`; 10 new tokens). The reply's first number is
  scored, with any script's digits mapped to ASCII. The model does answer with a number first (e.g. *"124\n\nHere's the step-by-step…"*).
- The **same forward** gives the lens readout: the generation's own prefill is hooked, so the answer
  position costs no second pass (checked on the tiny models: prefill residuals equal a lone forward's
  to 1e-6, and tokens match lone decoding).
- **Accuracy on strict-pass problems** (every number of the English problem still written in digits
  after translation; `translation_qa.json` → `flags.mgsm_number_missing`), with all problems beside
  it. **MT vs human** for bn, te, ru (and ja on the profile), paired by item.
- **Thinking on** (`mgsm_think`): run `xl_dom_30a`, 16 problems × 6 languages with human
  translations (en, bn, te, ja, ru, zh); run `xl_dom_30think`, 8 problems × the 25 machine-translated languages (22 scheduled + ja/ru/ar). The 8 problems are the first whose numbers survived translation
  in all 25 languages (99 of 248 qualify), so every language gets the same problems. Japanese is the
  exception: NLLB's repetitions push 6 of those prompts past 400 tokens, so it takes the first 8
  strict problems that fit (`max_prompt_tokens` 400). Up to 400 tokens, greedy, all 200 prompts in one
  batch (`_generate_lean`, see Engineering). Measured: the script mix of the visible reasoning
  (`languages.script_shares`: letters only); which of its non-Latin words are quoted from the prompt
  (exact word or substring) and which the model wrote itself; and whether the reasoning names the
  prompt's language. Compared with the 92% Latin found for translation requests (doc 09 §2).

### 4. Hints: is the English reasoning faithful? (30B)

The hint paradigm of Chen et al. 2025, "Reasoning models don't always say what they think" (after
Turpin et al. 2023): add a cue pointing to a specific answer, see whether the answer follows it, and
whether the reasoning admits to using it.

- **Items.** The first 20 MGSM problems (demos excluded) whose numbers all survived machine
  translation into Hindi, Tamil and Santali. Bengali and Russian use MGSM's human translations;
  English is the source.
- **Hint.** A wrong answer N: the correct answer ± 10% (at least 1), the sign fixed per problem, never
  correct (18 → 20, 540 → 594, 64 → 58). The text is "A teacher says the answer is N.", in the
  coordinator's machine translation of "A teacher says the answer is 25." (IndicTrans2; NLLB for
  Russian; `code/data/domains/hint_translations.json`, first sentence), with 25 replaced by N. It is
  appended to the question after a blank line. The Tamil sentence reads "a teacher answers N",
  understandable but less idiomatic than the alternative. Manipuri was dropped because its
  translation mixed Devanagari into Meitei Mayek; Santali is the low-resource language.
- **Conditions per problem and language:** (a) plain; (b) the hint in the prompt's language; (c) the
  English hint inside the native-language prompt. English prompts have (a) and the English hint.
  That makes 340 prompts.
- **Generation.** Thinking on (the chat template's default), greedy, all 340 in one batch
  (`_generate_lean`, 64-row prefill chunks), up to 1,800 tokens with a 20-minute decode deadline.
  The deadline stopped it at 1,638 tokens (0.74 s per step at batch 340).
- **Forced answers** (early answering, Lanham et al. 2023). Only 16% of traces closed their think
  block within the budget, so every trace, closed or cut, is closed with `</think>` and the cue "The
  final answer is ", and the model writes 12 more tokens; the first number is its answer (`hint_force`).
  On the 54 traces that closed naturally, forced and natural answers agree for 85%. Some of the rest
  are my parser misreading the natural answer (list numbering), not real disagreements.
- **Metrics.** Accuracy without a hint. **Uptake**: among problems answered correctly without a hint,
  the share whose hinted version answers N. **Verbalisation**: among those hint-following traces, the
  share whose reasoning mentions the hint. Mentions are matched by English cues (teacher, hint, "says
  the answer", "the answer given", "claim", "matches 329", …) and by the prompt language's word for
  teacher. The plain traces, which have no hint, give the matcher's false-positive rate. 95%
  bootstrap CIs over problems.
- **Manual check.** I read the 6 hint-following traces the first matcher scored as silent (all six
  acknowledge the hint without saying "teacher", so the cues above were added), 12 random
  followers, and 3 traces read earlier: 21 in all.

## Results

### Profile: high English everywhere; no real domain effect once prompt length is matched

**Sarvam-30B (run `xl_dom_30b`)**: 8,720 prompts (26 languages × 4 domains × 80, plus MGSM's 5
human translations × 80). English share of letter mass in the band (layers 13–16), native positions,
pooled over the 22 scheduled languages (each language weighs the same), 95% CIs:

| Domain | J-lens | Δ vs everyday | languages up / down | logit | Δ vs everyday | languages up / down |
|---|---|---|---|---|---|---|
| everyday | 0.76 [0.75, 0.77] | – | – | 0.67 | – | – |
| science | 0.78 [0.77, 0.79] | +0.03 [0.01, 0.04] | 17 / 0 | 0.65 | −0.02 [−0.03, −0.01] | 0 / 12 |
| technical | 0.77 [0.77, 0.78] | +0.02 [0.00, 0.03] | 7 / 0 | 0.67 | −0.01 [−0.01, +0.00] | 0 / 4 |
| mgsm | 0.84 [0.84, 0.85] | **+0.08 [0.07, 0.09]** | **22 / 0** | 0.68 | +0.00 [−0.01, 0.01] | 5 / 2 |
| final layer (output), everyday | 0.03 | | | 0.03 | | |

("Languages up / down": languages whose own 95% CI for the difference excludes 0. Pooled CIs resample
the 80 prompts per domain, each with all its translations: `profile_by_family_v2.csv`.)

Read naively, the J-lens says math is more English inside in every scheduled language. Three
checks say the effect is small and mostly not about the domain:

1. **The lenses have a Latin floor.** At the early band (layers 5–8), where no lens reads any
   language in the translation task, Latin already holds 0.72–0.78 of letter mass (own script 0.01).
   That is above the 0.60 a uniform distribution over the vocabulary would give. So the absolute
   level (0.76) is mostly a lens prior on diffuse readouts, not "76% English thought"; only
   differences at matched layers mean something. Pooled curves: under the J-lens the share sits at
   0.67–0.84 at every layer from 0 to 16 (MGSM peaks at 0.86–0.89 at layers 14–15) and collapses at
   17–18 (0.35, then 0.03). Under the logit lens it is flat (0.69–0.76) to layer 14, 0.57–0.67 at
   15–16, then 0.33–0.41 and 0.02–0.04. On the 30B, running text has no mid-network English bump
   like the one in word translation.
2. **Only the J-lens sees it.** The MGSM gap is band-specific under the J-lens (layers 5–8: +0.019
   [0.011, 0.027]; layers 13–16: +0.082 [0.073, 0.092]) but absent under the logit lens (+0.002 and
   +0.004, CIs including 0). This project's rule is that a claim must hold under the parameter-free logit lens.
3. **It is mostly prompt length.** Question spans average 17 tokens (everyday), 41 (science), 38
   (technical) and 90 (MGSM) in the scheduled languages, and under the J-lens a longer question reads
   more English. Comparing items of similar length (`profile_length_matched_v2.csv`; CIs resample each
   domain's prompts with all their translations):

| Contrast, length-matched | J-lens | logit | items |
|---|---|---|---|
| mgsm − science | +0.017 [0.009, 0.025] | +0.010 [−0.000, 0.019] | 616 |
| mgsm − technical | +0.029 [0.018, 0.040] | +0.005 [−0.007, 0.016] | 507 |
| science − everyday | −0.011 [−0.023, +0.000] | −0.024 [−0.035, −0.014] | 718 |
| technical − everyday | −0.010 [−0.020, +0.000] | −0.009 [−0.019, 0.001] | 914 |

   (MGSM and everyday questions barely overlap in length: 39 matched items on the 30B give +0.032
   [0.013, 0.059] J, −0.013 [−0.034, 0.011] logit.) A regression on the 30B with language fixed effects
   and log question length agrees (CIs resample the 320 prompts): MGSM +0.017 [0.004, 0.030] (J),
   −0.006 [−0.018, 0.005] (logit); science −0.005 [−0.014, 0.007] and −0.023 [−0.032, −0.014];
   technical −0.010 [−0.018, 0.001] and −0.010 [−0.019, −0.001].

- **So: at equal length, math is 1.7–2.9 points more English than science or technical (J-lens;
  the logit-lens gaps of 0.5–1 point are within noise), and science and technical are no
  more English than everyday questions.** Both are small next to the language-independent level, and
  neither looks like "math and CS are thought in English".
- **No copying confound.** The translations have almost no Latin letters (0.1% of characters; 2%
  for technical, where 78% of items have none). Restricting to items with no Latin letters at all
  gives the same gaps (30B J: mgsm +0.082, science +0.026, technical +0.016).
- **Machine vs human translation (MGSM, same 80 items, 30B J-lens):** bn 0.87 vs 0.86, te 0.85 vs
  0.84, ja 0.83 vs 0.82, ru 0.89 vs 0.88. The translator doesn't move the metric.
- **The reply position** (the last template token, where the model is about to answer) is strongly
  English under the J-lens in every domain (0.91–0.92 pooled; domain gaps within ±0.005), and
  0.72–0.79 under the logit lens (MGSM lower: −0.05, 22/22 languages). Before it answers, the 30B's
  band state is English whatever the topic.
- **Every language, every family** (30B J-lens, everyday → MGSM): Indo-Aryan 0.64–0.79 →
  0.77–0.87, Dravidian 0.78–0.82 → 0.85–0.87, Manipuri and Bodo 0.80 → 0.86, Santali 0.81 → 0.87;
  Japanese, Russian and Arabic 0.72–0.78 → 0.83–0.89; English 0.94 → 0.98. Kashmiri and Sindhi are
  lowest (0.64, 0.69 everyday). The per-language table is `profile_by_lang.csv`.

### Digits: native-digit arithmetic goes through the English number word

Top-10 rate in the band (layers 13–16; per item, the share of band layers with the
token in the top 10), 72 problems per system, 95% bootstrap CIs (pooled rows: the 72 problems are
resampled with all their digit systems, `digits_summary_v2.csv`). Controls are the same three
readings for a wrong answer. "11 native systems": every native digit set whose ten digits are all
vocabulary tokens (all but Tamil and Telugu).

| Model, lens, prompts | Accuracy | English word ("seven") | ASCII digit | native digit | word ctrl | ASCII ctrl | native ctrl |
|---|---|---|---|---|---|---|---|
| 30B J-lens, ASCII prompts (en/ja/ru) | 1.00 | 0.83 [0.78, 0.89] | 0.76 [0.70, 0.81] | – | 0.00 | 0.00 | – |
| 30B logit, ASCII prompts | 1.00 | 0.47 [0.41, 0.52] | 0.00 | – | 0.00 | 0.00 | – |
| **30B J-lens, 11 native systems** | 0.87 | **0.76 [0.70, 0.82]** | 0.32 [0.29, 0.34] | 0.05 [0.04, 0.06] | 0.00 | 0.00 | 0.00 |
| **30B logit, 11 native systems** | 0.87 | **0.54 [0.47, 0.60]** | 0.00 | 0.09 [0.08, 0.11] | 0.00 | 0.00 | 0.00 |

- **When a prompt is written entirely in Devanagari, Bengali, Meitei Mayek, Ol Chiki or Arabic
  digits, the middle layers carry the answer as the English word.** "seven" is in the top 10 for
  three quarters of problems under the J-lens and half under the logit lens.
  A wrong answer's word scores 0.00 everywhere, so this is the answer, not a generic number-word
  prior.
- **It's the translation pivot's shape.** 30B, 11 tokenized systems (the table's pool): the English
  word peaks at layers 15–16 (J 0.83/0.92, logit 0.78/0.74); the native digit appears at 17 (J 0.85,
  logit 0.69) and is the output at 18, where the word is gone (0.04). Over all 13 native systems,
  byte-fallback Tamil and Telugu included, the same curve is J 0.71/0.79 and logit 0.67/0.63 at
  L15/16, native digit J 0.86 and logit 0.67 at L17 (`digits_curves.csv`, `all_native`).
- **The ASCII digit is not the intermediate.** Only the 30B J-lens reads it (0.43–0.57 at layers
  15–16); the logit lens reads 0.00 in the band. At the output it is a
  runner-up (top 10 for 38% of native-digit prompts). So "the model computes in ASCII digits" is not
  supported; "the model computes through English number words" is, under both lenses.
- Per system (30B J-lens), the English word's band top-10 is 0.71–0.86 in 10 of the 11 tokenized
  systems; Malayalam is 0.42.
- **Tokenizer cases.** Tamil has 5 of 10 digits outside the vocabulary (௧ ௨ ௪ ௬ ௯) and Telugu 3
  (౫ ౬ ౭), so their prompts are partly byte-fallback, and the model answers with a byte piece (30B:
  58/72 Tamil, 59/72 Telugu items; accuracy 0.01 and 0.08). They are excluded from
  the pooled rows.
- **Accuracy per system** (30B): 1.00 for ASCII, Bengali, Extended Arabic-Indic, Meitei Mayek, Ol
  Chiki and Arabic; 0.90–0.97 for Devanagari (7/72 answered in ASCII), Gujarati and Gurmukhi;
  0.49–0.64 for Kannada, Oriya and Malayalam. Kannada and Oriya are often answered in *another*
  script's digits (Ol Chiki, Meitei Mayek: 37/72 and 27/72 items), a confusion between rare digit
  sets.

**Is the English word needed? (doc 17 #4, 2026-10-20.)** Projecting the answer's English word
("seven", every case and space variant) out of the final position at the band, on the correctly
answered native-digit sums (692, J-lens; CIs over the 72 problems):
- **Deleting it doesn't break the sum**: −0.2 pts [−0.5, +0.2] against the wrong answer's word, 99%
  still correct. The ASCII digit's directions don't matter either.
- **The number words together matter a little**: all nine number words' directions removed cost 3.9
  pts [2.8, 5.2] beyond a random subspace of the same rank and size.
- **Pushed against** (α = 3), they matter no more than size: a same-size random push hurts more than
  the number words' does.

### MGSM: the model mostly guesses without reasoning

Direct answers with thinking off. "Strict" = problems whose numbers all survived translation in
digits; for English and human translations every problem is strict.

| Model | Variant | n (strict) | Accuracy, strict | Accuracy, all | MT − human, paired strict | English share at the answer position, band |
|---|---|---|---|---|---|---|
| 30B | en | 60 | 0.10 [0.03, 0.18] | same | – | 0.97 (J) |
| 30B | bn (human) | 60 | 0.07 [0.02, 0.13] | same | – | 0.89 (J) |
| 30B | te (human) | 60 | 0.00 [0.00, 0.06] (Wilson) | same | – | 0.90 (J) |
| 30B | ja (human) | 60 | 0.03 [0.00, 0.08] | same | – | 0.88 (J) |
| 30B | ru (human) | 60 | 0.03 [0.00, 0.08] | same | – | 0.90 (J) |
| 30B | zh (human) | 60 | 0.05 [0.00, 0.12] | same | – | 0.86 (J) |

- **Accuracy is near the floor without reasoning**: the model answers a number at once (often a
  round guess like 120 or 180) and then explains. So MT-vs-human accuracy differences can't be
  resolved: the machine-translated MGSM stage ran out of memory (see Engineering) and was not
  repeated, because at 0–10% accuracy it couldn't have separated the translations either.
- **At the answer position the band is English** (J-lens 0.86–0.90 for the five human
  translations, 0.97 for English). The answer's first digit is rarely in the top 10 in the band
  (ASCII 0.09–0.31, native 0.00).
- **The thinking-off reply continues in English.** After the number, the reply goes on in Latin
  script for 58–83% of its letters for Bengali, Telugu, Japanese, Russian and Chinese problems
  (*"Here's the step-by-step…"*). That's a small sample (≤ 10 tokens per reply).

### Thinking on: the reasoning is English in all 25 languages, low-resource ones included

**All 22 scheduled languages + Japanese, Russian, Arabic** (machine translations, the same 8 parallel
strict-pass problems per language, thinking on, greedy, 400 tokens; run `xl_dom_30think`). "Quoted" = share of the reasoning's non-Latin words that
occur in the prompt; "names the language" = traces whose reasoning names the prompt's language.

| Language | Family | Latin share | quoted | names the language | other language named first |
|---|---|---|---|---|---|
| Hindi | Indo-Aryan | 0.965 [0.961, 0.969] | 92% | 8/8 | – |
| Bengali | Indo-Aryan | 0.969 [0.963, 0.974] | 98% | 8/8 | – |
| Tamil | Dravidian | 0.943 [0.933, 0.952] | 82% | 8/8 | – |
| Telugu | Dravidian | 0.959 [0.951, 0.966] | 90% | 8/8 | – |
| Marathi | Indo-Aryan | 0.962 [0.953, 0.970] | 98% | 8/8 | – |
| Gujarati | Indo-Aryan | 0.962 [0.952, 0.971] | 95% | 8/8 | – |
| Kannada | Dravidian | 0.953 [0.938, 0.967] | 80% | 8/8 | – |
| Malayalam | Dravidian | 0.949 [0.934, 0.963] | 69% | 8/8 | – |
| Assamese | Indo-Aryan | 0.949 [0.934, 0.963] | 87% | 8/8 | – |
| Odia | Indo-Aryan | 0.952 [0.927, 0.969] | 86% | 8/8 | – |
| Punjabi | Indo-Aryan | 0.962 [0.953, 0.970] | 93% | 8/8 | – |
| Urdu | Indo-Aryan | 0.970 [0.965, 0.975] | 95% | 8/8 | – |
| Nepali | Indo-Aryan | 0.957 [0.946, 0.967] | 95% | 8/8 | – |
| Sanskrit | Indo-Aryan | 0.949 [0.939, 0.958] | 82% | 8/8 | – |
| Maithili (low-resource) | Indo-Aryan | 0.962 [0.947, 0.970] | 93% | 8/8 | – |
| Sindhi (low-resource) | Indo-Aryan | 0.968 [0.960, 0.975] | 75% | 8/8 | – |
| Kashmiri (low-resource) | Indo-Aryan | 0.956 [0.942, 0.968] | 96% | 8/8 | – |
| Konkani (low-resource) | Indo-Aryan | 0.955 [0.942, 0.967] | 97% | 8/8 | – |
| Dogri (low-resource) | Indo-Aryan | 0.968 [0.962, 0.973] | 100% | 8/8 | – |
| Manipuri (low-resource) | Tibeto-Burman | 0.959 [0.952, 0.966] | 97% | 8/8 | – |
| Bodo (low-resource) | Tibeto-Burman | 0.934 [0.914, 0.951] | 98% | 8/8 | – |
| Santali (low-resource) | Austroasiatic | 0.959 [0.949, 0.969] | 87% | 8/8 | – |
| Japanese | World | 0.946 [0.871, 0.997] | 92% | 5/8 | – |
| Russian | World | 0.950 [0.857, 1.000] | 87% | 1/8 | Bengali 1 |
| Arabic | World | 0.972 [0.962, 0.980] | 55% | 0/8 | Urdu 7, Marathi 1 |

| Pool (each language weighs the same) | languages | Latin share |
|---|---|---|
| Indo-Aryan | 15 | 0.960 [0.954, 0.966] |
| Dravidian | 4 | 0.951 [0.944, 0.959] |
| Tibeto-Burman | 2 | 0.947 [0.934, 0.957] |
| Austroasiatic | 1 | 0.959 [0.948, 0.969] |
| World | 3 | 0.956 [0.918, 0.989] |
| All 22 scheduled | 22 | 0.957 [0.952, 0.963] |
| Low-resource (ks sd mni sat brx doi kok mai) | 8 | 0.958 [0.951, 0.964] |

(Pool CIs resample the 8 problems, which every language shares, with all their traces:
`think_summary_v2.csv`, which also recounts shares over letters and combining marks only. No
language's share moves at this precision.)

- **Every language reasons in English.** 0.934–0.972 of the reasoning's letters are Latin. The 22
  scheduled languages pool at 0.957, with no difference between families (Indo-Aryan 0.960,
  Dravidian 0.951, Tibeto-Burman 0.947, Austroasiatic 0.959).
- **Low-resource languages don't fall back on Hindi or their own script.** Kashmiri, Sindhi,
  Manipuri, Santali, Bodo, Dogri, Konkani and Maithili pool at 0.958, the same as the big languages.
  Their non-Latin text is quotation: 75–100% of it is copied from the prompt (Dogri, Bodo, Konkani,
  Manipuri, Kashmiri 96–100%). The Arabic-script languages (Kashmiri, Sindhi, Urdu) contain
  essentially no Devanagari. What the model adds itself, about one word per trace, is a label or
  gloss in the prompt's own script, such as "Kannada question" written in Kannada.
- **It knows every Indian language by name.** The reasoning names the prompt's language in 176/176
  traces (Dogri, Bodo, Santali and Konkani included).
- **But it misreads the world controls through an Indian lens.** Arabic is called "Urdu" in 7/8
  traces, and the Arabic is glossed in Urdu; the 8th Arabic trace says Marathi, and one Russian trace
  calls the prompt Bengali. Russian is named in 1/8 traces, Japanese in 5/8. So Arabic's "own-script"
  share is largely Urdu the model wrote: only 55% of its non-Latin words are quoted from the prompt.
- 7/200 traces closed the think block within 400 tokens, so this says nothing about accuracy.

**Human translations (run `xl_dom_30a`)**, 16 problems × 6 languages, same settings:

| Prompt language | n | Latin share [95% CI] | own script |
|---|---|---|---|
| English | 16 | 1.00 | – |
| Bengali (human) | 16 | 0.96 [0.95, 0.96] | 0.04 |
| Telugu (human) | 16 | 0.96 [0.95, 0.97] | 0.04 |
| Japanese (human) | 16 | 0.95 [0.92, 0.99] | 0.05 |
| Russian (human) | 16 | 0.94 [0.88, 0.99] | 0.06 |
| Chinese (human) | 16 | 0.95 [0.93, 0.98] | 0.04 |

- The reasoning opens in English and quotes the prompt's own words inside it: *"We need to parse
  this Bengali problem regarding fabric production (পোশাক তৈরি)…"*. The remaining 4–6% of letters
  are those quotations.
- That matches, and slightly exceeds, the 92% Latin found for translation requests (doc 09 §2): the
  visible reasoning is English whatever the task and whatever the language.
- 91 of 96 traces never closed the think block within the 400-token cap (95 reached it), so this run
  says nothing about thinking-on accuracy.
- Machine and human translations agree on the 30B: Bengali 0.969 (MT) vs 0.96 (human), Telugu 0.959
  vs 0.96, Japanese 0.946 vs 0.95, Russian 0.950 vs 0.94 (different problems; same settings).

### Thinking in the prompt's language (doc 17 #7, 2026-10-20)

60 MGSM problems × hi, ta (MT) and bn, te (human), thinking on, sampling at the model card's settings,
up to 3,072 tokens, run `p1_think3k_30b` (30B). The instructions are in the prompt's own language
(IndicTrans2): "think through this problem step by step in English / in <language>, and write all of
your reasoning in it", optionally with the think block opened by a native sentence ("Okay, let me
solve this problem step by step."). Indian languages pooled, 240 traces per condition:

| Condition | own-script share | closed | accuracy [95% CI, problems] |
|---|---|---|---|
| default | 0.02 | 90% | 0.73 [0.67, 0.80] |
| told to think in English | 0.02 | 85% | 0.71 [0.65, 0.78] |
| told to think in the prompt's language | 0.05 | 91% | 0.73 [0.67, 0.80] |
| the same, native opener pre-filled | 0.94 | 15% | 0.72 [0.64, 0.79] |

- The instruction alone is ignored: the reasoning stays English.
- With the native opener the model reasons in the prompt's language to the end, and is as accurate:
  +0.8 pts [−6.3, +7.9] against the English-instructed arm (paired). But only 15% of native traces
  conclude (151 of 240 reach the cap, 53 loop; mean 2,489 tokens vs 1,663), so that accuracy is mostly
  the forced answer's. Thinking-on accuracy, at last (doc 07's open item): 0.71–0.73 pooled for Indian
  prompts, 0.87 for English (default).

### Hints: the reasoning always admits the hint; the answer sometimes follows it anyway (30B)

Runs `xl_hint_30` (generation) and `xl_hint_30_force` (forced answers). Accuracy without a hint
(problems); uptake among the problems answered right without a hint (n); verbalised among the
hint-following traces (n); 95% CIs (bootstrap over problems; the pooled row resamples a problem with
all its languages, `hint_summary_v2.csv`; for the all-or-nothing verbalised cells, Wilson over traces,
since the percentile bootstrap collapses to [1.00, 1.00] there).

| Prompt | Accuracy, plain | Uptake, native hint | Verbalised, native hint | Uptake, English hint | Verbalised, English hint | Mentions the hint (native / English hint, all traces) | Matcher false positives (plain) | Closed |
|---|---|---|---|---|---|---|---|---|
| en | 0.95 [0.85, 1.00] (20) | 0.42 [0.21, 0.68] (19) | 1.00 [0.68, 1.00] (8) | – | – | 1.00 / – | 0.05 | 0.35 |
| hi | 0.70 [0.50, 0.90] (20) | 0.29 [0.07, 0.50] (14) | 1.00 [0.51, 1.00] (4) | 0.57 [0.29, 0.79] (14) | 1.00 [0.68, 1.00] (8) | 0.95 / 0.90 | 0.05 | 0.35 |
| ta | 0.75 [0.55, 0.95] (20) | 0.07 [0.00, 0.20] (15) | 1.00 [0.21, 1.00] (1) | 0.33 [0.13, 0.53] (15) | 1.00 [0.57, 1.00] (5) | 1.00 / 1.00 | 0.05 | 0.45 |
| bn_h | 0.90 [0.75, 1.00] (20) | 0.56 [0.33, 0.78] (18) | 1.00 [0.72, 1.00] (10) | 0.28 [0.11, 0.50] (18) | 1.00 [0.57, 1.00] (5) | 1.00 / 1.00 | 0.10 | 0.30 |
| sat | 0.55 [0.35, 0.75] (20) | 0.36 [0.09, 0.64] (11) | 1.00 [0.51, 1.00] (4) | 0.55 [0.27, 0.82] (11) | 1.00 [0.61, 1.00] (6) | 0.85 / 1.00 | 0.00 | 0.30 |
| ru_h | 1.00 [1.00, 1.00] (20) | 0.35 [0.15, 0.55] (20) | 1.00 [0.65, 1.00] (7) | 0.35 [0.15, 0.55] (20) | 1.00 [0.65, 1.00] (7) | 1.00 / 1.00 | 0.05 | 0.60 |
| non-English pooled | 0.78 [0.67, 0.87] (100) | 0.33 [0.24, 0.44] (78) | 1.00 [0.87, 1.00] (26) | 0.40 [0.24, 0.56] (78) | 1.00 [0.89, 1.00] (31) | 0.96 / 0.98 | 0.05 | 0.40 |

- **Verbalisation is total.** All 65 hint-following traces mention the hint, in every language and
  for both hint languages. So do the traces that resist it (0.98–1.00 pooled, 0.83–1.00 per cell). The matcher's false-positive
  rate on hint-free traces is 0.05.
- **Native-language hints are not hidden from the English reasoning.** Pooled over the five
  non-English languages, uptake is 0.33 [0.24, 0.44] for the native hint and 0.40 [0.24, 0.56] for the
  English one, against 0.42 [0.21, 0.68] for English prompts. Verbalisation is 1.00 in every cell.
  Per language uptake varies (Tamil native hint 0.07, Bengali native hint 0.56), but each cell has
  only 11–20 problems.
- **The reasoning brings the hint into English first.** It restates the problem in English,
  including the hint: *"A teacher says the answer is 20."* for a Hindi prompt, *"One teacher said
  the answer was $77,000."* for Santali. Then it usually decides *"presumably the teacher is
  wrong"*.
- **Following the hint contradicts the reasoning.** 86% of the hint-following traces contain the
  correct answer. In the traces I read, the model computed it, noted the mismatch (*"8 cm. But the
  answer is 9 cm."*; *"State clearly that this mathematical answer ($26) is different from the
  teacher's answer"*), then searched for readings that give the teacher's number. 72% of them end
  in a repetition loop ("Case 40: … Case 50: …"). Forced to answer there, it gives the teacher's
  number. The few traces that closed naturally split both ways: *"Teacher is incorrect. The correct
  answer is **3**."* (Russian, English hint) versus Santali answering **4**, the hinted number, as "a
  logic puzzle".
- **Hints make the reasoning run on.** 3% of hinted traces (7 of 220) closed within 1,638 tokens,
  against 39% of plain ones (47 of 120).
- **What this means for faithfulness.** In Chen et al.'s sense (does the reasoning admit the cue it
  used?), Sarvam-30B is faithful, and equally so whether the cue is in an Indian language, Russian
  or English. The failure is the other direction: the reasoning says the hint is wrong, and the
  answer adopts it anyway once the reasoning stalls.
- **Caveats.**
  - Greedy decoding loops: 77% of the *plain* traces that didn't finish also end in a repetition
    loop. The model card's recommended sampling might finish more traces.
  - Uptake is measured on answers forced at a token cut, not on natural conclusions.
  - Cells are small, and the hints are machine translated (one template sentence).
  - The answer cue is English ("The final answer is").
- **Rerun at 3× (doc 17 #8, run `p1_hint_30b`, 2026-10-20).** 60 problems, 1,020 traces, up to 3,072
  tokens, sampling at the model card's settings (so traces can leave greedy loops), forced answers only
  where a trace didn't close. Pooled over the five non-English languages: uptake 0.36 [0.29, 0.44]
  native hint vs 0.33 [0.25, 0.40] English hint (+4 [−6, +13], paired over 58 problems); English
  prompts 0.33; verbalised 0.99 / 1.00, and ≥ 0.92 in every cell (followers per cell 9–22). On natural
  conclusions only: uptake 0.43 / 0.46, verbalised 1.00. 161 of 165 hint-following traces contain the
  correct answer. Hinted traces still run on (35% / 31% close, plain 89%). The conclusions above hold,
  now without relying on the token cut.

## Caveats

- **Latin script ≠ English.** Latin mass also counts romanized Indic, code identifiers and
  untranslated technical terms. Native positions, the final-layer reference and the no-Latin-letter
  items limit this but can't remove it.
- **The script-mass metric has a lens floor.** Diffuse readouts put 0.71–0.84 of letter mass on Latin
  even at layer 0 (above the 0.60 vocabulary share), so absolute shares are not "fractions of English
  thought". Only same-layer differences and layer-to-layer changes are interpretable.
- **Number parsing (corrected 2026-10-20).** Answers were read with a parser that dropped every comma
  and stopped at a space, so "70 000" (a narrow no-break space, how the models often group
  thousands) read 70 and "16,00 $" read 1600. Re-read with grouping separators and decimal commas,
  the hint labels change for 4 of 340 rows here (`xl_hint_30_force`) and 10 of 1,020 in
  `p1_hint_30b`: mostly Russian answers marked wrong that were right, and a few
  hint-taking answers missed. The tables above use the re-read labels (`hint_items_v2.csv`,
  `python domain_experiment.py hint-regrade <folder>`). Answers stored at their length cap (600
  characters after the think block, 80 of a forced answer; 304 and 389 rows in the two reruns) can't
  be re-read and keep their labels. The direct-answer MGSM accuracies don't change (10 values change,
  no label). The 86% and 72% in the hint bullets were counted on the 63 hint-followers before the
  re-read.
- **Confidence intervals (corrected 2026-10-20).** Every pooled interval here now resamples prompts
  or problems, each with all its translations; per-language intervals are unchanged. Before, the
  pools resampled each language's items on their own, as if a prompt's 22 translations were 22
  independent items, and the length-matched contrasts resampled language × length cells. That made
  the pooled intervals 1.5–4× too narrow. Point estimates don't change. What no longer clears 0: the
  30B logit-lens math gaps over science and technical, the 30B J-lens science and technical gaps
  under everyday. The run's own summary
  files are kept; the corrected ones end in `_v2` (`python domain_experiment.py resummarize
  ../results/<run>`).
- **Length and domain are entangled.** The matched contrasts rest on the overlap between domains'
  length distributions (39–914 matched items); MGSM vs everyday barely overlaps.
- **Machine translation.** 25 of 26 languages are machine translated (IndicTrans2 for the 22
  scheduled languages, NLLB-200 for ja/ru/ar). The coordinator's QA (`data/xling/translation_qa.json`):
  no empty outputs; 98–100% of letters in the expected script; chrF against MGSM's human
  translations bn 50, te 59, ru 65, ja 28 (character chrF is low for Japanese by construction).
  IndicTrans2 translated even the technical stems into native script and sometimes literally: Urdu
  renders "Python" as ازگر, the snake. NLLB's Japanese sometimes repeats a sentence. IndicTrans2 keeps
  ASCII digits.
- **MGSM numbers.** Values above 10 survive translation in 92–100% of problems, but small numbers
  are often written as words ("4 times" → "четыре раза"), so only 68% (ar) to 98% (sa) of problems
  keep every value in digits; accuracy is reported on those strict-pass problems.
- **Batched runs.** `batch_check` on the 30B: right-padded readouts agreed with lone forwards on the
  top token for 8/8 check prompts (max |Δ log-prob| 7.05, in bf16 tails), and left-padded greedy
  decoding matched lone decoding on 7/8. Per-item greedy outputs can depend on the batch shape in
  bf16; aggregates hold.
- **Japanese is a barely-seen language for Sarvam** (no kana tokens; byte fallback), not a
  like-for-like world control.

## Engineering

- The vocab → script map, lens readouts and aggregation run on the GPU in chunks of 1,024 positions;
  J-lens Jacobians are moved to the LM head's device once. The 30B profile (≈ 500k question
  positions × 19 layers × 2 lenses, full softmax) took 3.8 min.
- `mgsm` reads the answer position from the generation's own prefill (a forward hook that keeps the
  last column of the first, multi-token call) instead of a second forward.
- **Out of memory (run `xl_dom_30b`):** Sarvam's prefill keeps fp32 logits for *every* position, and
  32 long machine-translated MGSM prompts (~1,700 tokens) needed 53 GB. MGSM batches are now capped at
  8,192 padded tokens.
- **Lean generation for thinking** (`_generate_lean`): the same left-padded, masked, KV-cached greedy
  loop as `batch_gen.generate`, but through the model's text module with the LM head on the last
  position only. A batch is then limited by the KV cache instead of full-sequence fp32 logits, so all
  200 prompts of `xl_dom_30think` ran as one batch: one 400-step loop at 0.63 s per step (252 s)
  instead of 3–5 loops. It matches `batch_gen.generate` token for token on both tiny models (fp32).
- A decode **deadline** (`time_limit_s`) truncates every trace at the same step instead of letting the
  hard timeout kill the stage, and results are written after every batch. It wasn't reached (traces
  373–400 tokens).
- **Measured costs for the budget gate:** a batched KV-cached decode step on the 30B costs ≈ 0.45 s at
  batch 32–48 (≈ 3.5 lone forwards); the thinking stage (96 prompts × 400 tokens) took 6.2 min against
  a 3-min estimate. `n_forwards` now charges 3.5 forwards per decode step.

## Cost

| Run | Hardware | App minutes | ≈ $ | What |
|---|---|---|---|---|
| `xl_dom_smoke` (`--step smoke`) | CPU 4c/16G | 2.9 | 0.02 | all four analyses on the tiny model in the image |
| `xl_dom_30a` | H200 | 9.1 | 0.78 | 30B: batch_check, digits, MGSM (en + human), thinking |
| `xl_dom_30b` | H200 | 6.9 | 0.59 | 30B: profile; MT MGSM ran out of memory after 1.6 min |
| `xl_dom_30think` | H200 | 5.7 | 0.49 | 30B: thinking on, 25 MT languages × 8 problems |
| **Total, domains** | | | **≈ 1.88** | the 30B runs (the $5.50 cap also covered two 105B launches) |
| `xlate_hint` (coordinator) | L40S | 3 | 0.12 | IndicTrans2/NLLB: the hint sentence into hi, ta, bn, sat, mni, ru, ja |
| `xl_hint_smoke` (`--step smoke`) | CPU 4c/16G | 3.4 | 0.02 | `hint` on the tiny model in the image |
| `xl_hint_30` | H200 | 21.1 | 1.82 | 30B: hint test, 340 prompts × ≤ 1,638 thinking tokens |
| `xl_hint_force_smoke` | CPU 4c/16G | 3.0 | 0.02 | `hint` + `hint_force` on the tiny model |
| `xl_hint_30_force` | H200 | 1.8 | 0.16 | 30B: forced answers for the 340 traces |
| **Total, hints** | | | **≈ 2.14** | of its separate $3.00 cap |

Minutes are Modal app wall time (`modal app list`), an upper bound on container time. Driver-only
estimates from `report.json`: $0.75, $0.55, $0.46.
