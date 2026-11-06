# 17. Follow-ups for blog post 1: is English special, and the gaps the post admits to

Code: `code/post1_experiment.py` (stages `post1_experiment.<analysis>`). Runs: see "Runs" below.
Budget: a hard cap of $20 for all nine items (Tier 1 ≈ $4, Tier 2 ≈ $16), approved 2026-10-20;
spent ≈ $10.04, of which ≈ $6.64 on the 30B runs here. Items and sentences about Sarvam-105B,
including its parts of the pre-registration, are left out of this release.

## Answer

**Post 1's readout claim holds; its "English is special" reading softens, and its push numbers
mostly go.**

- **#1 Every language Sarvam knows well is written in the middle layers; English is the most
  readable, not the only one.** For a ta→te item, Kannada, Bengali, Gujarati and the other Indian
  languages not in the prompt reach the top 10 for only 0.12 of items over the whole vocabulary
  (English 0.94, Hindi 0.58), but within their own script for 0.74, near the prompt's own
  languages (0.85). Readability tracks how well the model knows the language (big-10 0.73,
  low-resource 0.33). Pre-registered outcome: "graded": soften to "English is the strongest of several
  readable languages".
- **#2 The readout is word-specific.** Related English words (liquid for water) are elevated (median rank
  69) but rarely in the top 10 (0.10, on the pre-set boundary); the
  near-synonyms that do appear (hill, flame, home) are words the Indian source word also covers.
  This shows specificity. It doesn't separate an English word used internally from a shared concept
  decoded through English vocabulary, which would also favour "water" over "liquid".
- **#3 English, Hindi and the target word add up, and together still aren't a bottleneck**: 94% of
  translations survive removing all three (English + Hindi alone: −11.3 pts, 91% survive).
- **#5 The push results were mostly edit size.** A random direction pushed exactly as far as the
  English one costs nearly as much: 29.9 vs 39.9 pts at α = 3. The English-specific part is +5.7 to
  +9.9 pts. By the pre-registered rule the push doesn't stand as a single number.
- **#6 English is read at the final position** (the source word's tokens: 0.0 pts; all positions only
  +1.9 more than the last). Attention writes it first (L11, L13–14), the experts amplify it (L15–17),
  the last block removes it; the same heads write the Hindi and target directions. No English-writing
  experts by the pre-set criterion.
- **#4 Deleting "seven" doesn't break the sum** (−0.2 pts vs the wrong word); the nine number words
  together cost about 4 pts. Their push is magnitude only.
- **#7 Told to think in Tamil, Sarvam keeps thinking in English** (own-script share 0.05). Made to (a
  native opener), it reasons in Tamil to the end with no detectable accuracy difference (+0.8 pts
  [−6.3, +7.9], which can't rule out a ~6-point loss), but only 15% of those traces finish within
  3,072 tokens, so most of its answers are forced.
- **#8 The hint result holds at 3× n and on natural answers**: verbalised 99–100%, native vs English
  hint uptake 0.36 vs 0.33 (+4 [−6, +13]).

## Pre-registration (written 2026-10-20, before any run)

Each item below says what will be measured, the decision rule, and what each outcome would mean
for post 1's wording. The rules were written before the data existed; the results section reports
against them and flags anything decided afterwards.

Common conventions. Sarvam-30B: J-lens (`j_200p`) and logit lens, English band L13–16, early band
L5–8. "Band top-10" is, per item, the share
of band layers at which any of the role's tokens is in the readout-filtered top 10 (doc 11's
statistic), averaged over items. **Every CI resamples the 32 concepts** (all of a concept's items
together), 95%, as a pre-publication review asked; item-level CIs are not used.

### 1. Is every language readable, or is English special? (Tier 1 #1)

**Run.** The published 14-pair word-translation pivot (`results/final/pivot_all`'s prompts, items
and correctness filter), with the published roles unchanged, plus every other language in
`concepts_new.tsv` as a *third-language* role: for a ta→te item, the Kannada, Malayalam, Bengali,
Gujarati and Marathi words (the original 8 Indian languages), then the 14 added scheduled languages
and ja/ru/ar (exploratory). Each third-language role has a shuffled control (the same language's
word for the published control concept). A third-language role's tokens exclude every token a
published role claims and every token two third-language roles share. Three readouts per role:
- **prefix** (the published metric: tokens that begin the word, ≥ 2 characters, plus its first token);
- **whole-word** (C5): only tokens whose text is the whole word (`water`, `▁water`, `Water`…). For a
  word with no whole-word token, the role is absent (n reported);
- **own-script** (doc 15 §4): the prefix set's best rank among tokens of the role language's own
  script only.

**Reproduction check (stop rule).** The published roles must reproduce `results/final/pivot_all`
(ALL_INDIC, J-lens): English top-10 within 0.03 of 0.98/0.98/0.99 at L14–16, Hindi within 0.05 of
0.60/0.55/0.68. Otherwise stop and investigate.

**Decision metric.** ALL_INDIC (10 Indic→Indic pairs), 30B J-lens, band top-10, prefix readout,
pooled over the *original-8 third languages other than Hindi* (each item's mean over its third
languages). The logit lens as a replication.

| Outcome (30B J-lens, pooled other-Indian third languages) | Reading | Post 1 |
|---|---|---|
| ≤ 0.10 whole-vocabulary **and** own-script ≤ 0.25 (near its control) | English and Hindi are privileged; the other languages are not carried | "English is special" hardens; Hindi as the second anchor hardens |
| ≤ 0.10 whole-vocabulary, but own-script ≥ 0.50 | The concept is written into many languages' words; English wins the whole-vocabulary readout (vocabulary share plus data) | "Pivot" softens toward doc 15's shared space: say English is the most *readable* language, not the only one present |
| 0.10–0.30 whole-vocabulary | Graded: every language somewhat readable | Soften: "English is the strongest of several readable languages" |
| ≥ 0.30 whole-vocabulary | Readable in proportion to data | Soften strongly; the post's "thinks in English" framing would need rework |

Data proportionality (exploratory, no single cut-off): Sarvam publishes no language mix, so the
proxy is its own statement that "a substantial portion" of data is in the 10 largest Indian
languages. If readability tracks data, the big-10 third languages (bn, ta, te, mr, gu, kn, ml, pa,
or) read above the added low-resource ones (ks, sd, mni, sat, brx, doi, kok, mai). Whole-word
(C5): if English's band top-10 stays ≥ 0.90 whole-word, its lead is not a prefix-set artefact.

### 2. Lexical or semantic? A semantic-neighbour control (Tier 1 #2)

**Run.** In the same pivot, two English neighbour roles per concept (`code/data/neighbours.tsv`):
*close* (near-synonym, hypernym or co-hyponym: water → liquid, fire → flame, king → queen) and
*assoc* (thematic: water → ice, fire → smoke). None is a translation or one of the 32 concept words;
all are single Sarvam tokens, like the concept words. Controls: the published control concept's
close and assoc neighbours. Tokens claimed by any published role are excluded.

**Decision metric.** ALL_INDIC, 30B J-lens, band top-10, **whole-word** readout (prefix sets of a
neighbour include generic 2-letter prefixes; the prefix readout is reported beside it).

| Outcome | Reading | Post 1 |
|---|---|---|
| close ≤ 0.10 and assoc ≤ 0.05 (near their controls) | Lexical: the band holds the translation word, not its neighbourhood | "Pivot" wording hardens |
| close 0.10–0.30 | Partly semantic | Keep "pivot", add that related English words are also active |
| close ≥ 0.30 or assoc ≥ 0.20 | Semantic: the lens reads a neighbourhood of English words | Soften "pivot" to "an English-labelled concept region" |

Secondary: top-100 rate and median rank of the neighbours vs controls.

### 3. Remove English, Hindi and the target word together (Tier 1 #3)

**Run.** `causal_test`'s projection at the final position, J-lens directions, the published 10
pairs, conditions as rows of one forward per item: English alone and Hindi alone (the published
token rule, to reproduce −5.7 and −4.4), then with each word's most specific tokens first: English,
Hindi, target, English + Hindi, **all three**, the same for a different concept's three words
(rank-matched per language), a random subspace of the joint rank, all three at L5–8, and all three
pushed to −2× (α = 3) beside its control.

**Reproduction (stop rule).** English alone within [−7.5, −4.0] pts; Hindi alone within [−6.0, −2.8].

| Outcome (all three, L13–16, minus its control) | Reading | Post 1 |
|---|---|---|
| ≥ 2× the sum of the single-language drops | Redundant carriers: each language's words hold the concept and stand in for the others | "Deleting English barely hurts because Hindi and the target word carry it too" |
| ≈ the sum (CI covers it) | Separate, additive parts | Same sentence, weaker |
| still correct ≥ 80% | The concept is mostly carried outside these word directions | "Not a bottleneck" holds even for all three; the post should say the concept lives beyond any one language's words |
| still correct ≤ 50% | Together they are a bottleneck | The post can say what carries the concept |

### 4. Does native-digit arithmetic need the English number word? (Tier 2 #4)

**Run.** doc 12's native-digit sums (11 tokenized native systems × 72 problems, plus ASCII),
correctly answered items. At the final position, project out the answer's English number word
("seven", every case and space variant) at the band, vs the wrong answer's word (same size), vs
the same at the early band; all nine number words together vs a random subspace of that rank; the
ASCII answer digit; and the answer word pushed to −2× vs the wrong word pushed. Metric: drop in
P(the correct native digit) and the share still correct.

| Outcome (answer word minus wrong word, band) | Post 1 |
|---|---|
| ≥ 5 pts, CI excluding 0 | "Deleting 'seven' hurts the sum": the word is used |
| CI includes 0, push also null | "It is there but not needed", like the translation latent |
| projection null, push ≥ 10 pts beyond control | "Used, but redundant", the translation pattern |

### 5. A push curve with a norm-matched random control (Tier 2 #5)

**Run.** The published steering pairs (`ta-te,bn-gu,kn-ml,mr-bn,te-kn`): the English
component pushed by α ∈ {1, 1.5, 2, 2.5, 3}, beside (a) a different concept's English (the published
control) and (b) a **norm-matched random direction**: at each layer, a fixed random unit vector
scaled to the English edit's norm (α · ‖English component‖), so it moves the residual exactly as
far. The mean edit norm relative to ‖h‖ is logged per condition.

**Reproduction.** α = 1 and α = 3 within the published CIs (−6.3 / −41.2, control −8.9), up to
batching (rows of one forward).

**Decision.** Read the English-specific effect at the largest α at which **both** controls cost ≤ 5
pts. If, at that α, English minus the random control is ≥ 10 pts with a CI excluding 0, the push
result stands.

### 6. Where is the English direction written? (Tier 2 #6)

**(a) Positions.** 30B, J-lens, published 10 pairs: the English projection at L13–16 at the last
position (published), at the query's source-word tokens only, at every position but the last, at
every position (BOS excluded), the control concept's English at every position, and every position
at L5–8.

| Outcome (all positions vs last only) | Post 1 |
|---|---|
| all positions ≥ 3× last-only, or ≥ 15 pts | English is also held upstream and re-read; "95% survive" was partly a final-position artefact and needs a qualifier |
| within the CI of last-only | English sits at the final position and is redundant there; "not a bottleneck" firms up |

**(b) Writers.** 30B, kept items of the 10 pairs, one forward each. At the final position, each
layer's attention output (and each head's, via the output projection), each selected routed
expert's weighted output, the shared expert and the dense layer-0 MLP are scored on the English
word's J-lens readout direction (centred, divided by the final RMS), against the target word, the
Hindi word and the control concept's English. Also the direct (logit-lens) score.

Reported: the layers and component types that write English. Pre-set criterion for "English-writing
experts": the top 5 (layer, expert) units carry ≥ 2× the share of total positive English
attribution that the top 5 carry for the target word, and each writes English (positive score) in
≥ 50% of the items that route to it.

### 7. Think in the prompt's language (Tier 2 #7)

**Run.** 30B, thinking on, sampling at the model card's settings (temperature 0.8, top-p 0.95),
MGSM problems in hi, ta (machine translated, strict-pass) and bn, te (human), up to 3,072 tokens.
Conditions: default; a system turn asking for the reasoning in English; the same asking for the
prompt's language; the latter plus the think block pre-filled with a short native-language opener.
Both instructions and the opener are machine translated into the prompt's language (IndicTrans2), so
the instructed prompts contain no English. Unfinished traces get the forced-answer probe (doc 12).

**Compliance first.** An arm counts as "native thinking" if its reasoning is ≥ 50% own-script
letters (pooled). The main contrast is the compliant native arm vs the English-instructed arm,
paired by problem, CI over problems.

| Outcome (native minus English accuracy) | Post 1 |
|---|---|
| ≤ −5 pts, CI excluding 0 | "Made to think in Tamil, it does math worse": the English habit carries capability |
| CI includes 0 | "It can think in Tamil just as well; English is a habit, not a need" |
| no arm compliant | "It won't think in Tamil even when asked" (a finding in itself) |

*Note added 2026-10-20:* the middle row needed an equivalence margin; "CI includes 0" only means no
difference was detected. The results below are read against the −5-pt threshold as the margin.

### 8. A robust hint test (Tier 2 #8)

**Run.** doc 12's design (plain, native-language hint, English hint; en, hi, ta, bn, sat, ru) on 60
problems instead of 20 (3×), up to 3,072 tokens, sampling at the model card's settings so traces
can leave greedy loops; forced answers only for the traces that still don't finish.

| Outcome | Post 1 |
|---|---|
| verbalisation ≥ 90% in every hint language, native vs English uptake CI includes 0 | The post's hint paragraph holds, now on natural answers |
| verbalisation < 80% for native-language hints | Native hints are partly hidden in the English reasoning: the post's claim reverses |
| uptake on natural answers far below the forced 34–40% | "Follows the hint" was mostly a token-cut artefact: rewrite the paragraph |

Items about Sarvam-105B (#9, and an addendum running #7 and #8 on it) are left out of this release.

## Deviations from the pre-registration

Stated here so the results can be read against the rules as written.

1. **#7's first run used 2,048 tokens, not the pre-registered 3,072.** I lowered the cap for cost
   before launch (the model card's own `max_new_tokens`). Decoding turned out cheap, so the
   pre-registered 3,072-token run was also made (`p1_think3k_30b`); it is the primary #7 result, and
   the 2,048 run is kept as a cap-sensitivity check.
2. **#3's target-word directions behave unlike the other two.** Projecting the target word's own
   L13–16 directions out *raises* P(target) (−5.4 pts "drop" on the 30B). So "all three together"
   mixes a harmful and a helpful edit. The pre-registered contrast is reported as written; English +
   Hindi without the target word (`en_hi`, run in the same forward) is the readable joint test, and is
   labelled post hoc wherever it is used.
3. **#2's 30B outcome lands on a boundary**: close neighbours read 0.1016 against a pre-set cut of
   0.10. The rule places it (by 0.002) in "partly semantic"; the per-concept breakdown that explains
   it is post hoc.
4. Post hoc and labelled so: the per-concept neighbour breakdown, the big-10 minus low-resource
   contrast, English + Hindi superadditivity, the per-layer writer table, and a 30B Chinese ablation.
5. **#5's first random control was norm-matched at the first edited layer only.** The pre-registration
   asked for a random direction "scaled to the English edit's norm … so it moves the residual exactly
   as far". The first implementation (`random@`, run `p1_t2_30b`) scaled each row's random
   push to *that row's own* English component, which equals the English row's edit only at the band's
   first layer. The English push compounds over the band (its later edits are larger), so the random
   row moved h less far (mean |Δh| / |h| 0.26 vs 0.43 at α = 3).
   I found this while writing up, fixed it (`random_en@`: each layer's |Δ| copied from the English
   row's realized edit in the same forward; checked equal to the last digit on the tiny model), and
   reran `push` (`p1_push2_30b`). **The decision below uses the corrected control**; the
   first one is reported for comparison. The same size-matched partner (`~rand`) was then added to
   every other push condition (#3, #4, #6a and the Chinese ablation: run `p1_matched_30b`), because
   their α = 3 contrasts had the same weakness.

## Runs

| Run | Model | Stages | Driver min | App min | ≈ $ (app wall) |
|---|---|---|---|---|---|
| `xlate_think` (`modal_translate.py --step list`) | L40S | #7's instructions and opener into hi/ta/bn/te (IndicTrans2) | – | 0.4 | 0.02 |
| `post1_smoke` (`--step smoke`) | CPU 4c/16G | the smoke test + a dry run of all nine stages on the tiny model, in the image | – | 5.2 | 0.03 |
| `p1_t1_30b` | 30B, H200 | `roles` (#1, #2), `joint` (#3) | 5.6 | 5.9 | 0.51 |
| `p1_t2_30b` | 30B, H200 | `digits` (#4), `push` (#5), `positions` (#6a), `writers` (#6b), `zh` | 6.2 | 6.5 | 0.56 |
| `p1_think_30b` | 30B, H200 | `think` (#7), 2,048 tokens | 11.1 | 11.4 | 0.98 |
| `p1_hint_30b` | 30B, H200 | `hint` (#8), 3,072 tokens | 16.0 | 16.3 | 1.40 |
| `p1_think3k_30b` | 30B, H200 | `think` (#7), 3,072 tokens (pre-registered) | 26.5 | 26.9 | 2.32 |
| `p1_push2_30b` | 30B, H200 | `push` with the corrected norm-matched controls | 2.2 | 2.6 | 0.23 |
| `p1_matched_30b` | 30B, H200 | the push conditions of `joint`, `positions`, `digits`, `zh2`, each with a size-matched random partner | 6.6 | 6.9 | 0.59 |
| **total, the 30B runs here** | | | | | **≈ 6.64** |

All reproduction checks passed. The 30B pivot: English top-10 0.980 / 0.980 / 0.996 at L14–16
(published 0.980 / 0.980 / 0.992), Hindi 0.622 / 0.573 / 0.687 (0.602 / 0.551 / 0.676), 382 items
kept (382). The 30B ablation: English −5.2 [−7.0, −3.4] (published
−5.7), Hindi −4.5 (−4.4). The push: 30B −6.3 / −42.6 / control −8.6 at α 1 / 3 / 3 (published
−6.3 / −41.2 / −8.9). Every edited run is
batched (conditions as rows of one forward, through the text module with the LM head on the last
position); the published 30B ablations were batch-1, so these checks are also the batching check.

## Results

Each item: the numbers, the pre-registered outcome, and what it means for post 1. "Band top-10" as
defined above; [95% CI over concepts] unless stated. Points ("pts") are drops in P(correct answer)
× 100.

### 1. English is the most readable language, not the only one present

ALL_INDIC (10 Indic→Indic pairs; 256 items), band top-10. "Other
Indian" = each item's mean over the original-8 Indian languages that are neither the source, the
target nor Hindi (5 per item); their shuffled controls pooled the same way.

| Role | 30B J-lens, whole vocabulary | 30B J-lens, whole-word tokens | 30B J-lens, own script | 30B logit, whole vocabulary |
|---|---|---|---|---|
| English | **0.94** [0.90, 0.96] | 0.94 | 0.94 | 0.79 |
| Hindi | 0.58 [0.45, 0.70] | 0.60 | 0.92 | 0.38 |
| target word | 0.27 | 0.27 | 0.85 | 0.29 |
| source word | 0.22 | 0.23 | 0.85 | 0.23 |
| Chinese | 0.06 | 0.06 | 0.98 | 0.17 |
| **other Indian (pool)** | **0.12** [0.08, 0.17] | 0.14 [0.08, 0.21] | **0.74** [0.66, 0.82] | 0.15 [0.10, 0.19] |
| other Indian, shuffled controls | 0.00 | 0.00 | 0.03 | 0.00 |
| big-10 others (+ pa, or) | 0.13 | 0.19 | 0.73 [0.65, 0.80] | 0.14 |
| low-resource 8 (ks sd mni sat brx doi kok mai) | 0.00 | 0.01 | 0.33 [0.27, 0.40] | 0.01 |
| ja / ru / ar | 0.00 | 0.00 | 0.62 | 0.01 |

Per language (30B J-lens; whole vocabulary / own script): Tamil 0.21 / 0.87, Odia 0.22 / 0.73,
Kannada 0.14 / 0.74, Telugu 0.13 / 0.71, Urdu 0.13 / 0.56, Punjabi 0.12 / 0.71, Gujarati 0.11 / 0.79,
Bengali 0.10 / 0.89, Malayalam 0.08 / 0.66, Marathi 0.03 / 0.41 (its Devanagari tokens lose every
token Hindi claims), Russian 0.01 / 0.89, Manipuri 0.01 / 0.71, Santali 0.00 / 0.49, Arabic 0.00 /
0.33, Sindhi 0.00 / 0.10, Assamese 0.00 / 0.13, and Nepali, Konkani, Bodo, Maithili ≤ 0.09 own
script. The last five are low partly by construction: their words mostly share tokens with a
same-script language that is also tracked (Assamese with Bengali; the Devanagari languages with
Hindi, which claims them first), and shared tokens are dropped. Every shuffled control: 0.00–0.06.

- **Pre-registered outcome: "graded".** The pooled other-Indian whole-vocabulary rate, 0.12 on the
  J-lens (0.15 logit), falls in the 0.10–0.30 band: "English is the strongest
  of several readable languages". The own-script rate, 0.74, is far above the 0.50 line,
  so the second row's reading also holds: **the concept is written into many languages' words at
  once, and English wins the whole-vocabulary readout.**
- English's lead is large in absolute terms. English minus the other-Indian pool is +0.81 [0.77,
  0.86] (J-lens), +0.64 (logit); Hindi minus the pool +0.46 [0.34, 0.57] (J-lens). So
  English and then Hindi are privileged in what the whole-vocabulary readout shows. Within their own
  scripts the other languages' words reach 0.74, against 0.85 for the prompt's own two
  languages, 0.92 for Hindi, 0.94 for English and 0.98 for Chinese.
- **Readability tracks how well the model knows the language.** Own-script, the big-10 Indian
  languages read 0.73 against 0.33 for the eight low-resource ones (post hoc: +0.40 [0.33, 0.47]).
  Half of that pool (Bodo, Dogri, Konkani, Maithili) is depressed by
  the Devanagari token sharing above; the other four (Kashmiri 0.13, Sindhi 0.10, Manipuri 0.71,
  Santali 0.49) average 0.36, still half the big-10's rate. Russian (0.89) and Japanese (0.64), world
  languages in no prompt, read like the big Indian languages. Sarvam publishes no language mix, so
  this is consistent with "readable in proportion to data", not a test of it.
- **Not a tokenization artefact (C5).** Whole-word tokens only: English 0.94 (its words are single
  tokens, so it is unchanged), Hindi 0.60, the other-Indian pool 0.14. Fewer items have a whole-word
  token in the Indic languages (n drops from 256 to 70–200 per language), but the rates hold.
- The shuffled controls stay at 0.00 over the whole vocabulary and 0.02–0.06 within scripts, so none
  of this is generic script mass.

### 2. Semantic neighbours: mostly word-specific, near-synonyms leak in

ALL_INDIC, whole-word readout (decision metric), band top-10:

| | close neighbour (water → liquid) | associate (water → ice) | the control concept's close / associate |
|---|---|---|---|
| 30B J-lens | **0.10** [0.04, 0.18] | **0.01** [0.00, 0.04] | 0.00 / 0.00 |
| 30B logit | 0.06 | 0.01 | 0.00 / 0.00 |
| 30B J-lens, top-100 | 0.59 [0.47, 0.70] (median band rank 69) | 0.30 [0.20, 0.40] (rank 258) | 0.005 (rank 45,454) |
| (the translation, for scale) | English 0.94; top-100 0.98, median rank 1 | | |

- **Pre-registered outcome.** J-lens: close 0.1016, on the 0.10 boundary, so formally "partly semantic"
  ("keep 'pivot', add that related English words are also active"); the 30B logit lens gives 0.06.
- Post hoc, which neighbours reach the top 10 (30B J-lens): hill for mountain 0.78, flame for fire
  0.75, home for house 0.62, evening for night 0.53; all others ≤ 0.15 (queen, father, silver, cloud
  ≤ 0.07). These are near-synonyms that one Indian word usually covers (Hindi घर is both house and
  home, पहाड़ both mountain and hill), so they are arguably translations of the source word too.
  Associates reach the top 10 only for dentist (tooth, 0.42).
- Related English words are clearly *active* (close neighbours at median rank 69 of 262k, associates
  at 258, controls at 45,454), but the top of the readout is the translation word itself. The band
  holds the English word, surrounded by a semantic neighbourhood, not a neighbourhood with no centre.

### 3. English, Hindi and the target word together: additive, and still not a bottleneck

30B, J-lens directions, 10 pairs, 276 items correct at baseline:

| Removed at the band (α = 1) | drop, pts | still correct |
|---|---|---|
| English (published token rule) | 5.2 [3.4, 7.0] | 95.3% |
| Hindi (published rule) | 4.5 [2.7, 6.3] | 92.4% |
| English, most specific tokens | 4.6 [2.9, 6.3] | 96.0% |
| Hindi, most specific tokens | 4.0 [2.4, 5.6] | 93.1% |
| target word's own directions | **−5.4** [−6.9, −3.8] (P rises) | 99.6% |
| **English + Hindi** (post hoc as the joint test) | **11.3** [8.7, 13.7] | **90.6%** |
| all three | 2.8 [0.3, 5.4] | 93.8% |
| a different concept's three words, rank-matched | 1.7 [0.9, 2.5] | 95.3% |
| random subspace, same rank | 0.2 | 97.5% |
| all three at L5–8 | 0.5 | 97.5% |
| all three pushed to −2× (α = 3) | 61.8 [54.5, 68.7] | 27.5% |
| control pushed to −2× | 25.0 | 72.5% |

- **Pre-registered outcome.** All three minus its control: +1.1 [−1.6, +4.0]; all three minus the
  sum of the three single drops: −0.6 [−3.6, +2.5]: **additive,
  not redundant**. Still correct after removing all three: 94%, above the 80% line: **"not
  a bottleneck" holds even for all three; the concept lives beyond these words' readout directions.**
- The target-word directions spoil "all three": removing them helps the target (deviation 2). The
  clean joint test is English + Hindi: −11.3 pts, 91% still correct, about the sum
  of the two (+2.7 [1.4, 4.0] beyond it, mildly superadditive).
- Pushing against all three at once breaks most translations (28% left; the control push 73%),
  +36.9 [24.8, 46.9] beyond control. But a random direction pushed exactly as far (`p1_matched_30b`)
  costs 51.1 itself, so the push's specific part is **+11.6 [5.8, 17.2]**; the control concept's push
  is no worse than its own size-matched random push (−3.1). Most of what the push breaks, it breaks by
  moving h far (see #5).

### 4. Native-digit sums: the number words matter together, not the answer's word alone

doc 12's sums in the 11 tokenized native systems, correctly answered items (692, J-lens), 72 problems
as clusters:

| Removed at the band | drop | still correct |
|---|---|---|
| the answer's English word ("seven") | −1.0 (P rises) | 98.7% |
| the wrong answer's word (control) | −0.9 | 99.4% |
| answer word minus control | **−0.2 [−0.5, +0.2]** |  |
| answer word pushed (α = 3) minus wrong word pushed | **+8.1 [5.4, 10.7]** | 87.6% vs 93.5% |
| all nine number words, minus a random subspace of their rank | **+3.9 [2.8, 5.2]** | 97.0% |
| all nine pushed (α = 3), minus random | **+60.5 [57.4, 63.5]** | 19.5% |
| the ASCII answer digit | −1.1 | 98.1% |

- **Pre-registered outcome:** neither "deleting 'seven' hurts" (the answer word is null, CI includes
  0) nor "not needed" (the push is not null). The single-word push, +8.1 beyond control, falls short
  of the 10-pt bar for "used but redundant". Post 1's "I didn't test whether deleting 'seven' breaks
  the sum" can now read: **deleting it doesn't; pushing against the number words does.**
- Post hoc, the number-word subspace as a whole is used, a little: deleting all nine words' directions
  costs 3.9 pts beyond a random subspace of the same rank, and the two
  edits are the same size (|Δh| / |h| 0.105 vs 0.103). The ASCII digit is not used.
- **The pushes, against size-matched random partners** (`p1_matched_30b`): a random direction
  pushed as far hurts *more* than the number words do (answer word −3.6 [−5.1, −2.1], all nine −12.4
  [−15.2, −9.9] relative to their random partners), so the push rows above are magnitude.

### 5. The push curve: most of the push is edit size; English-specific only modestly

The published steering pairs (143 items, J-lens), run `p1_push2_30b`
(the corrected controls; deviation 5). Drop in pts (still correct):

| α | English | other concept's English | random, as far as English |
|---|---|---|---|
| 1 | 5.9 (94%) | 0.1 | 0.2 |
| 1.5 | 7.6 (94%) | 0.1 | 0.7 |
| 2 | 9.4 (94%) | 0.1 | 1.6 |
| 2.5 | 10.6 (90%) | 0.9 | 5.2 |
| 3 | **39.9 (59%)** | 8.1 (92%) | **29.9 (77%)** |

English minus the random push of the same size [CI over concepts]:

| α | 1 | 1.5 | 2 | 2.5 | 3 |
|---|---|---|---|---|---|
| 30B | **+5.7** [3.7, 7.8] | +6.9 [4.1, 10.1] | +7.8 [3.8, 12.0] | +5.4 [−0.1, 11.6] | **+9.9** [2.8, 17.9] |

- **Pre-registered decision** (the largest α with both controls ≤ 5 pts; the random control that
  moves h exactly as far): **α\* = 2.0**. There, English minus random is +7.8 [3.8, 12.0]: **it
  doesn't reach the 10-pt bar.** By the rule, the push result doesn't stand as a single number.
- **What the push measured was mostly how far it moved h.** At α = 3 a random direction moved
  exactly as far costs 29.9 pts (English 39.9). The different-concept control is no exception: it
  costs what a random push of *its* size costs (−5.5 [−9.5, −1.9] relative). The control looked mild
  only because a different concept's English component is smaller, so pushing it moves h less
  (|Δh| / |h| 0.28 vs 0.42). The published "+32 pts English-specific" compared pushes of different
  sizes.
- **The projection (α = 1) is English-specific**, against a random edit of the same size: +5.7
  [3.7, 7.8]. That is the published ablation, now with the control a pre-publication review asked for.
- So English carries about 6–10 pts of the answer that a same-size random edit doesn't touch, at
  every α; the jump at α = 3 (to −40) is mostly generic damage.
- The first run's control (`random@`, deviation 5) gave α\* = 2.5 and +6.9; it under-matched the edit
  size.

### 6a. Where is English read from? The final position

30B, J-lens, 10 pairs, 269 items:

| English projected out at L13–16 | drop, pts | still correct |
|---|---|---|
| last position only (the published edit) | 5.8 [4.0, 7.5] | 95.5% |
| the query's source-word tokens only | **0.03 [−0.35, +0.37]** | 99.6% |
| every position but the last | 1.4 [0.7, 2.2] | 97.4% |
| every position (BOS excluded) | **7.6 [5.2, 10.0]** | 93.3% |
| a different concept's English, every position | 1.3 | 97.0% |
| every position, L5–8 | 2.1 | 97.0% |
| every position, pushed to −2× / control | 50.2 / 17.8 | 47.6% / 86.6% |

- **Pre-registered outcome:** every position minus last only is +1.9 [1.1, 2.7] pts: far from the
  "≥ 3× or ≥ 15 pts" line, and just outside last-only's own CI. **English is held, and read, at the
  final position**; the source word's positions carry none of it. "Not a bottleneck" firms up: the
  95% that survive the last-position edit are not re-reading English from upstream (93% survive the
  all-position edit).
- The all-position push (α = 3) costs 49.5 (`p1_matched_30b`), against 42.7 for a random push of the
  same size at every position: +6.8 [0.3, 14.5] specific.

### 6b. Who writes it: a few attention heads, then the experts, then the last block removes it

30B, 272 kept items, scores on the English word's J-lens direction (centred, divided by the final
RMS: roughly the logit it adds once the average downstream Jacobian carries it to the output),
English minus the control concept's English, mean over items [CI over concepts]:

| Layer | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|
| attention | **+0.66** [0.56, 0.75] | +0.04 | **+1.37** [1.16, 1.58] | **+2.74** [2.36, 3.11] | +0.88 | +0.65 | −0.26 | +0.05 |
| MoE (routed + shared) | +0.11 | +0.16 | +0.61 | +0.89 | **+2.46** [1.94, 3.03] | **+3.90** [3.22, 4.58] | **+2.47** | **−0.93** [−1.60, −0.35] |
| the same attention, written on the target word | 0.42 | 0.10 | 0.92 | 1.79 | 0.76 | 0.60 | 0.17 | 0.78 |
| the same MoE, written on the target word | 0.02 | 0.11 | 0.50 | 0.78 | 1.60 | 3.47 | **5.73** | 0.84 |

Layers 0–10 write nothing English-specific (|score| < 0.08). The direct (logit-lens) scores give the
same order.

- **Attention writes English first** (L11, then L13–14), and **the experts amplify it** (L15–17); the
  **last block removes it** (−0.93) while writing the target. The same components write the Hindi and
  target-word directions at L13–14 almost as strongly (attention at L14: English 2.98, Hindi 2.31,
  target 1.79, control 0.25): they write the concept, which every language's word direction shares in
  part. The target only pulls ahead at L17 (MoE: target 5.73, English 1.51).
- **Heads:** the top 10 of 1,216 heads (L16 h23, L14 h10, L14 h24, L17 h10, L16 h29, L13 h2, L14 h2,
  L15 h52, L13 h14, L16 h31) carry 44% of the positive head attribution: attention's English write is
  concentrated.
- **Experts: pre-registered criterion not met** (no "English-writing experts"). The top 5 (layer,
  expert) units carry 24% of the positive English attribution, against 28% for the target word, so
  English is no more concentrated in experts than the answer is. Individually, L16 experts 106 and 114
  and L15 expert 7 write English in 94–98% of the items routed to them (a new candidate list for
  doc 09 §11's routing work), but they are among many.
- Over the whole forward, attention and MoE each write about half of the English score (7.4 vs 7.5;
  the shared expert 2.7 of the MoE's 7.5). Recomputing each MoE output from its experts matched the
  block's output to 0.3% (bf16).

### 7. Asked to think in Tamil it doesn't; made to, it reasons as well but rarely stops

30B, thinking on, sampling (temperature 0.8, top-p 0.95, the model card's settings), 60 MGSM
problems × hi, ta (machine translated, strict-pass) and bn, te (human), up to 3,072 tokens (run
`p1_think3k_30b`). Accuracy = the natural answer where the trace closed, else the forced answer;
95% CIs over problems. Indian languages pooled (240 traces per condition):

| Condition (instruction in the prompt's own language) | own-script share of the reasoning | closed within 3,072 | accuracy | accuracy, natural answers only (n) | mean tokens |
|---|---|---|---|---|---|
| default (no instruction) | 0.02 | 90% | 0.73 [0.67, 0.80] | 0.76 (217) | 1,293 |
| "think … in English" | 0.02 | 85% | **0.71** [0.65, 0.78] | 0.73 (205) | 1,663 |
| "think … in Tamil" (the prompt's language) | **0.05** | 91% | 0.73 [0.67, 0.80] | 0.76 (219) | 1,608 |
| the same + the think block opened with a native sentence | **0.94** | **15%** | **0.72** [0.64, 0.79] | 0.76 (37) | 2,489 |
| English prompts: default / "think in English" | 1.00 | 95% / 97% | 0.87 / 0.77 | 0.88 / 0.78 | 1,024 / 1,285 |

- **The instruction alone is ignored.** Told, in its own language, to reason in Hindi, Tamil, Bengali
  or Telugu, the model still reasons in English (own-script share 0.05; per language 0.03–0.08).
- **Pre-filling a native opener is followed to the end**: own-script 0.94 overall and 0.93 in each
  trace's last 1,500 characters; the reasoning is real arithmetic in Tamil (or Hindi, Bengali, Telugu),
  not a translation that reverts to English.
- **Pre-registered outcome:** the compliant arm minus the English-instructed arm is **+0.8 pts [−6.3,
  +7.9]**: "it can think in Tamil just as well; English is a habit, not a need". On the 34 problems
  where both closed naturally, +5.9 [−8.6, +20.6].
  **Reading added 2026-10-20 (outside review).** "CI includes 0" is not an equivalence test. Taking the
  pre-registered harm threshold (−5 pts) as the margin: the 30B's lower bound, −6.3, is outside it, so a
  disadvantage of about 6 points can't be ruled out, and 85% of its native answers were forced. Say
  "no detectable difference", not "habit, not a need".
- **But native reasoning doesn't conclude.** Only 15% of native traces close within 3,072 tokens (151
  of 240 hit the cap, 53 loop), against 85–91% of English ones, and they run 50% longer before that.
  Their accuracy rests mostly on the forced answer (0.71 on the 203 unclosed traces, vs 0.57 on the 35
  unclosed English-instructed ones). So it reaches the right answer in Tamil about as often, but
  doesn't stop and say it.
- The 2,048-token run (`p1_think_30b`) agrees: compliant arm minus English −0.4 [−7.1, +6.3], 10% of
  native traces closed.
- Asking for English reasoning in a system turn costs English prompts 10 pts (0.87 → 0.77) and
  Indian prompts 2.5 (CI includes 0): any system instruction here is a small accuracy cost.

### 8. The hint test, 3× the problems: the post's paragraph holds, now on natural answers

doc 12's design on 60 problems (doc 12 had 20), 3,072 tokens, sampling at the model card's settings,
forced answers only where a trace didn't close (run `p1_hint_30b`; 1,020 traces). Uptake among problems
answered right without a hint; verbalised among the hint-following traces; 95% CIs over problems.

| | uptake, native hint | uptake, English hint | verbalised (native / English hint) | followers (native / English) |
|---|---|---|---|---|
| English prompts | 0.33 [0.20, 0.45] (55) | – | 1.00 / – | 18 / – |
| Hindi | 0.28 (47) | 0.28 (47) | 0.92 / 1.00 | 13 / 13 |
| Tamil | 0.37 (46) | 0.33 (46) | 1.00 / 1.00 | 17 / 15 |
| Bengali | 0.33 (48) | 0.44 (48) | 1.00 / 1.00 | 16 / 21 |
| Santali | 0.31 (29) | 0.32 (28) | 1.00 / 1.00 | 9 / 9 |
| Russian | 0.50 (44) | 0.27 (45) | 1.00 / 1.00 | 22 / 12 |
| **non-English pooled** | **0.36** [0.29, 0.44] (214) | **0.33** [0.25, 0.40] (214) | **0.99** [0.96, 1.00] / **1.00** | 77 / 70 |
| pooled, natural answers only | 0.43 [0.30, 0.54] (89) | 0.46 [0.36, 0.57] (67) | 1.00 / 1.00 | 38 / 31 |

- **Pre-registered outcome: the paragraph holds.** Verbalisation is ≥ 0.92 in every cell (pooled 0.99
  native, 1.00 English); native minus English uptake is +3.8 pts [−5.8, +13.3] (paired over 58
  problems; natural answers only −4.4 [−25, +14]). doc 12 had 0.33 / 0.40 and 65 of 65 verbalised.
- **Uptake on natural conclusions is not lower** (0.43 / 0.46): following the hint is not an artefact
  of forcing an answer at the token cut.
- 161 of the 165 hint-following traces (on problems answered right without the hint) contain the
  correct answer (98%; doc 12: 86%), and 164 of 165 mention the hint.
- Hints still make the reasoning run on: 35% / 31% of hinted traces close within 3,072 tokens, against
  89% of plain ones (doc 12, greedy, 1,638 tokens: 3% vs 39%).
- Accuracy without a hint: English 0.92, pooled 0.72 (Hindi 0.78, Tamil 0.77, Bengali 0.80, Santali
  0.48, Russian 0.76). The matcher's false-positive rate on hint-free traces is 0.05.

### 9. Chinese on the 30B (post hoc)

J-lens directions, L13–16, 10 pairs, 267 items:

| Chinese word's directions | drop, pts |
|---|---|
| projected out (α = 1) | −0.0 |
| control (different concept's Chinese) | 0.1 |
| **minus control** | −0.1 [−0.9, +0.4] |
| reflected (α = 2) minus control | +0.0 |
| pushed (α = 3) / control | 7.5 / 5.9 |
| **pushed minus control** | +1.6 [−0.9, +4.4] |
| L5–8 (early) | 0.3 |

- Projecting the 30B's Chinese-word directions out changes nothing. Against a random push of the same
  size (`p1_matched_30b`, engine rerun `zh2`), the push costs +3.5 [1.1, 6.0] more, which is small.
  The 30B's Chinese latent is weak to begin with (doc 09 §5).

## What this changes in post 1

| Post 1 says (or implies) | After these runs |
|---|---|
| English is *the* language the model thinks in; Hindi a quieter second | English is the most readable language; the concept is also written into every well-known language's words (own-script 0.74), Hindi most of all. "Strongest of several", not "the only" (#1) |
| The English latent might just be a semantic neighbourhood | Word-specific: the translation word itself (median rank 1), neighbours elevated but rarely top-10 (#2); doesn't show an English word is used internally |
| Deleting English costs ~6 pts, 95% survive | Holds (5.2–5.9 pts; a same-size random edit 0.2); removing English + Hindi + the target word together still leaves 94% (#3); the edit is at the right place, the final position (#6a) |
| Pushing English to −2× costs 41 pts; +32 beyond control | Withdraw as stated: a random direction moved as far costs 30. English-specific: +10 at α = 3 (#5) |
| The model computes with English number words; "I didn't test whether deleting 'seven' breaks the sum" | Deleting it doesn't (−0.2); the number-word subspace together costs about 4 pts (#4) |
| It reasons out loud in English (Latin script, mostly English) | Even when told not to (#7); when made to reason in Tamil, no accuracy difference is detected (a ~6-pt loss can't be ruled out), and it rarely concludes |
| A native-language hint isn't hidden in the English reasoning | Holds at 3× n and on natural answers (#8) |

## Caveats

- **Token-set construction.** Third-language roles lose every token a published role or another
  third language claims. That depresses same-script languages (Marathi, Nepali, Konkani, Bodo,
  Maithili, Dogri behind Hindi; Assamese behind Bengali), so their low rates are partly by
  construction. The own-script readout compares scripts of very different sizes; the same-script
  shuffled controls (0.02–0.06) are the guard.
- **The neighbour list** (`code/data/neighbours.tsv`) is mine, 2 words per concept, not normed for
  relatedness; a different list could move #2's 30B result across its boundary either way.
- **Pushes.** "Size-matched" means the same |Δh| at every edited layer and position in the same
  forward. It doesn't match the direction's alignment with the model's own variation, which a random
  direction lacks; a more manifold-aware control could cost less than a random one.
- **#7** rests on machine-translated instructions and openers (IndicTrans2, not native-checked) and on
  forced answers for 85% of the native traces. Sampling makes individual traces non-reproducible;
  rates are aggregates over 240 traces per condition.
- **#8** uses doc 12's machine-translated hints, and the per-language cells are still small (28–55
  problems, 9–21 followers).
- **Batching.** Every condition here runs as a row of one batched forward; per-item
  results differ from batch-1 runs by bf16 batch-shape noise (doc 12, `batch_check_deep`). The
  reproduction checks above bound the aggregate effect (≤ 0.5 pts on the ablations, ≤ 1.4 on the α = 3
  push; the two 30B push runs, with 16 and 26 rows per forward, differ by 0.5 and 2.7 pts at α = 1 and 3).

## Cost

The 30B runs and smokes above cost ≈ $6.64, inside a $20 cap for all nine items (the rest of the
≈ $10.04 spent under that cap went to 105B runs, which aren't part of this release). Modal app wall
time at list prices (H200 $5.17/h); the Modal dashboard has the exact bill. The decode runs were far
cheaper than budgeted: a continuous-batching generator (`generate_cb`: a preallocated KV slab,
grouped-query attention for the decode step, finished rows replaced at once) decoded ~1,000 traces
at 0.22–0.40 s per step.

## Reproduce

From `code/` (Git Bash: prefix with `MSYS_NO_PATHCONV=1`). `think_texts.json` is
`post1_experiment.think_source_texts()` written as a JSON list.

```bash
modal run modal_app.py --step smoke --stages load,post1_experiment.roles,post1_experiment.joint,post1_experiment.digits,post1_experiment.push,post1_experiment.positions,post1_experiment.writers,post1_experiment.zh,post1_experiment.think,post1_experiment.hint
modal run modal_app.py --step pilot --model sarvam --stages load,sanity,post1_experiment.roles,post1_experiment.joint --tag p1_t1_30b --budget-minutes 12 --timeout-margin 2 --spf-guess 0.12
modal run modal_app.py --step pilot --model sarvam --stages load,sanity,post1_experiment.digits,post1_experiment.push,post1_experiment.positions,post1_experiment.writers,post1_experiment.zh --tag p1_t2_30b --budget-minutes 18 --timeout-margin 2 --spf-guess 0.12
modal run modal_translate.py --step list --langs hi,ta,bn,te --texts-json think_texts.json --out-json data/domains/think_lang_translations.json
modal run modal_app.py --step pilot --model sarvam --stages load,sanity,post1_experiment.think --tag p1_think3k_30b --budget-minutes 40 --timeout-margin 3 --spf-guess 0.12 --stage-args '{"items": 60, "max_new_tokens": 3072, "time_limit_s": 1800, "force_reserve_s": 300}'
modal run modal_app.py --step pilot --model sarvam --stages load,sanity,post1_experiment.hint --tag p1_hint_30b --budget-minutes 40 --timeout-margin 3 --spf-guess 0.12 --stage-args '{"items": 60, "max_new_tokens": 3072, "time_limit_s": 1800, "force_reserve_s": 300}'
modal run modal_app.py --step pilot --model sarvam --stages load,sanity,post1_experiment.push --tag p1_push2_30b --budget-minutes 7 --timeout-margin 2 --spf-guess 0.12
modal run modal_app.py --step pilot --model sarvam --stages load,sanity,post1_experiment.joint,post1_experiment.positions,post1_experiment.digits,post1_experiment.zh2 --tag p1_matched_30b --budget-minutes 15 --timeout-margin 2 --spf-guess 0.12 --stage-args '{"matched_pushes": true, "only_pushes": true}'
modal volume get sarvam-lens runs/<tag> ../results/
```

Each stage writes its rows (`*_rows.csv`, `roles_metrics.npz`, `writers.npz`, `*_items.csv`) and a
summary JSON; the summaries can be recomputed from the rows (`post1_experiment.summarize_roles`,
`condition_table`, `summarize_writers`, `summarize_think`, `domain_experiment.summarize_hint`). Local
checks on the tiny models (CPU): the edit engine equals `causal_test.Ablation`; `generate_cb` equals
lone greedy decoding token for token, with admissions, compaction and slab shifts; the size-matched
random edits copy |Δ| exactly.
