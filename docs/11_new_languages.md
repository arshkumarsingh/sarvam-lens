# 11. All 22 scheduled languages (plus Japanese, Russian, Arabic) in the word-translation pivot

Run 2026-10-19. Code: `code/newlang_experiment.py`
(analysis `pivot`). Word lists: `code/data/concepts_new.tsv`, sources `code/data/concepts_new_sources.tsv`,
build inputs and script in `code/data/newlangs/`. Runs: `results/xl_new_30b/` (Sarvam-30B, J-lens + logit),
`results/xl_new_causal_30b/` (English ablation, §4b), `results/xl_new_misses_30b/` (English on failed
translations, §4c).
Logs: `logs/modal/xl_new_*.log`.

## TL;DR

- **Every Indian language we could test routes through English, as source and as target.** That is
  13 of the 14 new scheduled languages, plus Japanese, Russian and Arabic. On Sarvam-30B (J-lens,
  band L13–16) the English word is in the top 10 for 0.84–1.00 of correctly translated items in every
  language group with more than 3 kept items (Kashmiri and Bodo as targets rest on 1 kept item each).
  The same language's word for a *different* concept scores at most 0.03. The layer profile barely
  changes with script or resource level: English is at 7–15% at L9 (control 0–4%; 0% under the logit
  lens), in the top 10 for ~80–86% of items at L13 and for 96–100% at L14–16. That holds for
  Brahmic-script Indo-Aryan languages, the Perso-Arabic scripts (Urdu, Sindhi, Kashmiri), Meitei Mayek
  and Ol Chiki, and the world controls.
- **The English step is causally used, more than for the original languages.** Removing the English
  word's lens directions at the English band costs the correct target word 13.2 points [10.6, 16.2]
  of probability (J-lens). That is over six new-language pairs, against 5.7 published for the
  original pairs. A different concept's English costs 0.1, and the early layers 0.2 (§4b).
- **When translation into a low-resource language fails, the English concept is usually still
  there.** On the 30B's misses, English is in the J-lens top 10 for 0.88 of items, vs 0.96 when the
  translation is correct (controls ≤ 0.01). That holds even when the model writes the source word
  (0.89) or a neighbour language (0.97). It reaches the concept and fails at producing the target (§4c).
- **What fails for low-resource languages is translation into them, not the English step.** The
  model translates *from* every new language well except Bodo (4/13 kept). *Into* the low-resource
  ones it mostly fails (Kashmiri 1/32, Bodo 1/13, Konkani 3/18, Sindhi 7/31, Santali 12/64, Manipuri
  18/64). Its decoded answers show how it fails. It writes the **source word** (Telugu→Kashmiri: 29 of 31 misses are in Telugu script) or a
  **neighbour language** (Hindi for Nepali, Marathi/Hindi for Konkani, Urdu for Sindhi). With an English
  source the same targets do better on the 30B: en→ks 13/32, en→brx 10/13, en→sat 17/32 (against
  bn→sat 12/32, the CIs overlap).
- **The logit lens reads less English for Meitei Mayek and Ol Chiki sources** (0.71 [0.64, 0.77],
  vs 0.84 for the baseline pairs). The J-lens
  shows no such gap (0.93 vs 0.95), so the English representation is there, just less readable directly.
- **Japanese**, barely covered by the tokenizer, is translated well (26–31 of 32). As a 30B target it has
  one of the weakest English steps (J-lens 0.84 [0.76, 0.90]; Sindhi 0.86, n = 7; CIs overlap) and the weakest Hindi step (0.36), and a few of its
  answers are plain English words ("house", "road").
- **Hindi stays a concept-specific second latent everywhere** (J-lens 0.26–0.88 by group, the top
  value from Maithili as target with n = 2; shuffled ≤ 0.01).
- **Dogri could not be tested.** Wiktionary has 6 of the 32 words (two identical to Hindi), Wikidata has
  no Dogri labels, and there is no Dogri Wikipedia (only an Incubator wiki). Maithili (10 words) and
  Bodo (13) are thin.
- Labels: 16 of the 17 self-names in `languages.py` match the title of each language's own Wikipedia
  article (or Incubator wiki for Dogri and Bodo); Kashmiri's matches the bold name in the article's lead
  rather than its title (§1). **No corrections.**

## 1. Word lists: sources and coverage

The rule was never to invent words. Every cell is copied from a cited page, in this order of preference
(the tier is recorded per cell in `concepts_new_sources.tsv`):

1. **English Wiktionary translation tables** (`en.wiktionary.org/wiki/<word>#Translations`, the block
   for the intended sense). The pages were read with firecrawl. Lines were copied verbatim into
   `data/newlangs/wiktionary_raw/<concept>.txt`, with the page revision (`oldid`) and the block's gloss.
   The table's first word in the language's script is used, unless a manual override applies (tier 2).
2. **Manual overrides** (`data/newlangs/overrides.tsv`, 49 cells, each with a reason). Most pick the
   everyday word over a formal or loan synonym *from the same table*: Odia ପାଣି over Sanskrit ଜଳ,
   Punjabi ਘਰ over the Persian loan ਮਕਾਨ, Urdu سورج over سوریہ, Japanese 猫 over 家猫. The rest take
   words from Wiktionary entry pages (Bodo, Dogri, Maithili; see tier 4).
3. **Wiktionary's hand-written Swadesh-list appendices** for Manipuri (in Meitei Mayek) and Santali
   (`data/newlangs/swadesh_appendix.tsv`). The English tables mostly give Manipuri in Bengali script
   or not at all. 29 of the 37 appendix words that also appear in an independently fetched copy
   match it exactly; the other 8 are different word choices, not copying errors.
4. **Wiktionary entry pages** whose first definition is the concept. For the small Dogri, Maithili and
   Bodo noun categories (33, 46 and 54 entries) every entry was read (`category_nouns.tsv`). For
   Konkani, Kashmiri, Manipuri, Sindhi and Santali gaps, targeted search results were confirmed on
   the entry (`search_hits.tsv`).
5. **Wikidata item labels** (`wikidata_labels.tsv`), only for Kashmiri, Manipuri, Santali and Sindhi.
   Their labels read as native words (Kashmiri گام village, Santali ᱥᱟᱫᱚᱢ horse). Maithili's labels
   were **not used**: they copy Hindi (कुत्ता for dog, where Maithili has कुकुर). Konkani's mix Latin,
   Kannada and Devanagari scripts. Wikidata has no label codes for Dogri or Bodo.

Normalisation: NFC. Single words only, in the language's script (`languages.py`). Arabic, Urdu and
Sindhi short-vowel marks and the optional noon-ghunna mark (U+0658) are stripped, since everyday
text omits them; the vocalised Wiktionary form is kept as an alternate spelling. Russian stress marks
are stripped. Invisible format characters are dropped (one Urdu cell carried a U+2069), except
ZWNJ/ZWJ. The original 11 columns are copied unchanged from `data/concepts.tsv` (checked byte for byte).

| lang | script | filled /32 | sources (cells) | notes |
|---|---|---|---|---|
| as Assamese | Bengali | 32 | table 32 | |
| or Odia | Odia | 31 | table 27, override 4 | cow missing; fish is Sanskrit ମୀନ (everyday ମାଛ not listed) |
| pa Punjabi | Gurmukhi | 31 | table 28, override 3 | cow missing |
| ur Urdu | Perso-Arabic | 32 | table 30, override 2 | |
| ne Nepali | Devanagari | 32 | table 31, override 1 | many words identical to Hindi (see §2) |
| sa Sanskrit | Devanagari | 32 | table 27, override 5 | stems (जल, नामन्); mother is {{t-check}} on Wiktionary |
| mai Maithili | Devanagari | 10 | table 6, entry 4 | 22 empty |
| sd Sindhi | Perso-Arabic | 31 | table 23, override 3, entry 3, Wikidata 2 | fish missing |
| ks Kashmiri | Perso-Arabic | 32 | table 16, entry 10, Wikidata 6 | |
| kok Konkani | Devanagari | 26 | table 11, entry 13, override 2 | rain, horse, salt, king, star, tooth missing |
| doi Dogri | Devanagari | 6 | table 2, entry 4 | **not run** |
| mni Manipuri | Meitei Mayek | 32 | Swadesh 19, table 4, entry 8, Wikidata 1 | |
| brx Bodo | Devanagari | 13 | entry 12, table 1 | 19 empty; the table's Bodo is romanized (`dwi`) |
| sat Santali | Ol Chiki | 32 | table 15, override 6, Swadesh 9, entry 2 | |
| ja Japanese | kanji/kana | 32 | table 29, override 3 | |
| ru Russian | Cyrillic | 32 | table 32 | |
| ar Arabic | Arabic | 32 | table 32 | |

Loanwords: no chosen word is an English loan. English loans in the tables (Urdu بُک, Japanese ミルク,
Hindi वाटर) were passed over. Persian/Arabic loans are kept where they are the everyday word
(Urdu کتاب, Sindhi جبل). Odia ଅଗ୍ନି (fire) and ମୀନ (fish) are Sanskrit where the everyday words are
ନିଆଁ and ମାଛ, which the tables don't list; the decoded answers show the model uses the everyday ones
(§3.3).

**Label check** (`data/newlangs/label_check.md`): all 17 labels match the language's own Wikipedia
article title code point for code point, except Kashmiri. Its label کٲشُر matches the bold name in the
article's first line, while the title has a kasra (کٲشِر); we keep the label. Dogri (डोगरी) and Bodo
(बर') have only Incubator test wikis, and the labels match those.

## 2. Method

Same as the published pivot (doc 02 §2.3): 5-shot prompts with each language's self-name as the line
label (no English in the prompt unless English is the source), the same token sets (prefixes of at
least 2 characters plus the tokenizer's own first token), the disjoint-set rule for script collisions,
the shuffled-concept controls (the same language's word for the concept half-way round the list), the
correctness filter (the model's next token must begin a listed spelling), readout filtering, and 95%
bootstrap CIs over concepts.

**Pairs (43; `newlang_experiment.PAIRS`):**
- Each new language once as source and once as target, against an original language of another
  family and, where possible, script: as↔te/ta, or↔ta/kn, pa↔ml/te, ur→ml, kn→ur, ne↔ta/ml, sa↔kn/te,
  mai↔ml/kn, sd↔kn/ml, ks↔ta/te, kok↔te/ta, mni→bn, te→mni, sat→hi, bn→sat, brx→ta, bn→brx,
  ja↔ta/kn, ru↔ml/te, ar↔kn/ml.
- New↔new: mni→sat, sat→mni (the two rare native scripts), ur→ne (Perso-Arabic into Devanagari).
- English→X controls (the model's translation skill into the lowest-resource targets): en→mni,
  en→sat, en→brx, en→ks, en→ja.
- Three original pairs as a same-run baseline: ta→te, bn→ta, te→gu.

**Two rules specific to this list:**
- **Kanji rule.** Pairs with Japanese track no Chinese role and no shuffled-Chinese control. Japanese
  words are mostly the Chinese column's characters (水 山 手 猫), so the disjoint-set rule would drop most
  Japanese targets as shared with Chinese. The two can't be separated at the token level anyway.
- **Script collisions with Hindi.** Hindi is always tracked, so a Devanagari target whose word is (or
  begins like) the Hindi word loses those tokens from both sets. 49 items lose their whole target set
  and are skipped. Nepali as a target loses 15 of 32 (पानी, घर, नदी, राजा…), Sanskrit 7, Konkani 8,
  Maithili 4, and ur→ne 15. For those targets the Hindi latent is measured only on the remaining
  tokens.

**Readout, batched.** `pivot_experiment.main` runs one unbatched forward per lens per item. Instead,
`newlang_experiment.pivot` reads every lens off **one** forward per item, 16 items per forward
(`reader_questions.Reader.forward_batch`), and decodes all layers of a lens in one unembed. It uses
the same functions for prompts, token sets, roles and summaries.
- On a tiny CPU model, batch size 1 reproduces `pivot_experiment.main`'s
  rows exactly: 5,358 rows, Δprob 0.0, identical ranks. Batch 5 differs only by float rounding
  (≤ 4e-6, no top-10 changes).
- On the real 30B, this run's three baseline pairs reproduce the published unbatched pivot
  (`results/final/pivot_all`). English top-10 at the band, J-lens: ta→te 0.966 vs 0.966, te→gu 0.972 vs
  0.972, bn→ta 0.920 vs 0.938 (22 items kept vs 24). Probabilities agree to within 0.015.
- `batch_check` on the 30B: top-1 agreement 1.0 and greedy-decode match 0.875. The largest |Δlog-prob|
  is 7.05, the same value as in the two sibling sessions' runs, so it lies in the distribution's tail.

**Decoded answers (30B only).** Every item was also greedy-decoded for 8 tokens, and the answer up to
its closing quote was matched against the list. This gives an exact-word accuracy next to the
first-token one. The first-token filter is lenient where a word's first token is a bare letter: Ol
Chiki ᱫ, as in the original study's Tamil த. The decoding also shows what the model wrote when it
missed.

Band: the English band L13–16. Per item, the top-10 rate and probability
are averaged over the band. Groups pool pairs: `X>` = X as source, `>X` = X as target (new↔new pairs
count in both).

## 3. Sarvam-30B (`results/xl_new_30b`, J-lens `j_200p` + logit)

1,208 items tested over 43 pairs (49 skipped as shared), 780 kept by the first-token filter.

### 3.1 The English step is the same in every class

J-lens top-10 rate by layer. Each row pools the kept items of the pairs with that class at either end
(English→X excluded):

| class | role | L9 | L11 | L12 | L13 | L14 | L15 | L16 | L17 |
|---|---|---|---|---|---|---|---|---|---|
| baseline: ta→te, bn→ta, te→gu (n=69) | English | 13% | 33% | 46% | 84% | 99% | 99% | 100% | 90% |
| | English, other concept | 1% | 0% | 0% | 0% | 0% | 1% | 1% | 0% |
| | Hindi | 0% | 1% | 1% | 45% | 62% | 58% | 75% | 68% |
| new Indo-Aryan, Brahmic script (as or pa ne sa mai kok; n=265) | English | 10% | 15% | 32% | 82% | 99% | 98% | 100% | 80% |
| | English, other concept | 2% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| | Hindi | 0% | 0% | 0% | 36% | 55% | 57% | 67% | 54% |
| Perso-Arabic script (ur sd ks; n=119) | English | 7% | 30% | 45% | 86% | 98% | 98% | 99% | 84% |
| | English, other concept | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| | Hindi | 0% | 2% | 3% | 40% | 67% | 68% | 78% | 66% |
| own script, not Indo-Aryan (mni Meitei Mayek, sat Ol Chiki; n=78) | English | 15% | 23% | 49% | 85% | 96% | 97% | 97% | 90% |
| | English, other concept | 4% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| | Hindi (n=55) | 0% | 0% | 0% | 42% | 55% | 56% | 69% | 67% |
| world: ja ru ar (n=169) | English | 13% | 24% | 41% | 79% | 98% | 99% | 99% | 85% |
| | English, other concept | 1% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| | Hindi | 0% | 1% | 4% | 37% | 53% | 56% | 69% | 37% |

(Bodo alone: n=5, 100% from L13.)

### 3.2 Per language

Band L13–16. "kept" = first token correct (the correctness filter). "exact word" = the decoded answer
is a listed spelling (or an inflected form of one). EN = the English word; HI = the Hindi word. CIs are
95% bootstrap over concepts. Readouts cover kept items only.

| language, direction | kept / tested | exact word | EN top-10, J [95% CI] | EN P, J | EN other concept, J | HI top-10, J (n) | EN top-10, logit [95% CI] | EN onset J / logit |
|---|---|---|---|---|---|---|---|---|
| as> | 24/32 | 23/32 | 0.96 [0.917,0.990] | 0.53 | 0.00 | 0.69 (24) | 0.80 [0.708,0.885] | L13 / L14 |
| >as | 22/32 | 19/32 | 0.97 [0.932,1.000] | 0.56 | 0.00 | 0.65 (22) | 0.89 [0.830,0.943] | L13 / L13 |
| or> | 25/31 | 25/31 | 0.98 [0.950,1.000] | 0.60 | 0.00 | 0.64 (25) | 0.87 [0.820,0.910] | L13 / L14 |
| >or | 23/31 | 24/31 | 0.91 [0.859,0.967] | 0.47 | 0.00 | 0.57 (23) | 0.74 [0.641,0.837] | L13 / L14 |
| pa> | 30/31 | 30/31 | 0.93 [0.883,0.975] | 0.49 | 0.00 | 0.64 (30) | 0.74 [0.675,0.808] | L13 / L14 |
| >pa | 24/31 | 22/31 | 0.97 [0.938,1.000] | 0.61 | 0.00 | 0.60 (24) | 0.81 [0.760,0.854] | L12 / L14 |
| ur> (ur→ml, ur→ne) | 39/49 | 38/49 | 0.95 [0.891,0.988] | 0.56 | 0.00 | 0.68 (37) | 0.82 [0.768,0.875] | L13 / L14 |
| >ur | 24/32 | 23/32 | 0.95 [0.906,0.979] | 0.49 | 0.00 | 0.68 (24) | 0.79 [0.750,0.833] | L13 / L14 |
| ne> | 28/32 | 28/32 | 0.93 [0.884,0.964] | 0.46 | 0.01 | 0.53 (20) | 0.77 [0.696,0.830] | L13 / L14 |
| >ne (ml→ne, ur→ne) | 17/34 | 13/34 | 0.97 [0.933,1.000] | 0.56 | 0.00 | 0.62 (13) | 0.85 [0.789,0.926] | L13 / L14 |
| sa> | 29/32 | 29/32 | 0.93 [0.879,0.974] | 0.51 | 0.00 | 0.40 (25) | 0.79 [0.724,0.853] | L13 / L14 |
| >sa | 14/25 | 11/25 | 0.93 [0.875,0.982] | 0.54 | 0.00 | 0.54 (13) | 0.88 [0.804,0.929] | L13 / L13 |
| mai> | 8/10 | 8/10 | 0.91 [0.719,1.000] | 0.59 | 0.00 | 0.62 (6) | 0.84 [0.594,1.000] | L12 / L13 |
| >mai | 2/6 | 0/6 | 0.88 | 0.74 | 0.00 | 0.88 (2) | 0.75 | L12 / L14 |
| sd> | 26/31 | 26/31 | 0.97 [0.942,1.000] | 0.56 | 0.00 | 0.59 (26) | 0.75 [0.654,0.837] | L12 / L14 |
| >sd | 7/31 | 5/31 | 0.86 [0.714,0.964] | 0.47 | 0.00 | 0.54 (7) | 0.71 [0.464,0.893] | L13 / L14 |
| ks> | 22/32 | 22/32 | 0.98 [0.932,1.000] | 0.63 | 0.00 | 0.66 (22) | 0.94 [0.864,1.000] | L11 / L13 |
| >ks | 1/32 | 1/32 | (1 item) | | | | | |
| en>ks | 13/32 | 7/32 | (English is the source) | | | 0.65 (13) | | |
| kok> | 16/26 | 17/26 | 0.98 [0.953,1.000] | 0.58 | 0.00 | 0.50 (12) | 0.78 [0.625,0.906] | L11 / L13 |
| >kok | 3/18 | 2/18 | (3 items) 1.00 | 0.75 | 0.00 | 0.67 (3) | 0.83 | L12 / L14 |
| mni> (mni→bn, mni→sat) | 25/64 | 24/64 | 0.97 [0.940,1.000] | 0.47 | 0.00 | 0.55 (25) | 0.69 [0.590,0.770] | L12 / L14 |
| >mni (te→mni, sat→mni) | 18/64 | 13/64 | 0.97 [0.934,1.000] | 0.66 | 0.00 | 0.54 (18) | 0.85 [0.765,0.921] | L12 / L13 |
| en>mni | 17/32 | 15/32 | (English is the source) | | | 0.56 (17) | | |
| brx> | 4/13 | 6/13 | (4 items) 1.00 | 0.51 | 0.00 | 0.31 (4) | 0.62 | L11 / L14 |
| >brx | 1/13 | 0/13 | (1 item) | | | | | |
| en>brx | 10/13 | 9/13 | (English is the source) | | | 0.60 (10) | | |
| sat> (sat→hi, sat→mni) | 32/64 | 28/64 | 0.89 [0.784,0.968] | 0.51 | 0.00 | 0.42 (9) | 0.73 [0.637,0.814] | L13 / L14 |
| >sat (bn→sat, mni→sat) | 12/64 | 12/64 | 0.98 [0.938,1.000] | 0.60 | 0.00 | 0.58 (12) | 0.85 [0.792,0.917] | L12 / L14 |
| en>sat | 17/32 | 14/32 | (English is the source) | | | 0.57 (17) | | |
| ja> | 30/32 | 30/32 | 0.98 [0.958,1.000] | 0.59 | 0.00 | 0.53 (30) | 0.92 [0.850,0.967] | L12 / L13 |
| >ja | 26/32 | 25/32 | **0.84 [0.760,0.904]** | 0.44 | 0.00 | **0.36** (26) | 0.86 [0.769,0.923] | L13 / L13 |
| en>ja | 27/32 | 26/32 | (English is the source) | | | 0.26 (27) | | |
| ru> | 30/32 | 29/32 | 0.97 [0.950,1.000] | 0.64 | 0.00 | 0.64 (30) | 0.88 [0.800,0.933] | L12 / L13 |
| >ru | 29/32 | 27/32 | 0.94 [0.897,0.974] | 0.57 | 0.00 | 0.59 (29) | 0.84 [0.802,0.888] | L13 / L14 |
| ar> | 31/32 | 31/32 | 0.98 [0.960,1.000] | 0.67 | 0.00 | 0.55 (31) | 0.89 [0.815,0.952] | L11 / L13 |
| >ar | 23/32 | 20/32 | 0.88 [0.804,0.957] | 0.55 | 0.00 | 0.54 (23) | 0.78 [0.696,0.859] | L13 / L14 |
| **baseline** (ta→te, bn→ta, te→gu) | 69/96 | 69/96 | 0.95 [0.916,0.983] | 0.58 | 0.01 | 0.60 (69) | 0.84 [0.780,0.889] | L13 / L13 |

The shuffled-Hindi control is 0.00 in every group, and shuffled Chinese is ≤ 0.01. Chinese itself
reaches 0.00–0.14 under the J-lens (0.05–0.24 under the logit lens), as in the published pivot.
Full tables, including the pooled classes and every role's probability with CIs:
`results/xl_new_30b/report_tables.md` (regenerate with
`python code/newlang_experiment.py report results/xl_new_30b/newlang_experiment_pivot --band 13-16 --n-layers 19`).
Figures: `results/xl_new_30b/newlang_experiment_pivot/*.png`.

### 3.3 What the model wrote

From the decoded answers. A "miss" means the first token is not on the list.

| target | from | kept | exact word | unlisted word in the target script | the source word / source script | another listed language's word |
|---|---|---|---|---|---|---|
| as | Indian | 22/32 | 19 | 8 (পক্ষী, অগ্নি, হৃদয়: valid synonyms) | 0 / 0 | 2 (Bengali) |
| or | Indian | 23/31 | 24 | 7 (ନିଆଁ fire, ମାଛ fish: the everyday words) | 0 / 0 | 0 |
| ne | Indian | 17/34 | 13 | 2 | 2 / 2 | **11 (Hindi)** |
| sd | Indian | 7/31 | 5 | 6 | 2 / 1 | **11 (5 of them Urdu)** |
| kok | Indian | 3/18 | 2 | 6 | 0 / 0 | **8 (Marathi or Hindi)** |
| ks | English | 13/32 | 7 | 15 (Perso-Arabic, many close variants) | – | 4 (Urdu) |
| ks | Indian (te→ks) | 1/32 | 1 | 1 | **12 / 17 (Telugu)** | 1 |
| mni | English | 17/32 | 15 | 15 | – | 0 |
| mni | Indian | 18/64 | 13 | 22 | **16 / 8** | 0 |
| sat | English | 17/32 | 14 | 13 | – | 1 |
| sat | Indian | 12/64 | 12 | 11 | **23 / 16** | 1 |
| brx | English | 10/13 | 9 | 2 | – | 0 |
| brx | Indian (bn→brx) | 1/13 | 0 | 0 | **6 / 3 (Bengali)** | 3 |
| ja | Indian / English | 26/32 / 27/32 | 25 / 26 | 1 / 1 | – | 2 (Chinese 王, 黄金) + **2 English words in each direction** ("house", "road", "sun") |
| ru, ar, ur, pa | Indian | 23–29 of 31–32 | | 2–6 | ≤ 2 | ≤ 3 |

Full table (all categories): `results/xl_new_30b/report_tables.md`.

## 4. Sarvam-105B (not published yet)

Not part of this release; see the README's last section.

## 4b. Is the English step used? Causal check

**Sarvam-30B** (`results/xl_new_causal_30b`).

This is the published English ablation (doc 02 §2.4, `causal_test.run`), unchanged, on six
new-language pairs: one or two per class, each with mostly-correct translations in the pivot. At the
final position, after each block of the English band (L13–16), the subspace spanned by the J-lens
steering directions of the English word's tokens is projected out. The model's probability of the
correct target-word start is then measured. Controls: the same-size subspace for a different concept's
English word (the fixed half-way derangement), and the English subspace at an early band (L5–8).
Negative steering (α = 3) pushes the English component to −2× (doc 09; published as `causal_steer`).
The run is **unbatched**, like the published one, and covers correctly translated items only.
CIs are paired 2,000-resample bootstraps. Each pair runs on `concepts_new.tsv` cut to the concepts that
have both words, which leaves demos and controls as in the pivot (Odia lacks "cow").

ΔP(target) in points (negative = lower probability), [95% CI], and the share of translations still
correct:

| pair | n | base P | English L13–16 | other concept L13–16 | English L5–8 | English pushed to −2× | other concept pushed to −2× |
|---|---|---|---|---|---|---|---|
| or→ta | 26 | 0.78 | −10.2 [−17.6, −3.5], 92% | −1.5 [−3.2, −0.1], 96% | +0.6 [−1.4, +2.6] | −52.0 [−62.7, −41.8], 39% | −18.5 [−27.8, −9.7], 77% |
| ur→ml | 30 | 0.91 | −9.1 [−14.6, −4.3], 90% | +0.2 [−0.6, +1.1], 100% | −0.0 [−0.8, +0.7] | −52.5 [−65.5, −38.8], 40% | −15.4 [−24.2, −8.2], 93% |
| ks→ta | 22 | 0.79 | −16.9 [−23.4, −11.0], 86% | +1.6 [−2.5, +6.9], 95% | +0.7 [−0.4, +1.8] | −40.4 [−53.1, −27.6], 46% | −7.4 [−14.0, −0.7], 86% |
| mni→bn | 25 | 0.81 | −12.9 [−19.8, −6.4], 88% | −0.1 [−1.7, +1.4], 100% | +0.4 [−1.1, +1.7] | −47.1 [−61.9, −32.4], 40% | −9.4 [−19.7, −0.8], 92% |
| sat→hi | 23 | 0.89 | −16.2 [−25.8, −7.9], 91% | −0.5 [−1.8, +0.7], 100% | −1.1 [−2.1, −0.1] | −32.8 [−49.7, −17.2], 65% | −8.8 [−16.0, −3.3], 96% |
| ja→ta | 30 | 0.92 | −15.3 [−24.1, −8.0], 90% | −0.3 [−1.3, +0.6], 100% | −1.3 [−2.7, −0.2] | −45.3 [−57.6, −32.4], 57% | −21.5 [−31.8, −12.1], 80% |
| **all six** | **156** | 0.86 | **−13.2 [−16.2, −10.6], 90%** | **−0.1 [−0.9, +0.7], 99%** | −0.2 [−0.8, +0.3], 100% | **−45.6 [−50.9, −39.9], 47%** | −14.0 [−17.7, −10.5], 87% |
| published, 10 original pairs (`final/causal_j_200p`) | 272 | 0.88 | −5.7 [−7.1, −4.3], 95% | −0.4 [−0.8, +0.0], 98% | −0.3 [−0.8, +0.1] | | |
| published steering (`rq30b/causal_steer_j_200p`) | 143 | 0.91 | −6.3 [−8.1, −4.5], 94% | −0.1 [−0.6, +0.4] | 0.0 [−0.4, +0.4] | −41.2 [−47.5, −35.5], 59% | −8.9 [−12.3, −5.8], 90% |

- **The English step is causally used for every new language tested,** and the effect is specific to
  English. Removing the English word's directions at L13–16 costs 13.2 points. The same-size
  subspace of a different concept's English word costs 0.1, and the same removal at L5–8 costs 0.2.
  English costs 13.1 [10.4, 16.3] points more than the control.
- **It costs more than for the original languages.** The published pooled effect is 5.7 points, and
  per original pair it ranges from 1.8 to 10.7. All six new pairs fall between 9.1 and 16.9. Kashmiri,
  Santali, Japanese and Manipuri sources (16.9, 16.2, 15.3, 12.9) are above every original pair. The
  per-pair CIs (n = 22–31) overlap one another, so the new pairs can't be ranked among themselves. In
  relative terms it is 15% of the base probability vs 6%. Still, ~90% of translations survive, so it
  isn't a bottleneck here either.
- **Under negative steering the gap to the control matches the original languages.** English minus
  control at −2× is 31.6 points, vs 32.3 published. The larger α = 1 effect is specific to English
  (its control is ~0); the blunt push also disturbs the new pairs more generically (control −14.0 vs
  −8.9).
- Different pairs, different items and different base probabilities separate the two runs, so "more
  than" compares this run with the published one rather than within one run. The six pairs share
  the original study's English words, token sets and code path.

## 4c. Is English there when translation fails? (`results/xl_new_misses_30b`)

When Sarvam-30B fails to translate into a low-resource language, does it still reach the English
concept mid-network? If it does, it understands the word and fails only at writing the target language.
- **Method.** Analysis `misses` is the same pivot (prompts, token sets, roles, controls, J-lens + logit)
  with readout rows written for the items the correctness filter drops, too. Each miss is greedy-decoded
  (8 tokens) and typed by what the model wrote (§3.3):
  - the source word or source script ("source copy");
  - the same concept's word in another listed language ("neighbour language");
  - an unlisted word in the target script (a synonym, variant or wrong word);
  - a listed word whose first token the filter missed;
  - other.
- **Pairs.** The 30B's worst targets: te→ks, bn→sat, mni→sat, te→mni, sat→mni, bn→brx, ta→kok, ml→sd.
  Then en→ks, en→sat, en→mni, en→brx as controls. 331 items tested, 99 kept.
- **Stats.** Band L13–16, band-mean top-10 rate [95% CI over concepts] and probability. With an English
  source, English *is* the source word, so its presence there is the source column.

**Pooled, J-lens** (Indian or world source):

| status | items | English top-10 (P) | English, other concept | source word | target word | Hindi | Hindi, other concept |
|---|---|---|---|---|---|---|---|
| translated correctly | 44 | **0.96** [0.92, 0.99] (0.61) | 0.00 | 0.14 | 0.09 | 0.57 | 0.00 |
| missed, all | 178 | **0.88** [0.83, 0.94] (0.49) | 0.01 | 0.12 | 0.01 | 0.41 | 0.01 |
| missed: source copy | 104 | 0.89 [0.83, 0.95] (0.51) | 0.00 | 0.11 | 0.00 | 0.44 | 0.00 |
| missed: neighbour language | 24 | 0.97 [0.91, 1.00] (0.52) | 0.00 | 0.16 | 0.01 | 0.40 | 0.00 |
| missed: unlisted target-script word | 45 | 0.81 [0.70, 0.92] (0.43) | 0.04 | 0.13 | 0.01 | 0.32 | 0.02 |
| missed: other (+1 listed word) | 5 | 0.94–1.00 | 0.00 | – | – | – | – |

The logit lens gives the same picture at lower levels: correct 0.81 [0.73, 0.88] (P 0.26), missed 0.69
[0.62, 0.76] (P 0.20). By failure type: source copy 0.70 [0.65, 0.76], neighbour 0.81 [0.72, 0.89],
target-script word 0.58 [0.46, 0.73]. Controls 0.00.

With an English source (the English word is the source), J-lens: correct 55 items, 0.99 [0.97, 1.00];
missed 54 items, 0.97 [0.94, 1.00]. Logit: 0.91 and 0.88.

**Per target, J-lens English top-10, correct vs missed** (Indian/world source; n in brackets):

| target | correct | missed | by failure type |
|---|---|---|---|
| Kashmiri (te→ks) | 1.00 (1) | 0.93 [0.84, 0.99] (31) | source copy (Telugu) 0.92 (29) |
| Santali (bn, mni→sat) | 0.98 [0.94, 1.00] (12) | 0.87 [0.79, 0.95] (52) | source copy 0.84 (39); target-script word 0.95 (11) |
| Manipuri (te, sat→mni) | 0.97 [0.94, 1.00] (19) | 0.79 [0.68, 0.88] (45) | source copy 0.89 (24); target-script word 0.68 (21) |
| Bodo (bn→brx) | 1.00 (1) | 0.98 [0.94, 1.00] (12) | source copy (Bengali) 0.97 (9) |
| Konkani (ta→kok) | 1.00 (4) | 1.00 (14) | neighbour (Marathi/Hindi) 1.00 (8) |
| Sindhi (ml→sd) | 0.86 [0.71, 0.96] (7) | 0.91 [0.80, 0.98] (24) | neighbour (Urdu the most frequent, 5) 0.94 (12) |

- **Yes: English is present on most failures**, almost as often as on successes. That is 0.88 vs 0.96
  under the J-lens and 0.69 vs 0.81 under the logit lens, with the other-concept controls at 0.00–0.01.
  So on most misses Sarvam reaches the English concept and fails afterwards, at producing the target
  language. Within a target, the gap between correct and missed items is −0.05 to 0.18 (the CIs above).
- **Every failure type carries English.** Copying the source word (0.89) and switching to a neighbour
  language (0.97) both happen *after* the English step: the model knows the concept, then writes Telugu
  for Kashmiri or Marathi for Konkani. The weakest type is an unlisted word in the target script
  (0.81; Manipuri 0.68). It mixes valid synonyms with wrong words, so some of these may be concepts the
  model never reached.
- **The source word is not what the model holds mid-network, even when it copies it.** Source-word
  top-10 at the band is 0.11–0.16 for every status, far below English. The copy is decided late.
- **The target word is absent on misses** (0.00–0.01 vs 0.09 on correct items, where it is only
  starting to appear at these layers). Hindi drops a little (0.41 vs 0.57) and stays concept-specific
  (control ≤ 0.02).
- **Caveats.** Per-target n is small where the model almost never succeeds (Kashmiri and Bodo: 1 correct
  item each), so the per-target "correct" cells are anecdotal there; the pooled rows are the result.
  "Missed" still includes word-list misses (the target-script category). The pairs differ in how often
  they fail, so the pooled comparison mixes pairs; the per-target rows control for that.

## 4d. Language-neutral or English-centric? (`docs/15_concept_geometry.md`)

This tests a reviewer's objection to the readouts above: is the mid-network concept space English, or
does it only *read out* as English because 148k of Sarvam's 262k tokens are Latin? Two token-free
measures answer it, on 788 words (32 concepts × 27 languages, this list) in two neutral carriers, plus a
vocabulary-fair readout. Full write-up: `docs/15_concept_geometry.md`.
- **The concept space is shared.** Cross-language retrieval at Sarvam-30B's English band is 0.84 with
  the line carrier (chance 0.03; bare words 0.45–0.48), including Meitei Mayek and Ol Chiki.
- **English holds no large lead in it.** The centroid nearest-language test favours Hindi and
  the Indo-Aryan majority, and English ties with Hindi once shared-script languages are removed. As
  retrieval hubs, English, Hindi and Chinese are close (0.92 / 0.91 / 0.89; English − Hindi is
  significant only at L13, English − Chinese +0.03 [0.002, 0.067] at L15).
- **Most of English's readout lead is the vocabulary.** Within its own script, the Hindi word is in the
  top 10 nearly as often as the English word (0.92 vs 0.95, J-lens).
- **Qwen is the one with an English hub.** With the line carrier its English hub beats Chinese and Hindi
  (0.73 / 0.65 / 0.59), and Chinese is not privileged.

**Verdict:** shared, and only weakly skewed, not English-centric. English and Hindi are co-anchors.

## 5. Findings

1. **Does every Indian language route through English? Yes, every one we could test.**
   That covers 13 of the 14 new scheduled languages (Dogri had no usable word list) and the world
   controls, as source and, where enough items survive, as target. In every group with more than 3
   kept items, English is in the top 10 for ≥ 0.84 of items (J-lens). The shuffled control is at most
   0.03. The step sits at the same depth everywhere: onset L11–13, near-saturated at L14–16. This is
   the same as for the original 8 languages.
2. **Low-resource languages: the pivot isn't weaker, but translation into them is far weaker.**
   Among correctly translated items, English is as present for Kashmiri, Santali, Manipuri or Bodo as
   for Odia or Punjabi (J-lens 0.89–1.00). As sources they are translated well: Kashmiri 22 of 32,
   Sindhi 26 of 31, Konkani 16 of 26. As targets the model mostly fails: Kashmiri 1 of 32, Santali 12
   of 64, Manipuri 18 of 64, Sindhi 7 of 31, Konkani 3 of 18, Nepali 17 of 34. The
   misses are mostly a failure to switch into the target. It writes the source word (Telugu→Kashmiri,
   Bengali→Bodo, Meitei→Santali) or the nearest big neighbour (Hindi for Nepali, Marathi/Hindi for
   Konkani, Urdu for Sindhi). From English it does better on the same targets (en→Kashmiri 13/32
   vs te→Kashmiri 1/32; en→Bodo 10/13 vs bn→Bodo 1/13).
3. **Perso-Arabic scripts: no clear difference.** As sources, J-lens 0.95–0.98 (pooled 0.96
   [0.93, 0.99]).
4. **Meitei Mayek and Ol Chiki: the pivot holds, and the logit lens reads less of it.** With the
   J-lens they match the baseline: pooled 0.93 [0.86, 0.98] as sources and 0.97 [0.94, 1.00] as
   targets, vs 0.95. Under the logit lens they are the lowest class as sources: 0.71 [0.64, 0.77] vs
   0.84 [0.78, 0.89], with non-overlapping CIs. So the English representation is there (the J-lens
   finds it) but is less directly vocabulary-aligned when the input is in these scripts. Punjabi
   (0.74) and Sindhi (0.75) sources are also low under the logit lens, so it isn't only a rare-script
   effect.
5. **Japanese, barely tokenized: translated well.** Japanese and Sindhi (0.86, n = 7) are the targets
   with English below 0.88 under the J-lens (Maithili 0.875 rests on n = 2). Japanese: 0.84
   [0.76, 0.90] vs 0.95 [0.92, 0.98] for the baseline. Its Hindi step is also the weakest there
   (0.36), and a few answers are English words. As a source it behaves like any other language
   (0.98). Common kanji are single tokens (水, 犬, 猫), so for these 32 words Japanese is less
   byte-fallback than whole sentences.
6. **Hindi is the second latent for every language.** It is top-10 for 0.26–0.88 of items by group
   (the top value: Maithili as target, n = 2), including Japanese, Russian, Arabic and Ol Chiki
   sources. Its shuffled control is 0.00 everywhere.

7. **The English step is causally used for the new languages, and more than for the original ones.**
   Projecting the English word's lens directions out at the English band (J-lens, L13–16) costs the
   target 13.2 points [10.6, 16.2], over six new-language pairs (or, ur, ks, mni, sat, ja as source).
   The published figure for the original pairs is 5.7, so the ratio is ~2.3×. Controls (another
   concept's English; early layers) cost 0.1 / 0.2. Kashmiri has the largest raw effect (−16.9), but
   per-pair CIs (n = 22–31) overlap. ~90% of translations survive removal, so English is used but
   isn't a bottleneck; pushing it to −2× breaks 53% (§4b).

8. **When translation fails, the English concept is usually still there.** On the 30B's worst
   targets (ks, sat, mni, brx, kok, sd), English is in the J-lens top 10 at L13–16 on 0.88 of missed
   items vs 0.96 of correct ones (logit 0.69 vs 0.81; controls ≤ 0.01). That holds whether the model
   copied the source word (0.89) or wrote a neighbour language (0.97). Most low-resource failures are
   failures to *produce* the target language after the concept has been reached (§4c).

## 6. Caveats

- **Readouts cover only correctly translated items.** On the 30B the low-resource targets rest on 1–18
  kept items (Kashmiri 1, Bodo 1, Konkani 3, Maithili 2, Sindhi 7), so for them "routes through
  English" is shown mainly as *sources*. Whether English is present when
  translation *fails* is measured separately on the 30B (§4c): it is, on most misses.
- **Accuracy mixes model errors with word-list errors.** The lists come from Wiktionary and Wikidata
  and weren't checked by native speakers. The 30B's decoded answers show valid synonyms counted as
  misses (Odia ନିଆଁ, ମାଛ; Assamese পক্ষী). The first-token filter is lenient for Ol Chiki, whose first
  tokens are bare letters, so the 30B's exact-word accuracy is given beside it (§3.2).
- **The pivot is batched**, and reproduces the published unbatched pivot (§2).
- **Script collisions.** Hindi and Devanagari targets share tokens. 49 items are skipped, and the Hindi
  latent for Nepali, Sanskrit, Konkani and Maithili targets is measured on reduced token sets.
- **Coverage.** Dogri was not run (6/32 words). Maithili (10) and Bodo (13) are thin. Tier-5 cells
  (Wikidata labels: Kashmiri 6, Sindhi 2, Manipuri 1) are the least certain.
- **Pairs and lenses.** Each new language sits in about 2 directions against one or two partners, so
  per-language CIs are wide (n = 1–48 kept items); the class pools are the more reliable comparison.
  The 30B's J-lens was fitted on English WikiText, as in the paper. The logit lens reads less English
  than the J-lens across the board (0.84 vs 0.95 at baseline).

## 7. Cost

| run | hardware | minutes | ≈ $ (report.json) | what |
|---|---|---|---|---|
| `xl_new_smoke` | CPU 4c/16G | ~7 (one preemption restart) | 0.04 (estimate) | smoke test (48 checks) + dry run of load, batch_check, `newlang_experiment.pivot` |
| `xl_new_30b` | 1× H200 | 7.73 (load 0.99, batch_check 0.47, stage 5.92) | 0.67 | 43 pairs, J-lens + logit, every item decoded (8 tokens) |
| `xl_new_causal_smoke` | CPU 4c/16G | ~4 | 0.03 (estimate) | smoke test + dry run of load, `newlang_experiment.causal` |
| `xl_new_causal_30b` | 1× H200 | 3.85 (load 0.66, stage 3.04) | 0.33 | English ablation + steering, 6 pairs, 156 items, unbatched |
| `xl_new_misses_smoke` | CPU 4c/16G | ~4 | 0.03 (estimate) | smoke test + dry run of load, `newlang_experiment.misses` |
| `xl_new_misses_30b` | 1× H200 | 3.04 (load 0.58, stage 2.27) | 0.26 | readouts on missed items too, 12 pairs, 331 items, misses decoded (separate $0.56 approval) |
| **total** | | | **≈ 1.36** | the misses check ran under a separate $0.56 approval |

The 30B's stage time went almost entirely to decoding every item (the readout itself is seconds).
Word-list collection used firecrawl (no Modal cost).

## 8. Reproduce

From `code/` (in Git Bash, prefix the `modal` commands with `MSYS_NO_PATHCONV=1`):

```bash
python data/newlangs/build_concepts_new.py --write
python newlang_experiment.py check <tiny-sarvam dir> --lens <tiny_j_lens.pt> --out <dir>
modal run modal_app.py --step smoke --stages load,batch_check,newlang_experiment.pivot --stage-args '{"decode": "all"}'
modal run modal_app.py --step pilot --model sarvam --stages load,batch_check,newlang_experiment.pivot --budget-minutes 9 --timeout-margin 2 --spf-guess 0.12 --tag xl_new_30b --stage-args '{"decode": "all", "decode_steps": 8}'
python newlang_experiment.py report ../results/xl_new_30b/newlang_experiment_pivot --band 13-16 --n-layers 19
modal run modal_app.py --step smoke --stages load,newlang_experiment.causal
modal run modal_app.py --step pilot --model sarvam --stages load,newlang_experiment.causal --budget-minutes 5.2 --timeout-margin 2 --spf-guess 0.12 --tag xl_new_causal_30b
modal run modal_app.py --step smoke --stages load,newlang_experiment.misses
modal run modal_app.py --step pilot --model sarvam --stages load,newlang_experiment.misses --budget-minutes 4.1 --timeout-margin 2 --spf-guess 0.12 --tag xl_new_misses_30b
python newlang_experiment.py misses-report ../results/xl_new_misses_30b/newlang_experiment_misses --band 13-16
```

The first command rebuilds the word lists from the saved sources. The `check` command confirms the
batched readout equals `pivot_experiment.main` on a tiny model.
