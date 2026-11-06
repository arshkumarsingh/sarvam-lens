# 15. Is the middle-layer concept space language-neutral or English-centric?

Run 2026-10-19. Code: `code/newlang_experiment.py` (analysis `geometry`) and
`code/concept_geometry_controls.py` (local controls on the saved vectors). Runs: `results/xl_geom_30b/`
(Sarvam-30B) and `results/xl_geom_qwen/` (Qwen3.6-35B-A3B). Word lists: `code/data/concepts_new.tsv`
(docs/11 §1). Logs: `logs/modal/xl_geom_*.log`.

The question is a reviewer's objection to the pivot result. The lens readouts find the English word
mid-network (doc 04, docs/11), but 148k of Sarvam's 262k vocabulary tokens are Latin. So is the concept
space really English-centric, or is it language-neutral and only *reads out* as English because English
owns most of the vocabulary? We answer with two measures that never decode through tokens, plus a
vocabulary-fair version of the readout.

## Verdict

**Shared, and only weakly skewed, not English-centric. Within that shared space, English and Hindi
act as co-anchors.**

- **There is a strongly shared concept space.** A concept's vector in one language retrieves the same
  concept among 32 in another language 84% of the time at Sarvam-30B's English band with the line
  carrier (chance 3%; bare words 45–48%). That includes Meitei Mayek and Ol Chiki words
  (Manipuri→Bengali 0.70, Santali→Hindi 0.76, 32 concepts each, no CI computed per pair).
- **Geometrically, English holds no large lead in Sarvam.**
  - It is not the language closest to the plain cross-language concept centroid. Indo-Aryan languages
    are (pa, gu, hi with the line carrier; Hindi first with bare words), and with only other-script
    centroids English ties with Hindi. Only when families are also weighted equally does English
    come first (+0.07 to +0.12 over Hindi on the 30B line carrier), and then it does on both
    models.
  - It is not a better retrieval hub than Hindi or Chinese: at L15, 0.92 vs 0.91 vs 0.89, with
    English minus Hindi +0.015 [−0.008, +0.038] (at L13 +0.016 [0.001, 0.030] just excludes 0; over
    Chinese +0.030 [0.002, 0.067] at L15).
- **Most of English's lead in the ordinary readout comes from the vocabulary.** Ranked only against
  tokens of its own script, the Hindi word is in the top 10 nearly as often as the English word:
  0.92 vs 0.95 with the J-lens (all pairs pooled), and with the logit lens the two are level (Hindi
  0.87 vs 0.82, +0.04 [−0.03, +0.10], concept bootstrap). On the same pool the whole-vocabulary
  figures are 0.57 vs 0.94 (10 original pairs: whole vocabulary 0.58 vs 0.94, own script 0.92 vs
  0.94). A modest English lead survives at size-scaled thresholds (top 0.1% of the script: 0.99 vs
  0.93, +0.06 [+0.01, +0.14]).
- **Qwen is more English-centric than Sarvam.** It doesn't privilege Chinese, its main
  other training language. With the line carrier, English is its best retrieval hub: 0.73, vs Chinese
  0.65 and Hindi 0.59, CIs excluding 0. On the family-weighted other-script centroid English is also
  closest, ahead of Chinese by +0.04 to +0.08 (CIs excluding 0; corrected 2026-10-20, see §2: an
  earlier version let Japanese into Chinese's centroid and put Chinese first).
- **As a retrieval hub, English minus Hindi** is +0.010 to +0.018 on Sarvam-30B (CIs exclude 0 at
  L13 only) and +0.10 to +0.16 on Qwen.

This fits the causal results. Removing the English word's directions costs translations 5.7 points,
and removing the Hindi word's costs 4.4 (doc 04 §4.2). The model uses a shared space in which both
English and Hindi words are well aligned with the concept, rather than routing through an English-only
representation.

## Method

**Vectors.** For each of the 32 concepts, in each of 27 languages with a listed word (en, the 22
scheduled languages, zh, ja, ru, ar; 788 words), the word is placed in two neutral carriers:
- **word:** the word alone after the model's BOS (Sarvam's `[@BOS@]`; Qwen has no BOS, so its document
  separator `<|endoftext|>` keeps one-token words off position 0, the attention sink);
- **line:** the pivot prompt's source line `label: "word` (the language's self-name label, no target,
  no closing quote).

The residual after every block is read at the word's last token. Forwards are batched (16, right-padded,
read at each row's last real token).
- On the tiny models, batched equals lone exactly for the word carrier.
- On the tiny 30B, the line carrier differs by up to 1% relative: batch-shape rounding flips its random
  router's near-tied experts. Individual vectors carry that noise; every measure below is an aggregate.

Vectors are saved (`vectors_<carrier>.pt`, fp16) for re-analysis.

**Measures** (per layer; 95% bootstrap CIs over concepts):
1. **Cross-language retrieval.** Subtract each language's mean vector (removing language identity), then
   unit-normalise. For each ordered language pair (A, B) and concept c in both: is c's nearest neighbour
   among B's concepts c itself? Top-1 accuracy, pooled and per pair; chance is about 1/32.
2. **Centroid proximity.** For each language, the mean cosine of its centred concept vectors to the other
   languages' centroid of the same concept. The plain leave-one-out centroid has two confounds:
   - the sample is mostly Indo-Aryan (15 of 27 languages);
   - Devanagari languages share whole word strings with Hindi (पानी is Hindi, Nepali and Dogri).

   So two controls are added: an **other-script** centroid (languages whose scripts share nothing
   with the language's own; Japanese counts as kana + Han, so Chinese is excluded from its centroid and
   it from Chinese's), and the same **weighted equally by family** (each family's mean, then the mean
   over families).
3. **Hub retrieval.** From every non-hub language, how often is the concept found among English's
   concepts, Hindi's, or Chinese's? Paired differences, bootstrapped over concepts. This needs no centroid.
4. **Vocabulary-fair readout** (Sarvam-30B only). On the pivot prompts (the published 10 Indic pairs plus
   the new-language pairs; kept items; J-lens and logit; English band L13–16), the best rank of each role's
   tokens is taken twice:
   - over the whole readout-filtered vocabulary;
   - **within the role language's own script** (`languages.SCRIPTS`: Latin 154,841 tokens, Devanagari
     13,638, a Dravidian script ~4,200). The Latin count includes 6,322 special and `<unusedN>`
     placeholder tokens whose decoded text is Latin letters; without them it is 148,467, the count
     doc 12's script map and this doc's introduction use. The difference only shifts the size-scaled
     cutoffs slightly (top 0.1%: rank 154 instead of 148).

   Reported as top-10 rates, and at thresholds scaled to the script's size (top 0.1% / 1%). The
   same-language **shuffled-concept** controls carry the same script and a similar token-set size, so
   they absorb the advantage a larger token set gives.

   This readout is not run on Qwen. Its byte-level tokenizer leaves most Indic tokens as partial UTF-8
   with no script.

Bands: Sarvam-30B L13–16 (its English band). Qwen L27–32, inside its J-lens English plateau of L24–36
(`results/qwen_j`: English top-10 0.92–0.96).

## Results

### 1. A shared concept space

Pooled cross-language retrieval, top-1 (chance 0.034):

| model, carrier | L0 | early | middle | English band | last |
|---|---|---|---|---|---|
| Sarvam-30B, line | 0.24 | L4 0.83 | L8 0.84 | **L13–16 0.83–0.84** | L18 0.59 |
| Sarvam-30B, word | 0.22 | L4 0.47 | L8 0.47 | L13–16 0.45–0.48 | L18 0.16 |
| Qwen, line | | L8 0.67 | L16 0.60 | **L27–32 0.45–0.51** | L39 0.35 |
| Qwen, word | | L8 0.58 | L16 0.40 | L27–32 0.21–0.28 | L39 0.20 |

Selected pairs, band mean (line carrier, then word carrier):

| pair | Sarvam-30B | Qwen |
|---|---|---|
| English→Hindi, Hindi→English | 1.00, 1.00 (0.75, 0.76) | 0.92, 0.92 (0.44, 0.33) |
| Hindi→Urdu (one language, two scripts) | 1.00 (0.73) | 0.86 (0.59) |
| Tamil→Telugu | 0.93 (0.67) | 0.28 (0.23) |
| Manipuri→Bengali (Meitei Mayek) | 0.70 (0.15) | 0.05 (0.00) |
| Santali→Hindi (Ol Chiki) | 0.76 (0.45) | 0.04 (0.04) |
| Kashmiri→Tamil | 0.59 (0.17) | 0.11 (0.03) |
| Japanese→Chinese | 0.90 (0.86) | 0.90 (0.77) |
| English→Chinese | 0.98 (0.99) | 0.99 (0.59) |

Sarvam's concepts are shared across almost all of its languages, rare scripts included. Qwen shares
English, Chinese, Japanese, Russian and Hindi/Urdu, but barely the Dravidian and rare-script languages.

### 2. Centroid proximity: English's rank depends on who defines the centre

English's rank among 27 languages by mean centred cosine to the cross-language concept centroid
(band mean), and English minus Hindi per band layer [95% CI]:

| model, carrier | centroid | English rank | closest languages | English − Hindi |
|---|---|---|---|---|
| Sarvam, line | all others | 8 | pa, gu, hi, mr, kn, or | −0.04, −0.05, −0.03, −0.04 (L13 [−0.074, −0.000], L14 [−0.084, −0.013]) |
| Sarvam, line | other-script only | 6 | pa, gu, kn, hi, or, **en** | −0.01, −0.02, +0.00, −0.01 (CIs include 0) |
| Sarvam, line | other-script, families equal | **1** | **en**, kn, te, pa, or, gu | +0.07, +0.10, +0.11, +0.12 (L13 [0.037, 0.106]) |
| Sarvam, word | all others | 9 | hi, gu, or, pa, mr, kn | −0.09, −0.10, −0.08, −0.09 (CIs exclude 0) |
| Sarvam, word | other-script only | 6 | gu, or, pa, kn, hi, **en** | −0.02, −0.02, −0.00, −0.01 (CIs include 0) |
| Sarvam, word | other-script, families equal | **1** | **en**, kn, te, zh, gu, ml | +0.06, +0.07, +0.08, +0.08 |
| Qwen, line | all others | 7 | gu, hi, pa, bn, mr, ur | −0.04 to −0.07 |
| Qwen, line | other-script only | 5 | gu, pa, hi, bn, **en** | −0.01 to −0.04 (CIs include 0) |
| Qwen, line | other-script, families equal | **1** | **en**, zh, ru, ja, gu, hi | +0.10 to +0.13 |
| Qwen, word | all others | 16 | mr, hi, gu, ne, bn, doi | −0.18 to −0.22 |
| Qwen, word | other-script only | 11 | gu, pa, hi, bn, or, mr | −0.08 to −0.10 (CIs exclude 0) |
| Qwen, word | other-script, families equal | **1** | **en**, gu, zh, ru, bn, mr | +0.03 to +0.07 |

**Correction (2026-10-20).** Before this date the other-script controls compared script labels as
strings, so Japanese ("kana+han") and Chinese ("han") counted as different scripts and stayed in each
other's centroid, although 31 of the 32 Japanese words are pure Han and 13 are identical to the Chinese
word (水, 火, 山, …). Script membership is now a set, and two languages are other-script only when their
sets are disjoint. That lowers Chinese and Japanese only; English's and Hindi's values, the
English − Hindi column, the "all others" rows and §3 are unchanged. The first place changed in three
families-equal rows: Qwen's line carrier (Chinese 0.512 → 0.444, English 0.505, so English first
instead of Chinese), Sarvam-30B's bare words (Chinese 0.512 → 0.404; English first instead of second)
and Qwen's bare words (Chinese 0.352 → 0.201, Japanese 0.317 → 0.165; English first instead of
third). In two other-script-only rows English moved up one place, below the same leaders (Sarvam bare
words 7 → 6, Qwen bare words 12 → 11). `geometry_controls_v2.json` has
the corrected values; `geometry_controls.json` is the superseded version, kept for comparison.

- The plain centroid is mostly Indo-Aryan, and **Hindi and its neighbours are closest to it**.
- **Removing shared-script languages makes English and Hindi tie** in the 30B and Qwen.
- **Weighting families equally puts English first on every model and carrier.** This weighting gives
  each single world language (English, Chinese, Japanese, Russian, Arabic) as much weight as all 15
  Indo-Aryan languages, so it favours the high-resource world languages.
- **English − Chinese with families weighted equally** (band range; CIs over concepts): Sarvam-30B line
  +0.12 to +0.13, bare words +0.05 (CIs exclude 0 at every band layer); Qwen line **+0.04 to +0.08** (all six exclude 0; L27 [0.021, 0.065]), Qwen bare words
  +0.02 to +0.04 (3 of 6 exclude 0). Before the fix Chinese led English for Qwen and for Sarvam's bare
  words.
- So no definition of the centre gives English a large lead over Hindi. English comes first only when
  four other world languages are weighted like whole families, and then on both models.

### 3. Hub retrieval: English, Hindi and Chinese as targets

Mean over all non-hub source languages of top-1 retrieval into the hub's 32 concepts [95% CI]:

| model, carrier | layer | English | Hindi | Chinese | English − Hindi | English − Chinese |
|---|---|---|---|---|---|---|
| Sarvam, line | L13 | 0.922 | 0.905 | 0.904 | +0.016 [0.001, 0.030] | +0.018 [−0.004, 0.043] |
| Sarvam, line | L15 | 0.925 | 0.910 | 0.894 | +0.015 [−0.008, 0.038] | +0.030 [0.002, 0.067] |
| Sarvam, word | L14 | 0.625 | 0.584 | 0.638 | +0.041 [−0.017, 0.103] | −0.013 [−0.056, 0.028] |
| Sarvam, word | L16 | 0.606 | 0.563 | 0.632 | +0.043 [−0.010, 0.097] | −0.026 [−0.070, 0.016] |
| Qwen, line | L29 | 0.731 | 0.593 | 0.647 | **+0.139** [0.088, 0.193] | **+0.084** [0.032, 0.143] |
| Qwen, line | L30 | 0.751 | 0.619 | 0.647 | **+0.132** [0.082, 0.185] | **+0.104** [0.063, 0.154] |
| Qwen, word | L29 | 0.292 | 0.379 | 0.277 | −0.087 [−0.144, −0.037] | +0.014 [−0.035, 0.060] |

In Sarvam-30B, the three hubs are within 0.02–0.07 of each other, and the CIs mostly include 0.
Qwen with the line carrier has a clear English hub, ahead of Chinese by +0.07 to +0.10 and of
Hindi by +0.10 to +0.16 over its band (against Hindi, an order of magnitude more than Sarvam's
edge), while its bare words favour Hindi.

### 4. Vocabulary-fair readout (Sarvam-30B, L13–16)

Top-10 rate of each role's word, over the whole vocabulary vs within the role language's own script:

| lens, pairs | English | English, other concept | Hindi | Hindi, other concept | target | source |
|---|---|---|---|---|---|---|
| J-lens, 10 original pairs: whole vocabulary | 0.94 | 0.00 | 0.58 | 0.00 | 0.27 | 0.22 |
| J-lens, 10 original pairs: own script | 0.94 | 0.00 | **0.92** | 0.04 | **0.85** | **0.85** |
| J-lens, new-language pairs: whole vocabulary | 0.95 | 0.00 | 0.58 | 0.00 | 0.20 | 0.10 |
| J-lens, new-language pairs: own script | 0.95 | 0.00 | **0.92** | 0.02 | 0.76 | 0.67 |
| logit, 10 original pairs: whole vocabulary | 0.79 | 0.00 | 0.38 | 0.00 | 0.29 | 0.23 |
| logit, 10 original pairs: own script | 0.80 | 0.00 | **0.86** | 0.01 | 0.78 | 0.79 |

At thresholds scaled to each script's size (all pairs pooled, J-lens; logit in brackets):

| threshold within own script | English | English, other concept | Hindi | Hindi, other concept | target | source |
|---|---|---|---|---|---|---|
| top 10 | 0.95 (0.82) | 0.00 | 0.92 (0.87) | 0.02 | 0.78 (0.73) | 0.74 (0.68) |
| top 0.1% | 0.99 (0.96) | 0.05 | 0.93 (0.88) | 0.03 | 0.73 (0.68) | 0.70 (0.64) |
| top 1% | 1.00 (0.99) | 0.22 | 0.97 (0.95) | 0.20 | 0.88 (0.84) | 0.83 (0.79) |
| median rank / script size | 0.00001 | 0.047 | 0.00007 | 0.063 | 0.00036 | 0.00036 |

- Over the whole vocabulary, the English word dominates mid-network: 0.94 against Hindi 0.58 and
  the target 0.27.
- **Within each script, most of that gap disappears.** The Hindi word is in its script's top 10 for
  0.92 of items, and the target and source words for 0.74–0.85. At these layers the concept is written
  into many languages' words at once.
- English stays a little ahead at size-scaled thresholds (top 0.1%: 0.99 vs 0.93), but not on the plain
  own-script top 10 with the logit lens (0.82 vs 0.87; Hindi − English +0.04 [−0.03, +0.10], level).
- The shuffled controls stay at 0.00–0.05 at the top-10 and top-0.1% thresholds; at the top 1% they
  reach 0.20–0.22, so that threshold is loose.
- **Extended to the languages outside the prompt (doc 17 #1, 2026-10-20).** The same own-script
  readout, now for every language of the word list that is neither the source, the target nor Hindi:
  for a ta→te item, the Kannada, Bengali, Gujarati, Marathi and Malayalam words are in their script's
  top 10 for 0.74 [0.66, 0.82] of items (30B J-lens), against 0.12 over the whole
  vocabulary and 0.03 for their shuffled controls. Big-10 Indian languages read 0.73, low-resource ones
  0.33, Russian 0.89, Japanese 0.64. So "the concept is written into many languages' words at once"
  holds beyond the prompt's own languages: the middle layers carry the word in every language the model
  knows well, and English and Hindi stand out only when all scripts compete in one ranking.

## Caveats

- **Word lists.** One spelling per word, from Wiktionary and Wikidata, not checked by native speakers
  (docs/11). Dogri has 6 words, Maithili 10 and Bodo 13; they enter the pooled measures with fewer concepts.
- **Token counts.** English words are one token each (Hindi 1.91 on average, Manipuri 1.94, Santali 2.66,
  Japanese 1.31). A single-token word's last-token vector may be a cleaner concept vector, which would
  favour English, yet English is not favoured in Sarvam.
- **The centroid depends on the language sample,** as §2 shows. Hub retrieval is the centroid-free test.
- **The fair readout** compares percentiles across scripts of very different sizes. No single threshold
  is "fair", so several are shown, all beside same-script shuffled controls. It is a lens readout (J-lens
  and logit), not geometry.
- **Batching.** Individual vectors carry batch-shape noise (methods); the measures are aggregates.

## Cost

| run | hardware | minutes | ≈ $ | what |
|---|---|---|---|---|
| `xl_geom_smoke` | CPU 4c/16G | ~4 | 0.03 (estimate) | smoke test + dry run of load, `newlang_experiment.geometry` |
| `xl_geom_30b` | 1× H200 | 2.82 (load 0.62, stage 2.02) | 0.24 | vectors in 2 carriers (788 words × 19 layers), metrics, fair readout on 1,481 pivot items |
| `xl_geom_qwen` | 1× H200 | 1.92 (load 0.79, stage 0.96) | 0.17 | the same vectors and metrics on Qwen (no fair readout) |
| **total** | | | **≈ 0.44** | (the 0.98 in `docs/05_costs.md` also covers a 105B launch) |

## Reproduce

From `code/` (Git Bash: prefix the `modal` commands with `MSYS_NO_PATHCONV=1`):

```bash
modal run modal_app.py --step smoke --stages load,newlang_experiment.geometry
modal run modal_app.py --step pilot --model sarvam --stages load,newlang_experiment.geometry --budget-minutes 6.5 --timeout-margin 2 --spf-guess 0.12 --tag xl_geom_30b
modal run modal_app.py --step pilot --model qwen --stages load,newlang_experiment.geometry --budget-minutes 6.0 --timeout-margin 2 --spf-guess 0.12 --tag xl_geom_qwen
modal volume get sarvam-lens runs/xl_geom_30b ../results/
python concept_geometry_controls.py ../results/xl_geom_30b/newlang_experiment_geometry --band 13-16
python concept_geometry_controls.py ../results/xl_geom_qwen/newlang_experiment_geometry --band 27-32
```

Per-layer metrics for every layer are in `geometry_<carrier>.json` (retrieval with CIs, the full 27×27
retrieval matrix, and centroid proximity per language). The controls are in `geometry_controls_v2.json`
(set-based scripts, 2026-10-20; `geometry_controls.json` is the superseded string-label version).
`python concept_geometry_controls.py --self-check` checks that no two languages sharing a script
partner in the other-script centroids.
The fair readout is in `fair_rows.csv` and `fair_summary.csv`.
