# 14. bf16 unembedding in the lens readouts: does it change the published pivot?

*Run 2026-10-19. It answers the open item in `docs/07` ("Check bf16 quantization in the lens readouts").
Code: `code/quant_check.py` (stage `quant_check.pivot`). Run: `results/quant_check/`.*

## Why check

Sarvam-30B's final RMSNorm has outlier gains (docs/04 §4.5), so its logits sit in the hundreds.
At that size bf16 only resolves whole units: a bf16 number has 8 significant bits, so values in
[128, 256) are 1.0 apart. The doc 13 judge found the problem and switched to `Fp32Head`, which
re-applies the final norm and LM head in fp32. The shared readouts never switched:

- jlens `unembed` (`jpp src/jlens/hf.py`) casts the transported residual to the head's dtype, then runs
  the final norm and `lm_head` in bf16;
- `pivot_experiment.score` casts the logits to fp32 only after that, and ranks the best token by
  (#strictly greater logits) + 1, so a tied token gets the better rank;
- `reader_questions.Reader.score` does the same.

Rounding P values could move P(English), and ties could inflate top-10 rates. The 0% shuffled
controls already rule out a spurious signal, but nobody had measured the effect on the
published numbers.

## Method

The published pivot is rebuilt exactly as `results/final/pivot_all` built it:

- **Items and prompts.** `pivot_experiment.DEFAULT_PAIRS` (14 pairs) on `data/concepts.tsv` (32
  concepts), 5-shot prompts, the correctness filter on.
- **Token sets.** The same sets, roles, readout filter and shuffled-concept controls, from
  `newlang_experiment.build_items`, which mirrors `pivot_experiment.main`.
- **Model and lenses.** Sarvam-30B in bf16, one lone forward per item, as `pivot_experiment.main`
  ran it. Lenses `j_200p` (the published J-lens) and `logit`, at layers 0–17.

For every item × lens × layer × role, the same transported residual `h = lens.transport(resid)` (fp32)
is decoded twice:

- **(a) bf16, as published.** `model.unembed(h).float()`. The rank is recorded twice: the published
  optimistic rank, and a pessimistic one that counts every token outside the set whose logit ties
  the set's best logit as ranking ahead of it.
- **(b) fp32.** `refusal_experiment.Fp32Head`: the final norm deep-copied to fp32 and the `lm_head`
  weight cast to fp32, applied to the fp32 `h`. The ranks use the same two rules.

Both decodings use the same readout mask, the same token sets and the same softmax. The correctness
filter keeps the published (bf16) decision. Each item also records the fp32 head's top token, to show
whether the filter itself would change.

**Statistics.** ALL_INDIC (the 10 Indic→Indic pairs) and each of the 14 pairs. Each is reported per
layer and over the band L13–16 (each item's mean over the band).

- Top-10 rate and mean P under (a), (a)-pessimistic and (b).
- fp32 − bf16 with a paired bootstrap 95% CI: 2,000 resamples of items, the same resample for both
  heads.
- The share of rows whose rank changes, whose top-10 call flips, and whose bf16 rank depends on the
  tie rule (anywhere, or at the top-10 boundary).
- Logit magnitudes: the top logit, the bf16 spacing at that size, and |bf16 − fp32| over the vocabulary.

The verdict rule follows the brief. A docs/04 §4.1 headline number "moves" if fp32 changes it by more
than 2 points of top-10, or by more than 10% of P relative (for P below 0.01, more than 0.1 point).

**Tests.**
- **Tiny CPU model, in bf16.** (a) reproduces `pivot_experiment.main` bit for bit: 4,224 of 4,224
  rows match, |ΔP| = 0, 0 rank differences. (b) differs only by rounding: max |Δlogit| is 0.0145,
  one bf16 step at logits of about 2.9.
- **Modal smoke**, in the image (`logs/modal/quant_smoke.log`): 48 smoke checks pass and the stage
  runs. The dry run had only the logit lens: the tiny model has 4 layers, and `own_j_lens()` needs 19.
  The J-lens path is covered by the tiny check above.

## Verdict

**The bf16 readouts are confirmed.** None of the 92 headline numbers checked from docs/04 §4.1 moves
by more than 2 points of top-10 or 10% relative P. The 92 are the J-lens and logit-lens top-10
tables, peak P(English) and P(Hindi), the controls' peak, and each pair's peak P(English).

- **Largest top-10 move:** −1.57 points, J-lens target at L16 (42.2% → 40.6%).
- **Largest relative P move among headline numbers:** 0.35%, bn→ta peak P(English) (0.8307 → 0.8279).
- **Direction:** fp32 lowers top-10 rates by 0–1.6 points, because bf16 ties were counted in the
  set's favour. P is unbiased: the mean signed change per row is +0.15% under the J-lens and +0.02%
  under the logit lens (rows with P ≥ 0.01; over all rows +0.52% and +0.02%).

## Results (`results/quant_check/quant_check_pivot`, Sarvam-30B, 382 items)

### Cross-check: (a) reproduces the published run

- **Rows.** (a) matches all 92,880 j_200p/logit rows of `results/final/pivot_all/rows.csv`:
  - mean |ΔP| 5×10⁻⁷, max 1.1×10⁻⁴;
  - 12 rank differences, 0 top-10 differences;
  - the same 382 kept items.
- **Run-to-run noise.** The residual differences are of order 10⁻⁵ in P, against a bf16 effect of
  order 10⁻² per row. They are run-to-run numerical noise between containers.
- **The extra 13,752 rows** are `chinese_shuffled`, a control added after the published run.
- **Published summary.** Every published `summary.csv` value checked equals (a) to four digits.

### docs/04 §4.1 table (J-lens j_200p, ALL_INDIC, top-10 rate)

Each cell gives published = bf16 (a) / bf16 pessimistic / **fp32 (b)**:

| role | L11 | L13 | L14 | L15 | L16 | L17 |
|---|---|---|---|---|---|---|
| English | 19.9 / 18.0 / **19.1** | 79.7 / 78.1 / **78.9** | 98.0 / 98.0 / **98.0** | 98.0 / 98.0 / **98.0** | 99.2 / 99.2 / **99.2** | 86.3 / 84.8 / **85.5** |
| English control | 0 / 0 / **0** | 0 / 0 / **0** | 0.4 / 0 / **0** | 0 / 0 / **0** | 0.4 / 0.4 / **0.4** | 0 / 0 / **0** |
| Hindi | 0.4 / 0.4 / **0.4** | 42.2 / 41.4 / **41.8** | 60.2 / 58.6 / **59.0** | 55.1 / 54.3 / **54.3** | 67.6 / 64.8 / **66.4** | 61.7 / 60.5 / **60.5** |
| Hindi control | 0 / 0 / **0** | 0 / 0 / **0** | 0 / 0 / **0** | 0 / 0 / **0** | 0 / 0 / **0** | 0 / 0 / **0** |
| Target | 0 / 0 / **0** | 11.3 / 10.5 / **10.9** | 21.9 / 20.7 / **21.5** | 27.0 / 25.4 / **26.2** | 42.2 / 39.5 / **40.6** | 84.4 / 83.2 / **84.0** |
| Chinese | 1.2 / 0.4 / **0.8** | 9.0 / 8.2 / **8.6** | 4.7 / 4.7 / **4.7** | 3.9 / 3.5 / **3.9** | 5.9 / 5.9 / **5.9** | 10.5 / 10.2 / **10.2** |

- **Paired CIs (fp32 − bf16, points):**
  - target L16: −1.56 [−3.12, −0.39];
  - Hindi L14, L16, L17: −1.17 [−2.73, 0.00] each;
  - English L13: −0.78 [−1.95, 0.00];
  - English L14–16: 0 [0, 0].
- **Logit lens, the same cells:** at most −0.39 points (English L14 92.6 → 92.2, L17 70.3 → 69.9).
  Both controls stay at 0 at every layer.

| Headline P | published = bf16 (a) | fp32 (b) | change |
|---|---|---|---|
| peak P(English), J-lens (L16) | 0.7404 | 0.7401 | −0.03% |
| peak P(English), logit lens (L16) | 0.4498 | 0.4501 | +0.06% |
| peak P(English) per pair, J-lens (12 pairs with an English role) | 0.633 (bn→gu) – 0.849 (hi_latn→ta) | 0.634 – 0.850 | −0.35% … +0.18% |
| peak P(Hindi), J-lens, band L13–16 / any layer | 0.0249 (L16) / 0.0667 (L17) | 0.0248 / 0.0667 | −0.6% / −0.02% |
| peak P(Hindi), logit lens | 0.0354 (L16) | 0.0354 | −0.01% |

The brief's "Hindi 0.55–0.83" is the L14–16 Hindi top-10 range across the three fitted lenses (J,
R, J++; tables §C). The logit lens reads 0.41–0.57 there (fp32: 0.41 / 0.49 / 0.57), so the range over
all four lenses is 0.41–0.83.
- The J-lens end, 0.55–0.68, becomes 0.54–0.66 (above).
- The 0.83 is R-25 at L16. The R and J++ lenses were not in this check (see Not covered).

### Band L13–16, ALL_INDIC (each item's band mean; paired bootstrap 95% CI over items)

| lens | role | top-10 bf16 | bf16 pess. | fp32 | fp32 − bf16, pts | P bf16 | P fp32 | rel. ΔP | top-10 calls flipped | tied at top-10 boundary |
|---|---|---|---|---|---|---|---|---|---|---|
| J | target | 0.256 | 0.240 | 0.248 | −0.78 [−1.37, −0.29] | 0.0109 | 0.0110 | +0.8% | 0.78% | 1.56% |
| J | English | 0.938 | 0.934 | 0.935 | −0.20 [−0.49, 0.00] | 0.5256 | 0.5257 | +0.01% | 0.20% | 0.39% |
| J | Hindi | 0.562 | 0.548 | 0.554 | −0.88 [−1.56, −0.29] | 0.0181 | 0.0180 | −0.2% | 0.88% | 1.46% |
| J | English control | 0.002 | 0.001 | 0.001 | −0.10 [−0.29, 0.00] | 3.2e−5 | 3.2e−5 | +0.1% | 0.10% | 0.10% |
| J | Hindi control | 0 | 0 | 0 | 0 | 1.5e−5 | 1.5e−5 | −0.7% | 0 | 0 |
| logit | target | 0.287 | 0.282 | 0.285 | −0.20 [−0.49, 0.00] | 0.0304 | 0.0304 | +0.08% | 0.20% | 0.49% |
| logit | English | 0.783 | 0.782 | 0.782 | −0.10 [−0.39, 0.00] | 0.3186 | 0.3188 | +0.05% | 0.10% | 0.10% |
| logit | Hindi | 0.381 | 0.377 | 0.379 | −0.20 [−0.59, +0.20] | 0.0169 | 0.0169 | +0.02% | 0.39% | 0.39% |
| logit | English control | 0 | 0 | 0 | 0 | 1.2e−5 | 1.2e−5 | +0.11% | 0 | 0 |
| logit | Hindi control | 0 | 0 | 0 | 0 | 3e−6 | 3e−6 | −0.07% | 0 | 0 |

- No P difference in this table is significant: every P CI includes 0.
- **Ranks change but rarely matter.** Many rows change rank under fp32: 49% for target, 28% for
  Hindi and 7% for English under the J-lens. Almost all of these changes are deep in the ranking, among the
  many tied tail tokens.
- **Every ALL_INDIC cell** (per layer or band, all roles): at most 1.3% relative change in P where
  P ≥ 0.01.

### Where bf16 does matter: ties, and the J-lens at layers 6–9

Pooled over all 105,696 rows that have a token set (both lenses, every layer and role):

- **Top-10 calls.** 0.11% flip under fp32. Of those 120 flips, 119 go from in the top 10 (bf16) to
  out (fp32): the optimistic tie rule.
- **Tie rule.** The bf16 top-10 call depends on the tie rule for 0.20% of rows.
- **Ranks.** 89% of bf16 ranks involve a tie somewhere; in fp32, 3%. The fp32 rank lies between the
  bf16 optimistic and pessimistic ranks in 85% of rows. In the rest, rounding moved the logit by more
  than a tie.
- **P per row.** For rows with P ≥ 0.01 (n = 6,049), the relative change has median 0.8%, 95th
  percentile 6.7% and maximum 52%. It is zero-mean, so it cancels in every average reported.

**Early layers.** The one sizeable aggregate move is outside §4.1's table: **J-lens English at L8,
11.3% → 6.6%** (−4.7 points [−7.4, −2.3]; the pessimistic bf16 rate is 4.7%).
- There, the J-lens readouts have the largest logits: median top logit 90–108 at L6–8, where the
  bf16 spacing is 0.5.
- 83–93% of those readouts have a tie at the 10th-largest logit.
- Part of the small pre-L11 English bump is therefore tie artefact: about 40% of it at L8 (11.3% →
  6.6%) and about 12% at L9 (9.4% → 8.2%). Table §C starts at L9, so the L8 value is in
  `quant_summary.csv`, not there.
- This only strengthens §4.1's claim that nothing is readable before layer 11.

### Logit magnitudes: the lens readouts are not in the hundreds

| | J-lens, L13–16 | logit lens, L13–16 | all readouts |
|---|---|---|---|
| median top logit (bf16) | −9.8 … 50.1 by layer | 8.9 … 12.6 | 11.2 (p99 121.5, max 161) |
| bf16 spacing at that size | 0.06–0.25 | 0.06 | 0.5% of readouts have \|top logit\| ≥ 128 |
| \|bf16 − fp32\| over the vocab, per readout: median of max / overall max | 0.14 / 0.54 | 0.07 / 0.18 | |
| top-1 token agrees | 98.0% | 99.7% | |
| 10th-largest logit tied with another token | 40% | 15% | |

- **Where the hundreds are.** Doc 13's "logits in the hundreds" describes the model's own final
  output on chat prompts. The intermediate readouts here sit an order of magnitude lower, where bf16
  resolves about 0.06–0.25.
- **Why the error exceeds half a step.** |Δlogit| reaches about two bf16 steps because the bf16 path
  also rounds the residual and the norm output before the matmul.
- **Effect.** Ties at the boundary are common (15–40% of readouts), but they decide a set's top-10
  call only when the set's best token is the tied one (0.2% of rows).

### The correctness filter

- **Disagreements.** For 6 of 448 items, the fp32 head's top token at the final layer differs from
  bf16's. That is the final-layer argmax, where the logits are larger.
- **Effect on the filter.** Under fp32 the filter would drop 2 kept items (gu→mr village, bn→ta house)
  and keep 3 dropped ones (ml→ta dog, te→gu river, en→ta king): 383 items instead of 382.
- **Bound on the effect.** In ALL_INDIC, two items swap and n stays 256, so each top-10 rate could
  move by at most 2/256 = 0.8 points. The swapped-in items were never scored. This is a bound, not a
  measurement.

### Per pair

- **Peak P(English).** It moves by at most 0.35% in every pair (table above).
- **Top-10 cells.** Per-pair cells are coarse: one item at one layer is 3–8 points. 80 of 2,546
  per-pair cells (5 roles, every layer and the band) move by more than 2 points. All are one to
  three flips, and none is a §4.1 number.
- **The largest band moves** (J-lens): te→gu Hindi −4.2 points (n = 18), mr→bn target −3.6
  (n = 28), ml→ta Hindi −2.7 (n = 28). Under the logit lens no band cell moves by more than about 1 point (largest: bn→ta Hindi −1.04,
  n = 24).
- **The only cell with P ≥ 0.01 moving by more than 10%** is te→gu target at L15: 0.0202 → 0.0223.
- **Full tables:** `results/quant_check/quant_check_pivot/report.md`, generated by
  `python quant_check.py analyze`.

## Not covered

- **R-25 and J++-25 lenses** (the brief named j_200p and logit): §4.1's "all four lenses agree", the
  R-lens P(English) 0.75 and the R-25 Hindi 0.83. They share the same bf16 head, so the mechanism is
  the same. Their logit scale was not measured.
- **Other readouts with the same bf16 path:**
  - `reader_questions.Reader.score` and `newlang_experiment.score_layers` (docs 09, 11, 12): chat
    prompts, so possibly larger logits;
  
  The tie mechanism is the same everywhere, and the fix is cheap: decode with `Fp32Head`, or count
  ties pessimistically.

## Cost

- **Smoke:** CPU, 4.5 min, **$0.02** (app `ap-rVpRnjrficqj5kVPZKnFaQ`).
- **30B run:** `quant_check`, `--budget-minutes 4.3 --timeout-margin 2`, worst case $0.54.
  - Driver: 3.22 min, of which 0.65 min load and 2.40 min for the stage; **$0.28**.
  - App wall time: 3.5 min, **$0.30** (app `ap-CHrjPzzlpTjM6b06gt8jk9`).
- **Total ≈ $0.33** of the $0.60 cap.
- **Logs:** `logs/modal/quant_smoke.log` and `logs/modal/quant_check.log`.

## Files

- `code/quant_check.py`: the stage (`run`, `n_forwards`), the analysis, `check` (tiny model) and
  `analyze` (tables, verdict, cross-check).
- `code/modal_pilot.py`: `quant_check` added to `XL_MODULES`. This is the only shared-file change.
- `results/quant_check/quant_check_pivot/`:
  - `rows.csv`: item × lens × layer × role, both heads, both rank rules;
  - `logits.csv`: per readout;
  - `items.csv`: with the fp32 top token;
  - `summary_bf16.csv` and `summary_fp32.csv`: in `pivot_experiment.summarize` format;
  - `quant_summary.csv`, `verdict.json` and `report.md`;
  - `report.json`: the pilot driver's report.

Reproduce the analysis locally:

```
python quant_check.py analyze ../results/quant_check/quant_check_pivot --md ../results/quant_check/quant_check_pivot/report.md
```
