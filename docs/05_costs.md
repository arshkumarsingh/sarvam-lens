# 5. Costs

Modal list prices on 2026-10-18: H200 $4.54/h, CPU $0.0472/core·h, memory $0.0080/GiB·h.
The GPU containers used 8 cores + 32 GiB, so **≈ $5.17/h per H200**. Costs below are estimates
from container minutes. Driver minutes come from each run's `report.json`, plus ~1–2 min of startup.
**The Modal dashboard has the exact bill.**

| # | Run (Modal step) | Hardware | Minutes | ≈ $ | Purpose |
|---|---|---|---|---|---|
| 1 | `prepare` | CPU 4c/32G | 8.3 | 0.07 | fp32 → bf16 checkpoint on the Volume |
| 2 | pilot attempt 1 | H200 | ~1 | 0.08 | failed at import (path order) |
| 3 | `smoke` (in-image) | CPU 4c/16G | ~7 | 0.04 | smoke test + dry pilot in the real image |
| 4 | **pilot** | H200 | 9.1 | 0.95 | timing, checks; found the broken pivot |
| 5 | **bosfix** | H200 | 3.6 | 0.44 | BOS hypothesis (wrong) |
| 6 | **ropefix** | H200 | 3.6 | 0.44 | after the RoPE fix: BOS probe, first real result |
| 7 | `prepare-qwen` | CPU 2c/8G | 2.8 | 0.01 | Qwen checkpoint onto the Volume |
| 8 | **qwen** | H200 | 4.9 | 0.52 | Qwen logit-lens baseline |
| 9 | **lensfit** | H200 | 43.0 | 3.86 | router, R-lens 25p, J++ weights, 3-lens pivot |
| 10 | **qwen_zh** | H200 | 4.3 | 0.47 | Qwen with Chinese tracked |
| 11 | `cka` | CPU 8c/32G | ~12 | 0.15 | CKA explorer reproduction (R, J++) |
| 12 | `diagnose` ×2 | CPU 8c/32G | ~10 each | 0.20 | degeneracy diagnostics (R) |
| 13 | **rows** | H200 | 7.1 | 0.72 | rows-per-pass timing probe |
| 14 | **jfit_s0–s4** | 5× H200 parallel | 40–43 each | **18.80** | 200-prompt J-lens |
| 15 | `merge-j` | CPU 8c/64G | ~6 | 0.09 | merge 5 shards |
| 16 | `cka` (J-lens) | CPU | ~12 | 0.15 | CKA with the J-lens |
| 17 | `diagnose` v3 | CPU | ~12 | 0.12 | J-lens diagnostics |
| 18 | **final** | H200 | 15.0 | 1.40 | 4-lens pivot + controls + causal ablation |
| 19 | `cka-robust` | CPU | ~10 | 0.10 | gain variants, `1+w` check |
| 20 | **extra** | H200 | 12.2 | 1.15 | swap, Hindi ablation, sentences |
| 21 | `smoke` (in-image, + tiny-model checks) | CPU 4c/16G | 6.5 | 0.03 | 30B smoke + dry pilot + a tiny-model smoke |
| 26 | `smoke` (in-image, + patching checks) | CPU 4c/16G | 16.9 | 0.09 | smoke test (42 checks) + dry pilot incl. `patch` + tiny-MLA smoke |
| 27 | **patch** | H200 | 6.1 | 0.53 | activation patching, 230 items × 7 conditions (driver 5.8 min) |
| 28 | `smoke` (in-image, + split checks) | CPU 4c/16G | 25.3 | 0.13 | hit its 25-min timeout in the dry pilot's last stages; the smoke test itself passed |
| 29 | **patch2** | H200 | 4.4 | 0.38 | the split: 230 items × 5 conditions (driver 4.1 min) |
| 30 | `smoke --stages load,sanity,patch_rank` | CPU 4c/16G | 4.4 | 0.02 | smoke test (48 checks) + a dry pilot of only this run's stages |
| 31 | **patch3** | H200 | 4.5 | 0.38 | equal rank across languages: 230 items × 4 conditions (driver 4.2 min) |
| 34 | `smoke --stages ...rq_*` | CPU 4c/16G | 3.5 | 0.02 | 30B dry pilot of the reader-question stages |
| 37 | **rq30** | H200 | 16.2 | 1.40 | 30B: chat ×3, homonyms/kinship, cloze (zh, steering, thinking skipped: budget) |
| 39 | **rq30b** | H200 | 9.3 | 0.80 | 30B: Chinese skill, negative steering, thinking mode |
| 42 | `build-corpus`, **jfit_indic** (3 shards), `merge-j` ×2, **pivot_indic_lens** | CPU; 3× H200; H200 | — | 3.37 | Indic-corpus J-lens (27 prompts) + the comparison pivot. One fit launch ($0.12) failed on a mangled path |
| 43 | **qwen_j** ×2 | H200 | — | 1.03 | Qwen pivot with its published J-lens. The first run lost its lens to Git Bash path conversion ($0.45) |
| 44 | **neuron30** + a 105B launch + 2 smokes | H200; 2× H200 | — | 0.86 | expert routing by language + layer-0 neurons (the cost includes the 105B launch) |
| 46 | `modal_translate` smoke, run (OOM after Hindi), run | L40S | 24.0 | 0.95 | 1,082 prompts × 25 languages (IndicTrans2, NLLB-200) |
| 47 | `batch-probe` ×2 | 2× L4 | ~5 | 0.14 | batched vs lone on the tiny models (the first launch failed at import) |
| 48 | **xl_new_30b** + a 105B launch + smoke | H200; 2× H200 | 10.8 | 1.20 | doc 11: 43-pair pivot over the 22 languages + ja/ru/ar (the cost includes the 105B launch) |
| 49 | **xl_new_causal_30b** + two 105B launches + smoke | H200; 2× H200 | 10.7 | 1.48 | doc 11: English ablation on 6 new-language pairs. One 105B launch was skipped by the budget gate and wasted $0.50: its estimate left out driver start-up |
| 50 | **xl_dom_30a**, **xl_dom_30b** + a 105B launch + smoke | H200; 2× H200 | 23.3 | 2.11 | doc 12: digits, script profile, MGSM (the 30B's MT-MGSM stage ran out of memory; the cost includes the 105B launch) |
| 51 | **xl_dom_30think** + a 105B launch | H200; 2× H200 | 15.4 | 2.07 | doc 12: reasoning script in 25 languages (the cost includes the 105B launch) |
| 55 | `smoke`, **quant_check** | CPU; H200 | ~4 | 0.33 | doc 14: the published 30B pivot read with a bf16 vs an fp32 head |
| 57 | `smoke`, **xl_new_misses_30b** | CPU; H200 | 3.0 | 0.29 | doc 11 §4c: readouts on failed low-resource translations (the user's "$2 question") |
| 59 | `smoke`, **xl_geom_30b**, **xl_geom_qwen** + a 105B launch | CPU; H200; H200; 2× H200 | ~12 | 0.98 | doc 15: concept-space geometry (retrieval, centroids, hubs, vocabulary-fair readout; the cost includes the 105B launch) |
| 60 | hint translation, 2 smokes, **xl_hint_30**, **xl_hint_30_force** | L40S; CPU; H200 | ~25 | 2.14 | doc 12: hint-faithfulness test (20 MGSM problems × 6 languages × 3 conditions, thinking on) + forced-answer probe |
| 61 | `smoke`, **routing_abl30** | CPU; H200 | 5.6 | 0.26 | doc 09 §11: expert routing (L13–18) under the English ablation vs the control |
| 72 | `xlate_think` (`modal_translate.py --step list`), `smoke` (all nine doc-17 stages) | L40S; CPU 4c/16G | 0.4 + 5.2 | 0.05 | doc 17: #7's instructions and opener into hi/ta/bn/te; the in-image dry run |
| 73 | **p1_t1_30b** | H200 | 5.9 | 0.51 | doc 17 Tier 1 on the 30B: every language + neighbours as pivot roles (#1, #2); joint English + Hindi + target ablation (#3) |
| 74 | **p1_t2_30b** | H200 | 6.5 | 0.56 | doc 17 on the 30B: digits ablation (#4), push curve (#5, first control), all-position ablation and writer attribution (#6), Chinese ablation |
| 76 | **p1_think_30b**, **p1_think3k_30b** | H200 | 11.4 + 26.9 | 3.30 | doc 17 #7: MGSM thinking in English vs the prompt's language, sampled to completion (2,048 and the pre-registered 3,072 tokens; ~1,000 traces at 0.22–0.40 s per decode step) |
| 77 | **p1_hint_30b** | H200 | 16.3 | 1.40 | doc 17 #8: the wrong-hint test with 60 problems (1,020 traces), 3,072 tokens, sampling |
| 78 | **p1_push2_30b** + a 105B launch | H200; 2× H200 | 2.6 + 4.2 | 0.91 | doc 17 #5 rerun with the corrected size-matched random control (the cost includes the 105B launch) |
| 79 | **p1_matched_30b** + a 105B launch | H200; 2× H200 | 6.9 + 7.2 | 1.75 | doc 17: every push condition (#3, #4, #6a) beside a random push of exactly its size (the cost includes the 105B launch) |
| | **Total** | | | **≈ $115.8** | every run, including the withheld ones below: the rows listed here come to ≈ $59.2, the withheld runs to ≈ $56.2 |

Row numbers have gaps where a run is withheld from this release. **Withheld (≈ $56.2):** every
launch that ran only Sarvam-105B (≈ $37.8, including its smokes and the one-time 424 GB fp32 →
212 GB bf16 conversion), and the refusal and safety runs on the 30B and Qwen (≈ $18.4, docs 13 and 16,
which will be published with the second post). A 2× H200 container (8 cores, 32 GiB) is
**≈ $9.71/h**. Rows marked "the cost includes the 105B launch" ran both models and can't be split.

Rows 26–27 (activation patching on the 30B) total **≈ $0.62** against a $2 cap for that task.
Rows 28–29 (the follow-up that splits the effect) total **≈ $0.51** against a $1 cap. It ran with
`--timeout-margin 2` (hard timeout 9.6 min, worst case $0.83), so the step could not pass $1 even
after the smoke's timeout. Rows 30–31 (the rank-matched test) total **≈ $0.41** against a $0.90
cap. The stage-limited smoke took 4.4 min, where the full one hit its 25-min timeout. The H200 hard
timeout was 9.5 min (worst case $0.82). Activation patching in all: ≈ $1.54. Minutes are app wall
time from `modal app list`, an upper bound on container time.

Rows 34, 37 and 39 (the reader questions, `docs/09_reader_questions.md`) total **≈ $2.22**. Each
launch's hard cap kept the worst case inside the approval.

Rows 42–44 (doc 07's next steps) total **≈ $5.26**. Parallel sessions ran them: the Indic lens
$3.37, Qwen $1.03, routing and neurons $0.86. Their minutes are spread over several apps each; the
per-session totals come from each session's `modal app list` accounting.
Lesson: Git Bash rewrites `/vol/...` arguments into Windows paths, which cost $0.57 in two failed
launches. `modal_app.py` now undoes the rewrite (`_volume_path`), and `MSYS_NO_PATHCONV=1` avoids it.

Rows 46–51 (docs 11 and 12) were run by parallel sessions under one approval: translation $0.95, the
batching probe $0.14, new languages $2.68, prompt domains $4.18. Minutes are app wall time. Lesson:
the budget gate must count driver start-up (~0.4 min). One launch whose estimate left it out was
skipped after the load, which wasted $0.50.

Rows 72–79 (doc 17, the post-1 follow-ups) total **≈ $8.48** here, inside a $20 hard cap. The decode
runs used continuous batching over a preallocated KV slab with a grouped-query decode step, which
kept ~1,000 traces decoding at 0.22–0.40 s per step. ≈ $2.7 (rows 78–79) went to reruns with the
size-matched random control that the first push run lacked (doc 17, deviation 5). Per-item costs are
in doc 17's Cost table.

Two lessons:
- A control that is "norm-matched" at one layer is not matched after the edit compounds downstream:
  match each layer's realized edit (same forward, `Edit.match_row`).
- Before sizing a decode-heavy run, check whether the attention expands the KV heads (`repeat_kv`):
  at ~1,000 rows × 3,000 tokens Sarvam's stock path would materialize ~50 GB per layer.

## Cost controls used

- Hard `timeout` on every Modal function (budget + 6 min for the GPU pilot, or `--timeout-margin`
  for a tighter cap), and `retries=0`.
- The pilot driver skips any stage whose estimate doesn't fit the remaining budget.
- One-time bf16 conversion on cheap CPU, so GPU containers load 64 GB instead of 129 GB.
- A CPU smoke test in the same image before any GPU run (caught a path bug for $0.04). With
  `--stages`, it dry-runs only the stages the GPU run will use (~4 min instead of 17–25).
- The model loads once per container (a cached loader), so all stages share one load.
- Fits checkpoint every 20 prompts, which allows resuming after preemption.
- Measured fit costs: ~63–77 s/prompt on an H200, ≈ $0.10/prompt.

## Volume storage

All on Modal Volume `sarvam-lens`, inside the free 1 TiB:
- Sarvam bf16 weights: 64 GB.
- Sarvam-105B bf16 weights: 212 GB.
- Qwen weights: 72 GB.
- Lenses: 3 × 1.2 GB pooled + 2 × 5.4 GB experts.
- 5 J-fit shard checkpoints.
