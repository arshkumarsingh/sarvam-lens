# results/: every run's raw outputs

Downloaded from Modal Volume `sarvam-lens` (`runs/`). Runs on Sarvam-105B and the refusal runs (doc 13 onward) aren't part of this release.
statuses, timings, results, cost estimate) and `pilot.log` (driver log). The matching client-side
logs are in `../logs/modal/`.

| Folder | What | Key files |
|---|---|---|
| `pilot/` | First full pilot. **Broken RoPE**: timings valid, readouts invalid | `report.json` (75/72 s/prompt), `pivot_logit/` (2/448 kept) |
| `bosfix/` | BOS-mode probe, still broken RoPE (BOS wasn't the cause) | `report.json` |
| `ropefix/` | **After the RoPE fix**: sanity, BOS probe (`[@BOS@]` 96.3%), first valid logit-lens pivot, grading | `pivot_logit/` |
| `wiki/` | k-means router (8 clusters, layers 0–17) on 128 WikiText prompts, refit after the fix | `router.pt` (19 MB, on Hugging Face with the lenses, not in git), `router_stamp.json` |
| `lensfit/` | R-lens 25p fit + J++ weights + 3-lens pivot | `pivot_lenses/` |
| `qwen/` | Qwen3.6-35B-A3B logit-lens pivot | `pivot_logit/` |
| `qwen_zh/` | Same, with Chinese tracked | `pivot_logit/` |
| `rows/` | Jacobian rows-per-pass timing probe (32 / 64) | `report.json` |
| `jfit_s0` … `jfit_s4/` | The 5 parallel shards of the 200-prompt J-lens | `report.json` |
| `final/` | **4-lens pivot + shuffled controls**; **causal English ablation** (J and R directions) | `pivot_all/`, `causal_j_200p/`, `causal_r_moe_25p/` |
| `extra/` | Concept swap, Hindi ablation, sentence clozes | `swap_j_200p/`, `hindi_ablation_j_200p/`, `sentences/` |
| `cka/` | CKA explorer reproduction (recipe gain) + degeneracy diagnostics | `*_cka.png`, `*_blocks.json`, `cross_*.json`, `diagnose.json` |
| `cka_robust/` | CKA without the gain; published-vs-correct `1+w` gain for Qwen | `sarvam_J200_nogain_cka.png` |
| `correctness.csv` | Sarvam-30B greedy correctness on J++'s readout eval items | used by J++ `fit-weights` |
| `token_split/` | Free re-analyses: token-count split (reader Q1), the GPT-2 J-lens test on the 30B | `token_split.json`, `metric_a_30b.json` |
| `rq30/`, `rq30b/` | Sarvam-30B reader questions: chat word (source / English instruction), chat sentence, cloze, homonyms + kinship, Chinese skill, negative steering, thinking mode | `rq_*/`, `causal_steer_j_200p/` |
| `jfit_indic/`, `pivot_indic_lens/` | J-lens fitted on 27 Indian-language Wikipedia passages, and the pivot under it vs WikiText J-lenses (40, 200 prompts) and logit | `pivot_ext/` |
| `qwen_j/` | Qwen3.6-35B-A3B pivot under its published J-lens vs logit | `pivot_ext/` |
| `routing_abl30/` | Expert routing at L13–18 under the published English ablation vs the other-concept control, same 272 items as `final/causal_j_200p` (doc 09 §11) | `routing_ablation_english/` (`routing_ablation_summary_v2.json`: concept-clustered CIs, 2026-10-20; `routing_ablation_summary.json`: the run's item-bootstrap original; `per_layer.csv`) |
| `neuron30/` | Expert routing by language (bare words + translation prompts) and layer-0 cross-script neurons | `routing/`, `neurons/` (`neurons_summary_v2.json`, `layer0_neurons_v2.csv`: tied activations ranked and won correctly, 2026-10-20; the unsuffixed files are the originals) |
| `xl_new_30b/` | **All 22 scheduled languages + ja/ru/ar** in the word-translation pivot (doc 11): 43 pairs, J-lens + logit, with decoded answers | `newlang_experiment_pivot/`, `report_tables.md` |
| `xl_geom_30b/`, `xl_geom_qwen/` | Concept-space geometry without token decoding: cross-language retrieval, centroid proximity, hub retrieval, vocabulary-fair readout (doc 15) | `newlang_experiment_geometry/` (`geometry_*.json`, `geometry_controls_v2.json` (corrected 2026-10-20; `geometry_controls.json` = superseded v1), `fair_*.csv`; the `vectors_*.pt` files are not in git) |
| `xl_new_misses_30b/` | Readouts on FAILED translations into ks/sat/mni/brx/kok/sd, typed by failure (doc 11 §4c) | `newlang_experiment_misses/` (`misses_tables.md`) |
| `xl_new_causal_30b/` | English ablation + α=3 steering on 6 new-language pairs (doc 11 §4b) | `newlang_experiment_causal/` |
| `xl_dom_30a/`, `xl_dom_30b/` | Prompt domains (doc 12): native-digit arithmetic, the script profile over math/science/technical/everyday prompts in 26 languages, MGSM direct answers | `domain_experiment_{digits,profile,mgsm}/` |
| `xl_hint_30/`, `xl_hint_30_force/` | Hint faithfulness (doc 12): MGSM with a wrong "teacher" hint in 6 languages, thinking on; then every trace forced to an answer | `domain_experiment_hint/`, `domain_experiment_hint_force/` (`hint_summary.json`) |
| `xl_dom_30think/` | Thinking on: the reasoning's script for MGSM in 25 machine-translated languages (doc 12) | `domain_experiment_mgsm_think/` |
| `quant_check/` | The published 30B pivot scored with a bf16 and an fp32 head (doc 14): 0 of 92 headline numbers move | `quant_check_pivot/` (`verdict.json`, `report.md`, `summary_{bf16,fp32}.csv`) |
| **`p1_t1_30b/`** | Doc 17 #1–#3: the published pivot with every other language and two English neighbours as roles (prefix, whole-word and own-script readouts); English + Hindi + target ablated alone and together | `post1_experiment_roles/` (`roles_metrics.npz`, `roles_summary.csv`, `roles_pools.csv`, `roles_curves.csv`), `post1_experiment_joint/` |
| **`p1_t2_30b/`** | Doc 17 #4–#6 on the 30B: the number-word ablation on native-digit sums, the push curve (first control), all-position ablation, writer attribution, a Chinese ablation | `post1_experiment_{digits,push,positions,writers,zh}/` (`writers.npz`, `writers_layers.csv`) |
| **`p1_push2_30b/`** | Doc 17 #5, the rerun with the corrected size-matched random control (supersedes the push stage above) | `post1_experiment_push/` |
| **`p1_matched_30b/`** | Doc 17: every push condition of `joint`, `positions`, `digits` and `zh2` beside a random push of exactly its size | `post1_experiment_{joint,positions,digits,zh2}/` |
| **`p1_think3k_30b/`**, `p1_think_30b/` | Doc 17 #7: MGSM thinking in English vs instructed into the prompt's language, sampled to completion (3,072 tokens; the 2,048-token run is the sensitivity check) | `post1_experiment_think/` (`think_items.csv`, `think_summary.csv`, `think_generations.jsonl`) |
| **`p1_hint_30b/`** | Doc 17 #8: doc 12's hint test with 60 problems, sampled to completion; natural-only summary in `natural/` | `post1_experiment_hint/` (`hint_items.csv`, `hint_summary.csv`) |

Inside a pivot folder:
- `rows.csv`: item × lens × layer × role.
- `items.csv`: kept or skipped, and why.
- `summary.csv`: means with bootstrap CIs.
- `*_mean_prob.png` and `*_top10_rate.png`: one figure per lens.

All tables in `../docs/04_results_tables.md` are generated from these files.

**Re-analyses of 2026-10-20 (outside review).** Files ending in `_v2` beside a run's own outputs are $0 re-analyses of the saved data; the run's own files are kept unchanged. `_v2` CIs resample concepts, prompts or problems with all their rows (`routing_ablation_summary_v2.json`; `profile_by_family_v2.csv`, `profile_length_matched_v2.csv`, `profile_length_regression_v2.csv`, `digits_summary_v2.csv`, `think_summary_v2.csv`; `hint_summary_v2.csv`), `hint_items_v2.csv` re-reads the stored answers with the corrected number parser, `think_summary_v2.csv` also recounts script shares over letters only, and `geometry_controls_v2.json` and `neurons_summary_v2.json` fix the script-overlap control and tied neuron activations. Docs 04, 09, 12 and 15 cite the `_v2` files.
