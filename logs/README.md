# logs/

Logs are verbatim except for local paths, which were rewritten before publishing: `.\sarvam-lens`
is the code folder relative to the project root (it was named `sarvam-lens/` at run time and is
now `code/`), `<scratch>` is a scratch copy used for the first `prepare` run, and `%TEMP%` is the
local temp directory used by the smoke tests, `.` is the project root and `%USERPROFILE%` the user's home folder. Modal run URLs and image IDs are left as they were. Logs of
the Sarvam-105B runs and the refusal runs aren't part of this release; the tiny-model 105B smoke
tests are.

## modal/: client-side output of every `modal run`

Includes image builds, container logs and final JSON.

| Log | Run |
|---|---|
| `prepare.log` | fp32 → bf16 conversion of Sarvam-30B |
| `smoke_modal.log` | CPU smoke test + dry pilot inside the image |
| `pilot.log` | first full pilot (broken RoPE). The earlier failed-at-import attempt was overwritten by this rerun |
| `pilot2.log` | `bosfix` |
| `pilot3.log` | `ropefix` |
| `prepare_qwen.log`, `qwen.log`, `qwen_zh.log` | Qwen download and baselines |
| `lensfit.log`, `lensfit_live_applogs.txt` | R-lens + J++ run (and a live log snapshot taken mid-run) |
| `rows.log` | rows-per-pass probe |
| `jfit.log` | the 5 parallel J-lens shards |
| `merge.log` | J-lens shard merge |
| `cka.log`, `cka2.log` | CKA reproduction (R/J++; then with the J-lens) |
| `diagnose.log`, `diagnose2.log`, `diagnose3.log` | degeneracy diagnostics v1–v3 |
| `robust.log` | CKA robustness |
| `final.log` | 4-lens pivot + causal ablation |
| `extra.log` | swap, Hindi ablation, sentences |
| `smoke_2gpu_fail1.log`, `smoke_2gpu.log` | tiny 105B split over 2× L4. The first run found the cross-device index bug; the second passed |
| `smoke_rq.log`, `smoke_2gpu_rq.log` | image checks of the reader-question stages (30B dry pilot; tiny 105B on 2 GPUs) |
| `rq30.log`, `rq30b.log` | the reader-question GPU runs |
| `indic_corpus*.log`, `jfit_indic*.log`, `merge_indic27.log`, `merge_wiki40.log`, `pivot_indic_lens.log` | the Indic-corpus J-lens: corpus, fit (`_fail_pathconv`: the Git Bash path failure), merges, comparison |
| `qwen_j.log`, `qwen_j_run1_logit_only.log` | Qwen pivot under its published J-lens (run 1 lost the lens to path conversion) |
| `smoke_2gpu_zh.log` | 2-GPU smoke with the new `chinese_shuffled` role |
| `xlate_smoke.log`, `xlate_run1_oom.log`, `xlate_run.log` | prompt-set translation (`modal_translate.py`): smoke, the first full run (OOM after Hindi), the resumed run |
| `batch_probe.log` | batched vs lone on the tiny 105B (2 GPUs) and tiny 30B, fp32 and bf16 (engineering log item 67) |
| `xl_new_*.log` | doc 11: smokes, the 30B pivot, the causal run, the misses run |
| `xl_dom_*.log` | doc 12: smoke, the 30B domain runs, the thinking-on run |
| `xl_geom_*.log` | doc 15: concept-space geometry (smoke, 30B, Qwen) |
| `xl_hint_smoke.log`, `xl_hint_30.log`, `xl_hint_force_smoke.log`, `xl_hint_30_force.log` | doc 12: the hint-faithfulness run and its forced-answer probe |
| `xlate_hint.log` | the hint sentence for doc 12's faithfulness test, translated by the same pinned models |
| `quant_smoke.log`, `quant_check.log` | doc 14: the bf16-vs-fp32 readout check (smoke, 30B run) |

The line `Do you wish to run the custom code? [y/N]` that appears in several logs is benign. It
comes from the tokenizer seeing `auto_map` in Sarvam's config, and every stage passed regardless.

## local/: CPU smoke tests on the tiny random Sarvam

`smoke.log` … `smoke4.log`, in order, as fixes landed. The last one passes all 35 checks.
`smoke5.log` is the 30B regression after the 105B changes (35 checks; run with `PYTHONUTF8=1`).
`smoke_mla.log` is `smoke_test_mla.py` on CPU (all checks pass).
