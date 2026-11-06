# 6. Reproducing everything

## Pinned inputs

| Input | Version |
|---|---|
| Sarvam-30B | `sarvamai/sarvam-30b` @ `071ae95e933605ca1104a6b4524a36a98488efa4` |
| Qwen baseline | `Qwen/Qwen3.6-35B-A3B` @ `995ad96eacd98c81ed38be0c5b274b04031597b0` |
| Qwen published J/R lenses (CKA only) | `camilablank/workspace-lenses`, `qwen3.6-35b-a3b/{j-lens,r-lens}/lens.pt` (main, 2026-10-18) |
| J++ pipeline | `safety-research/jpp_lens` @ `d7efc0de372e20589e899188c0cfe8434acc63ec` |
| CKA explorer code | `eliebak/open-jlens-data` @ `49010193dee265e8521dda041a26d23d7a8bc027` |
| Fit corpus | `Salesforce/wikitext` `wikitext-103-raw-v1` (first N records ≥600 chars, via jpp's loader) |

Snapshots of Sarvam's model code (which our adapter patches around) are in `reference/`.

## Software

- **Modal image** (`code/modal_app.py`): Debian slim, Python 3.12, `torch==2.11.0` (cu128),
  `transformers==5.14.1`, `huggingface_hub==1.26.0`, `numpy==2.5.1`, `pandas==3.0.5`,
  `datasets==5.0.1`, `pyarrow==25.0.0`, `accelerate==1.12.0`, `safetensors`, `matplotlib`, plus git
  clones of the two repos above at their pinned commits.
- **Local CPU environment** (smoke tests): J++'s own `uv sync --extra cpu --extra dev` (torch 2.13.0+cpu,
  same transformers) plus matplotlib.
- **Modal client:** 1.6.2.

## Credentials

Copy `code/modal_env.example.sh`, point `KEYFILE` at a file containing your
`modal token set --token-id ak-… --token-secret as-…` line, and `source` it before each `modal`
command. The token is exported only for that shell. Nothing is written to `~/.modal.toml`.

## Steps (run from `code/`)

```bash
# 0. local, $0: the pipeline on a tiny random Sarvam (CPU, a few minutes)
JPP_DIR=/path/to/jpp_lens python smoke_test.py

# 1. one-time: fp32 -> bf16 Sarvam weights on Volume "sarvam-lens" (~8 min CPU)
modal run modal_app.py --step prepare
modal run modal_app.py --step prepare-qwen          # baseline weights

# 2. check the exact image on CPU before any GPU spend (~$0.04)
modal run modal_app.py --step smoke

# 3. first look: sanity, BOS probe, logit-lens pivot, grading (~$0.50)
modal run modal_app.py --step pilot --budget-minutes 15 --stages load,sanity,bos_probe,pivot_logit,grade --tag ropefix

# 4. R-lens (25 prompts) + J++ weights + 3-lens pivot (~$3.90)
modal run modal_app.py --step pilot --budget-minutes 45 --stages load,fit_router,fit_r_lens,fit_jpp,pivot_lenses --tag lensfit --fit-prompts 25

# 5. Qwen baseline (logit lens, with Chinese tracked)
modal run modal_app.py --step pilot --model qwen --budget-minutes 6 --stages load,pivot_logit --tag qwen_zh

# 6. J-lens, 200 prompts, 5 parallel H200 shards (~$19), then merge on CPU
#    keep the machine awake: `modal run` stops the job if the client disconnects
modal run modal_app.py --step pilot --budget-minutes 53 --stages load,fit_j_shard --tag jfit --fit-prompts 200 --fit-rows 64 --num-shards 5
modal run modal_app.py --step merge-j --fit-prompts 200

# 7. 4-lens pivot + controls + causal ablation (~$1.40)
modal run modal_app.py --step pilot --budget-minutes 22 --stages load,pivot_all,causal --tag final --fit-prompts 25

# 8. follow-ups: swap, Hindi ablation, sentences (~$1.15)
modal run modal_app.py --step pilot --budget-minutes 15 --stages load,swap,hindi_ablation,sentences --tag extra

# 9. CKA explorer reproduction + diagnostics + robustness (CPU)
modal run modal_app.py --step cka
modal run modal_app.py --step diagnose
modal run modal_app.py --step cka-robust

# 10. fetch results, regenerate tables
modal volume get sarvam-lens runs ./results
python summarize_results.py
```

### Reader questions (docs/09, ~$2)

```bash
# free: token-count split and the GPT-2 J-lens test on existing results (any copy of the tokenizer)
python token_split.py <dir with Sarvam's tokenizer files> ../results/token_split/token_split.json
# the metric-A comparison (results/token_split/metric_a_30b.json) is in docs/03 Phase 11, item 50

# image check (cents): the 30B dry pilot of the new stages
modal run modal_app.py --step smoke --stages load,sanity,rq_chat_word_src,rq_chat_sentence,rq_chat_word_en,rq_collapse,rq_cloze,rq_zh,causal_steer,rq_think

# Sarvam-30B: chat x3, homonyms/kinship, cloze, Chinese skill, negative steering, thinking mode
modal run modal_app.py --step pilot --model sarvam --budget-minutes 26 --timeout-margin 3 --spf-guess 0.12 \
    --stages load,sanity,rq_chat_word_src,rq_chat_sentence,rq_chat_word_en,rq_collapse,rq_cloze,rq_zh,causal_steer,rq_think --tag rq30
python summarize_results.py                                # section M
```

(The actual runs split the 30B across `rq30` + `rq30b`; see doc 05 rows 34–39.)

### Lens corpus, Qwen J-lens, routing and neurons (docs/09 §9–11, ~$6)

In Windows Git Bash, prefix every command that has a `/vol/...` argument with `MSYS_NO_PATHCONV=1`.
Otherwise Git Bash rewrites the path to `C:/Program Files/Git/vol/...`, which cost one failed fit
(`modal_app.py` now also un-mangles `--lenses` and `--prompts-json`).

```bash
# Indic J-lens (30B): 4 Wikipedia passages x 9 languages, round-robin; the fit takes the first 27
modal run modal_app.py --step build-corpus --per-lang 4
MSYS_NO_PATHCONV=1 modal run modal_app.py --step pilot --model sarvam --stages load,fit_j_shard \
    --fit-prompts 27 --fit-rows 64 --fit-name j_indic_27p --prompts-json /vol/corpora/indic_wiki.json \
    --budget-minutes 36.5 --tag jfit_indic
modal run modal_app.py --step merge-j --fit-name j_indic_27p
# a size-matched WikiText lens: the first shard (40 prompts) of the 200-prompt fit
modal run modal_app.py --step merge-j --fit-name j_200p --shard-glob 'shard0*' --out-name j_wiki_40p
MSYS_NO_PATHCONV=1 modal run modal_app.py --step pilot --model sarvam --stages load,pivot_ext --budget-minutes 10 \
    --lenses /vol/runs/lenses/j_indic_27p.pt,/vol/runs/lenses/j_wiki_40p.pt,/vol/runs/lenses/j_200p.pt --tag pivot_indic_lens

# Qwen under its published J-lens (logit is always included)
MSYS_NO_PATHCONV=1 modal run modal_app.py --step pilot --model qwen --stages load,pivot_ext --budget-minutes 9 \
    --lenses /vol/ext_lenses/qwen3.6-35b-a3b/j-lens/lens.pt --tag qwen_j

# expert routing by language + layer-0 cross-script neurons (neuron_experiment.py)
modal run modal_app.py --step pilot --model sarvam --stages load,routing,neurons --budget-minutes 6 --tag neuron30

python summarize_results.py                                # sections M and N
```

The fit's jpp stamp shows `prompts_json: null` because the corpus arrives through the
`JPP_FIT_PROMPTS_JSON` env var, not jpp's own flag. The run log (`logs/modal/jfit_indic.log`, line
18) confirms `/vol/corpora/indic_wiki.json`.

### All 22 languages and prompt domains (docs 11 and 12)

Stages named `<module>.<analysis>` run a module's analysis inside the pilot. `--stage-args '{...}'`
(JSON) passes its options. Run `batch_check` first on each model.

```bash
# 1. prompt sets (CPU, downloads from GitHub and the HF Hub) -> data/xling/en_sets.json
python build_xling_sets.py
# 2. translation: IndicTrans2 is gated. Accept its terms on Hugging Face, then store a read token as
#    the Modal secret "huggingface-secret" (key HF_TOKEN). L40S, ~$0.95 in all
modal run modal_translate.py --step smoke
modal run modal_translate.py --step run
modal volume get sarvam-lens corpora/xling/parts data/xling/
python -c "import translate, languages as L; print(translate.assemble('data/xling/en_sets.json', 'data/xling/parts', 'data/xling', list(L.XLING[1:])))"
python check_translations.py data/xling          # -> data/xling/translation_qa.json
# 3. batched vs lone on the tiny models (2xL4, ~$0.07)
modal run modal_app.py --step batch-probe
```

**Doc 11, the 22 languages in the word pivot.** The word lists rebuild from the saved
Wiktionary sources:

```bash
python data/newlangs/build_concepts_new.py --write
modal run modal_app.py --step pilot --model sarvam --stages load,batch_check,newlang_experiment.pivot --budget-minutes 9 \
    --timeout-margin 2 --spf-guess 0.12 --tag xl_new_30b --stage-args '{"decode": "all", "decode_steps": 8}'
modal run modal_app.py --step pilot --model sarvam --stages load,newlang_experiment.causal --budget-minutes 5.2 \
    --timeout-margin 2 --spf-guess 0.12 --tag xl_new_causal_30b
python newlang_experiment.py report ../results/xl_new_30b/newlang_experiment_pivot --band 13-16 --n-layers 19
```

**Doc 12, prompt domains.** `MT` stands for
`hi,bn,ta,te,mr,gu,kn,ml,as,or,pa,ur,ne,sa,mai,sd,ks,kok,doi,mni,brx,sat,ja,ru,ar`; write the
list out inside the JSON.

```bash
modal run modal_app.py --step pilot --model sarvam --stages load,batch_check,domain_experiment.digits,domain_experiment.mgsm,domain_experiment.mgsm_think \
    --budget-minutes 10 --timeout-margin 2 --spf-guess 0.12 --tag xl_dom_30a \
    --stage-args '{"mgsm": {"items": 60, "langs": "en,bn_h,te_h,ja_h,ru_h,zh_h"}, "mgsm_think": {"items": 16, "batch": 48, "lean": false, "variants": "en,bn_h,te_h,ja_h,ru_h,zh_h"}}'
modal run modal_app.py --step pilot --model sarvam --stages load,domain_experiment.profile --budget-minutes 12 --timeout-margin 2 \
    --spf-guess 0.12 --tag xl_dom_30b --stage-args '{"profile": {"items": 80}}'
modal run modal_app.py --step pilot --model sarvam --stages load,domain_experiment.mgsm_think --budget-minutes 7 --timeout-margin 2 \
    --spf-guess 0.12 --tag xl_dom_30think --stage-args '{"mgsm_think": {"items": 8, "batch": 256, "max_tokens": 1000000000, "max_new_tokens": 400, "strict": true, "max_prompt_tokens": 400, "variants": "MT"}}'
python domain_experiment.py summarize ../results/xl_dom_30b          # local: regenerate the summary CSVs
```

(`xl_dom_30b` originally also ran an MGSM stage that went out of memory; the current code caps its
batches at 8,192 padded tokens.) Download each run with `modal volume get sarvam-lens runs/<tag> ../results/`.

Without Modal, the same stages run on any GPU box through `run_jpp.py` and `pivot_experiment.py`
directly. The original runbook is in `code/README.md`.

## What each code file does

| File | Role |
|---|---|
| `sarvam_support.py` | Sarvam loader (pinned code, transformers-5 shims: tied weights, **RoPE rebuild**), BOS modes, bit-exact Sarvam LRP rules, jpp dispatch hook |
| `run_jpp.py` | J++ CLI wrapper: Sarvam support, JSON fit corpora, `prepare-evals`, `grade`, fit/held-out leak guard |
| `pivot_experiment.py` | Pivot experiment: prompts, token sets, controls, scoring, summaries, plots |
| `causal_test.py` | Subspace ablation (`run`), concept swap (`run_swap`) |
| `sentence_experiment.py` | Sentence-cloze readouts |
| `cka_compare.py`, `cka_diagnose.py` | Explorer-method CKA (gain modes, cross-model) and degeneracy diagnostics |
| `modal_app.py`, `modal_pilot.py` | Modal functions and the budgeted multi-stage driver |
| `build_corpus.py` | Language-interleaved Wikipedia fit corpus (used for the Indic J-lens, docs/09 §9) |
| `languages.py`, `batch_gen.py`, `batch_probe.py` | Language registry; batched greedy decoding; the batched-vs-lone probe (docs 11 and 12) |
| `build_xling_sets.py`, `translate.py`, `modal_translate.py`, `check_translations.py` | Cross-lingual prompt sets, their translation (IndicTrans2/NLLB on Modal), translation QA |
| `newlang_experiment.py`, `domain_experiment.py` | Docs 11 and 12 |
| `neuron_experiment.py` | Expert routing by language/concept and layer-0 cross-script neurons (docs/09 §11) |
| `smoke_test.py` | End-to-end CPU test on a tiny random Sarvam (35 checks) |
| `summarize_results.py` | Rebuilds `docs/04_results_tables.md` from `results/` |
| `data/concepts.tsv`, `data/sentences.tsv` | Experiment items |
