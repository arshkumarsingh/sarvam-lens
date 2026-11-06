---
license: apache-2.0
base_model: sarvamai/sarvam-30b
datasets:
- Salesforce/wikitext
- wikimedia/wikipedia
language:
- en
- hi
- bn
- ta
- te
- mr
- gu
- kn
- ml
tags:
- interpretability
- mechanistic-interpretability
- jacobian-lens
- j-lens
- r-lens
- jpp-lens
- workspace-lens
- indic
- multilingual
---

# Sarvam-30B workspace lenses (J-lens, R-lens, J++, Indic-fit J-lens)

Four fitted workspace lenses for [`sarvamai/sarvam-30b`](https://huggingface.co/sarvamai/sarvam-30b),
an Indic-first open-weights MoE, plus the router the J++ lens's expert split uses. A lens maps the
residual stream after any of layers 0–17 to the vocabulary, so you can read what the model is "about
to say" mid-network. As far as we know these are the first Jacobian-family lenses published for an
Indic model. A tuned lens exists for the smaller Sarvam-1 (Indic-TunedLens, VarDial 2026).

They were fit for a study of whether Sarvam-30B passes through English when it translates between
Indian languages. At layers 14–16 of 19, the English word is in the lens's top 10 for 97–100% of
Indic→Indic translation items under the fitted lenses, and 93–95% under the fit-free logit lens; a
different concept's English word scores 0–3%. Read that as "English is the most readable language
mid-network", not "the only one": counted within each language's own script, the right word in the
other Indian languages is in the top 10 for about 74% of items too. See Limitations. All top-10
rates here are computed after jpp's readout filtering, which drops every token with no letter or digit
before ranking; a raw `topk(10)` over the full vocabulary will not reproduce them (see How to load).

- **Write-up:** https://arshkumarsingh.com/writing/what-language-does-an-indian-ai-think-in (publishing soon)
- **Code, every run's logs and raw results:** https://github.com/arshkumarsingh/sarvam-lens (publishing soon)
- **Download and check:** `hf download arshkumarsingh/sarvam-lens --local-dir sarvam-lenses` (`huggingface-cli download` on older huggingface_hub), then
  `python sarvam-lenses/verify_lenses.py sarvam-lenses` (sha256 of every lens file, plus shapes and
  configs; needs torch only for the shape checks).

> [!WARNING]
> **Do not load Sarvam-30B with a stock transformers-5 `from_pretrained` and apply these lenses.**
> Under transformers 5, Sarvam's rotary embedding comes back with **`inv_freq` uninitialized**
> (`torch.empty_like`; usually all zeros, sometimes arbitrary huge values).
> transformers re-derives non-persistent buffers only for its own classes, and Sarvam ships as Hub
> custom code. The model then has no position information: it reads its prompt as a bag of words
> and loops on greedy decoding. Nothing errors, but **every lens readout will be wrong.**
>
> Load the model with `sarvam_support.load_sarvam_lens_model` from the GitHub repo
> (`code/sarvam_support.py`), which rebuilds the rotary embedding after loading. To check any
> loaded model, the following must be non-zero:
>
> ```python
> hf_model.model.rotary_emb.inv_freq.abs().sum()   # 0 (or a huge number) means RoPE is broken
> ```
>
> The safer check compares it with the value recomputed from `config.json`, as the repo's
> `code/smoke_test.py` does.
>
> Observed with transformers 5.14.1. Not checked on transformers 4.x. All four lenses were fit
> on the fixed model.

## Files

| File | What it is | Size | dtype |
|---|---|---|---|
| `j_200p.pt` | **J-lens.** Plain Jacobian lens, 200 WikiText prompts | 1.2 GB | fp32 |
| `r_moe_25p.pt` | **R-lens.** Jacobian through an LRP backward graph, `r+moe` preset, 25 WikiText prompts | 1.2 GB | fp32 |
| `jpp_25p.pt` | **J++ lens.** Learned weighting of the R-lens's 8 expert Jacobians, 25 prompts, with the multilingual eval excluded from weight fitting | 1.2 GB | fp32 |
| `j_indic_27p.pt` | **Indic-fit J-lens.** Plain Jacobian lens fit on 27 Wikipedia passages in English and 8 Indian languages | 1.2 GB | fp32 |
| `router.pt` | The k-means router that assigns positions to the 8 expert clusters (needed only to refit experts) | 19 MB | fp32 |
| `*_stamp.json`, `router_stamp.json` | Fit provenance: exact arguments, jpp git hash and timestamp of each fit stage | small | |
| `jpp_25p_expert_weights.json` | The J++ per-layer expert weights and their fitting curve | small | |
| `SHA256SUMS`, `verify_lenses.py` | Checksums of every file above, and a script that checks them plus each file's structure | small | |
| `LICENSE`, `NOTICE.md` | Apache-2.0, and the attributions | small | |

Each lens covers source layers 0–17 of Sarvam-30B's 19 blocks (`d_model` 4096): 18 Jacobians of
4096 × 4096. The target is the final block, so layer 18 has no lens. The files are in
[jpp_lens](https://github.com/safety-research/jpp_lens)'s lens format and load with
`torch.load(..., weights_only=True)`, so no pickled code runs. Lens entries are O(1), so re-saving
at fp16 halves the size safely (fp16 is `jlens`'s default).

Not uploaded: the per-expert Jacobian files the J++ lens and a J++-on-J would be built from
(`j_200p_experts.pt`, `r_moe_25p_experts.pt`, `j_indic_27p_experts.pt`, 5.4 GB each in fp16). Ask in
the Community tab if you need them.

## Sarvam-105B

Not here yet. [`sarvamai/sarvam-105b`](https://huggingface.co/sarvamai/sarvam-105b) is the obvious next
step, and a first look says it's worth doing properly. Fitting lenses on it takes more compute than this
project had: its bf16 weights (about 212 GB) need two H200s just to load, and with 32 layers and wider
experts, each fitting prompt costs roughly four times what it does on the 30B. Fitted 105B lenses will
be added here once there's compute for them.

## Base model and fit

| | |
|---|---|
| Base model | `sarvamai/sarvam-30b` @ `071ae95e933605ca1104a6b4524a36a98488efa4` (`SarvamMoEForCausalLM`, 19 layers, 128 routed experts top-6 + 1 shared, layer 0 dense) |
| Weights used | the Hub's fp32 checkpoint converted to bf16. Fit on H200s |
| Fit corpus (J, R, J++) | `Salesforce/wikitext`, `wikitext-103-raw-v1`: the first N records of ≥ 600 characters, through jpp's loader. English only |
| Fit corpus (Indic J) | `wikimedia/wikipedia`, dump 20231101: passages of 600–1,500 characters, round-robin over en, hi, bn, ta, te, mr, gu, kn, ml (after skipping each language's first 1,000 articles; template duplicates dropped); 27 used |
| Fitting code | `safety-research/jpp_lens` @ `d7efc0de372e20589e899188c0cfe8434acc63ec`, with the Sarvam adapter in `code/sarvam_support.py` and `code/run_jpp.py` |
| Estimator | `J_l = E[∂h_final/∂h_l]`. Prompts truncated to 128 tokens. A one-hot cotangent at every valid target position, averaged over source positions. The first 16 positions (attention sinks) and the last are excluded |
| BOS | `[@BOS@]` alone (`sarvam_bos`), the measured best of three BOS modes. Use the same mode at readout |
| J-lens | `--lrp-mode none`, 200 prompts, 64 Jacobian rows per backward pass, run as 5 parallel shards of 40 prompts |
| R-lens | `--lrp-mode r+moe` with Sarvam-specific rules, each bit-exact in the forward (0.0 max difference in bf16). The LN rule detaches the `rsqrt` in the 39 residual-stream RMSNorms, and the attention q/k norms are left alone. The identity and half rules apply to the dense layer-0 MLP, the shared expert and all 2,304 routed experts. The router's top-k weights are detached, and the shared expert's gradient is scaled ×4 |
| J++ | Router: k-means, 8 clusters, on a 64-dim PCA of unit-normalised activations, fit on 128 WikiText prompts per layer (`router.pt`). The weights over the R-lens's 8 expert Jacobians were learned with jpp `fit-weights` (200 steps, pass@10) on the `fit:0` split (20%) of J++'s readout evals, **with the "multilingual" eval removed**, filtered to the items Sarvam-30B answers correctly (43 items). The multilingual eval asks for English intermediates inside non-English prompts. Fitting on it would have trained the lens to find English, which is the thing being measured |
| Indic J-lens | `--lrp-mode none`, 27 prompts, 64 rows per pass, same estimator and BOS as the J-lens |

## How to load

You need a clone of [jpp_lens](https://github.com/safety-research/jpp_lens) at `d7efc0d` and the
`code/` folder of the GitHub repo (https://github.com/arshkumarsingh/sarvam-lens, publishing soon). Full environment and pinned versions:
`docs/06_reproduce.md` in that repo.

```python
import sys; sys.path[:0] = ["/path/to/jpp_lens/src", "/path/to/sarvam-lens/code"]
import run_jpp; jpp_cli = run_jpp._import_jpp_cli(); run_jpp.install(jpp_cli)
import sarvam_support
import torch as t
from workspace_lens.utils import load_lens_file
from lens_evals.eval_utils import all_non_semantic_token_ids

# RoPE and BOS fixes are applied here. Do not use a plain from_pretrained.
model = sarvam_support.load_sarvam_lens_model("sarvamai/sarvam-30b", device_map="cuda")
lens = load_lens_file("sarvam-lenses/j_200p.pt", hf_model_name="sarvamai/sarvam-30b")

# The study's five-shot prompt for "water", Tamil -> Telugu
# (build_prompt in code/pivot_experiment.py, seed 0).
prompt = "\n".join([
    'தமிழ்: "கண்" - తెలుగు: "కన్ను"',
    'தமிழ்: "இரவு" - తెలుగు: "రాత్రి"',
    'தமிழ்: "உப்பு" - తెలుగు: "ఉప్పు"',
    'தமிழ்: "மீன்" - తెలుగు: "చేప"',
    'தமிழ்: "கிராமம்" - తెలుగు: "గ్రామం"',
    'தமிழ்: "தண்ணீர்" - తెలుగు: "',
])
layers = [14, 15, 16]
logits, model_logits, ids = lens.apply(model, prompt, layers=layers,
                                       token_positions_for_residuals=[-1])

# Readout filtering, as in the study: drop tokens with no letter or digit, then rank.
tok = model.tokenizer
drop = t.tensor(sorted(all_non_semantic_token_ids(tok, logits[layers[0]][0].shape[-1])))
for layer in layers:
    row = logits[layer][0].float().cpu()
    row[drop] = float("-inf")
    print(layer, [tok.decode([i]) for i in row.topk(10).indices.tolist()])
```

In the study's run of this item, an English token for "water" ranks first at all three layers, the
Hindi word ranks 7–8, and the Telugu answer is still outside the top 10 (rank 11–16). The study used
prompts like this one throughout: five demonstrations, then the word to translate, with the language
labels in their own scripts.

Pass `hf_model_name="sarvamai/sarvam-30b"` as shown. The files record the path the model was
loaded from during fitting (`/vol/sarvam-30b-bf16`, a container path).

## Limitations

1. **The lenses are not independent witnesses.** All four (and the logit lens) decode through the
   model's own output layer, so when they agree that is one kind of evidence, not four. The study's
   independent checks are a geometry comparison of internal states that doesn't use the vocabulary,
   and size-matched ablations.
2. **"Readable" is not "used".** Deleting the English word's lens directions at layers 13–16 costs
   about 6 points of target probability against a same-size random edit, and about 95% of
   translations survive; deleting the English, Hindi and target-word directions together still
   leaves 94%. Pushing the English component further (to −2×) looks much larger, but a random
   direction pushed as far costs nearly as much.
3. **The readout is word-specific, and English is the most readable language, not the only one.**
   Near-synonyms (liquid for water) rarely reach the top 10, which shows specificity but doesn't
   show an English word is used internally: a shared concept decoded through a mostly Latin-script
   vocabulary (about 148,000 of 262,144 tokens) would also favour the exact English word.
4. **Modest fit budgets.** J-lens 200 prompts, Indic J-lens 27, R-lens and J++ 25. The CKA
   explorer's reference fits use 460–1,000, and 25-prompt lenses can be under-converged. The
   200-prompt J-lens agrees with the 25-prompt R-lens on the study's pivot readouts.
5. **J++ is partly built on pooled copies.** In the 25-prompt R fit, 16/144 layer-expert slots had
   fewer than 50 token positions and fell back to the pooled Jacobian.
6. **Corpus language.** J, R and J++ were fit on English WikiText. The Indic-fit J-lens still shows
   the English step at layers 14–16 (top-10 0.89–0.99, against 0.98–0.99 for the WikiText J-lens),
   reading it slightly later. The R-lens and J++ were not refit on Indic text.
7. **No held-out lens-quality evaluation.** On its own 43 fitting items, J++'s macro pass@10 peaks
   at 0.23 at layer 15 (`train_macro_pass` in `jpp_25p_stamp.json`). The lenses were validated on
   the study's word-translation readouts, not on a general benchmark.
8. **The study's evidence is mostly single words.** It rests on 5-shot word translation over 32
   concrete nouns. Sentence-level checks are smaller: 43 plain sentence completions and 366
   chat-template sentence translations on the 30B.
9. **Coarse depth.** With 19 layers, "mid-network" is a handful of layers.
10. **Unusual geometry on the CKA explorer.** Sarvam's J-lens CKA map is near-uniform (layer
    coupling 0.991), which the study traces mostly to outlier final-norm gains. Even without them
    Sarvam stays unusually unlike Qwen3.6-35B-A3B (0.27), and why is open.

## Provenance and cost

Every fit stage's arguments, timestamp and jpp commit are in the `*_stamp.json` files. The client
logs of every run and all raw outputs are in the GitHub repo (`logs/`, `results/`). The run that
fit the router, the R-lens and the J++ weights (plus a 3-lens pivot) took 43 minutes on one H200
(about $3.86; the R-lens fit itself was 32 minutes at 77 s/prompt). The J-lens took 5 parallel H200
shards of about 42 minutes each (about $19), on 2026-10-18. The Indic J-lens and its comparison pivot
cost about $3.37 on 2026-10-19.

## License and attribution

The lenses, the router and the scripts here are released under the Apache License 2.0 (`LICENSE`),
Copyright 2026 Arsh Kumar Singh. See `NOTICE.md` for attributions.

- **Base model:** derived from Sarvam-30B, Apache-2.0, Copyright Sarvam AI.
- **Fitting code:** `safety-research/jpp_lens` (Apache-2.0, Kola Ayonrinde), which includes
  Anthropic's `jacobian-lens` reference implementation (Apache-2.0) and LRP rules adapted from
  Camila Blank and Agam Bhatia's R-lens.
- **Fit corpora:** activations were computed on WikiText-103 (`Salesforce/wikitext`, CC BY-SA,
  derived from Wikipedia) and on Wikipedia passages (`wikimedia/wikipedia`, CC BY-SA). The lens
  files contain no corpus text, only Jacobians averaged over positions; we credit the sources here.

Lens methods:
- J-lens: Gurnee et al. 2026, [*Verbalizable Representations Form a Global Workspace in Language
  Models*](https://transformer-circuits.pub/2026/workspace/index.html) (Anthropic), code
  [`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens).
- R-lens: Blank, Bhatia & Nanda 2026, [*R-lens: Making J-lens More Faithful on Early
  Layers*](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers).
- J++: Ayonrinde & Lindsey 2026, [*J++ Lens: Jacobian Filtering Enables More Faithful Workspace
  Lenses*](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace),
  code [`safety-research/jpp_lens`](https://github.com/safety-research/jpp_lens).

## Citation

```bibtex
@misc{singh2026sarvamlens,
  author       = {Singh, Arsh Kumar},
  title        = {Sarvam-30B workspace lenses (J-lens, R-lens, J++, Indic-fit J-lens)},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/arshkumarsingh/sarvam-lens}}
}
```
