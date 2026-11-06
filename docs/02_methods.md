# 2. Methods: what was measured, how, and why

## 2.1 Models

| | Sarvam-30B (`sarvamai/sarvam-30b` @ `071ae95`) | Qwen3.6-35B-A3B (`Qwen/Qwen3.6-35B-A3B` @ `995ad96`) |
|---|---|---|
| Role | subject | English/Chinese-centric baseline, similar size |
| Architecture | `SarvamMoEForCausalLM` (Hub custom code), MoE | `Qwen3_5MoeForConditionalGeneration` (built into transformers 5) |
| Layers / d_model | 19 / 4096 | 40 / 2048 |
| Experts | 128 routed (top-6) + 1 shared; layer 0 dense | 256 routed + shared |
| Vocab | 262,144 (Gemma-style tokenizer) | 248,320 (byte-level BPE) |
| Checkpoint | 32.15B params stored **fp32** (129 GB) → converted to bf16 (64.3 GB) | bf16 on the Hub (71.9 GB) |
| Hardware | 1× H200 141 GB | 1× H200 |

## 2.2 Lenses (four readouts of the same residual stream)

All lenses read the residual stream after block `l` (`h_l`, the block's output) at the final
prompt position and decode it into vocabulary logits with the model's own final norm + LM head.

| Lens | Transport `T_l(h)` before `unembed` | Fit | Why |
|---|---|---|---|
| **Logit** | `h` (identity) | none | baseline; what Wendler et al. used |
| **J-lens** | `J_l h`, `J_l = E[∂h_final/∂h_l]` | **200** WikiText prompts, target = final block (18), all source layers 0–17 | Anthropic's lens; the CKA explorer's lens |
| **R-lens** | `J_l h` with `J_l` estimated through an LRP backward graph | **25** WikiText prompts | cleaner early layers per Blank et al.; 25 prompts matches the published R-lenses |
| **J++** | `Σ_e w_e J_{l,e}` over 8 k-means "expert" Jacobians from the R fit | weights learned on J++ readout evals **minus "multilingual"** | strongest lens per Ayonrinde & Lindsey; the multilingual eval is excluded to avoid an English bias |

Estimator details (jpp `LensTrainer`, identical to `jlens.fit`): each prompt is truncated to
128 tokens. For each output dimension, a one-hot cotangent is placed at *every* valid target
position and backpropagated, and the source gradient is averaged over source positions. The first
16 positions (attention sinks) and the last position are excluded. `J_l` is the mean over prompts.

**R-lens rules for Sarvam** (`sarvam_support.apply_sarvam_lrp_rules`, preset `r+moe`):
- LN-rule: the `rsqrt` in every residual-stream RMSNorm (`input_layernorm`, `post_attention_layernorm`,
  final `model.norm`: 39 modules) is detached. **Attention-internal q/k norms are excluded.**
  In Sarvam they're named `attention.query_layernorm`/`key_layernorm`, which jpp's name-based rule
  would have missed.
- Identity rule: SiLU's gate is treated as constant (gradient `sigmoid(z)`). Half rule: the gated
  product's gradient is split 50/50. Applied to the dense layer-0 MLP, the shared expert, and all
  2,304 routed experts (18 MoE layers × 128).
- Router: top-k mixing weights detached. Shared expert: backward-only gradient ×4 (the R-lens
  authors' best MoE arm).
- **Every rule is bit-exact in the forward.** Verified 0.0 max difference on the real model in bf16.
  jpp's own identity rule (`z * sigmoid(z).detach()`) is not, and on Sarvam it moved the residual by
  up to 20 (on values ≤135) because rounding flips top-k expert choices. We use
  `silu(z).detach() + (z - z.detach()) * sigmoid(z).detach()`: value `silu(z)`, gradient `sigmoid(z)`.

**J++ details:** a router of k-means with 8 clusters on 64-dim PCA of unit-normalised activations,
fit on 128 WikiText prompts per layer. Expert Jacobians are pooled per cluster in the R fit. Weights
are learned with jpp `fit-weights` on the `fit:0` split (20%) of the J++ readout evals with
multilingual emptied, filtered to items Sarvam answers correctly (`results/correctness.csv`).

**BOS for fitting and readout:** `[@BOS@]` alone (`sarvam_bos`), the measured best (§2.7).

## 2.3 The pivot experiment (`code/pivot_experiment.py`)

**Prompts:** 5-shot word translation, Wendler-style, with language labels in each language's own
script, so Indic prompts contain no English:

```
தமிழ்: "நெருப்பு" - తెలుగు: "నిప్పు"
… (5 demos, drawn per (pair, concept) with a crc32-seeded RNG for reproducibility)
தமிழ்: "தண்ணீர்" - తెలుగు: "
```

**Concepts:** 32 common nouns × 11 language columns (`code/data/concepts.tsv`: en, hi, romanized
hi, bn, ta, te, mr, gu, kn, ml, zh), with spelling alternates.

**Pairs (14):** 10 Indic→Indic (Dravidian↔Dravidian, Indo-Aryan↔Indo-Aryan, cross-family; together
`ALL_INDIC`), plus hi→ta, romanized-hi→ta (a script test), en→ta and ta→en (controls).

**Roles tracked per item:** target, source, English, Hindi, Chinese, plus **shuffled controls**
`english_shuffled` and `hindi_shuffled`: the same language's word for a *different* concept (a
fixed derangement, half-way round the sorted concept list).

**Token sets (the tokenizer control):** English words are single tokens, while Sarvam splits Indic
words into ~2 tokens (Qwen: 3–5). Each role's set is every vocab token whose text (NFC,
case-folded, leading space/quotes stripped) is a prefix of ≥2 characters of one of the word's
spellings, **plus the tokenizer's own first token** of the word, even if shorter. For Qwen, that
first token can be a partial-UTF-8 piece. Tokens claimed by two roles (e.g. Hindi पानी / Marathi
पाणी share "पा") are dropped from both. Control sets also exclude every token a real role claims.

**Correctness filter:** keep an item only if the model's actual next token is in the target set
(382/448 kept for Sarvam).

**Readout filtering:** as in J++, tokens with no letter or digit are excluded from the ranking,
except tracked tokens.

**Metrics** per (item, lens, layer, role): softmax probability mass on the set, best rank, and
whether any token is in the top 10. These are averaged over items, with 95% bootstrap CIs over concepts.

## 2.4 Causal ablation (`code/causal_test.py: run`)

For each correctly translated Indic→Indic item (272), at the **final position** after each block in
a band, project out the subspace spanned by the lens steering directions of the concept's words in
the latent language:

```
v_t(l) = J_l^T (g ⊙ W_U[t])   for tokens t beginning the word (≥3-char prefixes)
h ← h − Q Qᵀ h,               Q = orthonormal basis of {v_t(l)}
```

`g` is the final-norm gain. The conditions are: the latent word at the mid band (layers 13–16); the
same-size subspace for a **different concept's** word in the same language (control); and the latent
word at the early band (5–8). The metric is the change in the model's probability of the correct
target-word start, with 2,000-resample bootstrap CIs. It was run with English (J-lens and R-lens
directions) and with Hindi (J-lens).

## 2.5 Concept swap (`code/causal_test.py: run_swap`)

This follows J++'s probe-swap convention. At the band layers and every position but 0, take unit
readout vectors `u_A, u_B` for concept A's and concept B's latent token, compute the residual's
least-squares coordinates on `[u_A, u_B]`, and exchange them (times a scale of 2 or 4). Success means
the model's answer becomes **concept B's word in the target language** (e.g. Telugu "fire" instead
of "water"). Conditions:
- English swap at mid layers.
- Hindi swap at mid layers.
- **Answer swap** (target-language tokens directly): the positive-control ceiling.
- English swap at early layers.
- **Null**: A↔C, scored against B.

Coordinates come from the incoming residual. J++ uses a cached clean pass instead, but across a
4-layer band the difference is small.

## 2.6 Sentence clozes (`code/sentence_experiment.py`, `code/data/sentences.tsv`)

40 natural sentences, 10 concepts × Hindi/Tamil/Telugu/Bengali, that stop right before the concept
word with no translation instruction (e.g. Telugu "I'm very thirsty, please give me some ___").
Indic SOV order makes these natural. Roles, token sets, controls and the correctness filter are the
same as in §2.3. Only 11/40 survived the filter (Sarvam often inserts an adjective first).

## 2.7 BOS probe

Sarvam's tokenizer has two "BOS" tokens: `<bos>` (added by its post-processor) and `[@BOS@]` (its
`bos_token`, written by the chat template as `[@BOS@]\n`). Three prompt prefixes were compared on
5-shot translation accuracy (5 pairs × 32 concepts): `<bos>` 60.6%, `<bos>[@BOS@]\n` 91.2%,
**`[@BOS@]` 96.3%** (chosen).

## 2.8 J-space CKA (`code/cka_compare.py`, `code/cka_diagnose.py`)

This reproduces the CKA explorer with its own code (`eliebak/open-jlens-data` @ `4901019`,
`code/jlens_cka`):
- Probe `P = W_U[ids] ⊙ gain`, with 4,096 ids sampled from the vocabulary by `RandomState(0)`.
- `V_l = P J_l` gives one steering vector per token.
- Linear CKA between layers' centered `V_l` (equivalently their token-pair kernels).
- The explorer's frozen block segmentation (kernel-segmentation DP, blockiness, 0.02 per-block penalty).
- Cross-model matched-depth CKA uses the explorer's shared-string probe ids (4,096 shared tokens).

**Two corrections:**
1. Qwen3.5/3.6 norms compute `x·(1+w)`, but the published `cka_lib` only applies the offset for
   Gemma. We use the correct gain, and we also measured the published rule: the difference is negligible.
2. The published Qwen lenses target the penultimate block, so the true layer count (40) is passed in.

**Diagnostics:**
- Final-norm gain statistics.
- Participation ratio of `V_l` and the share of variance on its top direction.
- The top singular value's share of `J_l`.
- Coupling with the gain removed, with the top principal component removed, and with the top
  output dims of `J_l`'s top direction masked.

## 2.9 Statistics and robustness choices

- Every lens conclusion is reported across 4 lenses. Every role has a shuffled control.
- Causal effects are compared against same-size, same-language subspaces of other concepts, at
  matched layers, plus an early-layer control.
- CIs are 95% bootstrap (1,000 resamples for readouts, 2,000 for ablations).
- **Fit corpus:** the main lenses were fitted on WikiText. A later J-lens fitted on 27 Indian-language
  Wikipedia passages (`code/build_corpus.py`) still shows the English step. The R-lens and J++ weren't
  refit. See doc 09 §9 and doc 07.
- **CI caveat.** The pooled pivot and causal CIs above resample *items*, and the same 32
  concepts recur across the 10 pairs, so they are too narrow. Doc 17's CIs resample concepts (all of a
  concept's items together; MGSM: problems).

## 2.10 Doc 17's additions (`code/post1_experiment.py`)

- **Every language as a role.** The published pivot's prompts, roles and controls, plus every other
  language of `concepts_new.tsv` as a *third-language* role (minus every token a published role or
  another third language claims), each with a shuffled control, and two English semantic neighbours
  per concept (`code/data/neighbours.tsv`). Three readouts per role: the published prefix sets; only
  tokens that are the whole word; and the prefix set's rank within the role language's own script
  (doc 15 §4).
- **Edits as rows of one forward.** Every causal condition of an item runs as one batch row (an
  `Edit` per layer: a projection at the last position, at given positions or at all positions;
  α = 1 removes, 2 reflects, 3 pushes to −2×), through the text module with the LM head on the last
  position only. On the tiny model this equals `causal_test.Ablation` exactly.
- **Size-matched random controls.** For any push, a fixed random unit direction scaled at every layer
  (and position) to the push's own realized |Δh| in the same forward, so it moves the residual exactly
  as far. A different concept's English is *not* a size-matched control: its component, hence its push,
  is smaller.
- **Writer attribution.** At the final position, each attention output (and each head through the
  output projection), each selected expert's weighted output and the shared expert, scored on a word's
  centred J-lens readout direction `(w_R − mean_V w) J_l` (and the logit-lens one), divided by the final
  RMS.
- **Decoding to completion.** `generate_cb`: continuous batching over a preallocated KV slab, grouped-
  query attention for the one-token decode step (no `repeat_kv`), finished rows replaced by new prompts,
  sampling at the model card's settings (temperature 0.8, top-p 0.95; nucleus over the top 2,048
  tokens), exact-loop detection. On the tiny models, greedy `generate_cb` equals lone decoding token for
  token, with admissions, compaction and slab shifts. Traces that don't close get doc 12's forced answer.
