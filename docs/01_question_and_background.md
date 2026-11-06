# 1. Question and background

## The question

Does `sarvamai/sarvam-30b`, an Indic-first open-weights MoE trained from scratch, route
translation between Indian languages through an internal "pivot" language (English? Hindi?), or
does it process each language locally? Secondary questions:

- At what depth does the pivot appear, and is it *causally used*?
- Does an "Indic-first" model rely on English less than an English/Chinese-centric one?
- What does Sarvam's internal geometry look like on the J-space CKA explorer, next to 41 other open
  models?

## Why Sarvam-30B

| Option | Verdict |
|---|---|
| Sarvam-1 / Sarvam-2B (dense, ~2B) | Cheap, but at 2B the lens's workspace properties may be weak; Sarvam-1 already has a logit/tuned-lens study (Indic-TunedLens, VarDial 2026). |
| **Sarvam-30B** (32B total / 2.4B active MoE, 19 layers, d_model 4096, 262K vocab) | **Chosen**: modern, trained from scratch, fits one H200 in bf16. |
| Sarvam-105B (106B MoE, MLA attention, 32 layers) | Needs 2×H200 and new adapter work. Not covered here yet. |
| Sarvam-M | A Mistral fine-tune, so its internals are mostly Mistral's. Skipped. |

## Prior work this builds on

| Work | What it contributes here |
|---|---|
| Wendler et al. 2024, *Do Llamas Work in English?* (ACL) | The original English-pivot finding in Llama-2, via the logit lens on few-shot word translation. Our prompt design and first-token token control follow it. |
| Zhong et al. 2025, *What Language Do Non-English-Centric LLMs Think In?* (ACL Findings) | Japanese-centric models pass through English *and* Japanese. We see the same "second language = main training language" pattern (Qwen: Chinese; Sarvam: Hindi). |
| Indic-TunedLens (VarDial 2026) | Logit/tuned lens on Sarvam-1 (2B); mentions an English-centric latent space. No causal tests, no workspace lenses. |
| Gurnee, Lindsey et al. 2026, *Verbalizable Representations Form a Global Workspace in Language Models* (Anthropic), code `anthropics/jacobian-lens` | The **J-lens**: transport a residual through the average Jacobian `J_l = E[∂h_final/∂h_l]`, then unembed. |
| Blank, Bhatia, Nanda 2026, *R-lens* (LessWrong) | The **R-lens**: same estimator, LRP rules in the backward pass (LN-rule, identity rule, half rule; MoE: router frozen, shared-expert gradient ×4). Cleaner early layers. |
| Ayonrinde & Lindsey 2026, *J++ Lens* (LessWrong), code `safety-research/jpp_lens` | **J++**: k-means expert Jacobians + learned weights + LRP + readout filtering. **We use this codebase for all fitting.** |
| Bakouch, *J-lens CKA explorer* (eliebak.com/viz/jspace-open), data `eliebak/open-jlens-data` | Layer×layer CKA of J-lens token geometry for 41 open models. We reproduce it for Sarvam with its own code. |
| Cammarata (Goodfire), *Neural Networks Are Already Clean* | A neuron-level view ("pr" neuron firing on Bengali "প্র"). Noted as a phase-2 direction, not used. |

## Prior work for the later questions (docs 11–15)

| Work | Where it bears on this study |
|---|---|
| Lindsey, Gurnee, Ameisen et al. 2025, [*On the Biology of a Large Language Model*](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Anthropic) | Attribution graphs found shared, largely language-independent concept features, with language-specific parts near input and output. Doc 15 tests the same question on Sarvam without decoding through tokens: a shared space with English and Hindi as co-anchors. |
| Schut et al. 2025, *Do Multilingual LLMs Think in English?* | Logit lens over generation positions. Doc 12's whole-text profile is in this spirit; it finds a high Latin floor, so only differences count. |
| Chen, Benton, Radhakrishnan et al. 2025, [*Reasoning Models Don't Always Say What They Think*](https://arxiv.org/abs/2505.05410) · Lanham et al. 2023, [*Measuring Faithfulness in Chain-of-Thought Reasoning*](https://arxiv.org/abs/2307.13702) | The hint paradigm and early answering used in doc 12's faithfulness test. |
| Korbak et al. 2025, [*Chain of Thought Monitorability*](https://arxiv.org/abs/2507.11473) · Guan et al. 2026, [*Monitoring Monitorability*](https://arxiv.org/abs/2512.18311) | Doc 12's results bear on multilingual CoT monitorability. The reasoning is 93–97% English in all 25 languages and verbalizes native-language hints (63/63), but final answers can ignore it. |
| Smith, Rajamanoharan, Conmy et al. 2025, [*Negative Results for SAEs on Downstream Tasks*](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks) | Why this study uses probes and lenses rather than training sparse autoencoders. |

## One confound we designed around from the start

J++ learns its expert weights on a dev set of readout tasks. Its **"multilingual" task has
non-English prompts with English target intermediates** (`Lo opuesto de "grande" es` → `small`). A J++
lens trained on the default items is *optimized to find English inside non-English prompts*, which
is exactly our claim. So our J++ weights were fit with that eval **excluded**
(`run_jpp.py prepare-evals --exclude multilingual`), and we only trust conclusions that hold for
logit, J, R *and* J++. Only J++ learns anything from labelled data.
