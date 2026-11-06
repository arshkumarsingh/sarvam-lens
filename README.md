# Sarvam Lens: what language does Sarvam-30B think in?

An interpretability study of `sarvamai/sarvam-30b`, an Indic-first open-weights MoE. We used four
workspace lenses (logit, J-lens, R-lens, J++), shuffled-concept controls, causal ablations, a
Qwen3.6-35B-A3B baseline, and a reproduction of the J-space CKA explorer. The core study ran on
2026-10-18 for ≈ $30 of Modal GPU time. Follow-ups from 2026-10-18 to 2026-10-20 added activation
patching, the reader questions, an Indic-fit lens, the routing/neuron analysis, math and technical
prompts, all 22 scheduled Indian languages (docs 11 and 12), and checks on the first blog post's
claims (doc 17). [docs/05](docs/05_costs.md) has the cost of every run.

This repo goes with the first blog post. The refusal and safety work (docs 13, 16, 18 and 19) will be
published with the second post, so those numbers are missing from the doc list.

- **Read the write-up:** [What language does an Indian AI think in?](https://arshkumarsingh.com/writing/what-language-does-an-indian-ai-think-in)
- **Lenses on Hugging Face:** [arshkumarsingh/sarvam-lens](https://huggingface.co/arshkumarsingh/sarvam-lens). The fitted J-lens, R-lens and J++ lens (1.2 GB each),
  the expert Jacobians and the router. They're too big for git, so `lenses/` here holds only their
  provenance files and the [model card](lenses/MODEL_CARD.md).

## Headline findings

1. **Indic→Indic translation passes through English mid-network.** At layers 14–16 of 19, the
   English word is in the top 10 for 97–100% of items under the three fitted lenses (J, R, J++) and
   93–95% under the fit-free logit lens. A different concept's English word scores 0–3%. Before layer 11
   almost nothing is readable: the logit lens, R-lens and J++ stay under 2%, and the J-lens shows
   only a small English bump around layer 8, partly a tie artefact (doc 14). The four lenses all decode through the model's output layer, so their
   agreement is one kind of evidence, not four; the independent checks are the geometry (13) and the
   ablations (2). *How to read "through English": see 13.*
2. **It's causally used, but it isn't a bottleneck.** Removing the English word's lens directions at
   layers 13–16 costs −5.7 pts of target probability (control −0.4, early layers −0.3), and ~95%
   of translations survive; removing English, Hindi and the target word together still leaves 94%.
   **Changed 2026-10-20 ([doc 17](docs/17_post1_followups.md) #5):** this headline used to add that
   pushing the English component to −2× costs −41 pts (control −9). Most of that is edit size: a
   random direction pushed exactly as far costs −30. English's specific share is the removal's
   ~6 pts (+5.7 against a same-size random edit) and ~10 more under the push.
3. **Hindi is a quieter, real second latent.** At layers 14–16 it ranks in the top 10 for 41–83% of
   items, depending on the lens (logit 41–57%, fitted lenses 55–83%; control 0–1%), and removing it
   costs −4.4 pts.
4. **No less English than an English/Chinese model.** Qwen3.6-35B-A3B pivots through English
   (equally strong) *and* Chinese. Each model's hidden languages track its main training languages.
5. **The same pattern shows up in plain sentence completion:** English top-10 0.85 on 43 sentences
   (J-lens), control ≤ 0.02 ([doc 09 §4](docs/09_reader_questions.md)).
6. **Sarvam is an outlier on the CKA explorer** (uniform map, coupling 0.991). We traced that to
   outlier final-norm gains. Without them, its early block ends at layer 11, where English appears.
7. **A transformers-5 bug:** Sarvam-30B loads with its **RoPE frequencies left uninitialized**
   (usually all zeros), so it has no position information. The bad buffer was printed on a tiny
   random model; on the full checkpoint its consequences were measured. Fixed here, and reported
   upstream ([Hugging Face discussion](https://huggingface.co/sarvamai/sarvam-30b/discussions/15)).
8. **It holds in real use, and it has a cost.** The pivot persists in the model's own chat template
   (English top-10 up to 0.88, control 0), and the 30B's visible reasoning is 92% English. English
   routing also produces errors: translations pick the other meaning of the same English word
   (4.6% vs 0% for a control). [Reader questions →](docs/09_reader_questions.md)
9. **It isn't a lens artifact or an "English expert".** A J-lens fitted only on Indian-language text
    still shows the English step (P 0.61). Under matched J-lenses Sarvam shows more English than Qwen
    (0.74 vs 0.61). The MoE router partly groups words by concept and target language, not through
    English; removing the English direction doesn't steer later experts toward English or the target.
10. **Every Indian language we could test goes through English, and the added ones lean on it
    more.** That is 21 of the 22 scheduled languages: 13 of the 14 added ones (Dogri has only 6 of
    the 32 words), plus Japanese, Russian and Arabic. It includes Kashmiri, Santali (Ol Chiki),
    Manipuri (Meitei Mayek) and Bodo: English top-10 at least 0.84 (J-lens) in every language group
    with more than 3 kept items, control ≤ 0.03.
    On six added source languages (Odia, Urdu, Kashmiri, Manipuri, Santali, Japanese), removing
    English costs about 2.3× what it cost the original pairs (−13.2 vs −5.7 pts; separate runs on
    different items). What fails for low-resource languages is translating *into* them. Even
    on failed translations the English concept is still there mid-network (top-10 0.88 vs 0.96
    on successes): the model understands the word but fails to produce the language.
    [Doc 11 →](docs/11_new_languages.md)
11. **Arithmetic reads out in English, and visible reasoning is mostly English.** Single-digit sums written only in native
    digits (Devanagari, Ol Chiki, Meitei…) put the English number word ("seven") in the middle
    layers' top 10 for 76% of problems under the J-lens (54% under the logit lens; a wrong answer's
    word 0%). But deleting "seven" there doesn't break the sum (−0.2 pts; all nine number words
    together about 4 pts; doc 17 #4), so it's a readout, not a required step. With thinking on,
    93–97% of the reasoning is in Latin script (mostly English) in all 25 languages tested
    (8 problems per language, first 400 tokens; low-resource languages 96%). Math prompts are *not*
    much more English inside than science or technical prompts of the same length (1.7–2.9 pts
    under the J-lens; the logit gaps of 0.5–1 pt are within noise); math and everyday prompts
    barely overlap in length.
    - That reasoning discloses hints but isn't in control of the answer. (Disclosing a hint is
      necessary for faithful reasoning, not proof of it.) Given a wrong
      "a teacher says the answer is N" hint, the 30B follows it about a third of the time overall
      (7–57% by language and hint language, cells of 11–20 problems). Every hint-following trace
      mentions the hint (65/65, Wilson [94, 100]), whatever the language: the model translates the
      hint into English in its reasoning. But 86% of those traces had already found the right
      answer; when the trace was cut at 1,638 tokens and the model made to answer, it gave the
      teacher's number (almost no trace finished its reasoning on its own). **Changed 2026-10-20
      ([doc 17](docs/17_post1_followups.md) #8):** that last caveat no longer applies. With 3× the
      problems and traces run to completion (3,072 tokens, sampling), uptake is 0.36 (native hint)
      vs 0.33 (English hint), and 0.43 / 0.46 on natural conclusions alone, so following the hint is
      not a token-cut artefact; verbalisation stays 0.99–1.00.
    [Doc 12 →](docs/12_prompt_domains.md)
12. **Refusal shows up as English words first, and it's weaker outside English.**
    - When Sarvam-30B refuses an Indian-language request in that language, English refusal words
      ("Sorry", "I cannot") are readable at layers 13–16 for 56% of them. The language's own refusal
      words are at 0% until layer 17. Qwen refuses "in Chinese" inside.
    - The 30B refuses unsafe prompts 82% of the time in English but 42% across the 22 Indian
      languages (machine-translated XSTest; translation explains only a small part).
    - A harm signal is readable from the prompt, and the model answers anyway. Read before the
      reply, a harm probe trained on English flags 74% of the harmful Indian-language prompts the
      model complied with. Used as a monitor, it catches 97% of held-out harmful English prompts but
      58% of Indian-language ones.
    The full refusal study (doc 13 onward) will be published with the second post.
13. **The space is shared, not English-centric; English is the easiest language to *read* it in.**
    Measured without decoding through tokens, the middle layers hold one shared concept space: the
    same concept is retrieved across 27 languages at 0.83–0.84 with the word in its prompt line (chance
    0.03; bare words 0.45–0.48), Meitei Mayek and Ol Chiki included.
    - No measure gives English a large lead. English and Hindi are co-anchors: as a retrieval hub,
      English 0.925 vs Hindi 0.910, a gap significant only at one of four band layers. Qwen has a
      real English hub (+0.14, prompt-line carrier).
    - Ranked within each language's own script, Hindi and the target word reach top-10 0.92 and
      0.85, against English's 0.94. So most of English's lead in the whole-vocabulary readout
      comes from the Latin script owning 148k of the 262k tokens (155k counting special and unused
      placeholder tokens).
    - The ablations still show the English directions are used. The better reading is a shared
      space that reads out most easily as English, not a model "thinking in English".
    [Doc 15 →](docs/15_concept_geometry.md)

## Where to start

| If you want… | Read |
|---|---|
| the gist, no jargon | [docs/00_plain_language_summary.md](docs/00_plain_language_summary.md) |
| the question and prior work | [docs/01_question_and_background.md](docs/01_question_and_background.md) |
| exactly what was measured and why | [docs/02_methods.md](docs/02_methods.md) |
| how it was built, every bug and fix, in order | [docs/03_engineering_log.md](docs/03_engineering_log.md) |
| the results with numbers | [docs/04_results.md](docs/04_results.md) and the auto-generated [docs/04_results_tables.md](docs/04_results_tables.md) |
| what it cost | [docs/05_costs.md](docs/05_costs.md) |
| how to rerun it | [docs/06_reproduce.md](docs/06_reproduce.md) |
| caveats and next steps | [docs/07_limitations_and_next_steps.md](docs/07_limitations_and_next_steps.md) |
| **the questions a reader would ask** (tokenization, real chat use, does English cause errors, Chinese, J-lens trust, lens fit corpus, Qwen, expert routing) | [docs/09_reader_questions.md](docs/09_reader_questions.md) |
| all 22 scheduled Indian languages (+ Japanese, Russian, Arabic) in the pivot, with Wiktionary word lists and a causal test (21 testable; Dogri lacks a word list) | [docs/11_new_languages.md](docs/11_new_languages.md) |
| math, science, technical vs everyday prompts; native-digit arithmetic; the language of the visible reasoning | [docs/12_prompt_domains.md](docs/12_prompt_domains.md) |
| a check that bf16 lens readouts don't distort the published pivot (they don't) | [docs/14_bf16_readout_check.md](docs/14_bf16_readout_check.md) |
| is the middle-layer space language-neutral or English-centric? (shared, with English and Hindi as co-anchors) | [docs/15_concept_geometry.md](docs/15_concept_geometry.md) |
| the follow-ups behind the first blog post: is English special, a push curve with a random control, thinking in the prompt's language | [docs/17_post1_followups.md](docs/17_post1_followups.md) |

## Next: Sarvam-105B

Not here yet. Sarvam's larger open model is the obvious next step, and a first look says it's worth
doing properly. Its bf16 weights are about 212 GB, so it needs two H200s just to load, and fitting
lenses on it takes roughly four times the compute per prompt (an estimate from the two models'
configs: 32 vs 19 layers, and wider active experts). Results will be added once there's compute for it.

## Folder map

```
Sarvam Lens/
├── README.md            ← you are here
├── docs/                ← the write-ups; 04_results_tables.md is generated from results/
├── code/                ← every script (README.md inside is the original runbook)
│   ├── data/            ← concepts.tsv (32 concepts × 11 languages), sentences.tsv (40 clozes)
│   └── modal_env.example.sh
├── results/             ← raw outputs of every published run (README.md maps run tag → experiment)
├── lenses/              ← stamps + model card for the 3 fitted lenses and 2 expert-Jacobian files (the .pt files are on Hugging Face)
├── logs/                ← client logs of the published Modal runs + local smoke-test logs
├── reference/           ← pinned snapshots of Sarvam's model code (30B, 105B) that the adapter patches around
└── history/             ← one-off patch scripts used during development (already applied)
```

## Regenerate the tables

```bash
python code/summarize_results.py     # rewrites docs/04_results_tables.md from results/
```

## License

Released under the Apache License 2.0 ([LICENSE](LICENSE)), Copyright 2026 Arsh Kumar Singh, except
the included prompt data and word lists, which keep their sources' licenses (some CC BY-SA; the
NLLB-200 translations are non-commercial). [NOTICE](NOTICE) lists them and the other third-party
attributions. The Sarvam model-code snapshots in
`reference/` are Apache-2.0 (Sarvam AI) and keep their own license, in
[reference/LICENSE](reference/LICENSE).
