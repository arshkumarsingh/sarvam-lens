# reference/: third-party code this project patches around

Unmodified snapshots of Sarvam's Hub model code, kept so the fixes in `code/sarvam_support.py`
can be checked against the exact code they target. License: Apache-2.0 (Sarvam AI).

| Folder | Source |
|---|---|
| `sarvam-30b/` | `sarvamai/sarvam-30b` @ `071ae95` (`SarvamMoEForCausalLM`) |
| `sarvam-105b/` | `sarvamai/sarvam-105b` @ `abe264b` (`SarvamMLAForCausalLM`), for the paused 105B follow-up |

Where the relevant code lives:
- **30B rotary module:** `SarvamMoERotaryEmbedding`, a non-persistent `inv_freq`, which transformers 5
  leaves uninitialized (usually zeros).
- **30B attention q/k norms:** `query_layernorm` / `key_layernorm`.
- **30B MoE expert loop:** `SarvamMoEExperts.forward`, which runs in-place in eval mode and is
  differentiable.
- **105B rotary:** a per-attention YaRN `rotary_emb` whose cache is rebuilt from config scalars on the first
  forward, so it is not affected by the zeroing (verified with `code/check_sarvam105_tiny.py`).

Not vendored, pinned instead (see `docs/06_reproduce.md`): `safety-research/jpp_lens` @ `d7efc0d`
and `eliebak/open-jlens-data` @ `4901019`. The Modal image clones both at those commits.

## License and attribution

Everything in this folder except this README is Copyright Sarvam AI, redistributed unmodified
under the Apache License, Version 2.0. A copy of the license is in [`LICENSE`](LICENSE). Neither
Hub repository ships a NOTICE file. These files are not covered by the license of the rest of
this repository.

| File | Upstream file | Header |
|---|---|---|
| `sarvam-30b/modeling_sarvam_moe.py`, `configuration_sarvam_moe.py`, `config.json` | [`sarvamai/sarvam-30b`](https://huggingface.co/sarvamai/sarvam-30b/tree/071ae95e933605ca1104a6b4524a36a98488efa4) @ `071ae95e933605ca1104a6b4524a36a98488efa4` (model card: `license: apache-2.0`) | none upstream |
| `sarvam-105b/modeling_sarvam_moe.py` | [`sarvamai/sarvam-105b`](https://huggingface.co/sarvamai/sarvam-105b/tree/abe264bdb8fb2244daadaa59ebc2020b3c139e89) @ `abe264bdb8fb2244daadaa59ebc2020b3c139e89` (model card: `license: apache-2.0`) | Sarvam's Apache-2.0 header, kept as is. It notes the code is based on the Llama and DeepSeek MoE implementations in transformers |
| `sarvam-105b/configuration_sarvam_moe.py`, `config.json`, `special_tokens_map.json`, `tokenizer_config.json` | same revision | none upstream |
