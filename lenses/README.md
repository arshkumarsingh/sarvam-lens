# lenses/: the fitted Sarvam-30B lenses

The `.pt` files are not in git. Download them from Hugging Face: [arshkumarsingh/sarvam-lens](https://huggingface.co/arshkumarsingh/sarvam-lens).
[`MODEL_CARD.md`](MODEL_CARD.md) is that repo's model card. The small `*.json` files here are the
same provenance files that ship with the lenses.

**Publishing.** The Hugging Face repo holds `j_200p.pt`, `r_moe_25p.pt`, `jpp_25p.pt`, `j_indic_27p.pt`
(the Indic-fit J-lens, fetched from the Modal volume) and the router, with their stamps; the
per-expert files below stay local. [`hf_manifest.json`](hf_manifest.json) maps each local file to its
path in that repo. [`make_checksums.py`](make_checksums.py) writes [`SHA256SUMS`](SHA256SUMS),
[`verify_lenses.py`](verify_lenses.py) checks a downloaded copy (hashes, shapes, configs), and
[`stage_for_browser.py`](stage_for_browser.py) builds a flat upload folder (hard links, outside OneDrive) for
the website's "Upload files" page, and [`upload_to_hf.py`](upload_to_hf.py) uploads from the command line in one commit (dry run by default; it refuses while
`TODO-` placeholders remain in the card or [`NOTICE_HF.md`](NOTICE_HF.md)).

| File | Lens | Size | Fit |
|---|---|---|---|
| `j_200p.pt` | **J-lens**, 200 prompts | 1.2 GB (fp32) | plain Jacobian, WikiText, target = final block, 64 rows/pass, 5 parallel shards |
| `r_moe_25p.pt` | **R-lens**, 25 prompts | 1.2 GB | LRP `r+moe` with bit-exact Sarvam rules |
| `jpp_25p.pt` | **J++ lens**, 25 prompts | 1.2 GB | learned weights over the R-lens's 8 expert Jacobians; multilingual eval excluded |
| `j_indic_27p.pt` | **Indic-fit J-lens**, 27 prompts | 1.2 GB (fp32) | plain Jacobian on Wikipedia passages in en + 8 Indian languages (doc 09 §9) |
| `j_200p_experts.pt` | J-lens **expert Jacobians** | 5.4 GB (fp16) | 8 per-cluster Jacobians per layer + pooled + position counts. **0/144** experts fell back to pooled. Raw material for a J++ built on plain J |
| `r_moe_25p_experts.pt` | R-lens expert Jacobians | 5.4 GB (fp16) | the input `jpp_25p.pt` was built from. **16/144** layer-expert slots had < 50 positions and fell back to the pooled Jacobian |
| `*_stamp.json` | fit provenance | small | exact arguments, git hash, timestamp of each fit stage |
| `jpp_25p_expert_weights.json` | J++ weights | small | the per-layer expert weights and their fitting curve |

Each lens covers source layers 0–17 of Sarvam-30B's 19 blocks, with d_model 4096, in jpp's lens
format.

Load the expert files with `workspace_lens.fitting.condense_experts.expert_jacobians.ExpertJacobians.load(path)`.
For a quick look without loading 5.4 GB into memory, use `torch.load(path, weights_only=True, mmap=True)`.
Positions per expert at layer 9: J-200 676–6,360; R-25 57–816.

## Using a lens

```python
import sys; sys.path[:0] = ["/path/to/jpp_lens/src", "/path/to/Sarvam Lens/code"]
import run_jpp; jpp_cli = run_jpp._import_jpp_cli(); run_jpp.install(jpp_cli)
import sarvam_support
from workspace_lens.utils import load_lens_file

model = sarvam_support.load_sarvam_lens_model("sarvamai/sarvam-30b", device_map="cuda")  # RoPE + BOS fixes applied
lens = load_lens_file("j_200p.pt", hf_model_name="sarvamai/sarvam-30b")
logits, model_logits, ids = lens.apply(model, 'தமிழ்: "நீர்" - తెలుగు: "', layers=[14, 15, 16],
                                       token_positions_for_residuals=[-1])
```

Load the model through `sarvam_support`. A stock transformers-5 load leaves RoPE uninitialized (usually zeros), and every
readout would be wrong.

To halve the file sizes, re-save at fp16. The entries are O(1), so fp16 is safe; it's `jlens`'s
default.
