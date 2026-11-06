# history/: one-off patch scripts

Two scripts used during development to apply multi-file edits in one go. Their changes are already
part of `code/`, so they're kept only as provenance and **should not be rerun**.

| Script | What it added |
|---|---|
| `patch_stages.py` | The `fit_r_lens`, `fit_jpp`, `pivot_lenses` stages; Qwen support and `prepare-qwen` in the Modal app |
| `patch_jfit.py` | The `rows_probe` and sharded `fit_j_shard` stages; the CPU `merge_j` and `diagnose` Modal functions |
