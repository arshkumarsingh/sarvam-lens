"""One-off patch: rows-per-pass probe, sharded J-lens fit, CPU merge, CKA diagnostic."""
import sys
from pathlib import Path

root = Path(sys.argv[1])

# ---- modal_pilot.py --------------------------------------------------------------
p = root / "modal_pilot.py"
s = p.read_text(encoding="utf-8")
s = s.replace(
    '''              stages: list[str] | None = None, tag: str = "pilot", fit_prompts: int = 2) -> dict:''',
    '''              stages: list[str] | None = None, tag: str = "pilot", fit_prompts: int = 2,
              fit_rows: int = 16, shard_idx: int = 0, num_shards: int = 1) -> dict:''')
s = s.replace(
    '''    ``fit_prompts`` sizes the real lens fit (``fit_r_lens``)."""''',
    '''    ``fit_prompts`` sizes the real lens fits; ``fit_rows`` is jpp's --jacobian-rows-per-pass;
    ``shard_idx``/``num_shards`` split ``fit_j_shard`` across containers."""''')
old = '''    plan = (
        ("load", 2, load),'''
new = '''    def rows_probe() -> dict:
        """Seconds/prompt and peak GPU memory of a 2-prompt J fit at 32 and 64 Jacobian rows
        per backward pass (jpp default 16: 256 passes per prompt on d_model 4096)."""
        out: dict = {}
        for rows in (32, 64):
            t.cuda.empty_cache()
            t.cuda.reset_peak_memory_stats()
            probe_start = time.monotonic()
            try:
                code = run_jpp.main(["fit-shard", "--hf-model-name", model_dir, *device_args, "--layers", layers(),
                                     "--router-path", str(router_path), "--lrp-mode", "none", "--num-prompts", "2",
                                     "--jacobian-rows-per-pass", str(rows), "--checkpoint-every-n-prompts", "1000",
                                     "--artifacts-base-dir", f"/tmp/rows_{rows}", "--checkpoint-name", f"rows_{rows}"])
                out[rows] = {"exit": code, "seconds_per_prompt": round((time.monotonic() - probe_start) / 2, 1),
                             "peak_gpu_gb": round(t.cuda.max_memory_allocated() / 1e9, 1)}
            except BaseException as exc:  # OOM at the larger row count is an answer, not a failure
                out[rows] = {"error": repr(exc)[:300]}
            t.cuda.empty_cache()
        return out

    j_name = f"j_{fit_prompts}p"

    def fit_j_shard() -> dict:
        """One shard of the plain J-lens fit (lrp-mode none); checkpoints on the Volume so the
        CPU ``merge_j`` step can sum all shards."""
        shard_start = time.monotonic()
        code = run_jpp.main(["fit-shard", "--hf-model-name", model_dir, *device_args, "--layers", layers(),
                             "--router-path", str(router_path), "--lrp-mode", "none",
                             "--num-prompts", str(fit_prompts), "--num-shards", str(num_shards), "--shard-idx", str(shard_idx),
                             "--jacobian-rows-per-pass", str(fit_rows), "--checkpoint-every-n-prompts", "10",
                             "--artifacts-base-dir", str(runs / "jfit" / j_name), "--checkpoint-name", j_name])
        commit()
        minutes = (time.monotonic() - shard_start) / 60
        return {"exit": code, "shard": f"{shard_idx}/{num_shards}", "minutes": round(minutes, 1)}

    plan = (
        ("load", 2, load),'''
assert old in s, "plan head"
s = s.replace(old, new)
old = '''        ("fit_r_lens", 1.25 * fit_prompts + 3, fit_r_lens),'''
new = '''        ("rows_probe", 6, rows_probe),
        ("fit_j_shard", 1.3 * fit_prompts / num_shards + 2, fit_j_shard),
        ("fit_r_lens", 1.25 * fit_prompts + 3, fit_r_lens),'''
assert old in s, "plan rows"
s = s.replace(old, new)
p.write_text(s, encoding="utf-8")

# ---- modal_app.py ----------------------------------------------------------------
p = root / "modal_app.py"
s = p.read_text(encoding="utf-8")
old = '''def pilot(budget_minutes: float = 38.0, stages: list[str] | None = None, tag: str = "pilot",
          model: str = "sarvam", fit_prompts: int = 2) -> dict:
    import sys

    sys.path.insert(0, "/sarvam-lens")
    import modal_pilot

    return modal_pilot.run_pilot(MODEL_DIRS[model], RUNS_DIR, budget_minutes=budget_minutes, commit=volume.commit,
                                 stages=stages, tag=tag, fit_prompts=fit_prompts)'''
new = '''def pilot(budget_minutes: float = 38.0, stages: list[str] | None = None, tag: str = "pilot",
          model: str = "sarvam", fit_prompts: int = 2, fit_rows: int = 16, shard_idx: int = 0,
          num_shards: int = 1) -> dict:
    import sys

    sys.path.insert(0, "/sarvam-lens")
    import modal_pilot

    return modal_pilot.run_pilot(MODEL_DIRS[model], RUNS_DIR, budget_minutes=budget_minutes, commit=volume.commit,
                                 stages=stages, tag=tag, fit_prompts=fit_prompts, fit_rows=fit_rows,
                                 shard_idx=shard_idx, num_shards=num_shards)


@app.function(image=image, volumes={VOLUME_ROOT: volume}, cpu=8, memory=65536, timeout=40 * 60, retries=0)
def merge_j(fit_prompts: int) -> dict:
    """CPU: sum the J-lens shards into experts + the pooled J-Lens (lenses/j_<n>p.pt)."""
    import glob
    import os
    import sys

    sys.path.insert(0, "/sarvam-lens")
    os.chdir("/jpp")
    import run_jpp

    name = f"j_{fit_prompts}p"
    shard_dirs = sorted(glob.glob(f"{RUNS_DIR}/jfit/{name}/*/{name}/shard*"))
    code = run_jpp.main(["merge-experts", "--checkpoint-dirs", *shard_dirs, "--checkpoint-name", name, "--device", "cpu",
                         "--out", f"{RUNS_DIR}/lenses/{name}_experts.pt", "--pooled-lens-out", f"{RUNS_DIR}/lenses/{name}.pt"])
    volume.commit()
    return {"exit": code, "shards": shard_dirs}


@app.function(image=image, volumes={VOLUME_ROOT: volume}, cpu=8, memory=32768, timeout=30 * 60, retries=0)
def diagnose() -> dict:
    """CPU: why Sarvam's CKA is degenerate (gain outliers, spectrum, CKA variants)."""
    import sys

    sys.path.insert(0, "/sarvam-lens")
    import cka_diagnose
    import run_jpp

    run_jpp._import_jpp_cli()
    specs = [
        {"name": "sarvam_R25", "lens": f"{RUNS_DIR}/lenses/r_moe_25p.pt", "model_dir": MODEL_DIR},
        {"name": "qwen_J25", "lens": f"{VOLUME_ROOT}/ext_lenses/qwen3.6-35b-a3b/j-lens/lens.pt", "model_dir": QWEN_DIR},
    ]
    result = cka_diagnose.run(specs, f"{RUNS_DIR}/cka", None, "/ojd/code/jlens_cka")
    volume.commit()
    return result'''
assert old in s, "pilot fn"
s = s.replace(old, new)
old = '''def main(step: str = "pilot", budget_minutes: float = 38.0, stages: str = "", tag: str = "pilot",
         model: str = "sarvam", fit_prompts: int = 2) -> None:'''
new = '''def main(step: str = "pilot", budget_minutes: float = 38.0, stages: str = "", tag: str = "pilot",
         model: str = "sarvam", fit_prompts: int = 2, fit_rows: int = 16, num_shards: int = 1) -> None:'''
assert old in s, "main sig"
s = s.replace(old, new)
old = '''        selected = [name for name in stages.split(",") if name] or None
        result = pilot.with_options(timeout=timeout_s).remote(budget_minutes, selected, tag, model, fit_prompts)
        print(json.dumps(result, indent=1, ensure_ascii=False))'''
new = '''        selected = [name for name in stages.split(",") if name] or None
        fn = pilot.with_options(timeout=timeout_s)
        if num_shards > 1:  # one container per shard, in parallel; same total cost, 1/N wall time
            print(f"{num_shards} parallel shards -> worst case ${num_shards * timeout_s / 3600 * 5.17:.2f}")
            calls = [fn.spawn(budget_minutes, selected, f"{tag}_s{i}", model, fit_prompts, fit_rows, i, num_shards)
                     for i in range(num_shards)]
            for call in calls:
                print(json.dumps(call.get(), indent=1, ensure_ascii=False))
        else:
            result = fn.remote(budget_minutes, selected, tag, model, fit_prompts, fit_rows)
            print(json.dumps(result, indent=1, ensure_ascii=False))'''
assert old in s, "main pilot body"
s = s.replace(old, new)
old = '''    elif step == "cka":'''
new = '''    elif step == "merge-j":
        print(json.dumps(merge_j.remote(fit_prompts), indent=1))
    elif step == "diagnose":
        print(json.dumps(diagnose.remote(), indent=1, default=float))
    elif step == "cka":'''
assert old in s, "cka step"
s = s.replace(old, new)
s = s.replace('raise SystemExit(f"unknown --step {step!r} (prepare | prepare-qwen | smoke | cka | pilot)")',
              'raise SystemExit(f"unknown --step {step!r} (prepare | prepare-qwen | smoke | cka | diagnose | merge-j | pilot)")')
p.write_text(s, encoding="utf-8")
print("patched")
