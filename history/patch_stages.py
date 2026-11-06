"""One-off patch: lens-fit/J++/pivot stages in modal_pilot.py and Qwen support in modal_app.py."""
import sys
from pathlib import Path

root = Path(sys.argv[1])

p = root / "modal_pilot.py"
s = p.read_text(encoding="utf-8")
s = s.replace('''              stages: list[str] | None = None, tag: str = "pilot") -> dict:''',
              '''              stages: list[str] | None = None, tag: str = "pilot", fit_prompts: int = 2) -> dict:''')
s = s.replace('''    ``stages`` picks a subset (in the fixed order below); ``tag`` names the output dir."""''',
              '''    ``stages`` picks a subset (in the fixed order below); ``tag`` names the output dir;
    ``fit_prompts`` sizes the real lens fit (``fit_r_lens``)."""''')
old = '''    plan = (
        ("load", 5, load),'''
new = '''    import glob

    lens_dir = runs / "lenses"
    r_name = f"r_moe_{fit_prompts}p"
    r_experts, r_pooled, jpp_lens = (lens_dir / f"{r_name}_experts.pt", lens_dir / f"{r_name}.pt",
                                     lens_dir / f"jpp_{fit_prompts}p.pt")

    def fit_r_lens() -> dict:
        """The real R+MoE fit: ``fit_prompts`` WikiText prompts, merged into expert Jacobians
        (for J++) and their pooled average (the R-Lens), both saved to the Volume."""
        art, fit_start = f"/tmp/fit_{r_name}", time.monotonic()
        code = run_jpp.main(["fit-shard", "--hf-model-name", model_dir, *device_args, "--layers", layers(),
                             "--router-path", str(router_path), "--lrp-mode", "r+moe", "--num-prompts", str(fit_prompts),
                             "--checkpoint-every-n-prompts", "1000", "--artifacts-base-dir", art, "--checkpoint-name", r_name])
        fit_minutes = (time.monotonic() - fit_start) / 60
        shard_dirs = sorted(glob.glob(f"{art}/*/{r_name}/shard*"))
        lens_dir.mkdir(parents=True, exist_ok=True)
        merged = run_jpp.main(["merge-experts", "--checkpoint-dirs", *shard_dirs, "--checkpoint-name", r_name,
                               "--out", str(r_experts), "--pooled-lens-out", str(r_pooled)])
        commit()
        return {"fit_exit": code, "merge_exit": merged, "fit_minutes": round(fit_minutes, 1),
                "seconds_per_prompt": round(fit_minutes * 60 / fit_prompts, 1), "pooled_lens": str(r_pooled)}

    def fit_jpp() -> dict:
        """J++ weights over the R+MoE experts, learned WITHOUT the multilingual eval (whose
        English-target items would tune the lens to find English in non-English prompts)."""
        evals = "/tmp/evals_no_multilingual"
        run_jpp.main(["prepare-evals", "--out", evals])
        code = run_jpp.main(["fit-weights", "--hf-model-name", model_dir, *device_args, "--layers", layers(),
                             "--experts", str(r_experts), "--checkpoint-name", f"jpp_{fit_prompts}p",
                             "--eval-data-dir", evals, "--correctness-csv", str(runs / "correctness.csv"),
                             "--out", str(jpp_lens)])
        commit()
        return {"exit": code, "lens": str(jpp_lens)}

    def pivot_lenses() -> dict:
        lens_args = ["--lens", "logit"]
        for path in (r_pooled, jpp_lens):
            if path.is_file():
                lens_args += ["--lens", str(path)]
        out = pilot_dir / "pivot_lenses"
        code = pivot_experiment.main(["--hf-model-name", model_dir, *device_args, *lens_args, "--out", str(out)])
        peaks: dict[str, dict] = {}
        summary_path = out / "summary.csv"
        if summary_path.is_file():
            with summary_path.open(encoding="utf-8") as handle:
                for r in csv.DictReader(handle):
                    if r["pair"] == "ALL_INDIC":
                        peaks.setdefault(r["lens"], {}).setdefault(r["role"], []).append(round(float(r["mean_prob"]), 3))
        return {"exit": code, "lenses": lens_args[1::2], "all_indic_mean_prob_by_layer": peaks}

    plan = (
        ("load", 5, load),'''
assert old in s, "plan head not found"
s = s.replace(old, new)
old2 = '''        ("fit_shard_r", 8, fit_shard("r+moe")),
        ("grade", 5, grade),
    )'''
new2 = '''        ("fit_shard_r", 8, fit_shard("r+moe")),
        ("fit_r_lens", 1.25 * fit_prompts + 3, fit_r_lens),
        ("fit_jpp", 6, fit_jpp),
        ("pivot_lenses", 6, pivot_lenses),
        ("grade", 5, grade),
    )'''
assert old2 in s, "plan tail not found"
s = s.replace(old2, new2)
p.write_text(s, encoding="utf-8")

p = root / "modal_app.py"
s = p.read_text(encoding="utf-8")
s = s.replace('MODEL_DIR = f"{VOLUME_ROOT}/sarvam-30b-bf16"',
              'MODEL_DIR = f"{VOLUME_ROOT}/sarvam-30b-bf16"\n'
              'QWEN_REPO = "Qwen/Qwen3.6-35B-A3B"\n'
              'QWEN_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"\n'
              'QWEN_DIR = f"{VOLUME_ROOT}/qwen3.6-35b-a3b"\n'
              'MODEL_DIRS = {"sarvam": MODEL_DIR, "qwen": QWEN_DIR}')
s = s.replace("PILOT_TIMEOUT_MIN = 45", "PILOT_TIMEOUT_MIN = 60")
old_pilot = s[s.index('@app.function(image=image, gpu="H200"'):]
new_pilot = '''@app.function(image=image, volumes={VOLUME_ROOT: volume}, cpu=2, memory=8192, timeout=40 * 60, retries=0)
def prepare_qwen() -> dict:
    """The Qwen baseline's checkpoint (72 GB, already bf16 on the Hub) onto the Volume."""
    import time

    from huggingface_hub import snapshot_download

    start = time.monotonic()
    snapshot_download(QWEN_REPO, revision=QWEN_REVISION, local_dir=QWEN_DIR,
                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])
    volume.commit()
    size = sum(p.stat().st_size for p in Path(QWEN_DIR).glob("*.safetensors"))
    return {"gb": round(size / 1e9, 1), "minutes": round((time.monotonic() - start) / 60, 1)}


@app.function(image=image, gpu="H200", volumes={VOLUME_ROOT: volume}, cpu=PILOT_CPU,
              memory=PILOT_MEMORY_MIB, timeout=PILOT_TIMEOUT_MIN * 60, retries=0)
def pilot(budget_minutes: float = 38.0, stages: list[str] | None = None, tag: str = "pilot",
          model: str = "sarvam", fit_prompts: int = 2) -> dict:
    import sys

    sys.path.insert(0, "/sarvam-lens")
    import modal_pilot

    return modal_pilot.run_pilot(MODEL_DIRS[model], RUNS_DIR, budget_minutes=budget_minutes, commit=volume.commit,
                                 stages=stages, tag=tag, fit_prompts=fit_prompts)


@app.local_entrypoint()
def main(step: str = "pilot", budget_minutes: float = 38.0, stages: str = "", tag: str = "pilot",
         model: str = "sarvam", fit_prompts: int = 2) -> None:
    """``--stages load,bos_probe,pivot_logit`` runs a subset; the container's hard timeout is
    then cut to budget + 6 min, so a hang can never cost more than that."""
    if step == "prepare":
        print(json.dumps(prepare_weights.remote(), indent=1))
    elif step == "prepare-qwen":
        print(json.dumps(prepare_qwen.remote(), indent=1))
    elif step == "smoke":
        print(json.dumps(smoke.remote(), indent=1, ensure_ascii=False))
    elif step == "pilot":
        if model not in MODEL_DIRS:
            raise SystemExit(f"--model must be one of {sorted(MODEL_DIRS)}")
        if budget_minutes > PILOT_TIMEOUT_MIN - 6:
            raise SystemExit(f"--budget-minutes must leave >= 6 min under the {PILOT_TIMEOUT_MIN}-min timeout")
        timeout_s = int((budget_minutes + 6) * 60)
        print(f"hard timeout {timeout_s / 60:.0f} min -> worst case ${timeout_s / 3600 * 5.17:.2f}")
        selected = [name for name in stages.split(",") if name] or None
        result = pilot.with_options(timeout=timeout_s).remote(budget_minutes, selected, tag, model, fit_prompts)
        print(json.dumps(result, indent=1, ensure_ascii=False))
    else:
        raise SystemExit(f"unknown --step {step!r} (prepare | prepare-qwen | smoke | pilot)")
'''
s = s.replace(old_pilot, new_pilot)
s = s.replace("    modal run modal_app.py --step smoke        # CPU, cents: smoke test + dry pilot in this image",
              "    modal run modal_app.py --step smoke        # CPU, cents: smoke test + dry pilot in this image\n"
              "    modal run modal_app.py --step prepare-qwen # CPU: the Qwen3.6-35B-A3B baseline checkpoint onto the Volume")
p.write_text(s, encoding="utf-8")
print("patched")
