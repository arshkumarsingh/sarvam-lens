"""Run the Sarvam lens pilot on Modal, under a hard cost cap.

    modal run modal_app.py --step prepare      # CPU: fp32 Hub checkpoint -> bf16 copy on a Volume (once)
    modal run modal_app.py --step smoke        # CPU, cents: smoke test + dry pilot in this image
    modal run modal_app.py --step smoke --stages load,sanity,patch_rank   # the same, dry pilot of just these
    modal run modal_app.py --step prepare-qwen # CPU: the Qwen3.6-35B-A3B baseline checkpoint onto the Volume
    modal run modal_app.py --step cka          # CPU: CKA-explorer block structure, Sarvam vs Qwen
    modal run modal_app.py --step pilot        # H200: real-model checks, timing fits, logit-lens pivot
    modal volume get sarvam-lens runs/pilot ./pilot_results

Activation patching on Sarvam-30B (J-lens directions; needs runs/lenses/j_200p.pt on the Volume):
    modal run modal_app.py --step pilot --model sarvam --stages load,sanity,patch --tag patch --budget-minutes 12
    modal run modal_app.py --step pilot --model sarvam --stages load,sanity,patch_split --tag patch2 \
        --budget-minutes 7.6 --timeout-margin 2    # the follow-up, under a $1 cap (worst case $0.83)

Sarvam-105B (2xH200, logit lens only):
    modal run modal_app.py --step prepare-105b # CPU: 424 GB fp32 -> 212 GB bf16 on the Volume (once)
    modal run modal_app.py --step smoke-2gpu   # 2xL4, cents: a tiny 105B-shaped model split over 2 GPUs
    modal run modal_app.py --step pilot --model sarvam105 --budget-minutes 24 \
        --stages load,sanity,bos_probe,pivot_logit,causal_logit --bos-modes sarvam_bos,chat --tag sarvam105

Cost controls: each function has a hard ``timeout`` (Modal kills the container), retries
are off, and the pilot only starts a stage when its estimate fits the remaining budget.
Worst case at Modal's list prices (2026-10-18): prepare <= 90 min x $0.45/h, pilot
<= 54 min x $5.17/h (1xH200) or x $9.71/h (2xH200).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
JPP_COMMIT = "d7efc0de372e20589e899188c0cfe8434acc63ec"
OJD_COMMIT = "49010193dee265e8521dda041a26d23d7a8bc027"  # eliebak/open-jlens-data (CKA explorer code)
SARVAM_REPO = "sarvamai/sarvam-30b"
SARVAM_REVISION = "071ae95e933605ca1104a6b4524a36a98488efa4"
VOLUME_ROOT = "/vol"
MODEL_DIR = f"{VOLUME_ROOT}/sarvam-30b-bf16"
QWEN_REPO = "Qwen/Qwen3.6-35B-A3B"
QWEN_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
QWEN_DIR = f"{VOLUME_ROOT}/qwen3.6-35b-a3b"
SARVAM105_REPO = "sarvamai/sarvam-105b"
SARVAM105_REVISION = "abe264bdb8fb2244daadaa59ebc2020b3c139e89"
SARVAM105_DIR = f"{VOLUME_ROOT}/sarvam-105b-bf16"
MODEL_DIRS = {"sarvam": MODEL_DIR, "qwen": QWEN_DIR, "sarvam105": SARVAM105_DIR}
# Models too big for one H200 run in the 2xH200 function with device_map="auto".
TWO_GPU_MODELS = {"sarvam105"}
RUNS_DIR = f"{VOLUME_ROOT}/runs"

PILOT_TIMEOUT_MIN = 60
PILOT_CPU, PILOT_MEMORY_MIB = 8, 32768
PREPARE_TIMEOUT_MIN = 90
# Modal list prices on 2026-10-18 ($/h): H200 4.54, L4 0.80, CPU 0.0472/core, memory 0.0080/GiB.
_CPU_MEM_USD_PER_HOUR = PILOT_CPU * 0.0472 + PILOT_MEMORY_MIB / 1024 * 0.0080
USD_PER_HOUR = {1: 4.54 + _CPU_MEM_USD_PER_HOUR, 2: 2 * 4.54 + _CPU_MEM_USD_PER_HOUR}  # by H200 count
SMOKE2_USD_PER_HOUR = 2 * 0.80 + 2 * 0.0472 + 8 * 0.0080
SMOKE_USD_PER_HOUR = 4 * 0.0472 + 16 * 0.0080  # the CPU smoke container
SMOKE_STAGES_TIMEOUT_MIN = 15  # `--step smoke --stages ...`

app = modal.App("sarvam-lens")
volume = modal.Volume.from_name("sarvam-lens", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    # jpp_lens' pins (its pyproject at JPP_COMMIT), plus plotting.
    .pip_install(
        "transformers==5.14.1", "huggingface_hub==1.26.0", "numpy==2.5.1", "pandas==3.0.5",
        "datasets==5.0.1", "pyarrow==25.0.0", "accelerate==1.12.0", "safetensors", "matplotlib",
    )
    .run_commands(f"git clone https://github.com/safety-research/jpp_lens /jpp && cd /jpp && git checkout {JPP_COMMIT}")
    .run_commands(f"git clone https://github.com/eliebak/open-jlens-data /ojd && cd /ojd && git checkout {OJD_COMMIT}")
    .env({"HF_HOME": f"{VOLUME_ROOT}/hf", "JPP_DIR": "/jpp", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(HERE, "/sarvam-lens", ignore=["__pycache__", "pilot_results", "runs", "*.png"])
)


@app.function(image=image, volumes={VOLUME_ROOT: volume}, cpu=4, memory=32768,
              timeout=PREPARE_TIMEOUT_MIN * 60, retries=0)
def prepare_weights(repo: str = SARVAM_REPO, revision: str = SARVAM_REVISION, out_dir: str = MODEL_DIR) -> dict:
    """Download the fp32 shards one at a time to local disk, cast floats to bf16, and write
    them to the Volume with the original names, so the index's weight map still holds.
    Resumable: a shard already on the Volume is kept (a partial one never is: each is written
    under a temporary name, then renamed). The model's two code files are copied beside the
    weights, which is where the 105B loader imports them from."""
    import os
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from safetensors.torch import load_file, save_file

    start = time.monotonic()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = HfApi().list_repo_files(repo, revision=revision)
    shards = sorted(f for f in files if f.endswith(".safetensors"))
    code_files = ("configuration_sarvam_moe.py", "modeling_sarvam_moe.py")
    for name in files:
        if name.endswith((".json", ".jinja")) or name in code_files:
            shutil.copy(hf_hub_download(repo, name, revision=revision), out / name)

    def convert(name: str) -> int:
        if (out / name).is_file():
            return (out / name).stat().st_size
        raw = hf_hub_download(repo, name, revision=revision, local_dir="/tmp/raw")
        tensors = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in load_file(raw).items()}
        partial = out / f"{name}.partial"
        save_file(tensors, str(partial), metadata={"format": "pt"})
        del tensors
        os.replace(partial, out / name)
        os.remove(raw)
        return (out / name).stat().st_size

    sizes = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(convert, name) for name in shards]
        for done, future in enumerate(as_completed(futures), 1):
            sizes.append(future.result())
            if done % 10 == 0:  # so a timeout or crash keeps the shards written so far
                volume.commit()
                print(f"{done}/{len(shards)} shards, {sum(sizes) / 1e9:.0f} GB, {(time.monotonic() - start) / 60:.1f} min", flush=True)

    config = json.loads((out / "config.json").read_text())
    config["dtype"] = "bfloat16"
    (out / "config.json").write_text(json.dumps(config, indent=2))
    index_path = out / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index.setdefault("metadata", {})["total_size"] = sum(sizes)
    index_path.write_text(json.dumps(index, indent=2))
    volume.commit()
    return {"shards": len(shards), "bf16_gb": round(sum(sizes) / 1e9, 1), "minutes": round((time.monotonic() - start) / 60, 1)}


@app.function(image=image, cpu=4, memory=16384, timeout=25 * 60, retries=0)
def smoke(stages: list[str] | None = None, stage_args: dict | None = None) -> dict:
    """CPU-only, a few cents: the tiny-model smoke test, then the whole pilot driver on that
    tiny model, in this exact image. It catches container-only breakage before H200 time.
    ``stages`` limits the dry pilot to those stages and skips the 105B part. The smoke test's tiny
    J-lens stands in for the fitted lens, so a patch stage runs without the dry pilot's fits."""
    import shutil
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "/sarvam-lens/smoke_test.py", "--workdir", "/tmp/smoke"],
                          cwd="/sarvam-lens", capture_output=True, text=True)
    checks = [line for line in proc.stdout.splitlines() if line.startswith(("  ok", "  FAIL", "ALL CHECKS"))]
    if proc.returncode != 0:
        return {"smoke_test": "FAILED", "checks": checks, "stderr_tail": proc.stderr[-3000:]}
    sys.path.insert(0, "/sarvam-lens")
    import modal_pilot

    if stages:
        Path("/tmp/dry_runs/lenses").mkdir(parents=True, exist_ok=True)
        shutil.copy("/tmp/smoke/tiny_j_lens.pt", "/tmp/dry_runs/lenses/j_200p.pt")
    report = modal_pilot.run_pilot("/tmp/smoke/tiny-sarvam", "/tmp/dry_runs", budget_minutes=20,
                                   commit=lambda: None, device_map=None, stages=stages, stage_args=stage_args)
    stages_run = {s["stage"]: s["status"] for s in report["stages"]}
    if stages:
        return {"smoke_test": "passed", "n_checks": sum(line.startswith("  ok") for line in checks),
                "dry_pilot_stages": stages_run,
                "dry_pilot_results": {s["stage"]: {k: v for k, v in (s.get("result") or {}).items()
                                                   if k in ("lens", "n_items", "conditions")} if "." not in s["stage"]
                                      else str(s.get("result"))[:1500] for s in report["stages"]},
                "dry_pilot_errors": {s["stage"]: s.get("traceback", "")[-1500:] for s in report["stages"] if s["status"] != "ok"}}
    # Sarvam-105B: the tiny MLA model through the real adapter and the 105B stages, on CPU.
    mla = subprocess.run([sys.executable, "/sarvam-lens/smoke_test_mla.py", "--workdir", "/tmp/smoke_mla"],
                         cwd="/sarvam-lens", capture_output=True, text=True)
    mla_checks = [line for line in mla.stdout.splitlines() if line.startswith(("  ok", "  FAIL", "ALL CHECKS"))]
    return {"smoke_test": "passed", "n_checks": len(checks), "dry_pilot_stages": stages_run,
            "dry_pilot_errors": {s["stage"]: s.get("traceback", "")[-1500:] for s in report["stages"] if s["status"] != "ok"},
            "mla_smoke": "passed" if mla.returncode == 0 else "FAILED", "mla_checks": mla_checks,
            "mla_stderr_tail": "" if mla.returncode == 0 else mla.stderr[-3000:]}


@app.function(image=image, gpu="L4:2", cpu=2, memory=8192, timeout=15 * 60, retries=0)
def smoke_2gpu() -> dict:
    """2xL4, ~$0.15: the tiny 105B-shaped model split over two GPUs with an explicit device map
    (embeddings + first layer on GPU 0, the rest + norm + LM head on GPU 1, as device_map="auto"
    places the real model), through every 105B stage. It catches cross-device bugs that the CPU
    smoke test cannot."""
    import subprocess
    import sys

    proc = subprocess.run([sys.executable, "/sarvam-lens/smoke_test_mla.py", "--workdir", "/tmp/smoke_mla", "--two-gpu"],
                          cwd="/sarvam-lens", capture_output=True, text=True)
    checks = [line for line in proc.stdout.splitlines() if line.startswith(("  ok", "  FAIL", "ALL CHECKS"))]
    return {"mla_smoke_2gpu": "passed" if proc.returncode == 0 else "FAILED", "checks": checks,
            "stdout_tail": proc.stdout[-2500:], "stderr_tail": "" if proc.returncode == 0 else proc.stderr[-4000:]}


@app.function(image=image, gpu="L4:2", cpu=2, memory=8192, timeout=15 * 60, retries=0)
def post1_decode_2gpu() -> dict:
    """2xL4, ~$0.15: post1_check_decode.py on the tiny 105B split over two GPUs (fp32): the per-GPU KV slab, the absorbed
    MLA decode step and the in-slab forced answers must match lone decoding token for token (doc 17, 105B #7/#8)."""
    import subprocess
    import sys

    sys.path.insert(0, "/sarvam-lens")
    import smoke_test_mla

    tiny_dir = smoke_test_mla.build_tiny_mla(Path("/tmp/p1mla"))[0]
    proc = subprocess.run([sys.executable, "/sarvam-lens/post1_check_decode.py", "/sarvam-lens", "-", str(tiny_dir), "--two-gpu"],
                          cwd="/sarvam-lens", capture_output=True, text=True)
    lines = [l for l in proc.stdout.splitlines() if l.startswith(("  ok", "  FAIL", "FAILS", "tiny"))]
    return {"exit": proc.returncode, "checks": lines, "stderr_tail": "" if proc.returncode == 0 else proc.stderr[-3000:]}


@app.function(image=image, gpu="L4:2", cpu=2, memory=8192, timeout=15 * 60, retries=0)
def batch_probe() -> dict:
    """2xL4, ~$0.05: batched vs lone runs on the tiny 105B (split over both GPUs) and the tiny 30B,
    fp32 and bf16, with and without padding (batch_probe.py): a padding bug or bf16 batch-variance?"""
    import sys

    sys.path.insert(0, "/sarvam-lens")
    import batch_probe as probe

    return probe.main()


@app.function(image=image, volumes={VOLUME_ROOT: volume}, cpu=2, memory=8192, timeout=40 * 60, retries=0)
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


@app.function(image=image, volumes={VOLUME_ROOT: volume}, cpu=8, memory=32768, timeout=25 * 60, retries=0)
def cka() -> dict:
    """CPU only: the CKA explorer's within-model block structure for Sarvam's R-lens and J++,
    calibrated by Qwen3.6-35B-A3B's published J- and R-lens, plus Sarvam x Qwen matched depth."""
    import sys

    from huggingface_hub import hf_hub_download

    sys.path.insert(0, "/sarvam-lens")
    import cka_compare
    import run_jpp

    run_jpp._import_jpp_cli()  # puts jpp's src/ on sys.path (lens loading)
    qwen_lens = {kind: hf_hub_download("camilablank/workspace-lenses", f"qwen3.6-35b-a3b/{kind}/lens.pt",
                                       local_dir=f"{VOLUME_ROOT}/ext_lenses") for kind in ("j-lens", "r-lens")}
    specs = [
        {"name": "sarvam_J200", "lens": f"{RUNS_DIR}/lenses/j_200p.pt", "model_dir": MODEL_DIR, "n_layers": 19},
        {"name": "sarvam_R25", "lens": f"{RUNS_DIR}/lenses/r_moe_25p.pt", "model_dir": MODEL_DIR, "n_layers": 19},
        {"name": "sarvam_JPP25", "lens": f"{RUNS_DIR}/lenses/jpp_25p.pt", "model_dir": MODEL_DIR, "n_layers": 19},
        {"name": "qwen_J25", "lens": qwen_lens["j-lens"], "model_dir": QWEN_DIR, "n_layers": 40},
        {"name": "qwen_R25", "lens": qwen_lens["r-lens"], "model_dir": QWEN_DIR, "n_layers": 40},
    ]
    result = cka_compare.run(specs, f"{RUNS_DIR}/cka", cross=[("sarvam_J200", "qwen_J25"), ("sarvam_R25", "qwen_R25")])
    volume.commit()
    return result


def _run_pilot(budget_minutes: float, stages: list[str] | None, tag: str, model: str, fit_prompts: int,
               fit_rows: int, shard_idx: int, num_shards: int, *, device_map: str, usd_per_hour: float,
               bos_modes: list[str] | None, spf_guess: float, extra_lenses: list[str] | None = None,
               fit_name: str = "", prompts_json: str = "", stage_args: dict | None = None) -> dict:
    import os
    import sys

    sys.path.insert(0, "/sarvam-lens")
    import modal_pilot

    if prompts_json:  # fit-router / fit-shard draw their prompts from this JSON list (run_jpp.FIT_PROMPTS_ENV)
        os.environ["JPP_FIT_PROMPTS_JSON"] = prompts_json
    return modal_pilot.run_pilot(MODEL_DIRS[model], RUNS_DIR, budget_minutes=budget_minutes, commit=volume.commit,
                                 device_map=device_map, stages=stages, tag=tag, fit_prompts=fit_prompts,
                                 fit_rows=fit_rows, shard_idx=shard_idx, num_shards=num_shards,
                                 usd_per_hour=usd_per_hour, bos_modes=bos_modes, seconds_per_forward_guess=spf_guess,
                                 extra_lenses=extra_lenses, fit_name=fit_name or None, stage_args=stage_args)


@app.function(image=image, gpu="H200", volumes={VOLUME_ROOT: volume}, cpu=PILOT_CPU,
              memory=PILOT_MEMORY_MIB, timeout=PILOT_TIMEOUT_MIN * 60, retries=0)
def pilot(budget_minutes: float = 38.0, stages: list[str] | None = None, tag: str = "pilot",
          model: str = "sarvam", fit_prompts: int = 2, fit_rows: int = 16, shard_idx: int = 0,
          num_shards: int = 1, bos_modes: list[str] | None = None, spf_guess: float = 0.5,
          extra_lenses: list[str] | None = None, fit_name: str = "", prompts_json: str = "",
          stage_args: dict | None = None) -> dict:
    return _run_pilot(budget_minutes, stages, tag, model, fit_prompts, fit_rows, shard_idx, num_shards,
                      device_map="cuda", usd_per_hour=USD_PER_HOUR[1], bos_modes=bos_modes, spf_guess=spf_guess,
                      extra_lenses=extra_lenses, fit_name=fit_name, prompts_json=prompts_json, stage_args=stage_args)


@app.function(image=image, gpu="H200:2", volumes={VOLUME_ROOT: volume}, cpu=PILOT_CPU,
              memory=PILOT_MEMORY_MIB, timeout=PILOT_TIMEOUT_MIN * 60, retries=0)
def pilot2(budget_minutes: float = 24.0, stages: list[str] | None = None, tag: str = "pilot",
           model: str = "sarvam105", fit_prompts: int = 2, fit_rows: int = 16, shard_idx: int = 0,
           num_shards: int = 1, bos_modes: list[str] | None = None, spf_guess: float = 1.0,
           extra_lenses: list[str] | None = None, fit_name: str = "", prompts_json: str = "",
           stage_args: dict | None = None) -> dict:
    """``pilot`` on 2xH200 (282 GB) for Sarvam-105B's 212 GB of bf16 weights: device_map="auto"
    splits the decoder layers across both GPUs (``pilot``'s "cuda" would put all on one and OOM)."""
    return _run_pilot(budget_minutes, stages, tag, model, fit_prompts, fit_rows, shard_idx, num_shards,
                      device_map="auto", usd_per_hour=USD_PER_HOUR[2], bos_modes=bos_modes, spf_guess=spf_guess,
                      extra_lenses=extra_lenses, fit_name=fit_name, prompts_json=prompts_json, stage_args=stage_args)


@app.function(image=image, volumes={VOLUME_ROOT: volume}, cpu=8, memory=65536, timeout=40 * 60, retries=0)
def merge_j(fit_prompts: int, fit_name: str = "", out_name: str = "", shard_glob: str = "shard*") -> dict:
    """CPU: sum the J-lens shards into experts + the pooled J-Lens (lenses/<out_name>.pt).
    ``fit_name`` is the fit's checkpoint name (default j_<n>p); ``out_name`` the lens name (default:
    the same); ``shard_glob`` picks shard dirs, e.g. the 200-prompt WikiText fit's first shard alone
    (``--fit-name j_200p --shard-glob shard0* --out-name j_wiki_40p``) is a 40-prompt WikiText J-lens."""
    import glob
    import os
    import sys

    sys.path.insert(0, "/sarvam-lens")
    os.chdir("/jpp")
    import run_jpp

    name = fit_name or f"j_{fit_prompts}p"
    out = out_name or name
    shard_dirs = sorted(glob.glob(f"{RUNS_DIR}/jfit/{name}/*/{name}/{shard_glob}"))
    code = run_jpp.main(["merge-experts", "--checkpoint-dirs", *shard_dirs, "--checkpoint-name", name, "--device", "cpu",
                         "--out", f"{RUNS_DIR}/lenses/{out}_experts.pt", "--pooled-lens-out", f"{RUNS_DIR}/lenses/{out}.pt"])
    volume.commit()
    return {"exit": code, "shards": shard_dirs, "lens": f"{RUNS_DIR}/lenses/{out}.pt"}


@app.function(image=image, volumes={VOLUME_ROOT: volume}, cpu=2, memory=8192, timeout=30 * 60, retries=0)
def build_corpus_fn(per_lang: int = 40, langs: str = "en,hi,bn,ta,te,mr,gu,kn,ml", name: str = "indic_wiki") -> dict:
    """CPU: a language-interleaved Wikipedia prompt pool on the Volume (``corpora/<name>.json``),
    for ``--prompts-json`` lens fits (see build_corpus.py)."""
    import subprocess
    import sys

    out = f"{VOLUME_ROOT}/corpora/{name}.json"
    proc = subprocess.run([sys.executable, "/sarvam-lens/build_corpus.py", "--langs", langs, "--per-lang", str(per_lang),
                           "--out", out], capture_output=True, text=True)
    volume.commit()
    return {"exit": proc.returncode, "out": out, "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-1500:]}


@app.function(image=image, volumes={VOLUME_ROOT: volume}, cpu=8, memory=32768, timeout=25 * 60, retries=0)
def cka_robust() -> dict:
    """CPU: is Sarvam's flat CKA a norm-gain effect, and how much does the published cka_lib's
    Gemma-only offset rule change a Qwen3.5-family model's picture?"""
    import sys

    sys.path.insert(0, "/sarvam-lens")
    import cka_compare
    import run_jpp

    run_jpp._import_jpp_cli()
    qwen_j = f"{VOLUME_ROOT}/ext_lenses/qwen3.6-35b-a3b/j-lens/lens.pt"
    sarvam_j = f"{RUNS_DIR}/lenses/j_200p.pt"
    specs = [
        {"name": "sarvam_J200_nogain", "lens": sarvam_j, "model_dir": MODEL_DIR, "n_layers": 19, "gain": "none"},
        {"name": "qwen_J25_nogain", "lens": qwen_j, "model_dir": QWEN_DIR, "n_layers": 40, "gain": "none"},
        {"name": "qwen_J25_explorergain", "lens": qwen_j, "model_dir": QWEN_DIR, "n_layers": 40, "gain": "explorer"},
        {"name": "qwen_J25_correctgain", "lens": qwen_j, "model_dir": QWEN_DIR, "n_layers": 40, "gain": "auto"},
    ]
    result = cka_compare.run(specs, f"{RUNS_DIR}/cka_robust",
                             cross=[("sarvam_J200_nogain", "qwen_J25_nogain"),
                                    ("qwen_J25_explorergain", "qwen_J25_correctgain")])
    volume.commit()
    return result


@app.function(image=image, volumes={VOLUME_ROOT: volume}, cpu=8, memory=32768, timeout=30 * 60, retries=0)
def diagnose() -> dict:
    """CPU: why Sarvam's CKA is degenerate (gain outliers, spectrum, CKA variants)."""
    import sys

    sys.path.insert(0, "/sarvam-lens")
    import cka_diagnose
    import run_jpp

    run_jpp._import_jpp_cli()
    specs = [
        {"name": "sarvam_J200", "lens": f"{RUNS_DIR}/lenses/j_200p.pt", "model_dir": MODEL_DIR},
        {"name": "sarvam_R25", "lens": f"{RUNS_DIR}/lenses/r_moe_25p.pt", "model_dir": MODEL_DIR},
        {"name": "qwen_J25", "lens": f"{VOLUME_ROOT}/ext_lenses/qwen3.6-35b-a3b/j-lens/lens.pt", "model_dir": QWEN_DIR},
    ]
    result = cka_diagnose.run(specs, f"{RUNS_DIR}/cka", None, "/ojd/code/jlens_cka")
    volume.commit()
    return result


def _volume_path(arg: str) -> str:
    """Undo Git Bash's MSYS path conversion, which turns ``/vol/x`` into ``C:/Program Files/Git/vol/x``
    (a mangled ``--lenses`` path made a stage silently run without its lens)."""
    marker = "/vol/"
    norm = arg.replace("\\", "/")
    if not norm.startswith("/") and marker in norm and "/Git/" in norm:
        return norm[norm.index(marker):]
    return arg


@app.local_entrypoint()
def main(step: str = "pilot", budget_minutes: float = 38.0, stages: str = "", tag: str = "pilot",
         model: str = "sarvam", fit_prompts: int = 2, fit_rows: int = 16, num_shards: int = 1,
         bos_modes: str = "", spf_guess: float = 0.0, timeout_margin: float = 6.0, lenses: str = "",
         fit_name: str = "", out_name: str = "", shard_glob: str = "shard*", prompts_json: str = "",
         per_lang: int = 40, stage_args: str = "") -> None:
    """``--stages load,bos_probe,pivot_logit`` runs a subset; the container's hard timeout is
    then cut to budget + ``--timeout-margin`` (default 6) min, so a hang can never cost more than
    that. The margin only has to cover container start and the last stage overrunning its
    estimate; a tight cap can lower it (>= 2). ``--model sarvam105`` runs on 2xH200.
    ``--bos-modes sarvam_bos,chat`` limits the BOS probe; ``--spf-guess`` is the seconds-per-forward
    guess that sizes stage estimates until ``sanity`` measures it. ``--lenses a.pt,b.pt`` adds lens
    files to the ``pivot_ext`` stage; ``--fit-name`` names a J-lens fit (``fit_j_shard`` / ``merge-j``);
    ``--prompts-json /vol/corpora/x.json`` fits on that prompt list instead of WikiText;
    ``--step build-corpus --per-lang N`` builds /vol/corpora/indic_wiki.json; ``--stage-args '{...}'``
    (JSON) reaches the cross-lingual stages (``<module>.<analysis>``) as ``ctx.args``."""
    stage_kwargs = json.loads(stage_args) if stage_args else None
    if step == "build-corpus":
        print("CPU 2c/8G, hard timeout 30 min -> worst case $0.09")
        print(json.dumps(build_corpus_fn.remote(per_lang), indent=1, ensure_ascii=False))
        return
    if step == "prepare":
        print(json.dumps(prepare_weights.remote(), indent=1))
    elif step == "prepare-105b":
        print(f"hard timeout {PREPARE_TIMEOUT_MIN} min -> worst case ${PREPARE_TIMEOUT_MIN / 60 * (4 * 0.0472 + 32 * 0.0080):.2f}")
        print(json.dumps(prepare_weights.remote(SARVAM105_REPO, SARVAM105_REVISION, SARVAM105_DIR), indent=1))
    elif step == "batch-probe":
        print(f"hard timeout 15 min -> worst case ${15 / 60 * SMOKE2_USD_PER_HOUR:.2f}")
        print(json.dumps(batch_probe.remote(), indent=1))
    elif step == "post1-decode-2gpu":
        print(f"hard timeout 15 min -> worst case ${15 / 60 * SMOKE2_USD_PER_HOUR:.2f}")
        print(json.dumps(post1_decode_2gpu.remote(), indent=1, ensure_ascii=False))
    elif step == "smoke-2gpu":
        print(f"hard timeout 15 min -> worst case ${15 / 60 * SMOKE2_USD_PER_HOUR:.2f}")
        print(json.dumps(smoke_2gpu.remote(), indent=1, ensure_ascii=False))
    elif step == "prepare-qwen":
        print(json.dumps(prepare_qwen.remote(), indent=1))
    elif step == "merge-j":
        print(json.dumps(merge_j.remote(fit_prompts, fit_name, out_name, shard_glob), indent=1))
    elif step == "cka-robust":
        print(json.dumps(cka_robust.remote(), indent=1, default=float))
    elif step == "diagnose":
        print(json.dumps(diagnose.remote(), indent=1, default=float))
    elif step == "cka":
        print(json.dumps(cka.remote(), indent=1, default=float))
    elif step == "smoke":
        selected = [name for name in stages.split(",") if name] or None
        if selected:  # the smoke test plus a dry pilot of just these stages: far under the full 25 min
            print(f"hard timeout {SMOKE_STAGES_TIMEOUT_MIN} min -> worst case ${SMOKE_STAGES_TIMEOUT_MIN / 60 * SMOKE_USD_PER_HOUR:.2f}")
            print(json.dumps(smoke.with_options(timeout=SMOKE_STAGES_TIMEOUT_MIN * 60).remote(selected, stage_kwargs), indent=1,
                             ensure_ascii=False))
        else:
            print(json.dumps(smoke.remote(), indent=1, ensure_ascii=False))
    elif step == "pilot":
        if model not in MODEL_DIRS:
            raise SystemExit(f"--model must be one of {sorted(MODEL_DIRS)}")
        if timeout_margin < 2:
            raise SystemExit("--timeout-margin must be >= 2 min (container start + a stage overrunning its estimate)")
        if budget_minutes > PILOT_TIMEOUT_MIN - timeout_margin:
            raise SystemExit(f"--budget-minutes must leave >= {timeout_margin} min under the {PILOT_TIMEOUT_MIN}-min timeout")
        timeout_s = int((budget_minutes + timeout_margin) * 60)
        n_gpus = 2 if model in TWO_GPU_MODELS else 1
        rate = USD_PER_HOUR[n_gpus]
        print(f"{n_gpus}xH200 at ${rate:.2f}/h, hard timeout {timeout_s / 60:.0f} min -> worst case ${timeout_s / 3600 * rate:.2f}")
        selected = [name for name in stages.split(",") if name] or None
        modes = [mode for mode in bos_modes.split(",") if mode] or None
        extra = {"bos_modes": modes, **({"spf_guess": spf_guess} if spf_guess > 0 else {}),
                 "extra_lenses": [_volume_path(p) for p in lenses.split(",") if p] or None, "fit_name": fit_name,
                 "prompts_json": _volume_path(prompts_json) if prompts_json else "", "stage_args": stage_kwargs}
        if extra["extra_lenses"] or extra["prompts_json"]:
            print(f"lenses {extra['extra_lenses']}, prompts_json {extra['prompts_json']!r}")
        fn = (pilot2 if n_gpus == 2 else pilot).with_options(timeout=timeout_s)
        if num_shards > 1:  # one container per shard, in parallel; same total cost, 1/N wall time
            print(f"{num_shards} parallel shards -> worst case ${num_shards * timeout_s / 3600 * rate:.2f}")
            calls = [fn.spawn(budget_minutes, selected, f"{tag}_s{i}", model, fit_prompts, fit_rows, i, num_shards, **extra)
                     for i in range(num_shards)]
            for call in calls:
                print(json.dumps(call.get(), indent=1, ensure_ascii=False))
        else:
            result = fn.remote(budget_minutes, selected, tag, model, fit_prompts, fit_rows, **extra)
            print(json.dumps(result, indent=1, ensure_ascii=False))
    else:
        raise SystemExit(f"unknown --step {step!r} (prepare | prepare-105b | prepare-qwen | smoke | smoke-2gpu | "
                         "cka | cka-robust | diagnose | merge-j | build-corpus | pilot)")
