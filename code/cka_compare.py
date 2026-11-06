"""Within-model "J-space" CKA (the method behind eliebak.com/viz/jspace-open) for Sarvam-30B,
calibrated against Qwen3.6-35B-A3B.

Uses the explorer's own code (eliebak/open-jlens-data ``code/jlens_cka``, frozen definitions):
probe P = W_U[ids] * final-norm gain (4096 ids, RandomState(0)), V_l = P @ J_l, linear CKA
between layers, and its frozen block segmentation (sensory_end / motor_start / blockiness /
layer_coupling).

The catch is the lens type. The explorer uses plain J-lenses; Sarvam so far has an R-lens and
a J++ lens (LRP backward), which change the geometry. Qwen3.6-35B-A3B has *both* a published
J-lens and R-lens (camilablank/workspace-lenses), so its J-vs-R difference calibrates how to
read Sarvam's R-lens CKA. A Sarvam J-lens can be added via --lens later.

Two corrections to the explorer code, both applied here:
- Qwen3.5/3.6 RMSNorm computes x * (1 + w), but ``load_probe_weights`` only treats Gemma
  as offset-from-1. The probe would use w (near 0) as the gain. We set the offset for
  any model whose norm stores it.
- ``Lens.n_layers`` assumes source layers run to L-2 (final-block target); the published
  Qwen lenses target the penultimate block, so the true layer count is passed explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OJD_CKA_DIR = "/ojd/code/jlens_cka"


def _imports(ojd_dir: str):
    sys.path.insert(0, ojd_dir)
    import analyze_blocks
    import cka_lib

    return cka_lib, analyze_blocks


def to_cka_lens(cka_lib, path: str, hf_model_name: str):
    """A jpp lens file (or a released legacy J/R lens) as cka_lib's ``Lens``."""
    from workspace_lens.utils import load_lens_file

    lens = load_lens_file(path, hf_model_name=hf_model_name)
    jacobians = {int(layer): J.float() for layer, J in lens.jacobians_L_dict_FN.items()}
    layers = sorted(jacobians)
    d_model = next(iter(jacobians.values())).shape[0]
    return cka_lib.Lens(jacobians=jacobians, source_layers=layers, d_model=d_model,
                        n_prompts=int(getattr(lens.config, "num_prompts_trained_on", 0) or 0), path=path)


def norm_is_offset(model_dir: str) -> bool:
    """True when the final RMSNorm multiplies by (1 + w): Gemma, Qwen3.5/3.6."""
    config = json.loads((Path(model_dir) / "config.json").read_text())
    text = config.get("text_config", config)
    kinds = f"{config.get('model_type', '')}/{text.get('model_type', '')}".lower()
    return "gemma" in kinds or "qwen3_5" in kinds


def run(specs: list[dict], out_dir: str, ojd_dir: str = OJD_CKA_DIR, cross: list[tuple[str, str]] = ()) -> dict:
    """``specs``: [{name, lens, model_dir, n_layers}]. Writes <name>_cka.npz/.json/.png and
    cross_<a>_x_<b>.json. Returns the per-lens block metrics."""
    import torch as t

    cka_lib, analyze_blocks = _imports(ojd_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    probes: dict[str, object] = {}
    lenses: dict[str, object] = {}
    results: dict[str, dict] = {}
    import copy

    raw_probes: dict[str, object] = {}
    for spec in specs:
        name, model_dir, n_layers = spec["name"], spec["model_dir"], spec["n_layers"]
        # gain: "auto" = correct (1+w for Gemma and Qwen3.5/3.6), "explorer" = the published
        # cka_lib rule (1+w for Gemma only), "none" = no gain folded into the probe.
        gain_mode = spec.get("gain", "auto")
        if model_dir not in raw_probes:
            raw_probes[model_dir] = cka_lib.load_probe_weights(model_dir)
        pw = copy.copy(raw_probes[model_dir])
        if gain_mode == "auto":
            pw.is_gemma = norm_is_offset(model_dir)  # "is_gemma" == "gain is 1 + w"
        elif gain_mode == "none":
            pw.gamma, pw.is_gemma = t.ones_like(pw.gamma), False
        elif gain_mode != "explorer":
            raise ValueError(f"unknown gain mode {gain_mode!r}")
        probes[name] = pw
        P, ids = cka_lib.build_probe(pw)
        lens = to_cka_lens(cka_lib, spec["lens"], model_dir)
        lenses[name] = lens
        with t.no_grad():
            C = cka_lib.cka_matrix(lens, P)
        blocks = analyze_blocks.segment_blocks(C, lens.source_layers, n_layers)
        rho = np.array([layer / (n_layers - 1) for layer in lens.source_layers])
        np.savez(out / f"{name}_cka.npz", C=C, source_layers=np.array(lens.source_layers), rho=rho, ids=ids)
        analyze_blocks.plot_cka_heatmap(C, blocks, rho, f"{name} (J-space CKA)", str(out / f"{name}_cka.png"))
        keep = ("n_blocks", "boundaries_rho", "sensory_end", "motor_start", "blockiness", "layer_coupling",
                "block_layers")
        results[name] = {k: blocks[k] for k in keep if k in blocks}
        results[name].update(n_layers=n_layers, fitted_layers=len(lens.source_layers), gain_offset=pw.is_gemma,
                             gain_mode=gain_mode, n_prompts=lens.n_prompts)
        (out / f"{name}_blocks.json").write_text(json.dumps(results[name], indent=1, default=float))
        print(name, json.dumps(results[name], default=float), flush=True)

    from transformers import AutoTokenizer

    spec_by_name = {s["name"]: s for s in specs}
    for name_a, name_b in cross:
        a, b = spec_by_name[name_a], spec_by_name[name_b]
        shared = cka_lib.shared_token_ids(AutoTokenizer.from_pretrained(a["model_dir"]),
                                          AutoTokenizer.from_pretrained(b["model_dir"]))
        P_a, _ = cka_lib.build_probe(probes[name_a], np.asarray(shared["ids_a"]))
        P_b, _ = cka_lib.build_probe(probes[name_b], np.asarray(shared["ids_b"]))
        with t.no_grad():
            res = cka_lib.cross_model_cka(lenses[name_a], P_a, lenses[name_b], P_b, full_matrix=True)
        summary = {"matched_mean": float(res["matched_mean"]), "n_shared_tokens": len(shared["ids_a"]),
                   "matched_rho": [float(x) for x in res["matched_rho"]],
                   "matched_cka": [float(x) for x in res["matched_cka"]]}
        if "cross" in res:
            np.save(out / f"cross_{name_a}_x_{name_b}.npy", np.asarray(res["cross"]))
        (out / f"cross_{name_a}_x_{name_b}.json").write_text(json.dumps(summary, indent=1))
        print(f"cross {name_a} x {name_b}: matched-depth CKA {summary['matched_mean']:.3f} "
              f"over {summary['n_shared_tokens']} shared tokens", flush=True)
        results[f"cross:{name_a}x{name_b}"] = {"matched_mean": summary["matched_mean"],
                                               "n_shared_tokens": summary["n_shared_tokens"]}
    (out / "summary.json").write_text(json.dumps(results, indent=1, default=float))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", action="append", required=True,
                        help="name=...,lens=...,model_dir=...,n_layers=... (repeatable)")
    parser.add_argument("--cross", action="append", default=[], help="nameA:nameB (repeatable)")
    parser.add_argument("--ojd-dir", default=OJD_CKA_DIR)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    specs = []
    for raw in args.spec:
        spec = dict(item.split("=", 1) for item in raw.split(","))
        spec["n_layers"] = int(spec["n_layers"])
        specs.append(spec)
    run(specs, args.out, args.ojd_dir, [tuple(c.split(":")) for c in args.cross])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
