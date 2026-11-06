"""Why is Sarvam's J-space CKA degenerate (layer coupling ~0.97, ~0.09 vs Qwen)?

For each lens: final-norm gain statistics, how much of the probe P = W_U[ids] * gain and of
V_l = P @ J_l the top singular directions carry (participation ratio), and the layer
coupling / cross-model matched-depth CKA under three probe variants:
  standard   the explorer's frozen definition (gain folded in)
  no_gain    P = W_U[ids]  (is a few huge gain dimensions dominating?)
  minus_pc1  standard, with each V_l's top principal component projected out
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch as t


def participation_ratio(X: t.Tensor) -> tuple[float, float]:
    """(PR of the centered singular spectrum, share of variance on the top direction)."""
    s2 = t.linalg.svdvals(X - X.mean(0, keepdim=True)).double() ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum()), float(s2[0] / s2.sum())


def drop_pc1(V: t.Tensor) -> t.Tensor:
    Vc = V - V.mean(0, keepdim=True)
    _, _, Vh = t.linalg.svd(Vc, full_matrices=False)
    top = Vh[0:1]
    return Vc - (Vc @ top.T) @ top


def run(specs: list[dict], out_dir: str, cross: tuple[str, str] | None, ojd_dir: str) -> dict:
    import cka_compare

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cka_lib, _ = cka_compare._imports(ojd_dir)
    report: dict = {}
    grams: dict[str, dict[str, list[t.Tensor]]] = {}
    for spec in specs:
        name, model_dir = spec["name"], spec["model_dir"]
        pw = cka_lib.load_probe_weights(model_dir)
        pw.is_gemma = cka_compare.norm_is_offset(model_dir)
        gain = pw.gain.abs()
        top = t.topk(gain, 5)
        lens = cka_compare.to_cka_lens(cka_lib, spec["lens"], model_dir)
        ids = cka_lib.sample_probe_ids(pw.vocab_rows)
        P = pw.W_U[t.as_tensor(ids)] * pw.gain
        P_raw = pw.W_U[t.as_tensor(ids)]
        entry: dict = {
            "gain_median": float(gain.median()), "gain_max": float(gain.max()),
            "gain_top5_dims": top.indices.tolist(), "gain_top5": [round(float(v), 2) for v in top.values],
            "P_pr_top1": participation_ratio(P), "P_raw_pr_top1": participation_ratio(P_raw),
            "V_pr_top1_by_layer": {},
        }
        variants = {"standard": [], "no_gain": [], "minus_pc1": []}
        for layer, J in lens:
            V = P @ J
            if layer % 3 == 0:
                entry["V_pr_top1_by_layer"][layer] = participation_ratio(V)
            variants["standard"].append(cka_lib._normalized_gram(V))
            variants["no_gain"].append(cka_lib._normalized_gram(P_raw @ J))
            variants["minus_pc1"].append(cka_lib._normalized_gram(drop_pc1(V)))
        coupling = {}
        for variant, gs in variants.items():
            G = t.stack([g.flatten() for g in gs])
            C = (G @ G.T).numpy()
            L = C.shape[0]
            coupling[variant] = float((C.sum() - np.trace(C)) / (L * (L - 1)))
        entry["layer_coupling"] = coupling
        grams[name] = variants
        report[name] = entry
        print(name, json.dumps(entry, default=float), flush=True)

        # Massive-activation test: which OUTPUT dims carry the dominant direction of J_l, and
        # does masking them from the probe (zeroing those columns of P) restore normal CKA?
        mid = lens.source_layers[len(lens.source_layers) // 2]
        U, S, _ = t.linalg.svd(lens.jacobians[mid].float(), full_matrices=False)
        u1 = U[:, 0].abs()
        dims = t.topk(u1, 16).indices
        entry["J_mid_layer"] = mid
        entry["J_top_sv_share"] = float(S[0] ** 2 / (S ** 2).sum())
        entry["J_u1_top_dims"] = dims[:8].tolist()
        entry["J_u1_mass_top8"] = float((u1[dims[:8]] ** 2).sum())
        masked = {}
        for k in (1, 2, 4, 8, 16):
            Pm = P.clone()
            Pm[:, dims[:k]] = 0
            gs = [cka_lib._normalized_gram(Pm @ J) for _, J in lens]
            G = t.stack([g.flatten() for g in gs])
            C = (G @ G.T).numpy()
            L = C.shape[0]
            masked[k] = float((C.sum() - np.trace(C)) / (L * (L - 1)))
            if k == 8:
                np.save(Path(out_dir) / f"{name}_cka_mask8.npy", C)
        entry["layer_coupling_masked_top_dims"] = masked
        print(name, "massive-dim test", json.dumps({k: entry[k] for k in ("J_top_sv_share", "J_u1_top_dims",
              "J_u1_mass_top8", "layer_coupling_masked_top_dims")}), flush=True)

    if cross:
        # Same-id probes only make sense for a shared vocab; across tokenizers use cka_compare's
        # shared-string cross. Here we just report how the variants move the within-model picture.
        report["note"] = "cross-model CKA under variants: see cka_compare (shared-string probes)"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "diagnose.json").write_text(json.dumps(report, indent=1, default=float))
    return report
