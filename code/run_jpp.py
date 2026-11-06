"""Run the J++ pipeline (``scripts/jpp_cli.py`` of safety-research/jpp_lens) with Sarvam support.

    python run_jpp.py <jpp stage> [jpp flags...]    # any stock stage, Sarvam-aware
    python run_jpp.py prepare-evals --out DIR [--exclude multilingual]
    python run_jpp.py grade --hf-model-name M --eval-data-dir DIR --out CSV

What changes relative to stock ``jpp_cli.py``:

- ``--hf-model-name`` may be a Sarvam checkpoint (``sarvamai/sarvam-30b`` or a local
  copy); it is loaded by :func:`sarvam_support.load_sarvam_lens_model`.
- ``fit-shard --lrp-mode rlens|r+moe`` on Sarvam uses :func:`sarvam_support.apply_sarvam_lrp_rules`.
- With ``JPP_FIT_PROMPTS_JSON=path`` set, ``fit-router`` and ``fit-shard`` draw their
  prompts from that JSON list (e.g. the Indic corpus from ``build_corpus.py``) instead
  of WikiText. ``fit-shard --prompts-json`` still works and takes precedence.
- Two extra stages: ``prepare-evals`` copies jpp's eval items, leaving out the evals you
  name (an excluded eval keeps an empty file, since jpp loads all five by name);
  ``grade`` writes the model-correctness CSV that ``fit-weights`` and ``evaluate``
  filter on (jpp ships one only for Qwen3.6-27B).

Set ``JPP_DIR`` to your jpp_lens checkout (default: ``./jpp_lens``). Run this from that
checkout's root, since jpp reads ``data/`` by relative path.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JPP_DIR = Path(os.environ.get("JPP_DIR", "jpp_lens")).resolve()
FIT_PROMPTS_ENV = "JPP_FIT_PROMPTS_JSON"


def _import_jpp_cli():
    for path in (JPP_DIR / "scripts", JPP_DIR / "src", HERE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import jpp_cli

    return jpp_cli


def install(jpp_cli) -> None:
    """Patch jpp_cli's model loader and fit-prompt loader, and jpp's LRP dispatch."""
    import sarvam_support

    sarvam_support.install_jpp_hooks()
    if getattr(jpp_cli, "_sarvam_installed", False):
        return
    stock_get_hf_model = jpp_cli.get_hf_model
    stock_load_fit_prompts = jpp_cli.load_fit_prompts

    def get_hf_model(model_name: str, **kwargs):
        model_type = sarvam_support.sarvam_model_type(model_name)
        if model_type is not None:
            spec = sarvam_support.SARVAM_MODELS[model_type]
            # The 105B's MLA is eager-only, whatever a stage's --attn-implementation default says.
            attn = spec["attn"] if spec["force_attn"] else (kwargs.get("attn_implementation") or spec["attn"])
            return sarvam_support.load_sarvam_lens_model(
                model_name,
                attn_implementation=attn,
                device_map=kwargs.get("device_map"),
            )
        return stock_get_hf_model(model_name, **kwargs)

    def load_fit_prompts(num_prompts: int) -> list[str]:
        path = os.environ.get(FIT_PROMPTS_ENV)
        if not path:
            return stock_load_fit_prompts(num_prompts)
        return jpp_cli.load_fit_prompts_from_json(path, num_prompts)

    jpp_cli.get_hf_model = get_hf_model
    jpp_cli.load_fit_prompts = load_fit_prompts
    jpp_cli._sarvam_installed = True


def prepare_evals(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="run_jpp.py prepare-evals")
    parser.add_argument("--src", default=str(JPP_DIR / "data/jlens/evaluations"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--exclude", default="multilingual", help="comma-separated eval slugs")
    args = parser.parse_args(argv)
    excluded = {slug for slug in args.exclude.split(",") if slug}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(args.src).glob("lens-eval-*.json")):
        slug = path.stem.removeprefix("lens-eval-")
        if slug in excluded:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["items"] = []
            (out / path.name).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{slug}: excluded (empty items)")
        else:
            shutil.copy(path, out / path.name)
            print(f"{slug}: copied")
    return 0


def grade(argv: list[str], jpp_cli) -> int:
    """One greedy step per gradeable item, merged as a new column into ``--out``."""
    import pandas as pd
    from lens_evals.readout_evals.readout_eval_items import load_readout_eval_items

    parser = argparse.ArgumentParser(prog="run_jpp.py grade")
    parser.add_argument("--hf-model-name", required=True)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--eval-data-dir", default=str(JPP_DIR / "data/jlens/evaluations"))
    parser.add_argument("--max-seq-len", type=int, default=jpp_cli.RECIPE_READOUT_MAX_SEQ_LEN)
    parser.add_argument("--out", required=True, help="correctness CSV to write or extend")
    args = parser.parse_args(argv)

    model = jpp_cli.get_hf_model(
        args.hf_model_name, attn_implementation=args.attn_implementation, device_map=args.device_map
    )
    items = load_readout_eval_items(args.eval_data_dir)
    runner = jpp_cli.readout_runner(model, layers=[model.n_layers - 1], max_seq_len=args.max_seq_len)
    graded = runner.grade_model_correctness(items, hf_model_name=args.hf_model_name)
    out = Path(args.out)
    if out.is_file():
        existing = pd.read_csv(out).drop(columns=[args.hf_model_name], errors="ignore")
        graded = existing.merge(graded, on=["eval", "item"], how="outer")
    out.parent.mkdir(parents=True, exist_ok=True)
    graded.to_csv(out, index=False)
    column = graded[args.hf_model_name]
    for slug, group in graded.groupby("eval"):
        gradeable = group[args.hf_model_name].dropna()
        if len(gradeable):
            print(f"{slug}: {int(gradeable.astype(bool).sum())}/{len(gradeable)} correct")
    print(f"wrote {out} ({column.notna().sum()} gradeable rows)")
    return 0


def check_no_fit_item_leak(argv: list[str], jpp_cli) -> None:
    """Refuse an ``evaluate`` that would score a J++ lens on its own weight-fitting items.

    jpp splits fit/held-out items with one RNG stream across evals in slug order, so the
    split depends on the whole item list. Fitting weights on a ``prepare-evals`` dir (one
    eval emptied) and evaluating ``held-out:<seed>`` on a different dir does not give the
    complement. With multilingual emptied, 31 of the 76 fit items (poetry, typo) land in
    the "held-out" split. ``--items all`` on the fit dir includes them all."""
    args = jpp_cli.build_parser().parse_args(argv)
    eval_dir = Path(args.eval_data_dir).resolve()
    for spec in args.lens:
        stamp = _fit_weights_stamp(jpp_cli, spec)
        if stamp is None:
            continue
        fit_args = stamp["args"]
        fit_dir = Path(fit_args["eval_data_dir"]).resolve()
        same_dir = fit_dir == eval_dir
        # held-out is the complement only of the very split the weights used: same dir, seed and fraction
        fit_split = str(fit_args.get("items", "fit:0"))
        same_split = (args.items.startswith("held-out:") and fit_split.startswith("fit:")
                      and args.items.partition(":")[2] == fit_split.partition(":")[2]
                      and float(args.fit_fraction) == float(fit_args.get("fit_fraction", args.fit_fraction)))
        if (args.items.startswith("held-out") and not (same_dir and same_split)) or (args.items == "all" and same_dir):
            raise SystemExit(
                f"refusing to evaluate {spec}: its expert weights were fitted on --items {fit_split} "
                f"(fit fraction {fit_args.get('fit_fraction')}) of {fit_dir}, and --items {args.items} "
                f"(fit fraction {args.fit_fraction}) on {eval_dir} would score some of those same items. "
                f"Use --eval-data-dir {fit_dir} --items held-out:{fit_split.partition(':')[2]} --fit-fraction "
                f"{fit_args.get('fit_fraction')} for the held-out split, and --items all only on a dir holding evals "
                "the weights never saw (e.g. prepare-evals --exclude association,multihop,order-ops,poetry,typo)."
            )


def _fit_weights_stamp(jpp_cli, spec: str) -> dict | None:
    """The fit-weights stamp beside a lens file, or None (logit lens, no stamp, another stage)."""
    stamp_path = Path(jpp_cli.stamp_path_beside(spec))
    if spec == jpp_cli.LOGIT_LENS_SPEC or not stamp_path.is_file():
        return None
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    return stamp if stamp.get("stage") == "fit-weights" else None


def refuse_in_sample_items(jpp_cli, lens_specs, items) -> None:
    """The item-level guard, run once ``evaluate`` has its scored items (the split depends on the
    scoreable list, so it can only be known after the model loads): refuse if any scored item is one the
    lens's expert weights were fitted on, as the fit-weights stamp records them (``fitting_item_names``).
    jpp itself only warns (``warn_about_in_sample_items``); the directory check above can't see a seed
    or fraction change on the same dir (held-out:1 after fit:0 scores 32 of the 43 fitting items of
    ``lenses/jpp_25p`` on the raw list). A stamp without the names is refused too."""
    scored = {item.name for item in items}
    for spec in lens_specs:
        stamp = _fit_weights_stamp(jpp_cli, spec)
        if stamp is None:
            continue
        if "fitting_item_names" not in stamp:
            raise SystemExit(f"refusing to evaluate {spec}: its fit-weights stamp does not record fitting_item_names")
        leaked = sorted(scored & set(stamp["fitting_item_names"]))
        if leaked:
            raise SystemExit(
                f"refusing to evaluate {spec}: {len(leaked)} of the {len(scored)} items to score were used to fit its "
                f"expert weights (stamp {jpp_cli.stamp_path_beside(spec)}), e.g. {leaked[:5]}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "prepare-evals":
        return prepare_evals(argv[1:])
    jpp_cli = _import_jpp_cli()
    install(jpp_cli)
    if argv and argv[0] == "grade":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        return grade(argv[1:], jpp_cli)
    if argv and argv[0] == "evaluate":
        check_no_fit_item_leak(argv, jpp_cli)
        # jpp calls this with the scored items before scoring; ours refuses where jpp's only warns
        jpp_cli.warn_about_in_sample_items = lambda specs, items: refuse_in_sample_items(jpp_cli, specs, items)
    return jpp_cli.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
