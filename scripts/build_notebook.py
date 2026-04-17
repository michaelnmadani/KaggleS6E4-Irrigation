"""Assemble a self-contained Kaggle notebook for a given iteration.

Kaggle's `kernels push` uploads only the single notebook file + metadata —
support files staged alongside are NOT sent. So every iteration's notebook
must be self-contained: we inline pipeline/src/*.py into cells of the notebook.

Produced staging dir:
    build/kernel/
        notebook.ipynb          — self-contained; inlined pipeline code + runner
        kernel-metadata.json    — Kaggle kernel config
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path


def _strip_relative_imports(src: str) -> str:
    """Remove `from . import X as Y` and rewrite `Y.attr` -> `attr`
    so all modules collapse into one notebook namespace."""
    aliases: dict[str, str] = {}
    for m in re.finditer(r"^from\s+\.\s+import\s+(\w+)\s+as\s+(\w+)\s*$", src, re.MULTILINE):
        aliases[m.group(2)] = m.group(1)
    out = re.sub(r"^from\s+\.\s+import\s+\w+(\s+as\s+\w+)?\s*$", "", src, flags=re.MULTILINE)
    for alias in aliases:
        out = re.sub(rf"\b{re.escape(alias)}\.", "", out)
    return out


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def build_notebook(repo: Path, iteration: str, comp_slug: str) -> dict:
    src_dir = repo / "pipeline" / "src"
    modules = ["data", "features", "models", "train"]

    cells: list[dict] = [_md_cell(f"# Pipeline iteration: {iteration}\n")]
    for mod in modules:
        code = (src_dir / f"{mod}.py").read_text(encoding="utf-8")
        cells.append(_md_cell(f"## `{mod}.py`\n"))
        cells.append(_code_cell(_strip_relative_imports(code)))

    config_text = (repo / "iterations" / iteration / "config.yaml").read_text(encoding="utf-8")
    runner = (
        'import json, pathlib\n'
        'CONFIG_YAML = r"""\n' + config_text + '"""\n'
        f'COMP_SLUG = "{comp_slug}"\n'
        f'ITERATION = "{iteration}"\n'
        'cfg_path = pathlib.Path("/kaggle/working/iteration_config.yaml")\n'
        'cfg_path.write_text(CONFIG_YAML)\n'
        'metrics = run(\n'
        '    config_path=str(cfg_path),\n'
        '    input_dir=f"/kaggle/input/{COMP_SLUG}",\n'
        '    output_dir="/kaggle/working",\n'
        ')\n'
        'print(json.dumps(metrics, indent=2))\n'
    )
    cells.append(_md_cell("## run\n"))
    cells.append(_code_cell(runner))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("iteration")
    ap.add_argument("--staging", default="build/kernel")
    ap.add_argument("--comp-slug", default=os.environ.get("COMP_SLUG", ""))
    ap.add_argument("--kaggle-user", default=os.environ.get("KAGGLE_USERNAME", ""))
    args = ap.parse_args()

    if not args.comp_slug:
        sys.exit("COMP_SLUG env var or --comp-slug is required")
    if not args.kaggle_user:
        sys.exit("KAGGLE_USERNAME env var or --kaggle-user is required")

    repo = Path(__file__).resolve().parent.parent
    iter_dir = repo / "iterations" / args.iteration
    if not iter_dir.exists():
        sys.exit(f"iteration dir not found: {iter_dir}")

    stage = repo / args.staging
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    nb = build_notebook(repo, args.iteration, args.comp_slug)
    (stage / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

    meta = json.loads((repo / "pipeline" / "kernel_metadata.json").read_text())
    kernel_slug = f"pipeline-iter-{args.iteration.replace('_', '-')}"
    meta["id"] = f"{args.kaggle_user}/{kernel_slug}"
    meta["title"] = kernel_slug
    meta["competition_sources"] = [args.comp_slug]
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"staged self-contained kernel at {stage}")


if __name__ == "__main__":
    main()
