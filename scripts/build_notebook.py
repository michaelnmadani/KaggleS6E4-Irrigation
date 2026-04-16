"""Assemble a Kaggle-pushable kernel directory for a given iteration.

Given an iteration name (e.g. 001_baseline), produces a staging dir containing:
    notebook.ipynb           (compiled from pipeline/notebook_template.py via jupytext)
    iteration_config.yaml    (copy of iterations/<iter>/config.yaml)
    kernel-metadata.json     (filled-in copy of pipeline/kernel_metadata.json)
    pipeline/                (the pipeline package — src/, __init__.py)

`kaggle kernels push -p <staging_dir>` uses this directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("iteration", help="iteration directory name, e.g. 001_baseline")
    ap.add_argument("--staging", default="build/kernel", help="output staging dir")
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

    # 1. compile template .py -> notebook.ipynb via jupytext, after templating.
    tmpl = (repo / "pipeline" / "notebook_template.py").read_text()
    tmpl = tmpl.replace("{COMP_SLUG}", args.comp_slug).replace("{ITERATION}", args.iteration)
    tmp_py = stage / "_notebook.py"
    tmp_py.write_text(tmpl)
    subprocess.check_call([sys.executable, "-m", "jupytext", "--to", "ipynb", str(tmp_py), "-o", str(stage / "notebook.ipynb")])
    tmp_py.unlink()

    # 2. copy iteration config alongside.
    shutil.copy(iter_dir / "config.yaml", stage / "iteration_config.yaml")

    # 3. copy pipeline package so it's importable from /kaggle/working.
    shutil.copytree(repo / "pipeline" / "src", stage / "pipeline" / "src")
    (stage / "pipeline" / "__init__.py").write_text("")

    # 4. fill in kernel-metadata.json.
    meta = json.loads((repo / "pipeline" / "kernel_metadata.json").read_text())
    kernel_slug = f"pipeline-iter-{args.iteration.replace('_', '-')}"
    meta["id"] = f"{args.kaggle_user}/{kernel_slug}"
    meta["title"] = kernel_slug
    meta["competition_sources"] = [args.comp_slug]
    (stage / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"staged kernel at {stage}")
    print(f"push with: kaggle kernels push -p {stage}")


if __name__ == "__main__":
    main()
