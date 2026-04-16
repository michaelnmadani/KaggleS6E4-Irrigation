"""Pre-flight check: CLI tools present, comp slug set, iteration dirs valid."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    results: list[bool] = []

    results.append(check("kaggle CLI on PATH", shutil.which("kaggle") is not None,
                         "install via: pip install kaggle"))
    results.append(check("jupytext on PATH", shutil.which("jupytext") is not None,
                         "install via: pip install jupytext"))

    comp = os.environ.get("COMP_SLUG", "")
    results.append(check("COMP_SLUG env var set", bool(comp),
                         f"current: {comp or '<unset>'}"))

    iters = sorted((repo / "iterations").glob("*/config.yaml"))
    results.append(check("iterations present", len(iters) > 0,
                         f"found {len(iters)}"))

    # Shallow kaggle API probe — won't submit, just lists
    if shutil.which("kaggle"):
        try:
            subprocess.run(["kaggle", "competitions", "list", "-s", "playground"],
                           check=True, capture_output=True, timeout=15)
            results.append(check("kaggle API auth", True))
        except Exception as e:
            results.append(check("kaggle API auth", False, f"{type(e).__name__}: {e}"))

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
