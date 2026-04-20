from __future__ import annotations

from pathlib import Path

from common import ensure_dir, resolve_repo_path
import analyze_camera_test as analyze
from constants import ANALYZE_RUNS_ROOT


def main() -> None:
    runs_root = resolve_repo_path(ANALYZE_RUNS_ROOT)
    ensure_dir(runs_root)

    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir() and (p / "metadata.json").exists()])
    if not run_dirs:
        raise RuntimeError(f"No run folders found in: {runs_root}")

    analyzed = 0
    for run_dir in run_dirs:
        analyze.ANALYZE_RUN_DIR = str(run_dir)
        analyze.ANALYZE_USE_LATEST_RUN_IF_EMPTY = False
        print(f"Analyzing: {run_dir.name}")
        analyze.main()
        analyzed += 1

    print(f"Analyze-all complete. Runs analyzed: {analyzed}")


if __name__ == "__main__":
    main()
