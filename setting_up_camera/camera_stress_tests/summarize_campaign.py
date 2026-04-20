from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from common import ensure_dir, read_json, resolve_repo_path
from constants import SUMMARY_OUTPUT_DIR, SUMMARY_RUNS_ROOT, TEST_RUNS


def _collect_summaries(runs_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        summary_path = run_dir / "analysis_summary.json"
        try:
            if not summary_path.exists():
                continue
            item = read_json(summary_path)
        except (PermissionError, OSError):
            continue
        item["_run_dir"] = str(run_dir)
        summaries.append(item)
    return summaries


def _flatten(item: dict[str, Any]) -> dict[str, Any]:
    c = item.get("condition", {})
    m = item.get("metrics", {})
    return {
        "run_id": item.get("run_id"),
        "title": c.get("title"),
        "link_distance_m": c.get("link_distance_m", c.get("distance_m")),
        "motion": c.get("motion"),
        "overall_pass": item.get("overall_pass"),
        "actual_fps": m.get("actual_fps"),
        "fps_ratio": m.get("fps_ratio"),
        "drop_pct": m.get("drop_pct"),
        "freeze_count": m.get("freeze_count"),
        "dt_p95_ms": m.get("dt_p95_ms"),
        "blur_frame_pct": m.get("blur_frame_pct"),
        "run_dir": item.get("_run_dir"),
    }


def _line_for(run_prefix: str, rows: list[dict[str, Any]]) -> str:
    matches = [r for r in rows if str(r.get("run_id", "")).startswith(run_prefix)]
    if not matches:
        return f"- {run_prefix}: no analyzed run"

    row = sorted(matches, key=lambda x: str(x.get("run_id")), reverse=True)[0]
    link_distance = row.get("link_distance_m")
    return (
        f"- {run_prefix}: link_distance_m={link_distance}, pass={row.get('overall_pass')}, "
        f"fps={row.get('actual_fps')}, drop%={row.get('drop_pct')}, "
        f"freeze={row.get('freeze_count')}, blur%={row.get('blur_frame_pct')}"
    )


def main() -> None:
    runs_root = resolve_repo_path(SUMMARY_RUNS_ROOT)
    out_dir = resolve_repo_path(SUMMARY_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_dir(runs_root)

    summaries = _collect_summaries(runs_root)
    if not summaries:
        raise RuntimeError(f"No analysis_summary.json found under: {runs_root}")

    rows = [_flatten(s) for s in summaries]
    expected_prefixes = [str(run.get("id", "")) for run in TEST_RUNS]
    rows = [
        r
        for r in rows
        if any(str(r.get("run_id", "")).startswith(prefix) for prefix in expected_prefixes)
    ]

    csv_path = out_dir / "campaign_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "title",
                "link_distance_m",
                "motion",
                "overall_pass",
                "actual_fps",
                "fps_ratio",
                "drop_pct",
                "freeze_count",
                "dt_p95_ms",
                "blur_frame_pct",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    passed = sum(1 for r in rows if bool(r.get("overall_pass")))

    report_lines = [
        "# Camera Stress Campaign Summary",
        "",
        f"Runs analyzed: {total}",
        f"Passing runs: {passed}",
        "",
        "## Core Evidence Lines",
    ]
    for run in TEST_RUNS:
        report_lines.append(_line_for(str(run.get("id", "")), rows))

    report_lines += [
        "",
        "## Interpretation Template",
        "Use these run lines to conclude:",
        "- whether effective FPS degrades as camera-drone distance increases",
        "- whether drops/freezes increase with distance",
        "- whether stream timing (lag/choppiness) becomes unstable with distance",
        "",
        "## Outputs",
        f"- campaign_summary.csv: {csv_path}",
    ]

    report_path = out_dir / "campaign_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Campaign summary complete.")
    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
