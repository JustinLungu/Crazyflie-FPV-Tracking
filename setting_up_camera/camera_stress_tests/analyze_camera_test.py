from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any

from common import ensure_dir, read_json, resolve_repo_path, write_json
from constants import (
    ANALYZE_BLUR_LAPLACIAN_THRESHOLD,
    ANALYZE_FREEZE_DIFF_THRESHOLD,
    ANALYZE_FREEZE_MIN_FRAMES,
    ANALYZE_MAX_BLUR_FRAME_PCT,
    ANALYZE_MAX_DROP_PCT,
    ANALYZE_MAX_DT_P95_MS,
    ANALYZE_MAX_FREEZE_COUNT,
    ANALYZE_MIN_FPS_RATIO,
    ANALYZE_RUN_DIR,
    ANALYZE_RUNS_ROOT,
    ANALYZE_USE_LATEST_RUN_IF_EMPTY,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(fmean(values))


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(pstdev(values))


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round(0.95 * (len(ordered) - 1)))
    return float(ordered[idx])


def _resolve_run_dir() -> Path:
    configured = str(ANALYZE_RUN_DIR).strip()
    if configured:
        run_dir = resolve_repo_path(configured)
        if not run_dir.exists():
            raise RuntimeError(f"ANALYZE_RUN_DIR does not exist: {run_dir}")
        return run_dir

    if not ANALYZE_USE_LATEST_RUN_IF_EMPTY:
        raise RuntimeError("ANALYZE_RUN_DIR is empty and ANALYZE_USE_LATEST_RUN_IF_EMPTY is False")

    runs_root = resolve_repo_path(ANALYZE_RUNS_ROOT)
    ensure_dir(runs_root)

    candidates = [p for p in runs_root.iterdir() if p.is_dir() and (p / "metadata.json").exists()]
    if not candidates:
        raise RuntimeError(f"No run folders found in: {runs_root}")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _estimate_drop_pct(dts: list[float], nominal_fps: float) -> tuple[int, float | None]:
    if nominal_fps <= 0:
        return 0, None
    nominal_dt = 1.0 / nominal_fps
    dropped = 0
    for dt in dts:
        if dt > 1.5 * nominal_dt:
            dropped += max(0, int(round(dt / nominal_dt)) - 1)
    total = len(dts) + 1
    if total <= 0:
        return dropped, None
    drop_pct = 100.0 * float(dropped) / float(total + dropped)
    return dropped, drop_pct


def _count_freezes(frame_diffs: list[float]) -> tuple[int, int]:
    freeze_count = 0
    freeze_frames = 0
    streak = 0
    for diff in frame_diffs:
        if diff <= ANALYZE_FREEZE_DIFF_THRESHOLD:
            streak += 1
        else:
            if streak >= ANALYZE_FREEZE_MIN_FRAMES:
                freeze_count += 1
                freeze_frames += streak
            streak = 0
    if streak >= ANALYZE_FREEZE_MIN_FRAMES:
        freeze_count += 1
        freeze_frames += streak
    return freeze_count, freeze_frames


def _build_markdown(summary: dict[str, Any]) -> str:
    checks = summary["checks"]
    link_distance = summary["condition"].get("link_distance_m", summary["condition"].get("distance_m"))
    lines = [
        "# Camera Run Analysis",
        "",
        f"Run ID: `{summary['run_id']}`",
        f"Overall pass: `{summary['overall_pass']}`",
        "",
        "## Setup",
        f"- title: {summary['condition'].get('title')}",
        f"- link_distance_m: {link_distance}",
        f"- motion: {summary['condition'].get('motion')}",
        "",
        "## Metrics",
        f"- actual_fps: {summary['metrics'].get('actual_fps')}",
        f"- fps_ratio: {summary['metrics'].get('fps_ratio')}",
        f"- drop_pct: {summary['metrics'].get('drop_pct')}",
        f"- freeze_count: {summary['metrics'].get('freeze_count')}",
        f"- dt_p95_ms: {summary['metrics'].get('dt_p95_ms')}",
        f"- blur_frame_pct: {summary['metrics'].get('blur_frame_pct')}",
        f"- sharpness_mean: {summary['metrics'].get('sharpness_mean')}",
        "",
        "## Checks",
    ]

    for check in checks:
        status = "PASS" if bool(check["passed"]) else "FAIL"
        lines.append(f"- {status} `{check['name']}`: value={check['value']} threshold={check['threshold']}")

    lines += ["", "## Notes", "All metrics above are computed automatically from stream logs.", ""]
    return "\n".join(lines)


def main() -> None:
    run_dir = _resolve_run_dir()

    metadata = read_json(run_dir / "metadata.json")
    stream_rows = _read_csv(run_dir / "stream_log.csv")

    times = [_as_float(r.get("t_mono_s")) for r in stream_rows]
    times = [t for t in times if t is not None]

    dts = [_as_float(r.get("dt_s")) for r in stream_rows]
    dts = [dt for dt in dts if dt is not None and dt > 0]

    frame_diffs = [_as_float(r.get("frame_diff_mean")) for r in stream_rows]
    frame_diffs = [x for x in frame_diffs if x is not None]

    lap_vars = [_as_float(r.get("laplacian_var")) for r in stream_rows]
    lap_vars = [x for x in lap_vars if x is not None]

    duration_s = None
    actual_fps = None
    if len(times) >= 2:
        duration_s = float(times[-1] - times[0])
        if duration_s > 0:
            actual_fps = float((len(times) - 1) / duration_s)

    nominal_fps = float(metadata.get("camera", {}).get("fps_hint", 0.0))
    fps_ratio = None
    if nominal_fps > 0 and actual_fps is not None:
        fps_ratio = actual_fps / nominal_fps

    dropped_frames_est, drop_pct = _estimate_drop_pct(dts, nominal_fps=nominal_fps)
    freeze_count, freeze_frames = _count_freezes(frame_diffs)

    dt_ms = [dt * 1000.0 for dt in dts]
    dt_p95_ms = _p95(dt_ms)

    blur_frames = 0
    for lv in lap_vars:
        if lv < ANALYZE_BLUR_LAPLACIAN_THRESHOLD:
            blur_frames += 1
    blur_frame_pct = None
    if lap_vars:
        blur_frame_pct = 100.0 * float(blur_frames) / float(len(lap_vars))

    metrics = {
        "frame_count": len(times),
        "duration_s": duration_s,
        "nominal_fps": nominal_fps,
        "actual_fps": actual_fps,
        "fps_ratio": fps_ratio,
        "dropped_frames_est": dropped_frames_est,
        "drop_pct": drop_pct,
        "freeze_count": freeze_count,
        "freeze_frames": freeze_frames,
        "dt_mean_ms": _mean(dt_ms),
        "dt_std_ms": _std(dt_ms),
        "dt_p95_ms": dt_p95_ms,
        "sharpness_mean": _mean(lap_vars),
        "sharpness_std": _std(lap_vars),
        "blur_frame_pct": blur_frame_pct,
    }

    checks = [
        {
            "name": "fps_ratio",
            "passed": fps_ratio is not None and fps_ratio >= ANALYZE_MIN_FPS_RATIO,
            "value": fps_ratio,
            "threshold": f">= {ANALYZE_MIN_FPS_RATIO}",
        },
        {
            "name": "drop_pct",
            "passed": drop_pct is not None and drop_pct <= ANALYZE_MAX_DROP_PCT,
            "value": drop_pct,
            "threshold": f"<= {ANALYZE_MAX_DROP_PCT}",
        },
        {
            "name": "freeze_count",
            "passed": freeze_count <= ANALYZE_MAX_FREEZE_COUNT,
            "value": freeze_count,
            "threshold": f"<= {ANALYZE_MAX_FREEZE_COUNT}",
        },
        {
            "name": "dt_p95_ms",
            "passed": dt_p95_ms is not None and dt_p95_ms <= ANALYZE_MAX_DT_P95_MS,
            "value": dt_p95_ms,
            "threshold": f"<= {ANALYZE_MAX_DT_P95_MS}",
        },
        {
            "name": "blur_frame_pct",
            "passed": blur_frame_pct is not None and blur_frame_pct <= ANALYZE_MAX_BLUR_FRAME_PCT,
            "value": blur_frame_pct,
            "threshold": f"<= {ANALYZE_MAX_BLUR_FRAME_PCT}",
        },
    ]
    overall_pass = all(bool(c["passed"]) for c in checks)

    summary = {
        "run_id": metadata.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "condition": {
            "title": metadata.get("title"),
            "link_distance_m": metadata.get("link_distance_m", metadata.get("distance_m")),
            "distance_m": metadata.get("distance_m"),
            "receiver_distance": metadata.get("receiver_distance"),
            "motion": metadata.get("motion"),
            "purpose": metadata.get("purpose"),
        },
        "metrics": metrics,
        "checks": checks,
        "overall_pass": overall_pass,
    }

    summary_path = run_dir / "analysis_summary.json"
    report_path = run_dir / "analysis_report.md"

    write_json(summary_path, summary)
    report_path.write_text(_build_markdown(summary), encoding="utf-8")

    print(f"Analysis complete: {run_dir}")
    print(f"Overall pass: {overall_pass}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
