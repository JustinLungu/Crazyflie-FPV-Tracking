from __future__ import annotations

import csv

from common import ensure_dir, resolve_repo_path
from constants import BONUS_RUNS, CAMERA_STRESS_TEST_PLAN_CSV, TEST_RUNS


def _join_list(raw: object) -> str:
    if isinstance(raw, (list, tuple)):
        return " | ".join(str(x).strip() for x in raw if str(x).strip())
    return str(raw) if raw is not None else ""


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for run in TEST_RUNS:
        link_distance = run.get("link_distance_m", run.get("distance_m"))
        rows.append(
            {
                "set": "core",
                "run_id": run["id"],
                "title": run["title"],
                "link_distance_m": link_distance,
                "motion": run["motion"],
                "duration_s": run["duration_s"],
                "purpose": run["purpose"],
                "check_focus": run["check_focus"],
                "setup_checklist": _join_list(run.get("setup_checklist", [])),
                "observe_realtime": _join_list(run.get("observe_realtime", [])),
                "how_to_run": (
                    f"Set RUN_SELECTED_ID='{run['id']}' in constants.py, then run "
                    "uv run python setting_up_camera/camera_stress_tests/run_camera_test.py"
                ),
            }
        )

    for run in BONUS_RUNS:
        link_distance = run.get("link_distance_m", run.get("distance_m"))
        rows.append(
            {
                "set": "bonus",
                "run_id": run["id"],
                "title": run["title"],
                "link_distance_m": link_distance,
                "motion": run["motion"],
                "duration_s": run["duration_s"],
                "purpose": run["purpose"],
                "check_focus": run["check_focus"],
                "setup_checklist": _join_list(run.get("setup_checklist", [])),
                "observe_realtime": _join_list(run.get("observe_realtime", [])),
                "how_to_run": (
                    f"Set RUN_SELECTED_ID='{run['id']}' in constants.py, then run "
                    "uv run python setting_up_camera/camera_stress_tests/run_camera_test.py"
                ),
            }
        )

    return rows


def main() -> None:
    output_path = resolve_repo_path(CAMERA_STRESS_TEST_PLAN_CSV)
    ensure_dir(output_path.parent)

    fieldnames = [
        "set",
        "run_id",
        "title",
        "link_distance_m",
        "motion",
        "duration_s",
        "purpose",
        "check_focus",
        "setup_checklist",
        "observe_realtime",
        "how_to_run",
    ]

    rows = _rows()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote test plan: {output_path}")
    print(f"Core runs: {len(TEST_RUNS)}")
    print(f"Bonus runs: {len(BONUS_RUNS)}")


if __name__ == "__main__":
    main()
