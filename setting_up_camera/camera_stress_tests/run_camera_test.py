from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
import time
from pathlib import Path

import cv2

from common import ensure_dir, laplacian_variance, now_timestamp, open_camera, resolve_repo_path, write_json
from constants import (
    CAMERA_BUFFER_SIZE,
    CAMERA_DEVICE,
    CAMERA_FOURCC,
    CAMERA_FPS_HINT,
    CAMERA_HEIGHT,
    CAMERA_STRESS_RUNS_ROOT,
    CAMERA_WIDTH,
    RUN_NOTES,
    RUN_OPERATOR,
    RUN_RECORD_RAW_VIDEO,
    RUN_SELECTED_ID,
    RUN_SHOW_PREVIEW,
    TEST_RUNS,
    BONUS_RUNS,
)


RUN_ID_ENV_KEY = "CAMERA_STRESS_RUN_ID"


def _find_run(run_id: str) -> dict[str, object]:
    all_runs = [*TEST_RUNS, *BONUS_RUNS]
    for run in all_runs:
        if str(run.get("id")) == str(run_id):
            return run
    known = ", ".join(str(r.get("id")) for r in all_runs)
    raise RuntimeError(f"Unknown RUN_SELECTED_ID='{run_id}'. Known: {known}")


def _select_run_id() -> tuple[str, str]:
    env_value = str(os.environ.get(RUN_ID_ENV_KEY, "")).strip()
    if env_value:
        return env_value, "env"
    configured = str(RUN_SELECTED_ID).strip()
    if not configured:
        raise RuntimeError("RUN_SELECTED_ID in constants.py is empty.")
    return configured, "constants"


def _run_focus_lines(run_spec: dict[str, object]) -> list[str]:
    raw = run_spec.get("observe_realtime", [])
    if isinstance(raw, (list, tuple)):
        lines = [str(x).strip() for x in raw if str(x).strip()]
        if lines:
            return lines
    check_focus = str(run_spec.get("check_focus", "")).strip()
    if not check_focus:
        return []
    return [x.strip() for x in check_focus.split(",") if x.strip()]


def main() -> None:
    selected_run_id, selected_source = _select_run_id()
    run_spec = _find_run(selected_run_id)
    run_id = f"{run_spec['id']}_{now_timestamp()}"

    output_root = resolve_repo_path(CAMERA_STRESS_RUNS_ROOT)
    run_dir = ensure_dir(output_root / run_id)

    metadata = {
        "run_id": run_id,
        "selected_template": run_spec["id"],
        "selected_run_id_source": selected_source,
        "title": run_spec["title"],
        "link_distance_m": float(run_spec.get("link_distance_m", run_spec.get("distance_m", 0.0))),
        # Backward-compatible key kept for older summary tooling.
        "distance_m": float(run_spec.get("link_distance_m", run_spec.get("distance_m", 0.0))),
        "receiver_distance": str(
            run_spec.get("receiver_distance", f"{run_spec.get('link_distance_m', run_spec.get('distance_m', ''))}m")
        ),
        "motion": str(run_spec["motion"]),
        "duration_requested_s": float(run_spec["duration_s"]),
        "purpose": str(run_spec["purpose"]),
        "check_focus": str(run_spec["check_focus"]),
        "operator": str(RUN_OPERATOR),
        "notes": str(RUN_NOTES),
        "camera": {
            "device": CAMERA_DEVICE,
            "width": int(CAMERA_WIDTH),
            "height": int(CAMERA_HEIGHT),
            "fps_hint": float(CAMERA_FPS_HINT),
            "buffer_size": int(CAMERA_BUFFER_SIZE),
            "fourcc": str(CAMERA_FOURCC),
        },
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }

    metadata_path = run_dir / "metadata.json"
    write_json(metadata_path, metadata)

    stream_log_path = run_dir / "stream_log.csv"

    cap = open_camera(
        camera_device=CAMERA_DEVICE,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        fps_hint=CAMERA_FPS_HINT,
        buffer_size=CAMERA_BUFFER_SIZE,
        fourcc=CAMERA_FOURCC,
    )

    writer_raw = None
    if RUN_RECORD_RAW_VIDEO:
        writer_raw = cv2.VideoWriter(
            str(run_dir / "raw_video.avi"),
            cv2.VideoWriter_fourcc(*"MJPG"),
            float(CAMERA_FPS_HINT),
            (int(CAMERA_WIDTH), int(CAMERA_HEIGHT)),
        )

    print(f"Starting camera run: {run_id}")
    print(f"Selected template id: {run_spec['id']} ({selected_source})")
    print(f"Template: {run_spec['title']}")
    print(f"Purpose: {run_spec['purpose']}")
    print(f"Output: {run_dir}")
    print("Press q or ESC to stop early.")

    frame_idx = 0
    success_frames = 0
    prev_frame = None
    prev_t = None
    run_start = time.monotonic()

    with open(stream_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame_idx",
                "t_wall_s",
                "t_mono_s",
                "dt_s",
                "inst_fps",
                "frame_diff_mean",
                "laplacian_var",
                "read_ok",
                "width",
                "height",
            ]
        )

        try:
            while True:
                elapsed = time.monotonic() - run_start
                if elapsed >= float(run_spec["duration_s"]):
                    break

                ok, frame = cap.read()
                now_wall = time.time()
                now_mono = time.monotonic()

                if not ok:
                    time.sleep(0.005)
                    continue

                frame_idx += 1
                success_frames += 1

                dt_s = None if prev_t is None else (now_mono - prev_t)
                prev_t = now_mono
                inst_fps = None if dt_s is None or dt_s <= 0 else (1.0 / dt_s)

                frame_diff_mean = None
                if prev_frame is not None:
                    frame_diff_mean = float(cv2.absdiff(frame, prev_frame).mean())
                prev_frame = frame

                lap_var = laplacian_variance(frame)

                writer.writerow(
                    [
                        frame_idx,
                        f"{now_wall:.6f}",
                        f"{now_mono:.6f}",
                        "" if dt_s is None else f"{dt_s:.6f}",
                        "" if inst_fps is None else f"{inst_fps:.6f}",
                        "" if frame_diff_mean is None else f"{frame_diff_mean:.6f}",
                        f"{lap_var:.6f}",
                        1,
                        int(frame.shape[1]),
                        int(frame.shape[0]),
                    ]
                )

                if writer_raw is not None:
                    writer_raw.write(frame)

                if RUN_SHOW_PREVIEW:
                    view = frame.copy()
                    focus_lines = _run_focus_lines(run_spec)
                    lines = [
                        f"{run_spec['title']}",
                        f"frame {frame_idx}",
                        f"fps {0.0 if inst_fps is None else inst_fps:5.1f}",
                        f"sharpness {lap_var:7.1f}",
                        "watch now:",
                    ]
                    for item in focus_lines[:3]:
                        lines.append(f"- {item}")
                    y = 24
                    for line in lines:
                        cv2.putText(
                            view,
                            line,
                            (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        y += 24
                    cv2.imshow("Camera Stress Run", view)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {ord("q"), 27}:
                        break

        finally:
            cap.release()
            if writer_raw is not None:
                writer_raw.release()
            cv2.destroyAllWindows()

    elapsed_s = time.monotonic() - run_start

    metadata["status"] = "completed"
    metadata["run_finished_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["duration_measured_s"] = float(elapsed_s)
    metadata["successful_frames"] = int(success_frames)
    metadata["stream_log"] = str(stream_log_path)
    write_json(metadata_path, metadata)

    print(f"Run finished: {run_id}")
    print(f"Frames captured: {success_frames}")
    print(f"Duration: {elapsed_s:.2f}s")
    print(f"Logs: {run_dir}")
    print("Next: run analyze_camera_test.py")


if __name__ == "__main__":
    main()
