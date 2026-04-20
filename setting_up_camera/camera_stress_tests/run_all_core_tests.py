from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
import textwrap

import cv2

from common import open_camera
from constants import (
    BONUS_RUNS,
    CAMERA_BUFFER_SIZE,
    CAMERA_DEVICE,
    CAMERA_FOURCC,
    CAMERA_FPS_HINT,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    RUN_ALL_INCLUDE_BONUS,
    RUN_ALL_PREFLIGHT_WINDOW_NAME,
    RUN_ALL_PREFLIGHT_NO_FRAME_TIMEOUT_S,
    RUN_ALL_PREFLIGHT_DISPLAY_SCALE,
    RUN_ALL_PREFLIGHT_MIN_HEIGHT,
    RUN_ALL_PREFLIGHT_MIN_WIDTH,
    RUN_ALL_PROMPT_BETWEEN_RUNS,
    RUN_ALL_SHOW_PREFLIGHT_PREVIEW,
    RUN_ALL_STOP_ON_ERROR,
    TEST_RUNS,
)

RUN_ID_ENV_KEY = "CAMERA_STRESS_RUN_ID"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _link_distance_m(run: dict[str, object]) -> float | None:
    raw = run.get("link_distance_m", run.get("distance_m"))
    try:
        if raw is None:
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def _prompt_action() -> str:
    raw = input("Action: [Enter]=start, s=skip, q=quit: ").strip().lower()
    if raw in {"q", "quit"}:
        return "quit"
    if raw in {"s", "skip"}:
        return "skip"
    return "start"


def _print_run_header(index: int, total: int, run: dict[str, object]) -> None:
    link_distance = _link_distance_m(run)

    print("\n" + "=" * 78)
    print(f"Run {index}/{total}: {run['title']}")
    print(f"- id: {run['id']}")
    print(f"- link_distance_m: {link_distance}")
    print(f"- motion: {run['motion']}")
    print(f"- duration_s: {run['duration_s']}")
    print(f"- purpose: {run['purpose']}")
    print(f"- check_focus: {run['check_focus']}")
    setup = _text_list(run.get("setup_checklist"))
    observe = _text_list(run.get("observe_realtime"))
    if setup:
        print("- setup_checklist:")
        for item in setup:
            print(f"  * {item}")
    if observe:
        print("- observe_realtime:")
        for item in observe:
            print(f"  * {item}")


def _text_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _movement_guidance(run: dict[str, object]) -> list[str]:
    from_spec = _text_list(run.get("setup_checklist"))
    if from_spec:
        return from_spec

    motion = str(run.get("motion", "")).strip().lower()
    link_distance = _link_distance_m(run)

    lines = [
        f"Set camera-drone to receiver distance to ~{0.0 if link_distance is None else link_distance:.1f} m",
    ]

    if motion == "static":
        lines.append("Keep camera drone mostly static")
    else:
        lines.append(f"Motion profile: {motion}")

    lines.append("Keep receiver+laptop fixed in one place")
    lines.append("Keep drone orientation comparable across runs")

    lines.append("Controls: c/space/enter=continue, s=skip, q=quit")
    return lines


def _realtime_focus(run: dict[str, object]) -> list[str]:
    from_spec = _text_list(run.get("observe_realtime"))
    if from_spec:
        return from_spec
    return _text_list(run.get("check_focus"))


def _draw_preflight_overlay(frame, run: dict[str, object]) -> None:
    h, w = frame.shape[:2]

    # Semi-transparent panel to keep text readable.
    panel = frame.copy()
    cv2.rectangle(panel, (6, 6), (w - 6, h - 6), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.45, frame, 0.55, 0.0, frame)

    # Center crosshair for placement alignment.
    cx = w // 2
    cy = h // 2
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 255, 255), 2)
    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 255, 255), 2)

    lines: list[str] = []
    link_distance = _link_distance_m(run)
    lines.append(f"{run['title']} ({run['id']})")
    lines.append(
        f"link_distance={0.0 if link_distance is None else link_distance:.1f}m | motion={run['motion']}"
    )
    lines.append("")
    lines.append("SETUP NOW:")
    for item in _movement_guidance(run):
        lines.append(f"[ ] {item}")
    lines.append("")
    lines.append("WATCH DURING RUN:")
    for item in _realtime_focus(run):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Keys: c/space/enter=continue, s=skip, q=quit")

    wrapped: list[str] = []
    for line in lines:
        if not line.strip():
            wrapped.append("")
            continue
        parts = textwrap.wrap(line, width=66) or [line]
        wrapped.extend(parts)

    max_lines = max(1, (h - 24) // 22)
    display_lines = wrapped[:max_lines]
    if len(wrapped) > max_lines:
        display_lines[-1] = "... (resize window or shorten text)"

    y = 24
    for line in display_lines:
        color = (0, 255, 0)
        if line in {"SETUP NOW:", "WATCH DURING RUN:"}:
            color = (0, 255, 255)
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 24


def _prepare_preflight_view(frame):
    scale = max(1.0, float(RUN_ALL_PREFLIGHT_DISPLAY_SCALE))
    h, w = frame.shape[:2]

    out_w = max(int(round(w * scale)), int(RUN_ALL_PREFLIGHT_MIN_WIDTH))
    out_h = max(int(round(h * scale)), int(RUN_ALL_PREFLIGHT_MIN_HEIGHT))

    if out_w == w and out_h == h:
        return frame

    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def _show_preflight_preview(run: dict[str, object]) -> str:
    if not RUN_ALL_SHOW_PREFLIGHT_PREVIEW:
        return "ready"

    try:
        cap = open_camera(
            camera_device=CAMERA_DEVICE,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps_hint=CAMERA_FPS_HINT,
            buffer_size=CAMERA_BUFFER_SIZE,
            fourcc=CAMERA_FOURCC,
        )
    except Exception as exc:  # pragma: no cover - hardware-dependent
        print(f"Preflight preview unavailable: {exc}")
        return "no_preview"

    try:
        cv2.namedWindow(RUN_ALL_PREFLIGHT_WINDOW_NAME, cv2.WINDOW_NORMAL)
        window_w = max(
            int(round(float(CAMERA_WIDTH) * float(RUN_ALL_PREFLIGHT_DISPLAY_SCALE))),
            int(RUN_ALL_PREFLIGHT_MIN_WIDTH),
        )
        window_h = max(
            int(round(float(CAMERA_HEIGHT) * float(RUN_ALL_PREFLIGHT_DISPLAY_SCALE))),
            int(RUN_ALL_PREFLIGHT_MIN_HEIGHT),
        )
        cv2.resizeWindow(RUN_ALL_PREFLIGHT_WINDOW_NAME, window_w, window_h)
    except cv2.error as exc:  # pragma: no cover - GUI/display dependent
        cap.release()
        print(f"Preflight preview window could not open: {exc}")
        return "no_preview"

    start_wait = time.monotonic()
    timed_out = False
    timeout_enabled = float(RUN_ALL_PREFLIGHT_NO_FRAME_TIMEOUT_S) > 0.0
    waiting_notice_printed = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if (
                    timeout_enabled
                    and (time.monotonic() - start_wait) >= float(RUN_ALL_PREFLIGHT_NO_FRAME_TIMEOUT_S)
                ):
                    timed_out = True
                    return "no_preview"
                if not waiting_notice_printed:
                    if timeout_enabled:
                        print(
                            "Waiting for camera frames in preflight... "
                            f"timeout in {RUN_ALL_PREFLIGHT_NO_FRAME_TIMEOUT_S:.1f}s."
                        )
                    else:
                        print(
                            "Waiting for camera frames in preflight... "
                            "no timeout configured (will wait indefinitely)."
                        )
                    waiting_notice_printed = True
                time.sleep(0.01)
                continue

            # First valid frame received; no longer waiting for camera startup.
            start_wait = time.monotonic()
            if waiting_notice_printed:
                print("Camera frames received. Use preview keys to continue/skip/quit.")
                waiting_notice_printed = False

            view = _prepare_preflight_view(frame)
            _draw_preflight_overlay(view, run)

            cv2.imshow(RUN_ALL_PREFLIGHT_WINDOW_NAME, view)
            key = cv2.waitKey(1) & 0xFF

            if key in {ord("q"), 27}:
                return "quit"
            if key in {ord("s")}:
                return "skip"
            if key in {ord("c"), ord(" "), 10, 13}:
                return "ready"
    finally:
        cap.release()
        try:
            cv2.destroyWindow(RUN_ALL_PREFLIGHT_WINDOW_NAME)
        except cv2.error:
            pass
        if timed_out and timeout_enabled:
            print(
                "Preflight preview timed out waiting for camera frames "
                f"({RUN_ALL_PREFLIGHT_NO_FRAME_TIMEOUT_S:.1f}s)."
            )


def main() -> None:
    planned = list(TEST_RUNS)
    if RUN_ALL_INCLUDE_BONUS:
        planned.extend(BONUS_RUNS)

    if not planned:
        raise RuntimeError("No runs configured.")

    print("Camera stress batch runner")
    print(f"Planned runs: {len(planned)}")
    print("Minimal link-distance protocol:")
    print("- receiver+laptop fixed in one place")
    print("- only variable = camera-drone to receiver-node distance")
    print("- no target-motion scenarios in this campaign")
    if RUN_ALL_INCLUDE_BONUS:
        print("Mode: core + bonus")
    else:
        print("Mode: core only")
    if float(RUN_ALL_PREFLIGHT_NO_FRAME_TIMEOUT_S) <= 0.0:
        print("Preflight no-frame timeout: disabled (wait indefinitely for stream).")
    else:
        print(f"Preflight no-frame timeout: {RUN_ALL_PREFLIGHT_NO_FRAME_TIMEOUT_S:.1f}s")

    repo_root = _repo_root()
    script_path = repo_root / "setting_up_camera/camera_stress_tests/run_camera_test.py"

    done: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for idx, run in enumerate(planned, start=1):
        _print_run_header(idx, len(planned), run)

        if RUN_ALL_SHOW_PREFLIGHT_PREVIEW:
            print(
                "Opening preflight preview. "
                "Press c/space/enter in the preview window when setup is ready."
            )
        preview_action = _show_preflight_preview(run)
        if preview_action == "quit":
            print("Stopping batch by user request (preflight).")
            break
        if preview_action == "skip":
            skipped.append(str(run["id"]))
            print(f"Skipped: {run['id']} (preflight)")
            continue

        if preview_action == "no_preview":
            print(
                "Preflight preview unavailable; using terminal confirmation only. "
                "Check CAMERA_DEVICE in constants.py if this is unexpected."
            )

        if RUN_ALL_PROMPT_BETWEEN_RUNS or preview_action == "no_preview":
            action = _prompt_action()
            if action == "quit":
                print("Stopping batch by user request.")
                break
            if action == "skip":
                skipped.append(str(run["id"]))
                print(f"Skipped: {run['id']}")
                continue

        env = os.environ.copy()
        env[RUN_ID_ENV_KEY] = str(run["id"])

        cmd = [sys.executable, str(script_path)]
        result = subprocess.run(cmd, cwd=str(repo_root), env=env)

        if result.returncode == 0:
            done.append(str(run["id"]))
            print(f"Completed: {run['id']}")
            continue

        failed.append(str(run["id"]))
        print(f"Failed: {run['id']} (exit={result.returncode})")
        if RUN_ALL_STOP_ON_ERROR:
            print("Stopping batch because RUN_ALL_STOP_ON_ERROR=True")
            break

    print("\n" + "-" * 78)
    print("Batch summary")
    print(f"- completed: {len(done)}")
    print(f"- skipped: {len(skipped)}")
    print(f"- failed: {len(failed)}")
    if done:
        print(f"- completed_ids: {', '.join(done)}")
    if skipped:
        print(f"- skipped_ids: {', '.join(skipped)}")
    if failed:
        print(f"- failed_ids: {', '.join(failed)}")


if __name__ == "__main__":
    main()
