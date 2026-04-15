from pathlib import Path
from constants import *
from utils import *
import csv
import cv2


def _is_valid_bbox(box: tuple[float, float, float, float] | tuple[int, int, int, int]) -> bool:
    return float(box[2]) > 1.0 and float(box[3]) > 1.0


def _init_tracker_from_box(frame, box_xywh: tuple[int, int, int, int]):
    tracker = make_tracker(TRACKER_TYPE)
    x, y, w, h = map(int, box_xywh)
    tracker.init(frame, (x, y, w, h))
    return tracker


def main():
    video_path = Path(VIDEO_PATH)
    labels_root = Path(OUT_DIR)
    session_dir = create_unique_label_session_dir(labels_root, LABEL_CLASS_NAME)

    images_dir = session_dir / "images"
    labels_dir = session_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # FFMPEG backend gives stable decoding for recorded video files.
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        # Safety fallback when container metadata is missing/corrupt.
        src_fps = FPS_HINT

    # If your video is 30 fps and EXPORT_FPS is 10, export_step becomes 3, 
    # meaning you save every 3rd frame. That yields roughly 10 labeled frames per second of video.
    export_step = max(1, int(round(src_fps / EXPORT_FPS)))

    frame_index = 0
    # Read one frame up-front for ROI selection and frame size discovery.
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")

    if is_near_black(frame):
        # Some receivers output dark frames at startup; skip ahead a bit.
        skipped = 0
        for _ in range(STARTUP_PROBE_FRAMES):
            ok, candidate = cap.read()
            if not ok:
                break
            skipped += 1
            frame_index += 1
            # Once we get a non-black frame, we can start labeling.
            if not is_near_black(candidate):
                frame = candidate
                break
        if skipped:
            print(f"Skipped {skipped} near-black startup frame(s)")
        if is_near_black(frame):
            print("Warning: startup frame still near-black. Check receiver/channel before labeling.")

    H, W = frame.shape[:2]

    class_name = sanitize_class_folder_name(LABEL_CLASS_NAME)
    yolo_weights = find_labeling_yolo_weights(
        class_name=LABEL_CLASS_NAME,
        models_root=LABEL_YOLO_BEST_MODELS_DIR,
    )
    yolo_model = None
    if yolo_weights is not None:
        try:
            yolo_model = load_labeling_yolo_model(yolo_weights)
        except Exception as exc:
            print(f"Warning: failed to load YOLO weights {yolo_weights}: {exc}")

    yolo_enabled = bool(yolo_model is not None and LABEL_YOLO_ENABLED_AT_START)

    init_box_xywh: tuple[int, int, int, int] | None = None
    if yolo_enabled and yolo_model is not None:
        startup_det = yolo_best_detection_xywh(
            yolo_model=yolo_model,
            frame_bgr=frame,
            conf_threshold=LABEL_YOLO_CONF_THRESHOLD,
        )
        if startup_det is not None:
            x, y, w, h, conf = startup_det
            init_box_xywh = (x, y, w, h)
            print(
                "Initialized tracker from YOLO "
                f"(conf={conf:.2f}, box={init_box_xywh})."
            )

    if init_box_xywh is None:
        print("Draw bbox, press ENTER or SPACE to confirm, c to cancel")
        # Manual seed bbox for tracker initialization.
        init_box = cv2.selectROI("label", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("label")
        if not _is_valid_bbox(init_box):
            raise RuntimeError("ROI selection cancelled. Could not initialize tracker.")
        init_box_xywh = clamp_bbox(*init_box, W, H)

    # Fresh tracker instance bound to the chosen algorithm in constants.
    tracker = _init_tracker_from_box(frame, init_box_xywh)
    last_accepted_box = init_box_xywh

    meta_path = session_dir / "meta.csv"
    # Per-export bookkeeping: which source frame produced each saved sample.
    meta_f = open(meta_path, "w", newline="")
    meta = csv.writer(meta_f)

    # "tracker_ok" is useful for filtering out failed samples during training or analysis.
    # Trackers can hallucinate boxes when the view is obstructed or the target is out of frame.
    meta.writerow(
        [
            "export_index",
            "video_frame_index",
            "image_name",
            "label_name",
            "bbox_ok",
            "bbox_source",
            "tracker_ok",
            "yolo_enabled",
            "yolo_candidate_conf",
            "yolo_candidate_ok",
            "yolo_rejected_far",
            "yolo_jump_ratio",
        ]
    )

    export_index = 0

    print("Controls")
    class_dir = labels_root / class_name
    print(f"class folder: {class_dir}")
    print(f"all_data dir: {class_dir / LABEL_ALL_DATA_DIR}")
    print(f"session dir: {session_dir}")
    if yolo_weights is not None and yolo_model is not None:
        print(f"yolo weights: {yolo_weights}")
    elif yolo_weights is not None:
        print(f"yolo weights found but unavailable: {yolo_weights}")
    else:
        print(
            "yolo weights: not found "
            f"(searched in {resolve_repo_path(LABEL_YOLO_BEST_MODELS_DIR)})"
        )
    # Keep controls printed here so the operator can recover quickly if tracking drops.
    print("q quit")
    print("r redraw bbox and reinit tracker")
    print("p pause or resume")
    print("y toggle yolo assist on or off")
    print(f"review delay per frame: {TRACK_REVIEW_DELAY_S:.2f}s")

    paused = False
    # Keep UI responsive while intentionally slowing frame advance.
    review_delay_ms = max(1, int(TRACK_REVIEW_DELAY_S * 1000))

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                # End of file or decode failure.
                break
            frame_index += 1

        # Draw overlays on a copy so saved frame stays unmodified.
        display = frame.copy()

        # Update tracker on the current frame before rendering/export decisions.
        track_ok_raw, track_box_raw = tracker.update(frame)
        tracker_box_xywh: tuple[int, int, int, int] | None = None
        if track_ok_raw:
            tracker_box_xywh = clamp_bbox(*track_box_raw, W, H)

        yolo_candidate_xywh: tuple[int, int, int, int] | None = None
        yolo_candidate_conf = 0.0
        yolo_candidate_ok = False
        yolo_rejected_far = False
        yolo_jump_ratio = 0.0
        if yolo_enabled and yolo_model is not None:
            try:
                yolo_det = yolo_best_detection_xywh(
                    yolo_model=yolo_model,
                    frame_bgr=frame,
                    conf_threshold=LABEL_YOLO_CONF_THRESHOLD,
                )
            except Exception as exc:
                print(f"Warning: YOLO inference failed, disabling YOLO assist: {exc}")
                yolo_enabled = False
                yolo_det = None

            if yolo_det is not None:
                dx, dy, dw, dh, yolo_candidate_conf = yolo_det
                yolo_candidate_xywh = (dx, dy, dw, dh)
                yolo_candidate_ok = True
                if last_accepted_box is not None:
                    yolo_jump_ratio = bbox_center_jump_ratio(
                        prev_box=last_accepted_box,
                        next_box=yolo_candidate_xywh,
                        frame_w=W,
                        frame_h=H,
                    )
                    if yolo_jump_ratio > LABEL_YOLO_MAX_CENTER_JUMP_RATIO:
                        yolo_candidate_ok = False
                        yolo_rejected_far = True

        chosen_box_xywh: tuple[int, int, int, int] | None = None
        bbox_source = "none"
        if yolo_candidate_ok and yolo_candidate_xywh is not None:
            chosen_box_xywh = yolo_candidate_xywh
            bbox_source = "yolo"
            # Keep CSRT as immediate fallback if YOLO drifts or disappears.
            tracker = _init_tracker_from_box(frame, chosen_box_xywh)
        elif tracker_box_xywh is not None:
            chosen_box_xywh = tracker_box_xywh
            bbox_source = "tracker"

        if chosen_box_xywh is not None:
            x, y, w, h = chosen_box_xywh
            last_accepted_box = chosen_box_xywh
            if bbox_source == "yolo":
                color = (0, 200, 255)
                tag = f"YOLO {yolo_candidate_conf:.2f}"
            else:
                color = (0, 255, 0)
                tag = "TRACK"
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                display,
                tag,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        else:
            x = y = w = h = 0
            # Keep running even when both YOLO and tracker fail so operator can reinitialize with 'r'.
            cv2.putText(display, "TRACK LOST press r", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        yolo_mode_txt = "ON" if (yolo_enabled and yolo_model is not None) else "OFF"
        cv2.putText(display, f"frame {frame_index}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(display, f"yolo {yolo_mode_txt}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if yolo_rejected_far:
            cv2.putText(
                display,
                f"YOLO rejected jump={yolo_jump_ratio:.2f}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("track_label", display)
        # Slow playback to give operator correction time; block when paused.
        key = cv2.waitKey(review_delay_ms if not paused else 0) & 0xFF

        if key == ord("q"):
            break

        if key == ord("p"):
            # Pause freezes current frame; tracker/update resumes from same frame state.
            paused = not paused
            continue

        if key == ord("y"):
            if yolo_model is None:
                print("YOLO toggle ignored: no compatible weights loaded.")
                continue
            yolo_enabled = not yolo_enabled
            state = "ON" if yolo_enabled else "OFF"
            print(f"YOLO assist: {state}")
            continue

        if key == ord("r"):
            print("Redraw bbox, press ENTER or SPACE to confirm")
            new_box = cv2.selectROI("track_label", frame, fromCenter=False, showCrosshair=True)
            if not _is_valid_bbox(new_box):
                print("ROI redraw cancelled.")
                continue
            new_box_xywh = clamp_bbox(*new_box, W, H)
            # Recreate tracker to reset internal state after manual correction.
            tracker = _init_tracker_from_box(frame, new_box_xywh)
            last_accepted_box = new_box_xywh
            continue

        if frame_index % export_step == 0:
            # Stable zero-padded names keep image/label pairs aligned and sortable.
            img_name = f"frame_{export_index:06d}.jpg"
            lbl_name = f"frame_{export_index:06d}.txt"
            img_path = images_dir / img_name
            lbl_path = labels_dir / lbl_name

            # Save RGB frame and matching YOLO label file at the same index.
            cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            bbox_ok = chosen_box_xywh is not None
            if bbox_ok:
                with open(lbl_path, "w") as f:
                    f.write(yolo_line(CLASS_ID, x, y, w, h, W, H) + "\n")
            else:
                # Keep 1:1 image/label file count even when localization fails.
                with open(lbl_path, "w") as f:
                    f.write("")

            meta.writerow(
                [
                    export_index,
                    frame_index,
                    img_name,
                    lbl_name,
                    int(bbox_ok),
                    bbox_source,
                    int(track_ok_raw),
                    int(yolo_enabled and yolo_model is not None),
                    f"{yolo_candidate_conf:.4f}" if yolo_candidate_xywh is not None else "",
                    int(yolo_candidate_ok),
                    int(yolo_rejected_far),
                    f"{yolo_jump_ratio:.4f}" if yolo_candidate_xywh is not None else "",
                ]
            )
            # Flush each sample so progress survives interruption.
            meta_f.flush()
            export_index += 1

    # Explicit cleanup avoids locked files/windows in repeated labeling sessions.
    meta_f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved to {session_dir}")


if __name__ == "__main__":
    main()
