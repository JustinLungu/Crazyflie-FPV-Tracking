from pathlib import Path
from constants import *
from utils import clamp_bbox, is_near_black, make_tracker, yolo_line
import csv
import cv2



def main():
    video_path=Path(VIDEO_PATH),
    out_dir=Path(OUT_DIR)

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
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

    print("Draw bbox, press ENTER or SPACE to confirm, c to cancel")
    # Manually draw one bounding box to seed the tracker
    # Tracker propagates this bbox over time.
    init_box = cv2.selectROI("label", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("label")

    # Fresh tracker instance bound to the chosen algorithm in constants.
    tracker = make_tracker(TRACKER_TYPE)
    tracker.init(frame, init_box)

    meta_path = out_dir / "meta.csv"
    # Per-export bookkeeping: which source frame produced each saved sample.
    meta_f = open(meta_path, "w", newline="")
    meta = csv.writer(meta_f)

    # "tacker_ok" is a useful column for filtering out failed samples during training or analysis.
    # Trackers can hallucinate boxes when the view is obstructed or the target is out of frame.
    meta.writerow(["export_index", "video_frame_index", "image_name", "label_name", "tracker_ok"])

    export_index = 0

    print("Controls")
    # Keep controls printed here so the operator can recover quickly if tracking drops.
    print("q quit")
    print("r redraw bbox and reinit tracker")
    print("p pause or resume")

    paused = False

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
        track_ok, box = tracker.update(frame)
        if track_ok:
            x, y, w, h = clamp_bbox(*box, W, H)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            x = y = w = h = 0
            # Keep running even when tracker fails so operator can reinitialize with 'r'.
            cv2.putText(display, "TRACK LOST press r", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.putText(display, f"frame {frame_index}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("track_label", display)
        # waitKey(0) while paused so stepping requires explicit key input.
        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord("q"):
            break

        if key == ord("p"):
            # Pause freezes current frame; tracker/update resumes from same frame state.
            paused = not paused
            continue

        if key == ord("r"):
            print("Redraw bbox, press ENTER or SPACE to confirm")
            new_box = cv2.selectROI("track_label", frame, fromCenter=False, showCrosshair=True)
            # Recreate tracker to reset internal state after manual correction.
            tracker = make_tracker(TRACKER_TYPE)
            tracker.init(frame, new_box)
            continue

        if frame_index % export_step == 0:
            # Stable zero-padded names keep image/label pairs aligned and sortable.
            img_name = f"frame_{export_index:06d}.jpg"
            lbl_name = f"frame_{export_index:06d}.txt"
            img_path = images_dir / img_name
            lbl_path = labels_dir / lbl_name

            # Save RGB frame and matching YOLO label file at the same index.
            cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            if track_ok:
                with open(lbl_path, "w") as f:
                    f.write(yolo_line(CLASS_ID, x, y, w, h, W, H) + "\n")
            else:
                # Keep 1:1 image/label file count even when tracking fails.
                with open(lbl_path, "w") as f:
                    f.write("")

            meta.writerow([export_index, frame_index, img_name, lbl_name, int(track_ok)])
            # Flush each sample so progress survives interruption.
            meta_f.flush()
            export_index += 1

    # Explicit cleanup avoids locked files/windows in repeated labeling sessions.
    meta_f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
