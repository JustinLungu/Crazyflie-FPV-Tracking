import csv
import time
from dataclasses import dataclass
from pathlib import Path
import cv2


@dataclass
class Config:
    video_path: Path
    out_dir: Path
    class_id: int = 0
    export_fps: float = 10.0
    tracker_type: str = "CSRT"  # CSRT is accurate, KCF is faster, MOSSE is fastest
    startup_probe_frames: int = 30  # skip startup frames that are near-black/blank


def make_tracker(tracker_type: str):
    t = tracker_type.upper()
    if t == "CSRT":
        return cv2.TrackerCSRT_create()
    if t == "KCF":
        return cv2.TrackerKCF_create()
    if t == "MOSSE":
        return cv2.TrackerMOSSE_create()
    raise ValueError(f"Unknown tracker_type {tracker_type}")


def clamp_bbox(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def yolo_line(class_id, x, y, w, h, W, H):
    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    nw = w / W
    nh = h / H
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def is_near_black(frame) -> bool:
    return float(frame.mean()) < 25.0 and (int(frame.max()) - int(frame.min())) < 8


def main():
    cfg = Config(
        video_path=Path("data/raw_data/video_session_20260211_174944/video.avi"),
        out_dir=Path("data/labels_session/"),
        export_fps=10.0,
        tracker_type="CSRT",    
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = cfg.out_dir / "images"
    labels_dir = cfg.out_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(cfg.video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {cfg.video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        src_fps = 30.0

    export_step = max(1, int(round(src_fps / cfg.export_fps)))

    frame_index = 0
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame")

    if is_near_black(frame):
        skipped = 0
        for _ in range(cfg.startup_probe_frames):
            ok, candidate = cap.read()
            if not ok:
                break
            skipped += 1
            frame_index += 1
            if not is_near_black(candidate):
                frame = candidate
                break
        if skipped:
            print(f"Skipped {skipped} near-black startup frame(s)")
        if is_near_black(frame):
            print("Warning: startup frame still near-black. Check receiver/channel before labeling.")

    H, W = frame.shape[:2]

    print("Draw bbox, press ENTER or SPACE to confirm, c to cancel")
    init_box = cv2.selectROI("label", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("label")

    tracker = make_tracker(cfg.tracker_type)
    tracker.init(frame, init_box)

    meta_path = cfg.out_dir / "meta.csv"
    meta_f = open(meta_path, "w", newline="")
    meta = csv.writer(meta_f)
    meta.writerow(["export_index", "video_frame_index", "image_name", "label_name", "tracker_ok"])

    export_index = 0

    print("Controls")
    print("q quit")
    print("r redraw bbox and reinit tracker")
    print("p pause or resume")

    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

        display = frame.copy()

        track_ok, box = tracker.update(frame)
        if track_ok:
            x, y, w, h = clamp_bbox(*box, W, H)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            x = y = w = h = 0
            cv2.putText(display, "TRACK LOST press r", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.putText(display, f"frame {frame_index}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("track_label", display)
        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord("q"):
            break

        if key == ord("p"):
            paused = not paused
            continue

        if key == ord("r"):
            print("Redraw bbox, press ENTER or SPACE to confirm")
            new_box = cv2.selectROI("track_label", frame, fromCenter=False, showCrosshair=True)
            tracker = make_tracker(cfg.tracker_type)
            tracker.init(frame, new_box)
            continue

        if frame_index % export_step == 0:
            img_name = f"frame_{export_index:06d}.jpg"
            lbl_name = f"frame_{export_index:06d}.txt"
            img_path = images_dir / img_name
            lbl_path = labels_dir / lbl_name

            cv2.imwrite(str(img_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            if track_ok:
                with open(lbl_path, "w") as f:
                    f.write(yolo_line(cfg.class_id, x, y, w, h, W, H) + "\n")
            else:
                with open(lbl_path, "w") as f:
                    f.write("")

            meta.writerow([export_index, frame_index, img_name, lbl_name, int(track_ok)])
            meta_f.flush()
            export_index += 1

    meta_f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved to {cfg.out_dir}")


if __name__ == "__main__":
    main()
