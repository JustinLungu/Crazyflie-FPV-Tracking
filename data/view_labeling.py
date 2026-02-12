import argparse
import csv
from pathlib import Path

import cv2

from constants import (
    LABEL_ALL_DATA_DIR,
    LABEL_CLASS_NAME,
    LABEL_SESSION_PREFIX,
    OUT_DIR,
    TRACK_REVIEW_DELAY_S,
)
from utils import sanitize_class_folder_name


# Keyboard mappings from cv2.waitKeyEx across common Linux/OpenCV backends.
KEY_QUIT = {ord("q"), 27}  # q, ESC
KEY_TOGGLE_PLAY = {ord(" ")}
KEY_PREV = {ord("a"), 2424832, 65361}
KEY_NEXT = {ord("d"), 2555904, 65363}
KEY_DELETE = {ord("x"), ord("X")}
KEY_CONFIRM_YES = {ord("y"), ord("Y")}
KEY_CONFIRM_NO = {ord("n"), ord("N"), 27}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review labeled frames with playback, navigation, and delete support."
    )
    parser.add_argument(
        "--session",
        type=str,
        default="",
        help=(
            "Path to a label session directory. "
            "If omitted, the latest session under labels/<class>/all_data is used."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=TRACK_REVIEW_DELAY_S,
        help="Playback delay in seconds between frames while in play mode.",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Start in paused mode.",
    )
    return parser.parse_args()


def find_latest_session(labels_root: Path, class_name: str) -> Path:
    class_dir = labels_root / sanitize_class_folder_name(class_name)
    all_data_dir = class_dir / LABEL_ALL_DATA_DIR
    if not all_data_dir.exists():
        raise RuntimeError(f"Missing class/all_data directory: {all_data_dir}")

    sessions = [d for d in all_data_dir.iterdir() if d.is_dir() and d.name.startswith(LABEL_SESSION_PREFIX)]
    if not sessions:
        raise RuntimeError(f"No label sessions found in: {all_data_dir}")
    return sorted(sessions, key=lambda d: d.name)[-1]


def session_paths(session_dir: Path) -> tuple[Path, Path, Path]:
    images_dir = session_dir / "images"
    labels_dir = session_dir / "labels"
    meta_path = session_dir / "meta.csv"
    if not images_dir.exists():
        raise RuntimeError(f"Missing images directory: {images_dir}")
    if not labels_dir.exists():
        raise RuntimeError(f"Missing labels directory: {labels_dir}")
    return images_dir, labels_dir, meta_path


def collect_entries(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    entries: list[tuple[Path, Path]] = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        entries.append((image_path, label_path))
    if not entries:
        raise RuntimeError(f"No images found in {images_dir}")
    return entries


def parse_yolo_labels(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int, int]]:
    boxes: list[tuple[int, int, int, int, int]] = []
    if not label_path.exists():
        return boxes

    for raw in label_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except ValueError:
            continue

        x1 = int((cx - bw / 2.0) * width)
        y1 = int((cy - bh / 2.0) * height)
        x2 = int((cx + bw / 2.0) * width)
        y2 = int((cy + bh / 2.0) * height)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def draw_overlay(
    image,
    boxes: list[tuple[int, int, int, int, int]],
    index: int,
    total: int,
    image_name: str,
    label_path: Path,
    playing: bool,
    delay_s: float,
    info_message: str,
):
    frame = image.copy()
    for cls, x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"class {cls}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    status = "PLAY" if playing else "PAUSE"
    cv2.putText(
        frame,
        f"{status}  frame {index + 1}/{total}  boxes={len(boxes)}  delay={delay_s:.2f}s",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(frame, image_name, (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if not label_path.exists():
        cv2.putText(
            frame,
            "Warning: label file missing",
            (15, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    if info_message:
        cv2.putText(frame, info_message, (15, frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


def confirm_delete(frame) -> bool:
    prompt = frame.copy()
    cv2.putText(prompt, "Delete this frame? y=yes n=no", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow("label_review", prompt)
    while True:
        key = cv2.waitKeyEx(0)
        if key in KEY_CONFIRM_YES:
            return True
        if key in KEY_CONFIRM_NO:
            return False


def remove_meta_rows(meta_path: Path, image_name: str, label_name: str) -> int:
    if not meta_path.exists():
        return 0

    with open(meta_path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0

    header = rows[0]
    data = rows[1:]
    img_col = header.index("image_name") if "image_name" in header else None
    lbl_col = header.index("label_name") if "label_name" in header else None

    kept_rows = []
    removed = 0
    for row in data:
        match = False
        if img_col is not None and img_col < len(row) and row[img_col] == image_name:
            match = True
        if lbl_col is not None and lbl_col < len(row) and row[lbl_col] == label_name:
            match = True
        if match:
            removed += 1
        else:
            kept_rows.append(row)

    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept_rows)
    return removed


def main():
    args = parse_args()

    labels_root = Path(OUT_DIR)
    if args.session:
        session_dir = Path(args.session)
    else:
        session_dir = find_latest_session(labels_root, LABEL_CLASS_NAME)

    images_dir, labels_dir, meta_path = session_paths(session_dir)
    entries = collect_entries(images_dir, labels_dir)

    delay_s = max(0.0, float(args.delay))
    delay_ms = max(1, int(delay_s * 1000))
    playing = not args.start_paused
    index = 0
    info_message = ""

    print(f"Reviewing session: {session_dir}")
    print("Controls")
    print("space: play/pause")
    print("a or Left Arrow: previous frame")
    print("d or Right Arrow: next frame")
    print("x: delete current image+label (with confirmation)")
    print("q or ESC: quit")

    while True:
        image_path, label_path = entries[index]
        image = cv2.imread(str(image_path))
        if image is None:
            info_message = f"Could not read image: {image_path.name}. Removing entry."
            entries.pop(index)
            if not entries:
                break
            index = min(index, len(entries) - 1)
            continue

        height, width = image.shape[:2]
        boxes = parse_yolo_labels(label_path, width, height)
        frame = draw_overlay(
            image=image,
            boxes=boxes,
            index=index,
            total=len(entries),
            image_name=image_path.name,
            label_path=label_path,
            playing=playing,
            delay_s=delay_s,
            info_message=info_message,
        )
        cv2.imshow("label_review", frame)
        info_message = ""

        key = cv2.waitKeyEx(delay_ms if playing else 0)

        if key == -1:
            if playing:
                if index < len(entries) - 1:
                    index += 1
                else:
                    playing = False
            continue

        if key in KEY_QUIT:
            break
        if key in KEY_TOGGLE_PLAY:
            playing = not playing
            continue
        if key in KEY_PREV:
            index = max(0, index - 1)
            playing = False
            continue
        if key in KEY_NEXT:
            index = min(len(entries) - 1, index + 1)
            playing = False
            continue
        if key in KEY_DELETE:
            if confirm_delete(frame):
                image_name = image_path.name
                label_name = label_path.name
                if image_path.exists():
                    image_path.unlink()
                if label_path.exists():
                    label_path.unlink()
                removed_rows = remove_meta_rows(meta_path, image_name=image_name, label_name=label_name)

                entries.pop(index)
                if not entries:
                    print("All frames deleted. Exiting.")
                    break
                index = min(index, len(entries) - 1)
                playing = False
                info_message = f"Deleted {image_name} (+{removed_rows} meta row(s))"
            else:
                info_message = "Delete cancelled"
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
