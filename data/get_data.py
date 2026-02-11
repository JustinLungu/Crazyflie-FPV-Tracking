import os
import time
import csv
from datetime import datetime
from pathlib import Path

import cv2


def make_session_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = root / f"images_session_{ts}"
    (session / "images").mkdir(parents=True, exist_ok=True)
    return session


def open_camera(device: str, width: int, height: int, fps_hint: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    # Ask for MJPEG if supported, reduces USB bandwidth and CPU in many cases
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_hint)

    # Try to reduce internal buffering if backend supports it
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def main():
    device = "/dev/video1"
    target_fps = 10  # set 5 or 10 for labeling
    width, height = 640, 480
    fps_hint = 30

    out_root = Path("camera_feed/raw_data")
    session_dir = make_session_dir(out_root)
    images_dir = session_dir / "images"
    meta_path = session_dir / "meta.csv"

    cap = open_camera(device, width, height, fps_hint)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at {device}")

    save_period = 1.0 / float(target_fps)
    next_save_t = time.monotonic()

    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "filename", "t_wall", "t_mono"])

        frame_idx = 0
        saved_idx = 0

        print(f"Saving to: {session_dir}")
        print("Press Ctrl+C to stop.")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    # If the stream hiccups, do not spin at 100 percent CPU
                    time.sleep(0.01)
                    continue

                now_mono = time.monotonic()
                if now_mono >= next_save_t:
                    t_wall = time.time()
                    filename = f"frame_{saved_idx:06d}.jpg"
                    out_path = images_dir / filename

                    # JPEG quality tradeoff, higher is larger and slower
                    cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

                    writer.writerow([frame_idx, filename, f"{t_wall:.6f}", f"{now_mono:.6f}"])
                    f.flush()

                    saved_idx += 1
                    next_save_t += save_period

                frame_idx += 1

        except KeyboardInterrupt:
            print("Stopped by user.")

    cap.release()


if __name__ == "__main__":
    main()
