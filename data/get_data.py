import os
import time
import csv
from datetime import datetime
from pathlib import Path
from constants import *

import cv2



def make_session_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = root / f"images_session_{ts}"
    (session / "images").mkdir(parents=True, exist_ok=True)
    return session


def open_camera() -> cv2.VideoCapture:
    # Explicitly use V4L2 backend (important on Linux for /dev/videoX devices)
    cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)

    # Each frame is compressed independently as a JPEG image.
    # So instead of sending raw pixels, the camera sends JPG frame
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    # Camera capture fps
    cap.set(cv2.CAP_PROP_FPS, FPS_HINT)

    # # Lower buffering = lower latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def main():
    session_dir = make_session_dir(Path(RAW_DATA_ROOT))
    images_dir = session_dir / "images"
    meta_path = session_dir / "meta.csv"

    cap = open_camera()
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at {DEVICE}")

    # How often (in seconds) we want to save an image (ex: 10FPS -> every 0.1s)
    save_period = 1.0 / float(TARGET_FPS)
    # Always increasing, never jumps backwards
    next_save_t = time.monotonic()

    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)

        # t_wall = gives real world clock time (good for syncing with other systems)
        # t_mono = continuously increasing stopwatch (good for latency measurement)
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
                    # Safer if crash occurs, but slightly slower
                    f.flush()

                    saved_idx += 1
                    next_save_t += save_period

                frame_idx += 1

        except KeyboardInterrupt:
            print("Stopped by user.")

    cap.release()


if __name__ == "__main__":
    main()
