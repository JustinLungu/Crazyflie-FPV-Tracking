import time
from datetime import datetime
from pathlib import Path
import cv2


def main():
    device = "/dev/video1"
    width, height = 640, 480
    fps = 30
    fourcc = "MJPG"

    out_root = Path("camera_feed/raw_data")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = out_root / f"video_session_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)

    video_path = session_dir / "video.avi"

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {device}")

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )

    print(f"Recording to {video_path}")
    print("Press q to stop")

    last_print = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        writer.write(frame)
        frames += 1

        cv2.imshow("record", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        now = time.time()
        if now - last_print > 2.0:
            print(f"frames: {frames}")
            last_print = now

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
