import time
from datetime import datetime
from pathlib import Path
from constants import *
from utils import make_session_dir, open_camera
import cv2


def main():
    session_dir = make_session_dir(Path(RAW_DATA_ROOT), "video")
    video_path = session_dir / VIDEO_FLIE_NAME

    cap = open_camera()

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*FOURCC),
        FPS_HINT,
        (WIDTH, HEIGHT),
    )

    print(f"Recording to {video_path}")
    print("Press q to stop")

    last_print = time.time()
    frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            # Prevent tight CPU spin
            time.sleep(0.01)
            continue
        
        # Append frame to video file
        writer.write(frame)
        frames += 1

        # # Show the frame in a GUI window
        cv2.imshow("record", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        now = time.time()
        # Every 2 seconds print total frames recorded
        if now - last_print > 2.0:
            print(f"frames: {frames}")
            last_print = now

    # Finalize video file
    writer.release()
    # Finalize camera
    cap.release()
    cv2.destroyAllWindows()
    print("Done")


if __name__ == "__main__":
    main()
