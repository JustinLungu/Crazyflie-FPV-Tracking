import cv2

dev = "/dev/video1"  # change to the working receiver device
cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

# Optional but often helps with USB receivers
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    raise RuntimeError(f"Could not open {dev}")

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed")
        continue

    cv2.imshow("FPV", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
