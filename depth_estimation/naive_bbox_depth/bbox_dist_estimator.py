import sys

from constants import IMAGE_PATH, MODEL_PATH, OUTPUT_DIR, YOLO_CONF_THRESHOLD
from utils import live_distance_inference, process_best_detection, yolo_inference


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        live_distance_inference(MODEL_PATH, YOLO_CONF_THRESHOLD)
        return

    image_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH

    yolo_results, resolved_image_path = yolo_inference(
        image_path,
        MODEL_PATH,
        YOLO_CONF_THRESHOLD,
    )
    process_best_detection(
        yolo_results,
        str(resolved_image_path),
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
