from constants import *
from utils import (
    live_distance_inference,
    ensure_output_dir,
    yolo_inference,
    process_best_detection
)

import sys




if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--live":
            # Live feed mode
            live_distance_inference(MODEL_PATH, YOLO_CONF_THRESHOLD)
        else:
            # Static image mode
            image_path = sys.argv[1]
            ensure_output_dir(OUTPUT_DIR)
            yolo_results = yolo_inference(image_path, MODEL_PATH, YOLO_CONF_THRESHOLD)
            process_best_detection(yolo_results, image_path)
    else:
        # Default: static image mode
        image_path = IMAGE_PATH
        ensure_output_dir(OUTPUT_DIR)
        yolo_results = yolo_inference(image_path, MODEL_PATH, YOLO_CONF_THRESHOLD)
        process_best_detection(yolo_results, image_path)