from constants import *
from utils import *


def main() -> None:
    dataset_yaml = require_dataset_yaml(
        labels_root=YOLO_LABELS_ROOT,
        target_class_name=YOLO_TARGET_CLASS_NAME,
        output_dataset_name=YOLO_OUTPUT_DATASET_NAME,
        dataset_yaml_name=YOLO_DATASET_YAML_NAME,
    )

    model_ref = resolve_model_reference(YOLO_TEST_WEIGHTS)
    YOLO = load_ultralytics_yolo()
    model = YOLO(model_ref)

    metrics = model.val(
        data=str(dataset_yaml),
        split=YOLO_TEST_SPLIT,
        imgsz=YOLO_IMG_SIZE,
        batch=YOLO_TEST_BATCH,
        device=YOLO_DEVICE,
        workers=YOLO_WORKERS,
        conf=YOLO_TEST_CONF,
        iou=YOLO_TEST_IOU,
        project=str(resolve_repo_path(YOLO_PROJECT_DIR)),
        name=YOLO_TEST_RUN_NAME,
        exist_ok=True,
    )

    results_dict = getattr(metrics, "results_dict", {}) or {}
    print("Evaluation complete.")
    print(f"- model: {model_ref}")
    print(f"- split: {YOLO_TEST_SPLIT}")
    print(f"- dataset: {dataset_yaml}")
    if results_dict:
        for key, value in sorted(results_dict.items()):
            print(f"- {key}: {value}")
    else:
        print("- No scalar metrics dictionary returned by Ultralytics.")


if __name__ == "__main__":
    main()
