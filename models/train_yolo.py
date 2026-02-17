from pathlib import Path

from constants import *
from utils import *


def main() -> None:
    dataset_yaml = require_dataset_yaml(
        labels_root=YOLO_LABELS_ROOT,
        target_class_name=YOLO_TARGET_CLASS_NAME,
        output_dataset_name=YOLO_OUTPUT_DATASET_NAME,
        dataset_yaml_name=YOLO_DATASET_YAML_NAME,
    )

    YOLO = load_ultralytics_yolo()
    model = YOLO(YOLO_TRAIN_MODEL)

    results = model.train(
        data=str(dataset_yaml),
        imgsz=YOLO_IMG_SIZE,
        epochs=YOLO_EPOCHS,
        batch=YOLO_BATCH,
        device=YOLO_DEVICE,
        workers=YOLO_WORKERS,
        patience=YOLO_PATIENCE,
        cache=YOLO_CACHE_IMAGES,
        project=str(resolve_repo_path(YOLO_PROJECT_DIR)),
        name=YOLO_TRAIN_RUN_NAME,
        exist_ok=True,
    )

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    print("Training complete.")
    print(f"- run dir: {save_dir}")
    print(f"- best weights: {best_weights}")
    print(f"- last weights: {last_weights}")


if __name__ == "__main__":
    main()
