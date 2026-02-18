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

    project_dir = resolve_repo_path(YOLO_RUNS_ROOT) / YOLO_MODELS_RUNS_DIR
    project_dir.mkdir(parents=True, exist_ok=True)
    desired_run_name = build_dated_run_name(YOLO_TRAIN_RUN_LABEL, YOLO_RUN_DATE_FORMAT)
    run_name = ensure_unique_run_name(project_dir, desired_run_name)

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
        project=str(project_dir),
        name=run_name,
        exist_ok=False,
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
