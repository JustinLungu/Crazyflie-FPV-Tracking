from pathlib import Path

from constants import *


def sanitize_class_folder_name(name: str) -> str:
    cleaned = name.strip()
    cleaned = cleaned.replace("/", "_").replace("\\", "_")
    return cleaned if cleaned else "unnamed_class"


def load_ultralytics_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed.\n"
            "Install it with:\n"
            "  uv add ultralytics\n"
            "Then run this script again."
        ) from exc
    return YOLO


def main() -> None:
    class_name = sanitize_class_folder_name(YOLO_TARGET_CLASS_NAME)
    dataset_root = Path(OUT_DIR) / class_name / YOLO_OUTPUT_DATASET_NAME
    dataset_yaml = dataset_root / YOLO_DATASET_YAML_NAME

    if not dataset_yaml.exists():
        raise RuntimeError(
            f"Missing YOLO dataset yaml: {dataset_yaml}\n"
            "Run `uv run python data/prepare_yolo_dataset.py` first."
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
        project=YOLO_PROJECT_DIR,
        name=YOLO_RUN_NAME,
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
