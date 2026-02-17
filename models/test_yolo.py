from pathlib import Path

try:
    from .constants import *
except ImportError:
    from constants import *

REPO_ROOT = Path(__file__).resolve().parent.parent


def sanitize_class_folder_name(name: str) -> str:
    cleaned = name.strip()
    cleaned = cleaned.replace("/", "_").replace("\\", "_")
    return cleaned if cleaned else "unnamed_class"


def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


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


def resolve_model_reference(model_ref: str) -> str:
    # Treat relative paths as repo-root based, so script works from any cwd.
    ref_path = resolve_repo_path(model_ref)
    if ref_path.exists():
        return str(ref_path)
    return model_ref


def main() -> None:
    class_name = sanitize_class_folder_name(YOLO_TARGET_CLASS_NAME)
    dataset_root = resolve_repo_path(YOLO_LABELS_ROOT) / class_name / YOLO_OUTPUT_DATASET_NAME
    dataset_yaml = dataset_root / YOLO_DATASET_YAML_NAME

    if not dataset_yaml.exists():
        raise RuntimeError(
            f"Missing YOLO dataset yaml: {dataset_yaml}\n"
            "Run `uv run python data/prepare_yolo_dataset.py` first."
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
