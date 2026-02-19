from constants import *
from utils import *


def main() -> None:
    dataset_yaml = require_dataset_yaml(
        labels_root=YOLO_LABELS_ROOT,
        target_class_name=YOLO_TARGET_CLASS_NAME,
        output_dataset_name=YOLO_OUTPUT_DATASET_NAME,
        dataset_yaml_name=YOLO_DATASET_YAML_NAME,
    )

    model_ref = resolve_model_reference(
        YOLO_TEST_WEIGHTS,
        runs_root=YOLO_RUNS_ROOT,
        models_runs_dir=YOLO_MODELS_RUNS_DIR,
    )
    model_token = sanitize_token(Path(model_ref).stem)
    project_dir = resolve_repo_path(YOLO_RUNS_ROOT) / YOLO_EVALUATION_RUNS_DIR
    project_dir.mkdir(parents=True, exist_ok=True)
    desired_run_name = build_dated_run_name(
        f"{YOLO_TEST_RUN_LABEL}_{model_token}_{YOLO_TEST_SPLIT}",
        YOLO_RUN_DATE_FORMAT,
    )
    run_name = ensure_unique_run_name(project_dir, desired_run_name)

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
        project=str(project_dir),
        name=run_name,
        exist_ok=False,
    )

    results_dict = getattr(metrics, "results_dict", {}) or {}
    print("Evaluation complete.")
    print(f"- model: {model_ref}")
    print(f"- split: {YOLO_TEST_SPLIT}")
    print(f"- dataset: {dataset_yaml}")
    print(f"- run dir: {project_dir / run_name}")
    if results_dict:
        for key, value in sorted(results_dict.items()):
            print(f"- {key}: {value}")
    else:
        print("- No scalar metrics dictionary returned by Ultralytics.")


if __name__ == "__main__":
    main()
