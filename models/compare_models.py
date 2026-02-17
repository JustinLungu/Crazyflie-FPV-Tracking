from pathlib import Path
from time import perf_counter

from constants import *
from utils import (
    format_metric,
    format_time_ms,
    load_ultralytics_yolo,
    print_comparison_table,
    require_dataset_yaml,
    resolve_model_reference,
    resolve_repo_path,
    sanitize_token,
)


def main() -> None:
    dataset_yaml = require_dataset_yaml(
        labels_root=YOLO_LABELS_ROOT,
        target_class_name=YOLO_TARGET_CLASS_NAME,
        output_dataset_name=YOLO_OUTPUT_DATASET_NAME,
        dataset_yaml_name=YOLO_DATASET_YAML_NAME,
    )

    if not YOLO_COMPARE_MODEL_REFS:
        raise RuntimeError("YOLO_COMPARE_MODEL_REFS is empty in models/constants.py.")

    YOLO = load_ultralytics_yolo()
    results: list[dict] = []

    print("Evaluating models on shared split...")
    print(f"- dataset: {dataset_yaml}")
    print(f"- split: {YOLO_TEST_SPLIT}")
    print()

    for idx, model_ref in enumerate(YOLO_COMPARE_MODEL_REFS, start=1):
        resolved_ref = resolve_model_reference(str(model_ref))
        model_name = Path(resolved_ref).name if Path(resolved_ref).exists() else str(model_ref)
        run_name = f"{YOLO_COMPARE_RUN_PREFIX}_{idx:02d}_{sanitize_token(Path(model_name).stem)}"

        print(f"[{idx}/{len(YOLO_COMPARE_MODEL_REFS)}] {model_name}")
        try:
            model = YOLO(resolved_ref)
            t0 = perf_counter()
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
                name=run_name,
                exist_ok=True,
            )
            elapsed = perf_counter() - t0
            metrics_dict = getattr(metrics, "results_dict", {}) or {}
            speed = getattr(metrics, "speed", {}) or {}

            results.append(
                {
                    "status": "ok",
                    "model_name": model_name,
                    "precision": metrics_dict.get("metrics/precision(B)"),
                    "recall": metrics_dict.get("metrics/recall(B)"),
                    "map50": metrics_dict.get("metrics/mAP50(B)"),
                    "map5095": metrics_dict.get("metrics/mAP50-95(B)"),
                    "inference_ms": speed.get("inference"),
                    "elapsed_s": elapsed,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "status": "failed",
                    "model_name": model_name,
                    "error": str(exc),
                }
            )

    print()
    print("Comparison results:")
    print_comparison_table(results)

    ok_rows = [r for r in results if r.get("status") == "ok"]
    if ok_rows:
        best = max(ok_rows, key=lambda r: r.get("map5095") or -1.0)
        print()
        print("Best by mAP50-95:")
        print(
            f"- {best['model_name']} | "
            f"mAP50-95={format_metric(best.get('map5095'))} | "
            f"mAP50={format_metric(best.get('map50'))} | "
            f"P={format_metric(best.get('precision'))} | "
            f"R={format_metric(best.get('recall'))} | "
            f"infer_ms={format_time_ms(best.get('inference_ms'))}"
        )
    else:
        print()
        print("No successful evaluations. Check errors above.")


if __name__ == "__main__":
    main()
