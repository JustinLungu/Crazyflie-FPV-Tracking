import csv
from pathlib import Path
from time import perf_counter

from constants import *
from utils import (
    build_comparison_session_name,
    ensure_unique_run_name,
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

    comparison_root = resolve_repo_path(YOLO_RUNS_ROOT) / YOLO_COMPARISON_RUNS_DIR
    comparison_root.mkdir(parents=True, exist_ok=True)
    desired_session = build_comparison_session_name(
        run_label=YOLO_COMPARE_RUN_LABEL,
        model_refs=list(YOLO_COMPARE_MODEL_REFS),
        date_format=YOLO_RUN_DATE_FORMAT,
    )
    comparison_session_name = ensure_unique_run_name(comparison_root, desired_session)
    comparison_session_dir = comparison_root / comparison_session_name
    comparison_session_dir.mkdir(parents=True, exist_ok=False)

    YOLO = load_ultralytics_yolo()
    results: list[dict] = []

    print("Evaluating models on shared split...")
    print(f"- dataset: {dataset_yaml}")
    print(f"- split: {YOLO_TEST_SPLIT}")
    print(f"- comparison dir: {comparison_session_dir}")
    print()

    for idx, model_ref in enumerate(YOLO_COMPARE_MODEL_REFS, start=1):
        resolved_ref = resolve_model_reference(
            str(model_ref),
            runs_root=YOLO_RUNS_ROOT,
            models_runs_dir=YOLO_MODELS_RUNS_DIR,
        )
        model_name = Path(resolved_ref).name if Path(resolved_ref).exists() else str(model_ref)
        run_name = f"{idx:02d}_{sanitize_token(Path(model_name).stem)}"

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
                project=str(comparison_session_dir),
                name=run_name,
                exist_ok=False,
            )
            elapsed = perf_counter() - t0
            metrics_dict = getattr(metrics, "results_dict", {}) or {}
            speed = getattr(metrics, "speed", {}) or {}

            results.append(
                {
                    "model_ref": str(model_ref),
                    "resolved_ref": resolved_ref,
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
                    "model_ref": str(model_ref),
                    "resolved_ref": resolved_ref,
                    "status": "failed",
                    "model_name": model_name,
                    "error": str(exc),
                }
            )

    print()
    print("Comparison results:")
    print_comparison_table(results)

    summary_csv = comparison_session_dir / "comparison_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "status",
                "model_name",
                "model_ref",
                "resolved_ref",
                "precision",
                "recall",
                "mAP50",
                "mAP50-95",
                "inference_ms",
                "elapsed_s",
                "error",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row.get("status", ""),
                    row.get("model_name", ""),
                    row.get("model_ref", ""),
                    row.get("resolved_ref", ""),
                    row.get("precision", ""),
                    row.get("recall", ""),
                    row.get("map50", ""),
                    row.get("map5095", ""),
                    row.get("inference_ms", ""),
                    row.get("elapsed_s", ""),
                    row.get("error", ""),
                ]
            )

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

    print(f"- summary csv: {summary_csv}")


if __name__ == "__main__":
    main()
