from pathlib import Path
from time import perf_counter

try:
    from .constants import *
except ImportError:
    from constants import *

REPO_ROOT = Path(__file__).resolve().parent.parent


def sanitize_class_folder_name(name: str) -> str:
    cleaned = name.strip()
    cleaned = cleaned.replace("/", "_").replace("\\", "_")
    return cleaned if cleaned else "unnamed_class"


def sanitize_token(name: str) -> str:
    safe = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name)
    safe = safe.strip("_")
    return safe if safe else "model"


def resolve_repo_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (REPO_ROOT / path)


def resolve_model_reference(model_ref: str) -> str:
    ref_path = resolve_repo_path(model_ref)
    if ref_path.exists():
        return str(ref_path)
    return model_ref


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


def f4(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def f2(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def print_table(rows: list[dict]) -> None:
    headers = [
        ("model", 42),
        ("P", 7),
        ("R", 7),
        ("mAP50", 9),
        ("mAP50-95", 11),
        ("infer_ms", 10),
        ("status", 10),
    ]

    def pad(text: str, width: int) -> str:
        if len(text) > width:
            return text[: width - 1] + "…"
        return text.ljust(width)

    header_line = " | ".join(pad(label, width) for label, width in headers)
    sep_line = "-+-".join("-" * width for _, width in headers)
    print(header_line)
    print(sep_line)

    for row in rows:
        model_name = row.get("model_name", "unknown")
        status = row.get("status", "ok")
        if status != "ok":
            line = [
                pad(model_name, headers[0][1]),
                pad("n/a", headers[1][1]),
                pad("n/a", headers[2][1]),
                pad("n/a", headers[3][1]),
                pad("n/a", headers[4][1]),
                pad("n/a", headers[5][1]),
                pad("FAILED", headers[6][1]),
            ]
            print(" | ".join(line))
            print(f"    error: {row.get('error', 'unknown error')}")
            continue

        line = [
            pad(model_name, headers[0][1]),
            pad(f4(row.get("precision")), headers[1][1]),
            pad(f4(row.get("recall")), headers[2][1]),
            pad(f4(row.get("map50")), headers[3][1]),
            pad(f4(row.get("map5095")), headers[4][1]),
            pad(f2(row.get("inference_ms")), headers[5][1]),
            pad("ok", headers[6][1]),
        ]
        print(" | ".join(line))


def main() -> None:
    class_name = sanitize_class_folder_name(YOLO_TARGET_CLASS_NAME)
    dataset_root = resolve_repo_path(YOLO_LABELS_ROOT) / class_name / YOLO_OUTPUT_DATASET_NAME
    dataset_yaml = dataset_root / YOLO_DATASET_YAML_NAME

    if not dataset_yaml.exists():
        raise RuntimeError(
            f"Missing YOLO dataset yaml: {dataset_yaml}\n"
            "Run `uv run python data/prepare_yolo_dataset.py` first."
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
    print_table(results)

    ok_rows = [r for r in results if r.get("status") == "ok"]
    if ok_rows:
        best = max(ok_rows, key=lambda r: r.get("map5095") or -1.0)
        print()
        print("Best by mAP50-95:")
        print(
            f"- {best['model_name']} | "
            f"mAP50-95={f4(best.get('map5095'))} | "
            f"mAP50={f4(best.get('map50'))} | "
            f"P={f4(best.get('precision'))} | "
            f"R={f4(best.get('recall'))} | "
            f"infer_ms={f2(best.get('inference_ms'))}"
        )
    else:
        print()
        print("No successful evaluations. Check errors above.")


if __name__ == "__main__":
    main()
