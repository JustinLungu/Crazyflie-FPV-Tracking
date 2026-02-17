from pathlib import Path

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


def build_dataset_yaml_path(
    labels_root: str,
    target_class_name: str,
    output_dataset_name: str,
    dataset_yaml_name: str,
) -> Path:
    class_name = sanitize_class_folder_name(target_class_name)
    dataset_root = resolve_repo_path(labels_root) / class_name / output_dataset_name
    return dataset_root / dataset_yaml_name


def require_dataset_yaml(
    labels_root: str,
    target_class_name: str,
    output_dataset_name: str,
    dataset_yaml_name: str,
) -> Path:
    dataset_yaml = build_dataset_yaml_path(
        labels_root=labels_root,
        target_class_name=target_class_name,
        output_dataset_name=output_dataset_name,
        dataset_yaml_name=dataset_yaml_name,
    )
    if not dataset_yaml.exists():
        raise RuntimeError(
            f"Missing YOLO dataset yaml: {dataset_yaml}\n"
            "Run `uv run python data/prepare_yolo_dataset.py` first."
        )
    return dataset_yaml


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


def format_metric(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def format_time_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def print_comparison_table(rows: list[dict]) -> None:
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
            pad(format_metric(row.get("precision")), headers[1][1]),
            pad(format_metric(row.get("recall")), headers[2][1]),
            pad(format_metric(row.get("map50")), headers[3][1]),
            pad(format_metric(row.get("map5095")), headers[4][1]),
            pad(format_time_ms(row.get("inference_ms")), headers[5][1]),
            pad("ok", headers[6][1]),
        ]
        print(" | ".join(line))
