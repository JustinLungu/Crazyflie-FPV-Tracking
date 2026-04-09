from datetime import datetime
from contextlib import contextmanager
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


def _resolve_latest_model_weights(
    model_ref: str,
    runs_root: str,
    models_runs_dir: str,
) -> str:
    models_dir = resolve_repo_path(runs_root) / models_runs_dir
    if not models_dir.exists():
        raise RuntimeError(
            f"Models runs directory not found: {models_dir}\n"
            "Run training first to create runs/models/<run_name>/weights/*.pt."
        )

    run_dirs = [p for p in models_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise RuntimeError(
            f"No training runs found in: {models_dir}\n"
            "Run training first to create at least one model run."
        )

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    weight_name = "best.pt" if model_ref == "latest_best" else "last.pt"
    weight_path = latest_run / "weights" / weight_name
    if not weight_path.exists():
        raise RuntimeError(
            f"Missing weights file: {weight_path}\n"
            f"Latest run is {latest_run.name}, but {weight_name} is not present."
        )
    return str(weight_path)


def resolve_model_reference(
    model_ref: str,
    runs_root: str | None = None,
    models_runs_dir: str | None = None,
) -> str:
    normalized = model_ref.strip().lower()
    if normalized in ("latest_best", "latest_last"):
        if runs_root is None or models_runs_dir is None:
            raise RuntimeError(
                "resolve_model_reference('latest_*') requires runs_root and models_runs_dir."
            )
        return _resolve_latest_model_weights(
            model_ref=normalized,
            runs_root=runs_root,
            models_runs_dir=models_runs_dir,
        )

    ref_path = resolve_repo_path(model_ref)
    if ("/" in model_ref or "\\" in model_ref or model_ref.startswith(".")) and not ref_path.exists():
        raise RuntimeError(
            f"Model path not found: {ref_path}\n"
            "Set YOLO_TEST_WEIGHTS to an existing .pt file path or use latest_best/latest_last."
        )
    if ref_path.exists():
        return str(ref_path)
    return model_ref


def build_dated_run_name(run_label: str, date_format: str) -> str:
    safe_label = sanitize_token(run_label)
    stamp = datetime.now().strftime(date_format)
    return f"{safe_label}_{stamp}"


def ensure_unique_run_name(project_dir: Path, desired_name: str) -> str:
    candidate = sanitize_token(desired_name)
    idx = 1
    while (project_dir / candidate).exists():
        candidate = f"{sanitize_token(desired_name)}_{idx:02d}"
        idx += 1
    return candidate


def build_comparison_session_name(
    run_label: str,
    model_refs: tuple[str, ...] | list[str],
    date_format: str,
) -> str:
    tags: list[str] = []
    for ref in model_refs:
        stem = Path(str(ref)).stem
        token = sanitize_token(stem)
        if token and token not in tags:
            tags.append(token)

    if not tags:
        tags = ["models"]
    if len(tags) > 3:
        model_part = "-".join(tags[:3]) + f"-plus{len(tags) - 3}"
    else:
        model_part = "-".join(tags)

    base = f"{sanitize_token(run_label)}_{model_part}"
    return build_dated_run_name(base, date_format)


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
    from ultralytics import YOLO
    return YOLO


def clamp_overlap_threshold_from_percent(overlap_percent: float) -> float:
    return max(0.0, min(1.0, overlap_percent / 100.0))


def compute_overlap_ratio_xyxy(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    min_area = min(area_a, area_b)
    if min_area <= 0:
        return 0.0
    return inter_area / min_area


def suppress_overlap_indices(
    boxes_xyxy: list[tuple[float, float, float, float]],
    confidences: list[float],
    overlap_threshold: float,
) -> list[int]:
    ordered = sorted(range(len(boxes_xyxy)), key=lambda i: confidences[i], reverse=True)
    kept: list[int] = []
    for idx in ordered:
        current_box = boxes_xyxy[idx]
        has_high_overlap = False
        for kept_idx in kept:
            if compute_overlap_ratio_xyxy(current_box, boxes_xyxy[kept_idx]) > overlap_threshold:
                has_high_overlap = True
                break
        if not has_high_overlap:
            kept.append(idx)
    return kept


def _filter_nms_outputs(outputs, overlap_threshold: float):
    if outputs is None:
        return outputs

    if isinstance(outputs, tuple):
        if not outputs:
            return outputs
        first = outputs[0]
        if isinstance(first, list):
            filtered_first = _filter_nms_outputs(first, overlap_threshold)
            return (filtered_first, *outputs[1:])
        return outputs

    if not isinstance(outputs, list):
        return outputs

    filtered = []
    for det in outputs:
        if det is None or len(det) <= 1:
            filtered.append(det)
            continue
        boxes = [tuple(map(float, row[:4].tolist())) for row in det]
        confs = [float(row[4]) for row in det]
        keep = suppress_overlap_indices(
            boxes_xyxy=boxes,
            confidences=confs,
            overlap_threshold=overlap_threshold,
        )
        filtered.append(det[keep])
    return filtered


@contextmanager
def patched_ultralytics_overlap_suppression(overlap_percent: float):
    overlap_threshold = clamp_overlap_threshold_from_percent(overlap_percent)
    if overlap_threshold <= 0.0:
        yield
        return

    patch_targets: list[tuple[object, str, object]] = []
    candidate_functions: list[object] = []

    # Ultralytics >=8.4.x exposes NMS in ultralytics.utils.nms.
    try:
        from ultralytics.utils import nms as nms_module
        if hasattr(nms_module, "non_max_suppression"):
            patch_targets.append((nms_module, "non_max_suppression", nms_module.non_max_suppression))
            candidate_functions.append(nms_module.non_max_suppression)
    except Exception:
        pass

    # Backward compatibility for older releases.
    try:
        from ultralytics.utils import ops as ops_module
        if hasattr(ops_module, "non_max_suppression"):
            patch_targets.append((ops_module, "non_max_suppression", ops_module.non_max_suppression))
            candidate_functions.append(ops_module.non_max_suppression)
    except Exception:
        pass

    if not patch_targets:
        # Keep evaluation runnable even if Ultralytics internals change.
        print("Warning: non_max_suppression hook not found; overlap suppression disabled for this run.")
        yield
        return

    # Also patch any module-level aliases imported via
    # `from ultralytics.utils.nms import non_max_suppression`.
    import sys

    candidate_set = set(candidate_functions)
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            mod_dict = vars(mod)
        except Exception:
            continue
        if "non_max_suppression" not in mod_dict:
            continue
        current = mod_dict["non_max_suppression"]
        if current in candidate_set:
            patch_targets.append((mod, "non_max_suppression", current))

    # Deduplicate identical targets.
    deduped: list[tuple[object, str, object]] = []
    seen = set()
    for target in patch_targets:
        key = (id(target[0]), target[1])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(target)

    def make_wrapper(original_fn):
        def wrapped_non_max_suppression(*args, **kwargs):
            outputs = original_fn(*args, **kwargs)
            return _filter_nms_outputs(outputs, overlap_threshold=overlap_threshold)
        return wrapped_non_max_suppression

    applied: list[tuple[object, str, object]] = []
    for obj, attr, original in deduped:
        setattr(obj, attr, make_wrapper(original))
        applied.append((obj, attr, original))

    try:
        yield
    finally:
        for obj, attr, original in reversed(applied):
            setattr(obj, attr, original)


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
