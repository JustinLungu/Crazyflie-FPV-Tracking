import math
import random
from pathlib import Path

import cv2
import numpy as np

from constants import *
from utils import *


def list_split_images(dataset_root: Path, split_name: str) -> list[Path]:
    images_dir = dataset_root / "images" / split_name
    if not images_dir.exists():
        raise RuntimeError(f"Split images directory not found: {images_dir}")

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_ext],
        key=lambda p: p.name,
    )
    if not image_paths:
        raise RuntimeError(f"No images found in split directory: {images_dir}")
    return image_paths


def choose_samples(image_paths: list[Path], count: int, rng: random.Random) -> list[Path]:
    if count <= 0:
        raise RuntimeError("YOLO_PREVIEW_IMAGE_COUNT must be > 0.")

    n = min(count, len(image_paths))
    return rng.sample(image_paths, n)


def draw_filename_banner(image_bgr: np.ndarray, filename: str) -> np.ndarray:
    output = image_bgr.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        filename,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def build_grid(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise RuntimeError("No annotated images to render.")

    n = len(images)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))

    cell_h, cell_w = images[0].shape[:2]
    grid = np.full((rows * cell_h, cols * cell_w, 3), 24, dtype=np.uint8)

    for idx, image in enumerate(images):
        r = idx // cols
        c = idx % cols
        if image.shape[:2] != (cell_h, cell_w):
            image = cv2.resize(image, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
        y0 = r * cell_h
        x0 = c * cell_w
        grid[y0 : y0 + cell_h, x0 : x0 + cell_w] = image

    return grid


def resolve_output_destination(model_ref: str, split_name: str, output_path_cfg: str) -> tuple[Path, str, str]:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    configured = output_path_cfg.strip()
    if configured:
        output_path = resolve_repo_path(configured)
        ext = output_path.suffix.lower()
        if ext in valid_ext:
            output_dir = output_path.parent
            base_name = output_path.stem
            final_ext = ext
        else:
            output_dir = output_path
            base_name = "random_predictions_grid"
            final_ext = ".jpg"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir, base_name, final_ext

    project_dir = resolve_repo_path(YOLO_RUNS_ROOT) / YOLO_EVALUATION_RUNS_DIR
    project_dir.mkdir(parents=True, exist_ok=True)

    model_token = sanitize_token(Path(model_ref).stem)
    desired_name = build_dated_run_name(
        f"{YOLO_TEST_RUN_LABEL}_{model_token}_{split_name}_preview",
        YOLO_RUN_DATE_FORMAT,
    )
    run_name = ensure_unique_run_name(project_dir, desired_name)
    run_dir = project_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, "random_predictions_grid", ".jpg"


def build_output_path(output_dir: Path, base_name: str, ext: str, run_idx: int, total_runs: int) -> Path:
    if total_runs <= 1:
        return output_dir / f"{base_name}{ext}"
    return output_dir / f"{base_name}_{run_idx:02d}{ext}"


def main() -> None:
    dataset_yaml = require_dataset_yaml(
        labels_root=YOLO_LABELS_ROOT,
        target_class_name=YOLO_TARGET_CLASS_NAME,
        output_dataset_name=YOLO_OUTPUT_DATASET_NAME,
        dataset_yaml_name=YOLO_DATASET_YAML_NAME,
    )
    dataset_root = dataset_yaml.parent
    split_name = str(YOLO_PREVIEW_SPLIT).strip()
    preview_count = int(YOLO_PREVIEW_IMAGE_COUNT)
    preview_runs = int(YOLO_PREVIEW_RUNS)
    preview_seed = YOLO_PREVIEW_RANDOM_SEED
    if preview_runs <= 0:
        raise RuntimeError("YOLO_PREVIEW_RUNS must be > 0.")

    all_split_images = list_split_images(dataset_root=dataset_root, split_name=split_name)

    model_ref = resolve_model_reference(
        YOLO_TEST_WEIGHTS,
        runs_root=YOLO_RUNS_ROOT,
        models_runs_dir=YOLO_MODELS_RUNS_DIR,
    )

    YOLO = load_ultralytics_yolo()
    model = YOLO(model_ref)
    output_dir, output_base_name, output_ext = resolve_output_destination(
        model_ref=model_ref,
        split_name=split_name,
        output_path_cfg=str(YOLO_PREVIEW_OUTPUT_PATH),
    )
    written_outputs: list[Path] = []
    for run_idx in range(1, preview_runs + 1):
        if preview_seed is None:
            rng = random.Random()
        else:
            rng = random.Random(int(preview_seed) + run_idx - 1)

        selected_images = choose_samples(
            image_paths=all_split_images,
            count=preview_count,
            rng=rng,
        )

        results = model.predict(
            source=[str(p) for p in selected_images],
            imgsz=YOLO_IMG_SIZE,
            conf=YOLO_TEST_CONF,
            iou=YOLO_TEST_IOU,
            device=YOLO_DEVICE,
            verbose=False,
        )

        rendered: list[np.ndarray] = []
        for image_path, result in zip(selected_images, results):
            annotated = result.plot(conf=True, labels=True)
            annotated = draw_filename_banner(annotated, image_path.name)
            rendered.append(annotated)

        grid = build_grid(rendered)
        output_path = build_output_path(
            output_dir=output_dir,
            base_name=output_base_name,
            ext=output_ext,
            run_idx=run_idx,
            total_runs=preview_runs,
        )
        if not cv2.imwrite(str(output_path), grid):
            raise RuntimeError(f"Failed to write output image: {output_path}")
        written_outputs.append(output_path)

    print("Random test preview created.")
    print(f"- model: {model_ref}")
    print(f"- split: {split_name}")
    print(f"- sampled images per run: {min(preview_count, len(all_split_images))} (from {len(all_split_images)})")
    print(f"- runs: {preview_runs}")
    print(f"- output directory: {output_dir}")
    for out in written_outputs:
        print(f"  - {out}")
    if preview_seed is not None:
        print(f"- base seed: {preview_seed}")


if __name__ == "__main__":
    main()
