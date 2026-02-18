import csv
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from constants import *


@dataclass(frozen=True)
class Sample:
    image_name: str
    label_name: str
    source_session: str


def sanitize_class_folder_name(name: str) -> str:
    cleaned = name.strip()
    cleaned = cleaned.replace("/", "_").replace("\\", "_")
    return cleaned if cleaned else "unnamed_class"


def normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise RuntimeError("Split ratios must sum to a positive value.")
    return train_ratio / total, val_ratio / total, test_ratio / total


def read_manifest(manifest_path: Path, included_sessions: set[str]) -> list[Sample]:
    samples: list[Sample] = []
    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required_cols = {"dataset_image", "dataset_label", "source_session"}
        if not required_cols.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(
                f"Manifest missing required columns: {required_cols}. Found: {reader.fieldnames}"
            )

        for row in reader:
            session = str(row["source_session"]).strip()
            if included_sessions and session not in included_sessions:
                continue

            image_name = str(row["dataset_image"]).strip()
            label_name = str(row["dataset_label"]).strip()
            if not image_name:
                continue
            if not label_name:
                label_name = f"{Path(image_name).stem}.txt"

            samples.append(
                Sample(
                    image_name=image_name,
                    label_name=label_name,
                    source_session=session,
                )
            )

    return samples


def split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    # Keep all splits non-empty for usable metrics when dataset size allows it.
    if total >= 3:
        if n_val == 0:
            n_val = 1
            n_train = max(1, n_train - 1)
        if n_test == 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1

    # Guard against corner cases from integer rounding.
    if n_train + n_val + n_test != total:
        n_test = total - n_train - n_val

    return n_train, n_val, n_test


def split_by_frame(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_train, n_val, _ = split_counts(len(shuffled), train_ratio, val_ratio, test_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def split_by_session(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.source_session].append(sample)

    sessions = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(sessions)

    total = len(samples)
    target_train, target_val, target_test = split_counts(total, train_ratio, val_ratio, test_ratio)
    targets = {"train": target_train, "val": target_val, "test": target_test}
    session_alloc: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    split_counts_by_image = {"train": 0, "val": 0, "test": 0}

    remaining_sessions = list(sessions)

    # Seed val/test with the smallest sessions to keep most data available for train.
    if len(sessions) >= 3:
        small_first = sorted(sessions, key=lambda s: len(grouped[s]))
        val_seed = small_first[0]
        test_seed = small_first[1]
        for split_name, seed_session in (("val", val_seed), ("test", test_seed)):
            session_alloc[split_name].append(seed_session)
            split_counts_by_image[split_name] += len(grouped[seed_session])
        remaining_sessions = [s for s in sessions if s not in {val_seed, test_seed}]

    for session_name in remaining_sessions:
        session_size = len(grouped[session_name])
        remaining_need = {
            split: targets[split] - split_counts_by_image[split]
            for split in ("train", "val", "test")
        }

        positive_need = [s for s in ("train", "val", "test") if remaining_need[s] > 0]
        if positive_need:
            max_need = max(remaining_need[s] for s in positive_need)
            candidates = [s for s in positive_need if remaining_need[s] == max_need]
        else:
            # If all targets are exceeded, keep the currently smallest split growing.
            min_count = min(split_counts_by_image.values())
            candidates = [s for s in ("train", "val", "test") if split_counts_by_image[s] == min_count]

        # Stable preference order for ties.
        for preferred in ("train", "val", "test"):
            if preferred in candidates:
                best_split = preferred
                break

        session_alloc[best_split].append(session_name)
        split_counts_by_image[best_split] += session_size

    # If possible, guarantee val/test are not empty.
    for split_name in ("val", "test"):
        if session_alloc[split_name]:
            continue
        donor = max(
            ("train", "val", "test"),
            key=lambda s: len(session_alloc[s]),
        )
        if len(session_alloc[donor]) <= 1:
            continue
        moved_session = min(session_alloc[donor], key=lambda s: len(grouped[s]))
        session_alloc[donor].remove(moved_session)
        session_alloc[split_name].append(moved_session)

    result = {"train": [], "val": [], "test": []}
    for split_name, session_names in session_alloc.items():
        for session_name in session_names:
            result[split_name].extend(grouped[session_name])
    return result


def write_yolo_label(src_label: Path, dst_label: Path, single_class_mode: bool, target_class_id: int) -> None:
    if not src_label.exists():
        dst_label.write_text("")
        return

    if not single_class_mode:
        shutil.copy2(src_label, dst_label)
        return

    out_lines: list[str] = []
    for raw_line in src_label.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            # Keep malformed lines untouched for manual inspection later.
            out_lines.append(line)
            continue
        parts[0] = str(target_class_id)
        out_lines.append(" ".join(parts))

    if out_lines:
        dst_label.write_text("\n".join(out_lines) + "\n")
    else:
        dst_label.write_text("")


def prepare_output_dirs(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise RuntimeError(
                f"Output directory already exists: {output_root}. "
                "Set YOLO_OVERWRITE_OUTPUT=True to recreate."
            )
        shutil.rmtree(output_root)

    for split_name in ("train", "val", "test"):
        (output_root / "images" / split_name).mkdir(parents=True, exist_ok=False)
        (output_root / "labels" / split_name).mkdir(parents=True, exist_ok=False)


def copy_split_files(
    split_samples: dict[str, list[Sample]],
    source_images_dir: Path,
    source_labels_dir: Path,
    output_root: Path,
    single_class_mode: bool,
    target_class_id: int,
) -> tuple[int, int]:
    copied = 0
    missing_images = 0

    for split_name, samples in split_samples.items():
        for sample in samples:
            src_image = source_images_dir / sample.image_name
            src_label = source_labels_dir / sample.label_name
            dst_image = output_root / "images" / split_name / sample.image_name
            dst_label = output_root / "labels" / split_name / sample.label_name

            if not src_image.exists():
                missing_images += 1
                continue

            shutil.copy2(src_image, dst_image)
            write_yolo_label(
                src_label=src_label,
                dst_label=dst_label,
                single_class_mode=single_class_mode,
                target_class_id=target_class_id,
            )
            copied += 1

    return copied, missing_images


def write_split_manifest(output_root: Path, split_samples: dict[str, list[Sample]]) -> Path:
    manifest_path = output_root / "split_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image", "label", "source_session"])
        for split_name in ("train", "val", "test"):
            for sample in split_samples[split_name]:
                writer.writerow([split_name, sample.image_name, sample.label_name, sample.source_session])
    return manifest_path


def write_dataset_yaml(output_root: Path, yaml_name: str, class_label: str) -> Path:
    yaml_path = output_root / yaml_name
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                f"  0: {class_label}",
                "",
            ]
        )
    )
    return yaml_path


def main() -> None:
    class_name = sanitize_class_folder_name(YOLO_TARGET_CLASS_NAME)
    labels_root = Path(OUT_DIR)
    class_dir = labels_root / class_name
    source_dataset_dir = class_dir / YOLO_SOURCE_DATASET_NAME
    output_dataset_dir = class_dir / YOLO_OUTPUT_DATASET_NAME

    source_images_dir = source_dataset_dir / "images"
    source_labels_dir = source_dataset_dir / "labels"
    source_manifest = source_dataset_dir / "manifest.csv"

    if not source_manifest.exists():
        raise RuntimeError(f"Missing manifest: {source_manifest}")
    if not source_images_dir.exists():
        raise RuntimeError(f"Missing source images directory: {source_images_dir}")
    if not source_labels_dir.exists():
        raise RuntimeError(f"Missing source labels directory: {source_labels_dir}")

    included_sessions = {s.strip() for s in YOLO_INCLUDED_SESSIONS if s.strip()}
    train_ratio, val_ratio, test_ratio = normalize_ratios(
        YOLO_TRAIN_RATIO,
        YOLO_VAL_RATIO,
        YOLO_TEST_RATIO,
    )

    samples = read_manifest(source_manifest, included_sessions)
    if not samples:
        raise RuntimeError("No samples found after filtering. Check YOLO_INCLUDED_SESSIONS.")

    session_count = len({s.source_session for s in samples})
    if session_count >= YOLO_MIN_SESSIONS_FOR_GROUP_SPLIT:
        split_samples = split_by_session(samples, train_ratio, val_ratio, test_ratio, YOLO_SPLIT_SEED)
        split_strategy = "session"
    elif YOLO_FALLBACK_TO_FRAME_SPLIT_IF_FEW_SESSIONS:
        split_samples = split_by_frame(samples, train_ratio, val_ratio, test_ratio, YOLO_SPLIT_SEED)
        split_strategy = "frame_fallback"
    else:
        raise RuntimeError(
            f"Only {session_count} sessions found, but group split requires "
            f"{YOLO_MIN_SESSIONS_FOR_GROUP_SPLIT}."
        )

    prepare_output_dirs(output_dataset_dir, YOLO_OVERWRITE_OUTPUT)
    copied, missing_images = copy_split_files(
        split_samples=split_samples,
        source_images_dir=source_images_dir,
        source_labels_dir=source_labels_dir,
        output_root=output_dataset_dir,
        single_class_mode=YOLO_SINGLE_CLASS_MODE,
        target_class_id=YOLO_TARGET_CLASS_ID,
    )
    split_manifest = write_split_manifest(output_dataset_dir, split_samples)
    yaml_path = write_dataset_yaml(
        output_root=output_dataset_dir,
        yaml_name=YOLO_DATASET_YAML_NAME,
        class_label=YOLO_TARGET_CLASS_NAME,
    )

    print("YOLO dataset prepared.")
    print(f"- class: {class_name}")
    print(f"- source dataset: {source_dataset_dir}")
    print(f"- output dataset: {output_dataset_dir}")
    print(f"- split strategy: {split_strategy}")
    print(
        "- split counts: "
        f"train={len(split_samples['train'])}, "
        f"val={len(split_samples['val'])}, "
        f"test={len(split_samples['test'])}"
    )
    print(f"- copied samples: {copied}")
    print(f"- missing source images skipped: {missing_images}")
    print(f"- split manifest: {split_manifest}")
    print(f"- dataset yaml: {yaml_path}")


if __name__ == "__main__":
    main()
