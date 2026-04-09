import argparse
import csv
import shutil
from pathlib import Path

from constants import LABEL_ALL_DATA_DIR, LABEL_CLASS_NAME, OUT_DIR
from utils import sanitize_class_folder_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine labeled sessions from a class into one dataset directory "
            "with contiguous filenames."
        )
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default=LABEL_CLASS_NAME,
        help="Class folder under labels root (default from constants.py).",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help=(
            "Optional list of session folder names (or paths). "
            "If omitted, all valid session folders in LABEL_ALL_DATA_DIR are included."
        ),
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        default=OUT_DIR,
        help="Root labels directory (default from constants.py).",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=LABEL_ALL_DATA_DIR,
        help=(
            "Source folder under labels/<class_name>/ containing session subfolders "
            "(default from constants.py LABEL_ALL_DATA_DIR)."
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="",
        help="Optional dataset folder name (default: <class>_dataset).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace dataset folder if it already exists.",
    )
    return parser.parse_args()


def is_session_dir(session_dir: Path) -> bool:
    return (session_dir / "images").is_dir() and (session_dir / "labels").is_dir()


def discover_all_sessions(all_data_dir: Path) -> list[Path]:
    sessions = [
        d
        for d in all_data_dir.iterdir()
        if d.is_dir() and is_session_dir(d)
    ]
    return sorted(sessions, key=lambda d: d.name)


def resolve_selected_sessions(all_data_dir: Path, requested: list[str]) -> list[Path]:
    selected: list[Path] = []
    for token in requested:
        by_name = all_data_dir / token
        by_path = Path(token)
        if by_name.exists() and by_name.is_dir():
            selected.append(by_name)
            continue
        if by_path.exists() and by_path.is_dir():
            selected.append(by_path)
            continue
        raise RuntimeError(
            f"Session not found: {token}. "
            f"Use folder names under {all_data_dir} or full paths."
        )

    invalid = [p for p in selected if not is_session_dir(p)]
    if invalid:
        formatted = "\n".join(f"- {p}" for p in invalid)
        raise RuntimeError(
            "Selected folders must contain both images/ and labels/.\n"
            f"Invalid selections:\n{formatted}"
        )
    return selected


def prepare_output_dirs(dataset_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    if dataset_dir.exists():
        if not overwrite:
            raise RuntimeError(
                f"Dataset directory already exists: {dataset_dir}. "
                "Use --overwrite to recreate it."
            )
        shutil.rmtree(dataset_dir)

    images_out = dataset_dir / "images"
    labels_out = dataset_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=False)
    labels_out.mkdir(parents=True, exist_ok=False)
    return images_out, labels_out


def collect_image_paths(images_dir: Path) -> list[Path]:
    return sorted(images_dir.glob("*.jpg"))


def combine_sessions(
    sessions: list[Path],
    images_out: Path,
    labels_out: Path,
    manifest_path: Path,
) -> tuple[int, int]:
    """
    Copy all session image/label pairs into contiguous dataset filenames.

    Returns:
        total_copied, missing_label_count
    """
    total_copied = 0
    missing_label_count = 0

    with open(manifest_path, "w", newline="") as mf:
        writer = csv.writer(mf)
        writer.writerow(
            [
                "dataset_index",
                "dataset_image",
                "dataset_label",
                "source_session",
                "source_image",
                "source_label",
            ]
        )

        for session_dir in sessions:
            images_dir = session_dir / "images"
            labels_dir = session_dir / "labels"
            if not images_dir.exists() or not labels_dir.exists():
                print(f"Warning: skipping malformed session {session_dir}")
                continue

            for image_path in collect_image_paths(images_dir):
                source_label = labels_dir / f"{image_path.stem}.txt"

                out_stem = f"frame_{total_copied:06d}"
                out_image = images_out / f"{out_stem}.jpg"
                out_label = labels_out / f"{out_stem}.txt"

                shutil.copy2(image_path, out_image)
                if source_label.exists():
                    shutil.copy2(source_label, out_label)
                else:
                    out_label.write_text("")
                    missing_label_count += 1

                writer.writerow(
                    [
                        total_copied,
                        out_image.name,
                        out_label.name,
                        session_dir.name,
                        image_path.name,
                        source_label.name,
                    ]
                )
                total_copied += 1

    return total_copied, missing_label_count


def main():
    args = parse_args()

    class_name = sanitize_class_folder_name(args.class_name)
    labels_root = Path(args.labels_root)
    class_dir = labels_root / class_name
    source_dir_token = args.source_dir.strip()
    if not source_dir_token:
        raise RuntimeError("source-dir cannot be empty.")
    all_data_dir = class_dir / source_dir_token

    if not all_data_dir.exists():
        raise RuntimeError(f"Missing source directory: {all_data_dir}")

    if args.sessions:
        sessions = resolve_selected_sessions(all_data_dir, args.sessions)
    else:
        sessions = discover_all_sessions(all_data_dir)

    if not sessions:
        raise RuntimeError(
            f"No valid session folders found in {all_data_dir}.\n"
            "Expected each selected folder to contain images/ and labels/."
        )

    dataset_name = args.output_name.strip() if args.output_name.strip() else f"{class_name}_dataset"
    dataset_dir = class_dir / dataset_name
    images_out, labels_out = prepare_output_dirs(dataset_dir, overwrite=args.overwrite)
    manifest_path = dataset_dir / "manifest.csv"

    print(f"Class: {class_name}")
    print(f"Source folder: {all_data_dir}")
    print(f"Selected sessions ({len(sessions)}):")
    for s in sessions:
        print(f"- {s}")
    print(f"Output dataset: {dataset_dir}")

    total_copied, missing_label_count = combine_sessions(
        sessions=sessions,
        images_out=images_out,
        labels_out=labels_out,
        manifest_path=manifest_path,
    )

    print("Done")
    print(f"Total samples copied: {total_copied}")
    print(f"Missing source labels (created empty labels): {missing_label_count}")
    print(f"Images dir: {images_out}")
    print(f"Labels dir: {labels_out}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
