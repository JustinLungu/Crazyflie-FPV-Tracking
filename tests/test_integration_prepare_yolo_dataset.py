import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


def load_module_from_file(module_file: Path, module_name: str) -> ModuleType:
    parent_dir = str(module_file.parent)
    sys.path.insert(0, parent_dir)
    try:
        for shadowed_name in ("constants", "utils", module_name):
            sys.modules.pop(shadowed_name, None)
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module: {module_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == parent_dir:
            sys.path.pop(0)


class PrepareYoloDatasetIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.script_path = self.repo_root / "data" / "prepare_yolo_dataset.py"

    def _build_source_dataset(self, labels_root: Path, class_name: str, dataset_name: str) -> None:
        src_root = labels_root / class_name / dataset_name
        images_dir = src_root / "images"
        labels_dir = src_root / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        idx = 0
        # 3 sessions so session split path is used.
        sessions = {
            "session_a": 3,
            "session_b": 3,
            "session_c": 3,
        }
        for session_name, count in sessions.items():
            for i in range(count):
                stem = f"frame_{idx:06d}"
                image_name = f"{stem}.jpg"
                label_name = f"{stem}.txt"
                (images_dir / image_name).write_bytes(b"fake_jpg")
                # Source labels use class id 1; output should be remapped to 0.
                (labels_dir / label_name).write_text("1 0.5 0.5 0.2 0.2\n")
                rows.append(
                    {
                        "dataset_index": idx,
                        "dataset_image": image_name,
                        "dataset_label": label_name,
                        "source_session": session_name,
                        "source_image": image_name,
                        "source_label": label_name,
                    }
                )
                idx += 1

        manifest = src_root / "manifest.csv"
        with manifest.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset_index",
                    "dataset_image",
                    "dataset_label",
                    "source_session",
                    "source_image",
                    "source_label",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_main_builds_split_and_remaps_labels(self) -> None:
        with tempfile.TemporaryDirectory(prefix="prepare_yolo_integration_") as tmp:
            tmp_path = Path(tmp)
            labels_root = tmp_path / "labels_root"
            class_name = "black_drone"
            src_dataset_name = "black_drone_dataset"
            out_dataset_name = "black_drone_yolo"
            self._build_source_dataset(labels_root, class_name, src_dataset_name)

            mod = load_module_from_file(self.script_path, "prepare_yolo_dataset_test_mod")

            # Override script constants for isolated temporary integration test.
            mod.OUT_DIR = str(labels_root)
            mod.YOLO_TARGET_CLASS_NAME = class_name
            mod.YOLO_SOURCE_DATASET_NAME = src_dataset_name
            mod.YOLO_OUTPUT_DATASET_NAME = out_dataset_name
            mod.YOLO_DATASET_YAML_NAME = "dataset.yaml"
            mod.YOLO_OVERWRITE_OUTPUT = True
            mod.YOLO_INCLUDED_SESSIONS = ()
            mod.YOLO_SPLIT_SEED = 7
            mod.YOLO_TRAIN_RATIO = 0.6
            mod.YOLO_VAL_RATIO = 0.2
            mod.YOLO_TEST_RATIO = 0.2
            mod.YOLO_MIN_SESSIONS_FOR_GROUP_SPLIT = 3
            mod.YOLO_FALLBACK_TO_FRAME_SPLIT_IF_FEW_SESSIONS = True
            mod.YOLO_SINGLE_CLASS_MODE = True
            mod.YOLO_TARGET_CLASS_ID = 0
            mod.YOLO_TARGET_CLASS_LABEL = class_name

            mod.main()

            out_root = labels_root / class_name / out_dataset_name
            self.assertTrue((out_root / "dataset.yaml").exists())
            self.assertTrue((out_root / "split_manifest.csv").exists())

            total_images = 0
            for split in ("train", "val", "test"):
                img_dir = out_root / "images" / split
                lbl_dir = out_root / "labels" / split
                self.assertTrue(img_dir.exists())
                self.assertTrue(lbl_dir.exists())
                images = sorted(img_dir.glob("*.jpg"))
                labels = sorted(lbl_dir.glob("*.txt"))
                self.assertEqual(len(images), len(labels))
                total_images += len(images)

                for label_file in labels:
                    lines = [line.strip() for line in label_file.read_text().splitlines() if line.strip()]
                    self.assertTrue(lines, f"Expected non-empty label file: {label_file}")
                    for line in lines:
                        self.assertEqual(line.split()[0], "0", f"Label was not remapped in {label_file}")

            self.assertEqual(total_images, 9)


if __name__ == "__main__":
    unittest.main()
