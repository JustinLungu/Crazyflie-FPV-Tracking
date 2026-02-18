import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace


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


def create_dataset_yaml(base_dir: Path, class_name: str, dataset_name: str) -> Path:
    dataset_root = base_dir / class_name / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_yaml = dataset_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                f"  0: {class_name}",
                "",
            ]
        )
    )
    return dataset_yaml


class FakeYOLO:
    train_calls: list[tuple[str, dict]] = []
    val_calls: list[tuple[str, dict]] = []

    def __init__(self, model_ref: str):
        self.model_ref = model_ref

    def train(self, **kwargs):
        FakeYOLO.train_calls.append((self.model_ref, kwargs))
        run_dir = Path(kwargs["project"]) / kwargs["name"]
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "best.pt").write_text("fake_best")
        (run_dir / "weights" / "last.pt").write_text("fake_last")
        return SimpleNamespace(save_dir=str(run_dir))

    def val(self, **kwargs):
        FakeYOLO.val_calls.append((self.model_ref, kwargs))
        return SimpleNamespace(
            results_dict={
                "metrics/precision(B)": 0.9,
                "metrics/recall(B)": 0.88,
                "metrics/mAP50(B)": 0.91,
                "metrics/mAP50-95(B)": 0.41,
            },
            speed={"inference": 4.2},
        )


class ModelsIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.train_script = self.repo_root / "models" / "train_yolo.py"
        self.test_script = self.repo_root / "models" / "test_yolo.py"
        self.compare_script = self.repo_root / "models" / "compare_models.py"

    def test_train_script_main_with_fake_yolo(self) -> None:
        with tempfile.TemporaryDirectory(prefix="models_train_integration_") as tmp:
            tmp_path = Path(tmp)
            labels_root = tmp_path / "labels"
            runs_root = tmp_path / "runs"
            create_dataset_yaml(labels_root, "black_drone", "black_drone_yolo")

            mod = load_module_from_file(self.train_script, "models_train_test_mod")
            mod.load_ultralytics_yolo = lambda: FakeYOLO
            mod.YOLO_LABELS_ROOT = str(labels_root)
            mod.YOLO_TARGET_CLASS_NAME = "black_drone"
            mod.YOLO_OUTPUT_DATASET_NAME = "black_drone_yolo"
            mod.YOLO_DATASET_YAML_NAME = "dataset.yaml"
            mod.YOLO_TRAIN_MODEL = "fake_model.pt"
            mod.YOLO_RUNS_ROOT = str(runs_root)
            mod.YOLO_MODELS_RUNS_DIR = "models"
            mod.YOLO_RUN_DATE_FORMAT = "%Y%m%d"
            mod.YOLO_TRAIN_RUN_LABEL = "train_run"
            mod.YOLO_IMG_SIZE = 640
            mod.YOLO_EPOCHS = 1
            mod.YOLO_BATCH = 2
            mod.YOLO_DEVICE = "cpu"
            mod.YOLO_WORKERS = 0
            mod.YOLO_PATIENCE = 1
            mod.YOLO_CACHE_IMAGES = False

            FakeYOLO.train_calls.clear()
            mod.main()

            self.assertTrue(FakeYOLO.train_calls, "Expected train() to be called.")
            _, kwargs = FakeYOLO.train_calls[-1]
            self.assertEqual(kwargs["project"], str(runs_root / "models"))
            self.assertTrue(kwargs["name"].startswith("train_run_"))
            self.assertTrue((Path(kwargs["project"]) / kwargs["name"] / "weights" / "best.pt").exists())

    def test_test_script_main_with_fake_yolo(self) -> None:
        with tempfile.TemporaryDirectory(prefix="models_test_integration_") as tmp:
            tmp_path = Path(tmp)
            labels_root = tmp_path / "labels"
            runs_root = tmp_path / "runs"
            weights = tmp_path / "weights" / "best.pt"
            weights.parent.mkdir(parents=True, exist_ok=True)
            weights.write_text("fake")
            create_dataset_yaml(labels_root, "black_drone", "black_drone_yolo")

            mod = load_module_from_file(self.test_script, "models_test_test_mod")
            mod.load_ultralytics_yolo = lambda: FakeYOLO
            mod.YOLO_LABELS_ROOT = str(labels_root)
            mod.YOLO_TARGET_CLASS_NAME = "black_drone"
            mod.YOLO_OUTPUT_DATASET_NAME = "black_drone_yolo"
            mod.YOLO_DATASET_YAML_NAME = "dataset.yaml"
            mod.YOLO_RUNS_ROOT = str(runs_root)
            mod.YOLO_COMPARISON_RUNS_DIR = "comparison"
            mod.YOLO_RUN_DATE_FORMAT = "%Y%m%d"
            mod.YOLO_TEST_RUN_LABEL = "eval"
            mod.YOLO_TEST_WEIGHTS = str(weights)
            mod.YOLO_TEST_SPLIT = "test"
            mod.YOLO_IMG_SIZE = 640
            mod.YOLO_TEST_BATCH = 2
            mod.YOLO_DEVICE = "cpu"
            mod.YOLO_WORKERS = 0
            mod.YOLO_TEST_CONF = 0.001
            mod.YOLO_TEST_IOU = 0.7

            FakeYOLO.val_calls.clear()
            mod.main()

            self.assertTrue(FakeYOLO.val_calls, "Expected val() to be called.")
            _, kwargs = FakeYOLO.val_calls[-1]
            self.assertEqual(kwargs["project"], str(runs_root / "comparison"))
            self.assertTrue(kwargs["name"].startswith("eval_best_test_"))
            self.assertEqual(kwargs["split"], "test")

    def test_compare_script_main_with_fake_yolo(self) -> None:
        with tempfile.TemporaryDirectory(prefix="models_compare_integration_") as tmp:
            tmp_path = Path(tmp)
            labels_root = tmp_path / "labels"
            runs_root = tmp_path / "runs"
            create_dataset_yaml(labels_root, "black_drone", "black_drone_yolo")
            model_a = tmp_path / "weights" / "a.pt"
            model_b = tmp_path / "weights" / "b.pt"
            model_a.parent.mkdir(parents=True, exist_ok=True)
            model_a.write_text("a")
            model_b.write_text("b")

            mod = load_module_from_file(self.compare_script, "models_compare_test_mod")
            mod.load_ultralytics_yolo = lambda: FakeYOLO
            mod.YOLO_LABELS_ROOT = str(labels_root)
            mod.YOLO_TARGET_CLASS_NAME = "black_drone"
            mod.YOLO_OUTPUT_DATASET_NAME = "black_drone_yolo"
            mod.YOLO_DATASET_YAML_NAME = "dataset.yaml"
            mod.YOLO_RUNS_ROOT = str(runs_root)
            mod.YOLO_COMPARISON_RUNS_DIR = "comparison"
            mod.YOLO_RUN_DATE_FORMAT = "%Y%m%d"
            mod.YOLO_COMPARE_RUN_LABEL = "cmp"
            mod.YOLO_COMPARE_MODEL_REFS = (str(model_a), str(model_b))
            mod.YOLO_TEST_SPLIT = "test"
            mod.YOLO_IMG_SIZE = 640
            mod.YOLO_TEST_BATCH = 2
            mod.YOLO_DEVICE = "cpu"
            mod.YOLO_WORKERS = 0
            mod.YOLO_TEST_CONF = 0.001
            mod.YOLO_TEST_IOU = 0.7

            FakeYOLO.val_calls.clear()
            out = io.StringIO()
            with redirect_stdout(out):
                mod.main()
            output = out.getvalue()

            self.assertIn("Comparison results:", output)
            self.assertIn("Best by mAP50-95:", output)
            self.assertGreaterEqual(len(FakeYOLO.val_calls), 2)

            # Compare flow should create one dated session folder + summary csv.
            comparison_root = runs_root / "comparison"
            session_dirs = [d for d in comparison_root.iterdir() if d.is_dir()]
            self.assertEqual(len(session_dirs), 1)
            self.assertTrue((session_dirs[0] / "comparison_summary.csv").exists())

            _, kwargs = FakeYOLO.val_calls[0]
            self.assertTrue(str(kwargs["project"]).startswith(str(comparison_root / "cmp_")))


if __name__ == "__main__":
    unittest.main()
