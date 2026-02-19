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
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module: {module_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == parent_dir:
            sys.path.pop(0)


class DepthUtilsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.utils_file = self.repo_root / "depth_estimation" / "utils.py"

    def test_resolve_existing_image_path_uses_fallback_extension(self) -> None:
        mod = load_module_from_file(self.utils_file, "depth_utils_test_mod")
        with tempfile.TemporaryDirectory(prefix="depth_utils_") as tmp:
            expected = Path(tmp) / "images" / "frame_000001.jpg"
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_bytes(b"test")

            requested_png = str(expected.with_suffix(".png"))
            resolved = mod.resolve_existing_image_path(requested_png, ("png", "jpg", "jpeg"))

            self.assertEqual(resolved, expected)

    def test_resolve_existing_image_path_raises_when_missing(self) -> None:
        mod = load_module_from_file(self.utils_file, "depth_utils_test_mod_missing")
        with tempfile.TemporaryDirectory(prefix="depth_utils_missing_") as tmp:
            requested = str(Path(tmp) / "images" / "missing.png")
            with self.assertRaises(RuntimeError) as err:
                mod.resolve_existing_image_path(requested, ("png", "jpg"))

            msg = str(err.exception)
            self.assertIn("Image not found. Tried:", msg)
            self.assertIn("missing.png", msg)


if __name__ == "__main__":
    unittest.main()
