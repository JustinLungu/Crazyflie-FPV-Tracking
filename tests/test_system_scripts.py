import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


class ScriptLauncherSystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.scripts_dir = self.repo_root / "scripts"

    def test_all_launchers_dispatch_via_repo_root(self) -> None:
        script_to_entrypoint = {
            "live_view.sh": "setting_up_camera/get_visual.py",
            "live_inference.sh": "inference/live_inference.py",
            "capture_images.sh": "data/images_get_data.py",
            "capture_video.sh": "data/videos_get_data.py",
            "label_video.sh": "data/track_label_video.py",
            "review_labels.sh": "data/view_labeling.py",
            "create_dataset.sh": "data/create_dataset.py",
            "prepare_yolo_dataset.sh": "data/prepare_yolo_dataset.py",
            "train_yolo.sh": "models/train_yolo.py",
            "test_yolo.sh": "models/test_yolo.py",
            "compare_models.sh": "models/compare_models.py",
            "depth_image.sh": "depth_estimation/depth_image_inference.py",
            "depth_video.sh": "depth_estimation/depth_video_inference.py",
            "upload_backup.sh": "data/upload_data_drive.py",
        }

        with tempfile.TemporaryDirectory(prefix="scripts_system_test_") as tmp:
            fake_bin = Path(tmp) / "bin"
            fake_bin.mkdir(parents=True, exist_ok=True)
            fake_uv = fake_bin / "uv"
            fake_uv.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "echo \"FAKE_UV cwd=$PWD args=$*\"\n"
            )
            fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

            env = dict(os.environ)
            env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"

            for script_name, entrypoint in script_to_entrypoint.items():
                script_path = self.scripts_dir / script_name
                self.assertTrue(script_path.exists(), f"Missing launcher: {script_path}")

                result = subprocess.run(
                    ["bash", str(script_path)],
                    cwd="/tmp",
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"{script_name} failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
                )
                self.assertIn("FAKE_UV", result.stdout)
                self.assertIn(f"cwd={self.repo_root}", result.stdout)
                self.assertIn(f"args=run python {entrypoint}", result.stdout)


if __name__ == "__main__":
    unittest.main()
