from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_estimation.camera_calibration.pipeline import CameraCalibrationPipeline


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    pipeline = CameraCalibrationPipeline()
    pipeline.run_checkerboard_detection(image_path=image_path)


if __name__ == "__main__":
    main()
