from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_estimation.midas.pipeline import MiDaSPipeline


def main() -> None:
    pipeline = MiDaSPipeline()
    pipeline.run_image()


if __name__ == "__main__":
    main()
