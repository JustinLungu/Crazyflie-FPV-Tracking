from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_estimation.naive_bbox_depth.constants import NAIVE_RUN_MODE
from depth_estimation.naive_bbox_depth.pipeline import NaiveBBoxDepthPipeline


def main() -> None:
    pipeline = NaiveBBoxDepthPipeline()
    mode = str(NAIVE_RUN_MODE).strip().lower()

    if mode == "live":
        pipeline.run_live()
        return
    if mode == "image":
        pipeline.run_image()
        return

    raise ValueError(
        f"Unsupported NAIVE_RUN_MODE='{NAIVE_RUN_MODE}'. Use 'image' or 'live'."
    )


if __name__ == "__main__":
    main()
