from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from depth_estimation.naive_bbox_depth.pipeline import NaiveBBoxDepthPipeline


def main() -> None:
    pipeline = NaiveBBoxDepthPipeline()

    if len(sys.argv) > 1 and sys.argv[1] == "--live":
        pipeline.run_live()
        return

    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    pipeline.run_image(image_path=image_path)


if __name__ == "__main__":
    main()
