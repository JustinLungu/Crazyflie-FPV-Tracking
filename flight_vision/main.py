from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run flight vision runtime (drone control + live YOLO)."
    )
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Run camera + YOLO only (skip Crazyradio/Crazyflie control).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    from flight_vision.app import ConcurrentFlightVisionApp

    app = ConcurrentFlightVisionApp(enable_drone_control=not args.vision_only)
    app.run()


if __name__ == "__main__":
    main()
