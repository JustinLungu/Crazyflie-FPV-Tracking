from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main() -> None:
    from flight_vision.app import ConcurrentFlightVisionApp

    app = ConcurrentFlightVisionApp()
    app.run()


if __name__ == "__main__":
    main()
