# Flight Vision Module

This module runs **drone control** and **live YOLO detection** together.

It is designed to be:
- OOP and separable by responsibility (camera source, detector, presenter, orchestrator),
- compatible with the existing `drone_control` and `inference` configuration defaults.

## Entry Point

- Python: `flight_vision/main.py`
- Shell launcher: `./scripts/flight_vision.sh`

## Quick Run

From repo root:

```bash
./scripts/flight_vision.sh
```

This starts:
- `DroneControlApp` for flight logic
- YOLO live detection window using defaults from `inference/constants.py`

For camera/model checks without a Crazyradio dongle:

```bash
./scripts/flight_vision.sh --vision-only
```

This runs YOLO live view only and skips Crazyflie connection.

## Frequently Switched Settings

Edit `flight_vision/constants.py`:
- `FLIGHT_DRONE_URI`
- `FLIGHT_MISSION`
- `VISION_CAMERA_DEVICE`
- `VISION_CAMERA_WIDTH`
- `VISION_CAMERA_HEIGHT`
- `VISION_MODEL_WEIGHTS`
- `VISION_INFER_DEVICE`

The module uses a single camera path: local USB receiver via `/dev/videoX` (V4L2).

## Extend With New Sources

To add a custom source later:
- implement `FrameSource` (`open/read/close`)
- add a creation helper in `flight_vision/camera_sources.py`
- wire it in `flight_vision/app.py`

This keeps camera integration independent from flight logic.
