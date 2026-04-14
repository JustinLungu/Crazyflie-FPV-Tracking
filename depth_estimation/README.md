# Depth Estimation

This folder contains the project depth pipelines.

Use this README as a quick overview. Detailed setup, constants, and usage live inside each pipeline folder.

## Live Runner

- `live_depth_estimation.py`
  - Main live entrypoint for depth estimation.
  - Select one or multiple methods with `--methods naive`, `--methods unidepth`, `--methods midas`, or combinations like `--methods naive,unidepth`.
  - Launch via `scripts/live_depth.sh`.
- `live_depth_review.py`
  - Constants-driven live reviewer with side telemetry panel (session-review style, but real-time camera).
  - Configure methods/camera/UI in `depth_estimation/constants.py` using `DEPTH_LIVE_REVIEW_METHODS`.
  - Designed for phased expansion (for example, start with `("naive",)` then move to `("naive", "unidepth")`).
  - Launch via `scripts/live_depth_review.sh`.

## Pipelines

- `camera_calibration/`
  - Checkerboard-based camera calibration.
  - Produces intrinsics/distortion outputs used by other depth/inference work.

- `naive_bbox_depth/`
  - Simple baseline distance estimation from YOLO bounding-box width.
  - Fast and lightweight, mainly for rough range estimates.
  - Includes a session playback reviewer (`session_depth_review.py`) for recorded test sessions.

- `unidepth/`
  - Monocular depth with UniDepth v2 for image and video input.

- `midas/`
  - Monocular depth with MiDaS for image and video input.

## OOP Structure

- Each pipeline exposes a class interface:
  - `CameraCalibrationPipeline`
  - `NaiveBBoxDepthPipeline`
  - `UniDepthPipeline`
  - `MiDaSPipeline`
- Live-capable pipelines implement `process_live_frame(...)`, so methods can be composed together in one live stream.
