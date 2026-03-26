# Depth Estimation

This folder contains the project depth pipelines.

Use this README as a quick overview. Detailed setup, constants, and usage live inside each pipeline folder.

## Pipelines

- `camera_calibration/`
  - Checkerboard-based camera calibration.
  - Produces intrinsics/distortion outputs used by other depth/inference work.

- `naive_bbox_depth/`
  - Simple baseline distance estimation from YOLO bounding-box width.
  - Fast and lightweight, mainly for rough range estimates.

- `unidepth/`
  - Monocular depth with UniDepth v2 for image and video input.

- `midas/`
  - Monocular depth with MiDaS for image and video input.
