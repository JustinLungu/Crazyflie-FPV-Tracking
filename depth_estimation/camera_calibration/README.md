# Camera Calibration

This folder contains checkerboard-based intrinsic calibration for the camera used in depth/inference pipelines.

## What Lives Here

- `calibration.py`: main calibration pipeline.
- `check_board_detection.py`: quick one-image checkerboard detection sanity check.
- `calibration_images/input_images/`: input calibration images (`.jpg`).
- `calibration_images/corner_detections/`: saved visualizations of detected corners.
- `calibration_images/camera_matrix.npy`: saved intrinsic matrix `K`.
- `calibration_images/dist_coeffs.npy`: saved distortion coefficients.
- `calibration_images/calibration_sample_original.jpg`: sample original frame.
- `calibration_images/calibration_sample_undistorted.jpg`: sample undistorted frame.

## Current Calibration Settings

Defined in `calibration.py`:

- Checkerboard internal corners: `CHECKERBOARD = (10, 7)`
- Square size: `SQUARE_SIZE = 0.025` meters (25 mm)

If your printed board is different, update these constants before running calibration.

## Folder Assumptions

`calibration.py` now uses your current folder structure (renamed from the old initial layout):

- Reads images from `depth_estimation/camera_calibration/calibration_images/input_images/`
- Writes outputs to `depth_estimation/camera_calibration/calibration_images/`

## Run Calibration

From repo root:

```bash
./scripts/camera_calibration.sh
```

Manual equivalent:

```bash
uv run python depth_estimation/camera_calibration/calibration.py
# or: python3 depth_estimation/camera_calibration/calibration.py
```

## Quick Checkerboard Test

```bash
uv run python depth_estimation/camera_calibration/check_board_detection.py
# optional: pass a specific image path
uv run python depth_estimation/camera_calibration/check_board_detection.py depth_estimation/camera_calibration/calibration_images/input_images/frame_1.jpg
```

This is useful to confirm the board pattern is detected before running full calibration.

## Quality Target

Reprojection RMSE guide:

- Good: below `1.0 px`
- Better: around `0.7 px` or lower
- Very strong: around `0.3-0.6 px`
