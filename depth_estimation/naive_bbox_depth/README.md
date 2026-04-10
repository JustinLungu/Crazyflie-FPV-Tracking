# Naive BBox Depth

A minimal depth baseline based on bounding-box width.

## What It Does

1. Runs YOLO on an image (or live camera frame).
2. Selects the highest-confidence detection.
3. Estimates distance with:
   - `z = (fx * real_width_m) / bbox_width_px`
4. Converts bbox center + distance into camera-relative position:
   - `x_rel_m, y_rel_m, z_rel_m`
   - `yaw_error_rad`
5. Draws bbox + estimated distance/relative position on the image.
6. Saves annotated image to output folder (non-live mode).

## Files

- `bbox_dist_estimator.py`: CLI entrypoint.
- `session_depth_review.py`: interactive playback over a recorded session (`images/`) with YOLO + naive depth overlay.
- `utils.py`: detection processing, distance math, drawing, camera loop.
- `constants.py`:
  - `MODEL_PATH`: YOLO weights path
  - intrinsics: `FX`, `FY`, `CX`, `CY`
  - intrinsics source switching: `NAIVE_INTRINSICS_SOURCE`, `NAIVE_CAMERA_MATRIX_PATH`
  - `DRONE_WIDTH_M`: assumed real drone width
  - `OUTPUT_DIR`: non-live save path
  - camera settings for `--live`
  - relative-position settings: `NAIVE_ENABLE_RELATIVE_POSITION`, `NAIVE_Y_AXIS_CONVENTION`
  - session-review settings for playback (`NAIVE_REVIEW_*`)

## Outputs

- Non-live output directory:
  - `depth_estimation/output/naive_bbox/`
- Output filename format:
  - `<input_image_stem>_distance_estimate.jpg`
- Session review logs (per-frame CSV):
  - `depth_estimation/output/naive_bbox/review_logs/`
  - columns include raw + filtered values:
  - `frame_index`, `infer_ms`, `track_state`, `frames_since_detection`, `estimate_source`, `filter_mode`
  - `raw_bbox_width_px`, `bbox_width_px`, `raw_bbox_center_x_px`, `bbox_center_x_px`
  - `raw_distance_m`, `distance_m`
  - `raw_x_rel_m`, `raw_y_rel_m`, `raw_z_rel_m`, `x_rel_m`, `y_rel_m`, `z_rel_m`
  - `raw_yaw_error_rad`, `yaw_error_rad`

## Filtering

All filtering behavior is controlled in `constants.py`:

- `NAIVE_FILTER_MODE`: `none`, `ema`, or `kalman`
- signal toggles: `NAIVE_FILTER_DISTANCE`, `NAIVE_FILTER_CENTER`, `NAIVE_FILTER_WIDTH`
- EMA params: `NAIVE_EMA_ALPHA_*`
- Kalman params: `NAIVE_KALMAN_*`
- dropout handling: `NAIVE_DROPOUT_HOLD_FRAMES`, `NAIVE_DROPOUT_STALE_FRAMES`
- runtime `track_state`: `tracked`, `held`, `stale`, `lost`

Suggested progression:

1. Version 1: set `NAIVE_FILTER_MODE="ema"`, `NAIVE_FILTER_DISTANCE=True`, center/width `False`.
2. Version 2: enable `NAIVE_FILTER_CENTER=True`.
3. Version 3: tune dropout hold/stale thresholds.
4. Version 4: switch `NAIVE_FILTER_MODE="kalman"` if EMA is not enough.

## Run

From repo root:

```bash
./scripts/naive_bbox_depth.sh
./scripts/naive_bbox_depth.sh <image_path>
./scripts/naive_bbox_depth.sh --live
./scripts/session_naive_depth_review.sh
./scripts/session_naive_depth_review.sh --session data/labels/brushless_drone/all_data/test/label_session_YYYYMMDD_HHMMSS
./scripts/session_naive_depth_review.sh --no-write-log
```
