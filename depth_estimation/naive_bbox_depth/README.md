# Naive BBox Depth

Baseline monocular depth from YOLO bbox width, with optional temporal filtering, gating/sanity checks, and session-review telemetry.

## Overview

For each frame:

1. YOLO detections are produced.
2. A candidate target is selected.
3. Raw distance is estimated with `z = (fx * real_width_m) / bbox_width_px`.
4. Optional temporal filtering is applied (`none`, `ema`, `kalman`) on distance/center/width.
5. Optional relative pose outputs are computed from intrinsics:
   - `x_rel_m`, `y_rel_m`, `z_rel_m`
   - `yaw_error_rad`, `yaw_error_deg`
6. Optional gating rejects implausible detections before they update temporal state.
7. Overlay + telemetry/log metrics are produced.

## Entry Points

- `bbox_dist_estimator.py`: constants-driven entrypoint.
  - `NAIVE_RUN_MODE = "image"` runs single-image inference.
  - `NAIVE_RUN_MODE = "live"` runs live camera inference.
- `session_depth_review.py`: replay recorded sessions (`<session>/images`) with interactive controls.

## Main Components

- `pipeline.py`
  - core inference pipeline (`NaiveBBoxDepthPipeline`)
  - intrinsics loading
  - filtering/dropout logic
  - gating logic
  - live run loop + runtime gating toggle (`g`)
- `filtering.py`
  - `ExponentialMovingAverage`
  - `ConstantVelocityKalman1D`
  - `ScalarSignalFilter`
- `utils.py`
  - geometry helpers (distance, relative position, yaw)
  - path/output helpers
- `constants.py`
  - all runtime configuration (modes, intrinsics, filters, gating, review UI/logging)

## Intrinsics and Geometry

- Intrinsics source is controlled by:
  - `NAIVE_INTRINSICS_SOURCE` (`"manual"` or `"calibration_npy"`)
  - `NAIVE_CAMERA_MATRIX_PATH`
  - `NAIVE_INTRINSICS_FALLBACK_TO_MANUAL`
- Manual values:
  - `FX`, `FY`, `CX`, `CY`
- Relative outputs:
  - enable with `NAIVE_ENABLE_RELATIVE_POSITION`
  - choose y sign convention with `NAIVE_Y_AXIS_CONVENTION` (`"up"` or `"down"`)
  - frame overlay toggle with `NAIVE_SHOW_RELATIVE_OVERLAY_ON_FRAME`

## Temporal Filtering and Dropout

Filtering is controlled in `constants.py`:

- `NAIVE_FILTER_MODE`: `none`, `ema`, `kalman`
- signal toggles:
  - `NAIVE_FILTER_DISTANCE`
  - `NAIVE_FILTER_CENTER`
  - `NAIVE_FILTER_WIDTH`
- EMA parameters:
  - `NAIVE_EMA_ALPHA_DISTANCE`
  - `NAIVE_EMA_ALPHA_CENTER`
  - `NAIVE_EMA_ALPHA_WIDTH`
- Kalman parameters:
  - `NAIVE_KALMAN_PROCESS_VAR_*`
  - `NAIVE_KALMAN_MEAS_VAR_*`

Dropout handling:

- `NAIVE_DROPOUT_HOLD_FRAMES`: keep last estimate for short misses
- `NAIVE_DROPOUT_STALE_FRAMES`: then mark stale before full loss
- `NAIVE_RESET_FILTER_ON_LOST`: reset temporal state after long loss
- Track states:
  - `tracked`, `held`, `stale`, `lost`

## Gating (Safety Layer)

Gating is optional and fully configurable:

- master switch:
  - `NAIVE_GATING_ENABLED`
- candidate count:
  - `NAIVE_GATING_MAX_CANDIDATES`
- checks:
  - `NAIVE_GATING_CHECK_CONFIDENCE`
  - `NAIVE_GATING_CHECK_MIN_WIDTH`
  - `NAIVE_GATING_CHECK_MAX_DISTANCE`
  - `NAIVE_GATING_CHECK_BORDER`
  - `NAIVE_GATING_CHECK_DISTANCE_JUMP`
  - `NAIVE_GATING_CHECK_X_JUMP`
  - `NAIVE_GATING_CHECK_Y_JUMP`
- thresholds:
  - `NAIVE_GATING_MIN_CONF_FOR_CONTROL`
  - `NAIVE_GATING_MIN_BBOX_WIDTH_PX`
  - `NAIVE_GATING_MAX_VALID_DISTANCE_M`
  - `NAIVE_GATING_BORDER_MARGIN_PX`
  - `NAIVE_GATING_MAX_DISTANCE_JUMP_M`
  - `NAIVE_GATING_MAX_X_JUMP_M`
  - `NAIVE_GATING_MAX_Y_JUMP_M`
- rejected-overlay toggle:
  - `NAIVE_GATING_SHOW_REJECTION_OVERLAY`

Runtime behavior:

- When gating is `OFF`, the pipeline uses only the top-confidence detection (`1` active candidate).
- When gating is `ON`, it evaluates up to top-`K` candidates (`K = NAIVE_GATING_MAX_CANDIDATES`) and accepts the first passing candidate.
- Press `g` in live mode or session review to toggle gating.

## Session Review UI and Controls

Session review constants:

- `NAIVE_REVIEW_SESSION_DIR`
- `NAIVE_REVIEW_DELAY_S`
- `NAIVE_REVIEW_START_PAUSED`
- `NAIVE_REVIEW_USE_SIDE_PANEL`
- `NAIVE_REVIEW_SIDE_PANEL_*`
- `NAIVE_REVIEW_WRITE_LOG`, `NAIVE_REVIEW_LOG_DIR`

Controls in review window:

- `space`: play/pause
- `a` or Left Arrow: previous frame
- `d` or Right Arrow: next frame
- `g`: toggle gating and reprocess timeline
- `q` or `ESC`: quit

Live mode control:

- `g`: toggle gating
- `q` or `ESC`: quit

## Real-Time Metrics

Current per-frame metrics include:

- runtime:
  - `infer_ms`, `infer_fps`
  - `process_ms`, `process_fps`
- detection/candidate:
  - `detection_count` (active considered candidates)
  - `yolo_detection_count` (total YOLO detections)
  - `candidate_pool`, `candidate_limit`, `selected_candidate_rank`
- track/filter:
  - `track_state`, `frames_since_detection`, `estimate_source`, `is_stale`, `filter_mode`
- bbox/range:
  - `confidence`
  - `raw_bbox_width_px`, `bbox_width_px`
  - `raw_bbox_center_x_px`, `raw_bbox_center_y_px`
  - `bbox_center_x_px`, `bbox_center_y_px`
  - `raw_distance_m`, `distance_m`
- relative outputs:
  - `raw_x_rel_m`, `raw_y_rel_m`, `raw_z_rel_m`
  - `x_rel_m`, `y_rel_m`, `z_rel_m`
  - `raw_yaw_error_rad`, `raw_yaw_error_deg`
  - `yaw_error_rad`, `yaw_error_deg`
- gating:
  - `gating_enabled`, `gating_passed`, `gating_reasons`
  - `gating_rejected_candidates`, `best_rejected_rank` (when rejected)

## FPS Note (Delay vs Compute)

`process_fps` excludes the manual playback delay (`NAIVE_REVIEW_DELAY_S`) because it is measured inside frame processing only. It can still change with different delays due to hardware scheduling and clock-state effects (for example CPU/GPU boost/idle behavior), even on the same frames.

## Output Paths

- Annotated image output:
  - `depth_estimation/output/naive_bbox/`
- Session review CSV logs:
  - `depth_estimation/output/naive_bbox/review_logs/`

## Run

From repo root:

```bash
./scripts/naive_bbox_depth.sh
./scripts/session_naive_depth_review.sh
```
