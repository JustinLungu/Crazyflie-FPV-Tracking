# Naive BBox Depth

A minimal depth baseline based on bounding-box width.

## What It Does

1. Runs YOLO on an image (or live camera frame).
2. Selects the highest-confidence detection.
3. Estimates distance with:
   - `z = (fx * real_width_m) / bbox_width_px`
4. Draws bbox + estimated distance on the image.
5. Saves annotated image to output folder (non-live mode).

## Files

- `bbox_dist_estimator.py`: CLI entrypoint.
- `session_depth_review.py`: interactive playback over a recorded session (`images/`) with YOLO + naive depth overlay.
- `utils.py`: detection processing, distance math, drawing, camera loop.
- `constants.py`:
  - `MODEL_PATH`: YOLO weights path
  - `FX`: focal length in pixels
  - `DRONE_WIDTH_M`: assumed real drone width
  - `OUTPUT_DIR`: non-live save path
  - camera settings for `--live`
  - session-review settings for playback (`NAIVE_REVIEW_*`)

## Outputs

- Non-live output directory:
  - `depth_estimation/output/naive_bbox/`
- Output filename format:
  - `<input_image_stem>_distance_estimate.jpg`
- Session review logs (per-frame CSV):
  - `depth_estimation/output/naive_bbox/review_logs/`
  - columns: `frame_index`, `infer_ms`, `confidence`, `bbox_width_px`, `bbox_center_x_px`, `bbox_center_y_px`, `distance_m`

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
