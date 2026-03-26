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
- `utils.py`: detection processing, distance math, drawing, camera loop.
- `constants.py`:
  - `MODEL_PATH`: YOLO weights path
  - `FX`: focal length in pixels
  - `DRONE_WIDTH_M`: assumed real drone width
  - `OUTPUT_DIR`: non-live save path
  - camera settings for `--live`

## Outputs

- Non-live output directory:
  - `depth_estimation/output/naive_bbox/`
- Output filename format:
  - `<input_image_stem>_distance_estimate.jpg`

## Run

From repo root:

```bash
./scripts/naive_bbox_depth.sh
./scripts/naive_bbox_depth.sh <image_path>
./scripts/naive_bbox_depth.sh --live
```
