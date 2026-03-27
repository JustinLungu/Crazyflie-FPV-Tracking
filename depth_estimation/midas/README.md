# MiDaS Pipeline

This folder runs monocular depth inference using MiDaS for:
- single images
- offline videos (`.avi`)

It mirrors the UniDepth folder structure and usage style.

## What This Folder Contains

- `constants.py`: all runtime configuration (input paths, output paths, visualization, logging).
- `midas_model.py`: model wrapper (`torch.hub` load + preprocessing transform + prediction).
- `depth_image_inference.py`: single-image inference entrypoint.
- `depth_video_inference.py`: frame-by-frame video inference entrypoint.
- `utils.py`: shared helpers for paths, center-depth stats, colormap rendering, and resizing.

## End-to-End Flow

### Single Image

1. Resolve image path from `MIDAS_IMAGE_INPUT_PATH` (with fallback extensions).
2. Load MiDaS model from torch hub.
3. Predict per-pixel depth map.
4. Compute center-patch depth statistic.
5. Save:
   - raw depth map (`.npy`)
   - colorized depth image (`.png`)

### Video

1. Open input video from `MIDAS_VIDEO_INPUT_PATH`.
2. For each frame:
   - run MiDaS inference
   - resize depth map to frame size
   - colorize depth
   - compose side-by-side output (RGB | depth)
   - overlay frame index, inference time, center depth
3. Optionally write video output (`MIDAS_VIDEO_WRITE_OUTPUT`).
4. Optionally show preview (`MIDAS_VIDEO_SHOW_PREVIEW`).

## Model Notes

`midas_model.py` currently:
- loads model with `torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)`
- loads matching MiDaS transform from `torch.hub` (`dpt_transform` or `small_transform`)
- supports `MIDAS_DEVICE = "auto"` (cuda/mps/cpu fallback)

Note: MiDaS output is relative depth (scale is not metric by default).

## Key Configuration (`constants.py`)

- `MIDAS_MODEL_TYPE`
- `MIDAS_DEVICE`
- `MIDAS_IMAGE_INPUT_PATH`
- `MIDAS_VIDEO_INPUT_PATH`
- `MIDAS_OUTPUT_ROOT`
- `MIDAS_IMAGE_OUTPUT_DIR`
- `MIDAS_VIDEO_OUTPUT_PATH`
- `MIDAS_COLORMAP`, `MIDAS_INVERT_COLORMAP`
- `MIDAS_CENTER_PATCH_SIZE`

## Outputs

- Image mode (flat):
  - `depth_estimation/output/midas/<DRONE_NAME>/<image_stem>_depth.npy`
  - `depth_estimation/output/midas/<DRONE_NAME>/<image_stem>_depth_vis.png`

- Video mode:
  - `depth_estimation/output/midas/<DRONE_NAME>/<CUSTOM_PATH_VIDEO>/video_depth_overlay.avi`

## Run

From repository root:

```bash
./scripts/midas_image.sh
./scripts/midas_video.sh
```

Manual equivalent:

```bash
uv run python depth_estimation/midas/depth_image_inference.py
uv run python depth_estimation/midas/depth_video_inference.py
```
