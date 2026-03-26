# UniDepth Pipeline

This folder runs monocular depth inference using UniDepth v2 for:
- single images
- offline videos (`.avi`)

The pipeline reads RGB frames, predicts a depth map, creates a colorized visualization, and optionally writes a side-by-side video output.

## What This Folder Contains

- `constants.py`: all runtime configuration (input paths, outputs, visualization, logging).
- `depth_image_inference.py`: single-image inference entrypoint.
- `depth_video_inference.py`: frame-by-frame video inference entrypoint.
- `unidepth_v2.py`: model wrapper around UniDepth v2 (`torch.hub` load + patch fix).
- `utils.py`: shared helpers for path handling, depth stats, colormap conversion, and resizing.

## End-to-End Flow

### Single Image

1. Resolve input image path from `DEPTH_IMAGE_INPUT_PATH` (with fallback extensions).
2. Load UniDepth v2 model.
3. Predict depth + intrinsics.
4. Compute center-patch depth statistic.
5. Save:
   - raw depth map (`.npy`)
   - colorized depth image (`.png`)

### Video

1. Open input video from `DEPTH_VIDEO_INPUT_PATH`.
2. For each frame:
   - run UniDepth inference
   - resize depth to frame size
   - colorize depth
   - compose side-by-side frame (RGB | depth)
   - overlay frame index, inference time, center depth
3. Optionally write output video (`DEPTH_VIDEO_WRITE_OUTPUT`).
4. Optionally show preview window (`DEPTH_VIDEO_SHOW_PREVIEW`).

## Model Notes

`unidepth_v2.py` currently:
- loads UniDepth from `torch.hub`:
  - repo: `lpiccinelli-eth/UniDepth`
  - model: `UniDepth`, `version="v2"`, `backbone="vitb14"`
- forces model to CUDA (`self.model.to("cuda")`)
- applies a monkey-patch for UniDepth padding bug fix (GitHub issue #139)
- supports optional `DEPTH_RESOLUTION_LEVEL` speed/detail tradeoff

## Configuration (`constants.py`)

Most important fields:

- `DEPTH_IMAGE_INPUT_PATH`: single-image input path.
- `DEPTH_VIDEO_INPUT_PATH`: video input path.
- `DEPTH_OUTPUT_ROOT`: output root folder.
- `DEPTH_IMAGE_OUTPUT_DIR`: where image outputs are saved.
- `DEPTH_IMAGE_OUTPUT_*_SUFFIX`: output filename suffixes for image mode.
- `DEPTH_VIDEO_OUTPUT_PATH`: output `.avi` path for video mode.
- `DEPTH_COLORMAP`, `DEPTH_INVERT_COLORMAP`: depth visualization style.
- `DEPTH_CENTER_PATCH_SIZE`: center-depth patch size.

## Outputs

Current output pattern:

- Image mode (flat in one folder):
  - `depth_estimation/output/unidepth/<DRONE_NAME>/<image_stem>_depth.npy`
  - `depth_estimation/output/unidepth/<DRONE_NAME>/<image_stem>_depth_vis.png`

- Video mode:
  - `depth_estimation/output/unidepth/<DRONE_NAME>/<CUSTOM_PATH_VIDEO>/video_depth_overlay.avi`

## Run

From repository root:

```bash
./scripts/unidepth_image.sh
./scripts/unidepth_video.sh
```

Manual equivalent:

```bash
uv run python depth_estimation/unidepth/depth_image_inference.py
uv run python depth_estimation/unidepth/depth_video_inference.py
```

## Quick Notes

- This pipeline is non-live (image file or video file input), though video mode can show a preview window.
- `KEY_QUIT` supports `q` or `ESC` for preview exit.
- If CUDA is unavailable, current model wrapper will need adjustment before running.
