# Crazyflie-FPV-Tracking

Real-time FPV data collection, labeling, training, inference, and depth-estimation pipeline for Crazyflie tracking.

The repository is focused on:
- collecting raw FPV data from a USB video receiver,
- exporting videos and sampled images,
- semi-automatic annotation with OpenCV trackers,
- organizing labels by class in a structure that supports later train/val/test splitting,
- training/testing/comparing YOLO models,
- running live YOLO inference on receiver feed,
- running flight control and YOLO live view concurrently via one integrated runtime,
- estimating depth with UniDepth v2 (and an additional bbox-width baseline).

## Goal

Build an end-to-end workflow for Crazyflie CV experiments: capture data, label it, prepare YOLO datasets, train/evaluate models, run live inference, and estimate depth.

## Repository Structure

```text
Crazyflie-FPV-Tracking/
├── data/
│   ├── constants.py                     # Data/label/dataset/backup config
│   ├── images_get_data.py               # Capture periodic still frames
│   ├── videos_get_data.py               # Record continuous video with FPS matching
│   ├── track_label_video.py             # Semi-auto tracker-based labeling
│   ├── view_labeling.py                 # Review/delete labeled frames
│   ├── create_dataset.py                # Merge label sessions into one dataset
│   ├── prepare_yolo_dataset.py          # Build YOLO train/val/test split
│   ├── upload_data_drive.py             # Zip+upload raw/labels backup to Drive
│   └── utils.py                         # Shared camera/tracker/session helpers
├── models/
│   ├── constants.py                     # YOLO train/test/compare config
│   ├── train_yolo.py                    # Train YOLO model
│   ├── test_yolo.py                     # Evaluate selected YOLO weights
│   ├── compare_models.py                # Compare multiple model refs on one split
│   └── utils.py
├── inference/
│   ├── constants.py                     # Live inference config
│   ├── live_inference.py                # Realtime YOLO inference on camera
│   └── utils.py
├── flight_vision/
│   ├── main.py                          # Combined drone_control + YOLO entrypoint
│   ├── app.py                           # Concurrent app orchestration
│   ├── camera_sources.py                # Frame-source interfaces and receiver factory
│   ├── vision_runtime.py                # YOLO + overlay + OpenCV presenter runtime
│   ├── constants.py                     # Integrated runtime defaults
│   └── README.md
├── depth_estimation/
│   ├── unidepth/
│   │   ├── constants.py                 # UniDepth image/video config
│   │   ├── depth_image_inference.py
│   │   ├── depth_video_inference.py
│   │   ├── unidepth_v2.py
│   │   └── utils.py
│   └── direct_depth_estimation/
│       ├── constants.py                 # Bbox-width distance baseline config
│       ├── bbox_dist_estimator.py
│       └── utils.py
├── scripts/
│   ├── *.sh                             # One launcher per feature
│   └── README.md                        # Launcher catalog
├── setting_up_camera/
│   ├── README.md                        # Receiver setup and debugging notes
│   └── get_visual.py                    # Minimal live camera test
├── tests/
│   └── test_*.py                        # Integration/system tests
├── runs/                                # Training/eval/comparison outputs
├── pyproject.toml
├── uv.lock
└── README.md
```

Generated data (not committed) is created under:
- `data/raw_data/`
- `data/labels/`
- `data/depth/`
- `runs/`

## Requirements

- Linux machine (tested with `/dev/videoX` + V4L2 flow)
- USB FPV receiver exposed as a video device
- Python 3.13
- `uv` package/environment manager
- GPU is recommended for UniDepth/YOLO inference and training

## Setup With uv

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create environment and install dependencies

Run from repository root:

```bash
uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate
uv sync
```

Alternative (no manual activation):

```bash
uv run python --version
```

## OpenCV Tracker Note

Tracker APIs like CSRT/KCF/MOSSE require contrib modules.
This project depends on:
- `opencv-contrib-python`

Quick check:

```bash
uv run python -c "import cv2; print(cv2.__version__, hasattr(cv2,'TrackerCSRT_create'))"
```

Expected final value is `True`.

## Data Collection Workflow

### 1. Capture still images

```bash
./scripts/capture_images.sh
```

Output:
- `data/raw_data/images_session_<timestamp>/images/*.jpg`
- `data/raw_data/images_session_<timestamp>/meta.csv`

### 2. Record a video

```bash
./scripts/capture_video.sh
```

Output:
- `data/raw_data/<DRONE_TYPE>_session_<timestamp>/video.avi`

Important:
- video writer FPS is matched to measured capture FPS to avoid sped-up playback.

## Labeling Workflow

### 1. Set labeling inputs in `data/constants.py`

At minimum update:
- `VIDEO_PATH`
- `LABEL_CLASS_NAME`
- `CLASS_ID`

### 2. Run labeling

```bash
./scripts/label_video.sh
```

### 3. Controls

- `q`: quit
- `p`: pause/resume
- `r`: redraw ROI and reinitialize tracker

### 4. Label output structure

Each run is saved to a unique session directory:

```text
data/labels/<class_name>/all_data/label_session_<timestamp>/
├── images/
├── labels/
└── meta.csv
```

This prevents overwrite across repeated labeling sessions of the same class.

Optional label review:

```bash
./scripts/review_labels.sh --session data/labels/<class_name>/all_data/<label_session_timestamp>
```

## Class Naming and IDs

- Use `LABEL_CLASS_NAME` for folder organization.
- Use `CLASS_ID` for YOLO numeric labels.
- If two visual variants are the same detection class, keep the same `CLASS_ID`.
- If they are different detection classes, use different `CLASS_ID` values consistently across the dataset.

## Dataset Preparation Workflow

### 1. Merge selected label sessions into one class dataset

```bash
./scripts/create_dataset.sh --class-name <class_name> --overwrite
```

By default this collects all direct subfolders under:

```text
data/labels/<class_name>/<LABEL_ALL_DATA_DIR>/
```

and includes folders that contain both `images/` and `labels/` (folder name can be anything, not only `label_session_*`).

Default output:

```text
data/labels/<class_name>/<class_name>_dataset/
├── images/
├── labels/
└── manifest.csv
```

### 2. Build YOLO train/val/test split dataset

Set YOLO split options in `data/constants.py` (`YOLO_*` keys), then run:

```bash
./scripts/prepare_yolo_dataset.sh
```

Output:

```text
data/labels/<class_name>/<class_name>_yolo/
├── images/train|val|test
├── labels/train|val|test
├── split_manifest.csv
└── dataset.yaml
```

## YOLO Model Workflow

### 1. Configure model and run settings

Edit `models/constants.py` for:
- dataset reference (`YOLO_TARGET_CLASS_NAME`, `YOLO_OUTPUT_DATASET_NAME`)
- training/eval settings
- model refs for comparison

### 2. Train

```bash
./scripts/train_yolo.sh
```

### 3. Evaluate selected weights

```bash
./scripts/test_yolo.sh
```

### 4. Compare multiple model refs

```bash
./scripts/compare_models.sh
```

Outputs are written under:
- `runs/models/`
- `runs/evaluation/`
- `runs/comparison/`

## Depth Estimation Workflow

### 1. UniDepth v2 on one image

Set `DEPTH_IMAGE_*` in `depth_estimation/unidepth/constants.py`, then run:

```bash
./scripts/depth_image.sh
```

Output:
- depth array (`.npy`)
- depth visualization (`.png`)

### 2. UniDepth v2 on one video

Set `DEPTH_VIDEO_*` in `depth_estimation/unidepth/constants.py`, then run:

```bash
./scripts/depth_video.sh
```

### 3. Direct bbox-width distance baseline (experimental)

Configure `depth_estimation/direct_depth_estimation/constants.py`, then run:

```bash
uv run python depth_estimation/direct_depth_estimation/bbox_dist_estimator.py
```

Live mode:

```bash
uv run python depth_estimation/direct_depth_estimation/bbox_dist_estimator.py --live
```

## Live Inference Workflow

Set `inference/constants.py` (camera + weights), then run:

```bash
./scripts/live_inference.sh
```

Controls:
- `q` or `ESC` to quit

## Script Launchers

Feature launchers are in `scripts/` (data capture, labeling, dataset prep, training, testing, depth, live inference, backups).

Full launcher catalog:
- `scripts/README.md`

## Testing

Run repository integration/system tests:

```bash
./scripts/run_tests.sh
```

## Troubleshooting

### Camera cannot open (`Could not open camera at /dev/videoX`)

- Verify device path with:
```bash
v4l2-ctl --list-devices
```
- See `setting_up_camera/README.md` for full receiver checks.

## Notes

- Generated data under `data/raw_data`, `data/labels`, and `data/depth` is ignored by `data/.gitignore`.
- `scripts/*.sh` can be run from repo root or via absolute path.
