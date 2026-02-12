# Crazyflie-FPV-Tracking

Real-time FPV data collection and labeling pipeline for Crazyflie tracking.

The repository is focused on:
- collecting raw FPV data from a USB video receiver,
- exporting videos and sampled images,
- semi-automatic annotation with OpenCV trackers,
- organizing labels by class in a structure that supports later train/val/test splitting.

## Goal

Build a clean dataset pipeline for drone detection/tracking experiments with low-friction capture and annotation.

Current scope is data acquisition and labeling. Training/inference integration can be added on top of this dataset structure.

## Repository Structure

```text
Crazyflie-FPV-Tracking/
├── data/
│   ├── constants.py           # All configurable parameters
│   ├── utils.py               # Shared camera/tracker/session helpers
│   ├── images_get_data.py     # Capture periodic still frames
│   ├── videos_get_data.py     # Record continuous video with FPS matching
│   ├── track_label_video.py   # Semi-auto tracker-based labeling
│   └── .gitignore             # Ignores generated data folders
├── setting_up/
│   ├── README.md              # Receiver setup and debugging notes
│   └── get_visual.py          # Minimal live camera test
├── pyproject.toml
├── uv.lock
└── README.md
```

Generated data (not committed) is created under:
- `data/raw_data/`
- `data/labels/`

## Requirements

- Linux machine (tested with `/dev/videoX` + V4L2 flow)
- USB FPV receiver exposed as a video device
- Python 3.13
- `uv` package/environment manager

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
uv run python -c "import cv2; print(cv2.__version__, hasattr(cv2,'legacy'), hasattr(cv2.legacy,'TrackerCSRT_create'))"
```

Expected final value for CSRT support is `True`.

## Data Collection Workflow

### 1. Capture still images

```bash
uv run python data/images_get_data.py
```

Output:
- `data/raw_data/images_session_<timestamp>/images/*.jpg`
- `data/raw_data/images_session_<timestamp>/meta.csv`

### 2. Record a video

```bash
uv run python data/videos_get_data.py
```

Output:
- `data/raw_data/video_session_<timestamp>/video.avi`

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
uv run python data/track_label_video.py
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

## Class Naming and IDs

- Use `LABEL_CLASS_NAME` for folder organization.
- Use `CLASS_ID` for YOLO numeric labels.
- If two visual variants are the same detection class, keep the same `CLASS_ID`.
- If they are different detection classes, use different `CLASS_ID` values consistently across the dataset.

## Troubleshooting

### Camera cannot open (`Could not open camera at /dev/videoX`)

- Verify device path with:
```bash
v4l2-ctl --list-devices
```
- See `setting_up/README.md` for full receiver checks.

## Notes

- Generated data under `data/raw_data` and `data/labels` is ignored by `data/.gitignore`.
- `main.py` is currently a placeholder entry point.
