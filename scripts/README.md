# Script Launchers

This folder provides one shell launcher per project feature so you can avoid
remembering Python entrypoints and import-path details.

All scripts:
- force execution from repository root, and
- run via `uv run python ...` when `uv` exists (fallback to `python3`).

## Usage

From repository root:

```bash
./scripts/<script_name>.sh
```

From anywhere:

```bash
/absolute/path/to/Crazyflie-FPV-Tracking/scripts/<script_name>.sh
```

## Available Launchers

- `live_view.sh`: live camera preview (`setting_up_camera/get_visual.py`)
- `capture_images.sh`: capture image session (`data/images_get_data.py`)
- `capture_video.sh`: capture video session (`data/videos_get_data.py`)
- `label_video.sh`: tracker-assisted labeling (`data/track_label_video.py`)
- `review_labels.sh`: interactive label review (`data/view_labeling.py`)
- `create_dataset.sh`: merge label sessions (`data/create_dataset.py`)
- `prepare_yolo_dataset.sh`: build YOLO split dataset (`data/prepare_yolo_dataset.py`)
- `train_yolo.sh`: train model (`models/train_yolo.py`)
- `test_yolo.sh`: evaluate selected model (`models/test_yolo.py`)
- `compare_models.sh`: compare multiple models (`models/compare_models.py`)
- `live_inference.sh`: run real-time YOLO inference on live feed (`inference/live_inference.py`)
- `upload_backup.sh`: upload raw/labels backups to Drive (`data/upload_data_drive.py`)
- `run_tests.sh`: run system/integration test suite (`tests/test_*.py`)

## Notes

- Scripts that require configuration still read their corresponding constants:
  - data pipeline: `data/constants.py`
  - model pipeline: `models/constants.py`
  - inference pipeline: `inference/constants.py`
- `create_dataset.sh` forwards optional CLI args to `data/create_dataset.py`.
