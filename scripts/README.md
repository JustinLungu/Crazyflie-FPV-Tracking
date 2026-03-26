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
- `camera_calibration.sh`: run camera calibration (`depth_estimation/camera_calibration/calibration.py`)
- `naive_bbox_depth.sh`: run naive bbox depth (`depth_estimation/naive_bbox_depth/bbox_dist_estimator.py`)
- `unidepth_image.sh`: run UniDepth on one image (`depth_estimation/unidepth/depth_image_inference.py`)
- `unidepth_video.sh`: run UniDepth on a .avi video (`depth_estimation/unidepth/depth_video_inference.py`)
- `midas_image.sh`: run MiDaS on one image (`depth_estimation/midas/depth_image_inference.py`)
- `midas_video.sh`: run MiDaS on a .avi video (`depth_estimation/midas/depth_video_inference.py`)
- `live_inference.sh`: run real-time YOLO inference on live feed (`inference/live_inference.py`)
- `upload_backup.sh`: upload raw/labels backups to Drive (`data/upload_data_drive.py`)
- `run_tests.sh`: run system/integration test suite (`tests/test_*.py`)

## Notes

- Scripts that require configuration still read their corresponding constants:
  - data pipeline: `data/constants.py`
  - model pipeline: `models/constants.py`
  - unidepth pipeline: `depth_estimation/unidepth/constants.py`
  - midas pipeline: `depth_estimation/midas/constants.py`
  - naive bbox depth pipeline: `depth_estimation/naive_bbox_depth/constants.py`
  - camera calibration pipeline: `depth_estimation/camera_calibration/constants.py`
  - inference pipeline: `inference/constants.py`
- `create_dataset.sh` forwards optional CLI args to `data/create_dataset.py`.
