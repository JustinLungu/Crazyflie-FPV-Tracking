# Report Completion Draft

This draft is written as report-ready material to extend the current semester report. It focuses on the work that is only briefly mentioned or not yet included: data backup/upload, YOLO dataset preparation, YOLO training and inference, depth estimation, live review tools, and flight-control integration.

## Suggested Updated Report Structure

1. Introduction and objective
2. Hardware platform and camera setup
3. Drone control foundation
4. Simulation exploration
5. Camera receiver setup and stress testing
6. Data collection and backup workflow
7. Dataset labeling and quality control
8. YOLO dataset preparation
9. YOLO training, checkpoints, and evaluation
10. Live YOLO inference
11. Depth estimation
12. Integrated flight and vision runtime
13. Drone follower demo
14. Testing and software engineering
15. Limitations
16. Future work
17. Conclusion

## Hardware Platform and Camera Setup

The project uses Bitcraze Crazyflie drones as the experimental platform. Both brushed and brushless variants were considered during development. The brushed platform was useful for initial familiarization because it is simpler and cheaper to work with, while the brushless platform became the main target for the final computer-vision pipeline due to its stronger flight performance and closer relevance to real evader-pursuer experiments.

The onboard vision setup was intentionally based on a low-cost analog FPV camera and USB receiver. This made the perception problem harder because the stream has limited resolution, compression artifacts, noise, occasional dropouts, and changing image quality depending on receiver distance and channel lock. However, this also made the system a useful robustness test: if detection and tracking can work reliably on this camera, the same pipeline should become easier to run on a better camera.

Before building the learning pipeline, I created a camera setup and debugging workflow. This included manual checks for the USB video device, V4L2 camera access, OpenCV live preview, receiver channel scanning, and fallback debugging through external tools such as VLC, guvcview, and ffplay. The repository contains a minimal live preview script to verify that the receiver stream can be opened and displayed before running heavier detection or depth pipelines.

TODO evidence to add here later:

- Figure: brushed Crazyflie and brushless Crazyflie side by side.
- Figure: FPV camera, receiver, antenna, battery/wiring, and camera mount.
- Table: brushed vs brushless comparison: price, weight, flight performance, payload capacity, stability, and why the brushless platform became the main perception target.
- Figure: final camera mounting position on the drone, including any 3D-printed holder if used.

## Simulation Exploration

During the first phase, I also briefly explored IsaacSim to understand the available drone simulation and training environment. This was not the main implementation path for the semester, but it helped build context for how simulated drone perception and control experiments could later complement the real-world FPV pipeline.

TODO evidence to add here later:

- Add 1 short paragraph explaining what was tested in IsaacSim and why it was not prioritized further.
- Figure: IsaacSim screenshot if available.
- Future-work note: simulation could later be used for synthetic data generation, control prototyping, or obstacle-avoidance experiments before real drone tests.

## Camera Receiver Stress Testing

To justify the reliability of the FPV camera link, I implemented a camera stress-test workflow under `setting_up_camera/camera_stress_tests/`. The current protocol is link-focused: the receiver and laptop stay fixed, while the camera-drone distance to the receiver is changed. The core planned distances are 0.5 m, 1.0 m, 2.0 m, and 3.0 m.

For each run, the system records stream metadata and optionally a raw video. The logged metrics include actual FPS, frame-to-frame timing, estimated dropped frames, freeze count, blur proxy using Laplacian variance, and timing stability through the 95th percentile frame interval. A separate analysis script converts each run into `analysis_summary.json` and `analysis_report.md`, while a campaign summary script combines runs into a table for the report.

This part of the project is important because the camera stream quality directly affects detection confidence, tracker stability, depth estimation, and ultimately closed-loop flight behavior. It also gives a repeatable way to compare the original cheap camera against the improved camera that was ordered later.

TODO evidence to add here later:

- Table: actual stress-test results for 0.5 m, 1.0 m, 2.0 m, and 3.0 m after rerunning/analyzing the current protocol.
- Figure: example raw frame from near distance and far distance.
- Figure or table: FPS ratio, drop percentage, freeze count, blur percentage, and `dt_p95_ms`.
- If results are not ready, phrase this as "evaluation framework implemented; final campaign results pending."

## Data Collection and Backup Workflow

After the camera feed was established, I implemented separate scripts for collecting still images and videos. Image capture stores frames at a target FPS into timestamped sessions and writes metadata such as frame index and timestamps. Video capture records continuous `.avi` files and measures the actual delivered camera FPS before creating the video writer. This avoids a common problem where recorded videos play too fast because the writer FPS does not match the real capture FPS.

The raw data is organized under `data/raw_data/`, with one timestamped folder per capture session. In the current workspace there are 48 raw-data session folders, covering early black/green drone experiments, brushless drone videos, and test brushless sessions.

I also added a backup/upload module for preserving experimental data. The script `data/upload_data_drive.py` creates zip archives for both raw data and labels, then uploads them to a configured Google Drive folder. The private Drive target is read from `.env`, while OAuth credentials and tokens are stored separately and ignored by Git. This provides a reproducible way to checkpoint the dataset outside the local machine without committing large experimental data into the repository.

The backup names follow a date-based convention:

- `dataset_backup_<date>_raw_data.zip`
- `dataset_backup_<date>_labels.zip`

This makes it possible to recover older labeling or data-collection checkpoints if later filtering, relabeling, or retraining changes the local dataset.

TODO evidence to add here later:

- Table: total recorded sessions, approximate total raw frames/video time, and final retained high-quality samples.
- Add the actual Google Drive backup date(s), archive names, and what was uploaded if the script was run.
- Figure: folder structure screenshot or small tree showing `raw_data`, `labels`, and backup archive naming.

## Dataset Labeling and Quality Control

A major part of the work was building a semi-automatic labeling pipeline for drone detection. The labeling script reads a recorded video, initializes a bounding box around the target drone, tracks the object frame by frame, and exports YOLO-format image-label pairs at a configured export FPS.

Initially, the tracker was based on OpenCV CSRT. CSRT is useful because it is more accurate than simpler correlation trackers when the object changes scale or appearance, although it is slower. This tradeoff is acceptable for offline labeling because labeling quality is more important than real-time speed. The code also supports the structure needed to compare or switch to other OpenCV trackers later.

The labeling pipeline was later improved with YOLO-assisted labeling. If trained class-matched YOLO weights are available under `yolo_best_models/`, the labeler can use YOLO detections as the primary bounding-box source and keep CSRT as a fallback. This makes the labeling process almost fully automated: YOLO corrects the box when it is confident, while the tracker maintains continuity when YOLO briefly fails. A jump-rejection check prevents YOLO from suddenly switching to a false detection far away from the previous accepted target.

Each labeling run is saved as a unique session folder:

```text
data/labels/<class_name>/all_data/<bucket>/label_session_<timestamp>/
├── images/
├── labels/
└── meta.csv
```

The metadata file records the source video frame, exported image name, label name, whether the bounding box was valid, whether the box came from YOLO or the tracker, and whether YOLO candidates were rejected.

I also implemented a label-review tool. It loads a labeled session, draws the YOLO bounding boxes on top of the images, and allows manual frame-by-frame review. Bad samples can be deleted together with their label files and metadata rows. This was used to remove images with no visible drone, bad quality, severe outliers, repeated frames, and frames containing the wrong target drone.

For the brushless drone specifically, the current labeled data contains approximately 54,000 image files across the `all_data` buckets. The merged brushless training/validation dataset contains 49,214 samples, and an additional manual test set contains 4,845 images.

TODO evidence to add here later:

- Figure: screenshot of the labeling window showing tracker/YOLO bounding box.
- Figure: screenshot of the label-review window.
- Add a short data-cleaning subsection: empty frames, wrong drone, blurred frames, heavy static, duplicate/repetitive images, partial target, and outlier boxes.
- Clarify whether negative/background-only images were included. If not, mention this as a limitation and future robustness improvement.
- Add one example of the `meta.csv` fields and explain why they help trace labeling quality.

## YOLO Dataset Preparation

The repository separates raw labeling sessions from YOLO-ready datasets. This makes the workflow safer because original label sessions are preserved, while the YOLO dataset can be regenerated with different split rules.

The script `data/create_dataset.py` merges selected label sessions into a contiguous dataset:

```text
data/labels/brushless_drone/brushless_drone_dataset/
├── images/
├── labels/
└── manifest.csv
```

The manifest records which original session and frame each merged sample came from. This is important for traceability and makes it possible to debug poor training samples later.

The script `data/prepare_yolo_dataset.py` then creates the YOLO train/validation/test structure:

```text
data/labels/brushless_drone/brushless_drone_yolo/
├── images/train
├── images/val
├── images/test
├── labels/train
├── labels/val
├── labels/test
├── split_manifest.csv
└── dataset.yaml
```

The current brushless YOLO split is:

| Split | Images |
|---|---:|
| Train | 43,735 |
| Validation | 5,455 |
| Test | 4,845 |
| Total | 54,035 |

The training and validation data are split using the configured split strategy, while the test set is taken from manually separated test sessions. This is useful because the final evaluation is then less affected by temporally adjacent frames from the training videos. The generated `dataset.yaml` maps class `0` to `brushless_drone`, and the preparation script can remap labels into single-class mode so the detector focuses on one target class.

TODO evidence to add here later:

- Figure/table: final dataset split counts.
- Add a small diagram showing `label_session_* -> brushless_drone_dataset -> brushless_drone_yolo`.
- Add one sample YOLO label line and explain normalized center-x, center-y, width, height.

## YOLO Training Process

The model pipeline is implemented under `models/`. The main training script is `models/train_yolo.py`, and all training settings are controlled through `models/constants.py`.

The current brushless model was trained using:

| Setting | Value |
|---|---|
| Base model | `yolo26s.pt` |
| Image size | 1024 |
| Epochs | 100 |
| Batch size | 8 |
| Device | GPU `0` |
| Workers | 2 |
| Early stopping patience | 10 |
| Cache images | False |
| Dataset | `data/labels/brushless_drone/brushless_drone_yolo/dataset.yaml` |

The training script automatically creates timestamped run folders under:

```text
runs/models/<run_name>/
```

Each training run stores:

- `args.yaml`, containing the exact Ultralytics training configuration;
- `results.csv`, containing epoch-by-epoch losses and metrics;
- `results.png`, training curves;
- confusion matrices and precision/recall curves when produced;
- sample training and validation batches;
- `weights/best.pt`;
- `weights/last.pt`.

This means each model checkpoint is traceable to the dataset, training hyperparameters, and output metrics. The `best.pt` checkpoint is the model selected by validation performance, while `last.pt` is the final epoch checkpoint. The repository also supports resuming training from a previous `last.pt` checkpoint.

For the final brushless model run `brushless_drone_yolo26s_20260410_220049`, the last epoch in `results.csv` reached approximately:

| Metric | Value |
|---|---:|
| Precision | 0.996 |
| Recall | 0.995 |
| mAP50 | 0.995 |
| mAP50-95 | 0.856 |

These numbers are validation metrics from the training run. They show that the detector fits the curated validation data very well. Separate test-set evaluation is still important because the test data comes from different held-out/manual sessions and is therefore more realistic.

TODO evidence to add here later:

- Figure: `results.png` training curves from the final brushless run.
- Figure: `labels.jpg` or training batch example to show dataset distribution.
- Figure: validation prediction examples, e.g. `val_batch*_pred.jpg`.
- Add a short explanation of why validation metrics are much higher than held-out test metrics: validation is closer to the training distribution, while manual test sessions are harder and more realistic.
- Add checkpoint note: `best.pt` used for deployment; `last.pt` available for resume/debugging.

## YOLO Evaluation and Model Comparison

The repository contains two evaluation paths:

- `models/test_yolo.py` evaluates one selected model on a chosen split;
- `models/compare_models.py` evaluates several model references on the same split and writes a comparison CSV.

The evaluation scripts use Ultralytics validation and additionally apply duplicate suppression for the single-drone use case. The extra suppression rule removes overlapping boxes when the intersection divided by the smaller box area exceeds a configured threshold. This is more suitable than standard IoU alone when one predicted box is mostly contained inside another.

The latest brushless model comparison in the workspace compares:

- `brushless_drone_yolo26s_20260410_220049/weights/best.pt`
- `brushless_drone_yolo26s_20260331_172052/weights/best.pt`
- `runs/models/backup/weights/best.pt`

On the current test split, the comparison results were:

| Model | Precision | Recall | mAP50 | mAP50-95 | Inference ms |
|---|---:|---:|---:|---:|---:|
| 20260410 best | 0.825 | 0.757 | 0.746 | 0.311 | 11.34 |
| 20260331 best | 0.823 | 0.787 | 0.758 | 0.317 | 11.35 |
| backup best | 0.839 | 0.796 | 0.764 | 0.315 | 11.32 |

An earlier comparison showed that the first brushless model had much lower recall, while later models improved significantly:

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 20260322 best | 0.763 | 0.366 | 0.585 | 0.279 |
| 20260331 best | 0.860 | 0.815 | 0.804 | 0.339 |
| backup best | 0.869 | 0.818 | 0.806 | 0.339 |

This shows that the training process improved substantially after more data and better dataset preparation. It also shows that the final model choice is not based only on the training validation metrics, but also on held-out test behavior and inference speed.

TODO evidence to add here later:

- Figure: confusion matrix from the selected final evaluation run.
- Figure: precision-recall curve or F1 curve.
- Add selected random test preview grid from `runs/evaluation/rand_preview/`.
- If you choose one final checkpoint, explicitly justify it using both metrics and live behavior.

## Live YOLO Inference

The live inference module is implemented in `inference/live_inference.py`. It opens the FPV receiver stream, loads the trained YOLO model, performs real-time prediction, draws detections, and overlays runtime information such as detection count, inference time, and display FPS.

The current live inference configuration uses:

- camera device: `/dev/video2`;
- resolution: 640x480;
- model: `runs/models/brushless_drone_yolo26s_20260410_220049/weights/best.pt`;
- image size: 1024;
- confidence threshold: 0.4;
- IoU threshold: 0.9;
- max detections: 10;
- GPU device: 0.

Because live drone video can have brief dropouts or low-quality frames, I implemented an optional tracker failsafe in `inference/tracker_failsafe.py`. When YOLO detects the drone, the failsafe initializes an OpenCV tracker from the accepted bounding box. If YOLO then fails for a small number of frames, the tracker can provide a temporary fallback box instead of immediately declaring the target lost. This is not intended to replace YOLO, but to smooth over short failures in a live stream.

There is also a recorded-session inference reviewer in `inference/session_inference_review.py`. It replays a labeled session frame by frame, runs YOLO on each image, and allows pause, previous/next navigation, and visual inspection. This was useful for checking detector behavior on real held-out sessions before relying on the model in live experiments.

TODO evidence to add here later:

- Figure: live inference screenshot with bounding box and overlay.
- Table: measured live FPS, inference time, and end-to-end latency if available.
- Figure: example where tracker failsafe bridges a YOLO dropout, if you have a recorded example.
- Add note on maximum usable live distance once experimentally measured.

## Depth Estimation

Depth estimation was developed as a separate package with a shared pipeline interface. The base interface is `LiveDepthPipeline`, which exposes a `process_live_frame(...)` method. This makes it possible to run different depth methods under the same live-review tool.

Three depth approaches were implemented or explored:

1. UniDepth V2
2. MiDaS
3. Naive bounding-box geometric depth

### UniDepth V2

The UniDepth pipeline loads UniDepth V2 through `torch.hub`, runs single-image or video inference, saves raw depth arrays, and writes colorized depth visualizations. It also supports side-by-side video output showing the original frame and the predicted depth map.

A practical issue was that the local package name `depth_estimation/unidepth` can shadow the external UniDepth repository package. To avoid this, the model wrapper temporarily removes the local shadowing path and adds the cached torch hub repository to `sys.path` when needed. I also added a monkey patch for a known UniDepth padding issue where certain aspect ratios can produce invalid padding.

UniDepth is useful as a learned monocular depth baseline, but in this application the target drone is often very small in the image. When the drone occupies only a few pixels, dense monocular depth models do not have enough visual information to produce stable metric estimates for the object itself.

### MiDaS

The MiDaS pipeline mirrors the UniDepth structure. It loads a MiDaS model from `torch.hub`, applies the correct transform, predicts a relative depth map, computes a center-patch depth statistic, and writes `.npy`, `.png`, or side-by-side video outputs.

MiDaS is useful as another learned baseline. However, its depth is relative rather than directly metric, so it is less directly useful for closed-loop control unless it is calibrated or combined with another scale source.

### Geometric Bounding-Box Depth

The most practical depth approach for this project was the geometric bounding-box method. It estimates forward distance using:

```text
distance = (focal_length_px * real_drone_width_m) / bounding_box_width_px
```

This method is simple but well matched to the problem because the target class has a known physical width and YOLO already provides the bounding box. It is also much faster than dense monocular depth models and gives metric distance directly.

To support this, I performed camera calibration using a checkerboard. The calibration produced:

```text
fx = 218.867 px
fy = 217.141 px
cx = 322.385 px
cy = 236.242 px
```

The distortion coefficients were:

```text
[0.2219, -0.2532, -0.0009, -0.0022, 0.0675]
```

The naive depth pipeline can load these intrinsics from `camera_matrix.npy`, with a manual fallback if the calibration file is not available. In addition to forward distance, the pipeline computes:

- horizontal relative position `x_rel_m`;
- vertical relative position `y_rel_m`;
- forward distance `z_rel_m`;
- yaw error in radians and degrees.

These values are the main control-oriented outputs used later by the drone follower demo.

TODO evidence to add here later:

- Figure: checkerboard calibration image with detected corners.
- Figure: original vs undistorted calibration sample.
- Figure: UniDepth depth visualization and MiDaS depth visualization on the same image/video frame.
- Figure: naive bbox-depth annotated frame.
- Table: real measured distance vs naive bbox-depth estimate vs UniDepth/MiDaS estimate if you run ground-truth tests.
- Add exact measured drone width used for `real_drone_width_m`.

## Temporal Filtering, Gating, and Review for Depth

The naive depth pipeline includes filtering and safety logic because raw YOLO boxes can jitter. It supports:

- no filter;
- exponential moving average;
- constant-velocity 1D Kalman filtering.

Filtering can be applied separately to distance, bounding-box center, and bounding-box width. The pipeline also handles missed detections through four track states:

- `tracked`: fresh accepted detection;
- `held`: short dropout, using the last estimate;
- `stale`: longer dropout, estimate exists but should be treated cautiously;
- `lost`: no usable estimate.

I also added optional gating checks that reject implausible measurements before they update the filtered state. These checks can reject low confidence detections, very small boxes, distances above a maximum threshold, boxes near the frame border, or sudden jumps in distance/position. Gating can be toggled live with the `g` key.

The session depth reviewer replays recorded test sessions and logs per-frame metrics to CSV. The logged metrics include inference time, FPS, track state, detection source, raw and filtered distance, relative position, yaw error, gating status, and tracker failsafe status. This makes it possible to analyze the depth behavior offline rather than only looking at the live window.

TODO evidence to add here later:

- Figure: depth session review UI with telemetry side panel.
- Plot/table: raw distance vs filtered distance over time from one review CSV.
- Table: examples of rejected gating reasons and why they are useful for safety.

## Live Depth Review

The top-level live depth reviewer can run one or more depth methods on the same camera stream. It supports:

- `naive`
- `unidepth`
- `midas`
- combinations such as `naive,unidepth`

Each pipeline returns a visualization frame and a metrics dictionary. The live reviewer combines the frames and adds a telemetry side panel. For the naive method, the panel shows tracking state, estimate source, detection source, inference time, FPS, gating state, relative X/Y/Z, and yaw error. This is useful for debugging before connecting the output to flight control.

TODO evidence to add here later:

- Figure: live depth review window.
- If comparing methods live, add side-by-side screenshot of naive vs UniDepth or MiDaS.

## Integrated Flight and Vision Runtime

The repository includes a `flight_vision` module that runs drone control and live YOLO visualization together. It has a vision-only mode for testing the camera and model without connecting to the Crazyflie radio.

The module is structured with separate responsibilities:

- `FrameSource`: camera abstraction;
- `YOLODetector`: model loading and prediction;
- `OverlayRenderer`: runtime text overlay;
- `OpenCVPresenter`: display and keyboard handling;
- `VisionRuntime`: camera loop;
- `ConcurrentFlightVisionApp`: starts vision and drone control together.

When drone control is enabled, the vision runtime runs in a separate thread while `DroneControlApp` runs the selected mission. If the vision thread fails before or during flight, the error is surfaced and the app shuts down. This module is mainly an integration layer for monitoring live detections while running flight-control experiments.

TODO evidence to add here later:

- Diagram: two-thread runtime, with drone control in one loop and YOLO vision in another.
- Figure: `flight_vision --vision-only` screenshot.
- Add measured behavior if tested together with the drone: whether vision affected control timing.

## Drone Control and Safety

The drone-control package implements both teleoperation and autonomous missions. The teleoperation controller reads a joystick mapping from JSON, so button and axis assignments can be configured without changing the control code. The current mapping is for a Sony Interactive Entertainment wireless controller.

Safety features include:

- flow-deck detection before flight;
- battery voltage logging;
- takeoff blocked below a configured voltage;
- automatic landing if voltage drops below a flight threshold;
- emergency landing button;
- controlled takeoff/land toggle;
- manual joystick takeover during autonomous missions.

Autonomous missions are run through `TakeoverRunner`. This runner starts teleoperation, gives a mission a takeover-aware context, and transfers control to manual teleoperation if joystick activity is detected. The implemented missions include square flight, height sequence, origin-to-point return, and roll/pitch/yaw response tests.

TODO evidence to add here later:

- Figure: joystick/controller and button mapping.
- Table: safety mechanisms: emergency land, low battery land, flow-deck check, joystick takeover.
- Add short results from autonomous mission tests: square, height, origin-to-point, roll/pitch/yaw.

## Drone Follower Demo

The drone follower demo combines the depth-estimation pipeline with the autonomous-control runner. The goal is to follow a detected drone while maintaining a target distance, keeping the target horizontally centered with yaw control, and optionally keeping it vertically centered with vertical velocity control.

The demo starts the computer-vision pipeline before takeoff. It warms up the camera/model for a configurable number of frames, shows the live telemetry preview, and then engages flight control. The control law uses naive depth metrics:

- `z_rel_m` for forward/backward velocity;
- `yaw_error_deg` for yaw-rate correction;
- `y_rel_m` for vertical correction.

The demo can be configured to move only when the estimate source is a fresh measurement. This is safer than commanding motion from stale or held estimates. If the target is lost, the drone commands zero motion or waits instead of continuing blindly. Joystick activity still triggers immediate takeover through the same `TakeoverRunner` mechanism used by the other autonomous missions.

TODO evidence to add here later:

- Figure: drone follower preview window.
- Diagram: control inputs from depth pipeline to `vx`, `vz`, and `yawrate`.
- Add real experiment outcome once tested: target distance, stable/unstable cases, takeover behavior, and failure modes.

## Software Engineering and Reproducibility

The repository is organized around separate modules for data, models, inference, depth estimation, flight vision, and drone control. Shell launchers under `scripts/` run each feature from the repository root using `uv run python` when available. This avoids import-path errors and makes the workflow easier to reproduce.

The project uses Python 3.13 and `uv` for dependency management. The `pyproject.toml` explicitly excludes `opencv-python` because the project needs `opencv-contrib-python` for CSRT and other OpenCV trackers. This avoids conflicts where both OpenCV wheels provide the same `cv2` module.

There is also a small test suite covering dataset preparation, model training/evaluation script integration with a fake YOLO class, tracker-failsafe behavior, and shell launcher behavior. At the time of writing, some tests are stale because they refer to an old `depth_estimation/utils.py` file and old `depth_image.sh`/`depth_video.sh` launchers. The codebase has since been reorganized into separate UniDepth and MiDaS modules, so these tests should be updated.

TODO evidence to add here later:

- Figure: repository module diagram.
- Add final test status after updating stale tests.
- Add command examples for the most important launchers in an appendix.

## Current Limitations

There are several limitations that should be mentioned in the report:

- The analog FPV camera is noisy and low resolution, which limits both detection and depth accuracy at longer distances.
- Dense monocular depth methods such as UniDepth and MiDaS struggle when the drone occupies only a small region of the image.
- The geometric bbox-depth method depends on accurate camera calibration, accurate real drone width, and reliable bounding-box width.
- YOLO validation metrics are very high, but held-out test metrics are lower, showing that real-world generalization is still challenging.
- Live flight integration exists, but the fully autonomous drone-follower behavior still needs careful controlled testing.
- Obstacle avoidance has not yet been fully implemented; the current system mainly covers detection, depth estimation, tracking, and initial follow-control integration.

## Future Work

The next steps should focus on closing the loop between perception and safe real-world autonomy:

1. Finish comparing the original and improved cameras using the stress-test protocol.
2. Update stale tests so the software validation suite matches the current module structure.
3. Select the best YOLO checkpoint based on test-set performance, live reliability, and inference speed rather than validation metrics alone.
4. Tune the naive depth gating and filtering parameters using session-review CSV logs.
5. Validate the follower demo in controlled experiments with short distances and manual takeover ready.
6. Add obstacle detection/avoidance, for example with balloons or other simple obstacles.
7. Consider retraining with more difficult negative examples, motion blur, partial occlusion, and longer-distance samples.
8. If a better camera is adopted, recalibrate intrinsics and rerun the detector/depth evaluation.

## Updated Conclusion

The project has progressed from basic Crazyflie control and camera setup to a complete computer-vision and control-oriented perception pipeline. The repository now supports raw FPV data collection, semi-automatic labeling, YOLO dataset generation, model training, checkpointing, evaluation, live inference, depth estimation, camera calibration, and initial integration with Crazyflie flight control.

The most mature parts of the system are the data pipeline, YOLO detector training/evaluation workflow, and the naive bounding-box depth estimator. The learned depth models were useful baselines, but the geometric method is currently more practical for small drone targets because it produces direct metric estimates from YOLO bounding boxes and calibrated camera intrinsics.

The remaining work is mainly integration and validation: improving live robustness, tuning depth filtering/gating, testing the drone follower safely, and extending the perception stack toward obstacle avoidance. Overall, the project now has the core infrastructure needed for real-time drone detection and depth-aware pursuit experiments outside the motion-capture room.
