# Drone Follower Demo

This demo combines:
- `depth_estimation` live tracking (currently `naive` bbox depth), and
- `drone_control` autonomous runner with joystick takeover safety.

## Goal

Follow a detected drone while:
- keeping target distance near `DEMO_FOLLOW_TARGET_DISTANCE_M`
- keeping target horizontally centered using yaw correction
- keeping target vertically centered using vz correction (`y_rel_m`)
- allowing instant teleop takeover at any time

## Run

```bash
./scripts/drone_follower_demo.sh
```

## Configuration

Edit:
- `demos/drone_follower/constants.py`

Main knobs:
- `DEMO_DEPTH_METHOD`: depth backend (`"naive"` now, structure is ready for future methods)
- `DEMO_DRONE_URI`: radio URI for the Crazyflie link (defaults to `flight_vision` URI)
- `DEMO_PRECONTROL_CV_WARMUP_FRAMES`: run CV preview/model warm-up before flight control engages
- `DEMO_FOLLOW_TARGET_DISTANCE_M`: desired follow distance
- `DEMO_FOLLOW_ONLY_ON_MEASUREMENT`: only move on fresh detections; otherwise hover/wait
- `DEMO_FOLLOW_KP_FORWARD`, `DEMO_FOLLOW_MAX_VX`: forward/back distance control
- `DEMO_FOLLOW_KP_YAW`, `DEMO_FOLLOW_MAX_YAWRATE_DEG_S`: centering yaw control
- `DEMO_FOLLOW_ENABLE_VERTICAL`, `DEMO_FOLLOW_KP_VERTICAL`, `DEMO_FOLLOW_MAX_VZ`: vertical centering control
- `DEMO_TAKEOVER_ON_ANY_INPUT`: immediate safety takeover on joystick input
- `DEMO_LAND_AFTER_MISSION_IF_NO_TAKEOVER`: post-mission landing behavior

## Live Controls

- `q` or `ESC`: close preview / finish mission
- `g`: toggle gating (if selected depth method supports gating)
- any joystick activity: takeover from autonomy to teleop

## Startup Order

This demo starts CV first:
1. open camera + initialize depth pipeline
2. show live preview/overlay (same style as `depth_estimation/live_depth_review.py`) and warm up for `DEMO_PRECONTROL_CV_WARMUP_FRAMES`
3. then engage takeoff + follow control loop
