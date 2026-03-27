# Drone Control

Teleoperation + autonomous mission control for Crazyflie, with joystick takeover and battery safety hooks.

## Entry Points

- `start_drone.py` is the main runtime entrypoint.
- `main.py` is a compatibility wrapper that forwards to `start_drone.py`.
- Recommended launcher from repo root:
  - `./scripts/drone_control.sh`
- `start_drone.py` exposes `DroneControlApp` so you can embed flight control in a larger pipeline.

## Folder Layout

- `constants.py`:
  - app-level defaults (mission choice, teleop defaults, mission presets)
- `autonomous/`:
  - `constants.py` for runner defaults
  - mission classes (`missions/`)
  - `missions/constants.py` for mission defaults
  - takeover orchestration (`takeover_runner.py`)
- `joystick/`:
  - `constants.py` for teleop defaults
  - teleoperation loop (`teleoperation.py`)
  - joystick mapping tools (`joystick_map.py`, `joystick_check.py`, `joystick_map.json`)
- `safety/`:
  - `constants.py` for battery thresholds
  - battery guard (`battery_guard.py`)
- `tutorials/`:
  - standalone cflib examples/experiments

## Quick Run

1. Choose mission in `start_drone.py` (`MISSION = ...`).
2. Ensure joystick mapping file is correct (`joystick/joystick_map.json`).
3. Run:
   - `./scripts/drone_control.sh`

## Dependencies and UV

This folder currently still contains `requirements.txt`. You can remove it after you confirm all required deps are in your project `pyproject.toml` and lockfile.

Recommended checks:
1. `uv --version`
2. `uv sync`
3. `uv run python -c "import cflib, pygame, numpy, scipy, cv2, usb"`

If all pass and `uv run` works for drone control scripts, `drone_control/requirements.txt` is no longer needed.
