# Camera-Receiver Link Stress Test (Minimal)

This is a **link-only** protocol.

You are not testing target detection/tracking/motion here.
You are only testing how stream quality changes with **camera-drone to receiver-node distance**.

## Exactly What Changes Per Run

- Keep receiver node + laptop fixed in one place.
- Keep camera settings fixed.
- Keep environment fixed as much as possible.
- Only change camera-drone distance to receiver node:
  - `0.5 m`
  - `1.0 m`
  - `2.0 m`
  - `3.0 m`

## Metrics You Get

- actual FPS
- FPS ratio (actual/nominal)
- estimated drop percentage
- freeze count
- timing stability (`dt_p95_ms`)
- blur proxy (`blur_frame_pct`)

No manual annotation is required.

## Run Definitions

Defined in `constants.py`:

1. `run_01_link_0p5m`
2. `run_02_link_1p0m`
3. `run_03_link_2p0m`
4. `run_04_link_3p0m`

## Commands

Build checklist CSV:

```bash
uv run python setting_up_camera/camera_stress_tests/build_test_matrix.py
```

Run all 4 link-distance runs:

```bash
uv run python setting_up_camera/camera_stress_tests/run_all_core_tests.py
```

Analyze latest run:

```bash
uv run python setting_up_camera/camera_stress_tests/analyze_camera_test.py
```

Analyze all run folders:

```bash
uv run python setting_up_camera/camera_stress_tests/analyze_all_runs.py
```

Summarize campaign:

```bash
uv run python setting_up_camera/camera_stress_tests/summarize_campaign.py
```

Menu launcher:

```bash
./scripts/camera_stress_tests.sh
```

## Preflight Overlay

The preflight window shows what to set up for each run.

Controls:
- `c` / `space` / `enter`: continue
- `s`: skip run
- `q` / `esc`: quit

Preflight waits indefinitely for camera frames by default.

## Outputs

Per run folder (`runs/<run_id_timestamp>/`):
- `metadata.json`
- `stream_log.csv`
- `raw_video.avi` (if enabled)
- `analysis_summary.json` (after analyze)
- `analysis_report.md` (after analyze)

Campaign outputs:
- `campaign_summary.csv`
- `campaign_summary.md`

## What You Should Conclude

From 4 runs only, you should be able to state:

- Does effective FPS stay stable from `0.5 m` to `3.0 m`?
- Do drops/freezes increase with distance?
- Does stream lag/choppiness become unacceptable at longer distance?

That gives direct evidence about camera+receiver link reliability.
