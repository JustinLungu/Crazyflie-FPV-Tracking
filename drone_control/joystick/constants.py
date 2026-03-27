from pathlib import Path

from cflib.utils import uri_helper

URI = uri_helper.uri_from_env(default="radio://0/80/2M/E7E7E7E7E7")
MAPPING_FILE = str(Path(__file__).resolve().with_name("joystick_map.json"))

# Height safety.
TELEOP_Z_MIN = 0.10
TELEOP_Z_MAX = 1.50
TELEOP_DEFAULT_TARGET_Z = 0.50

# Target height update rate (m/s at full stick).
TELEOP_TARGET_RATE_MPS = 0.30

# Z controller.
TELEOP_KP_Z = 1.2
TELEOP_MAX_VZ = 0.40
TELEOP_TOL_HOLD = 0.02

# XY + yaw limits.
TELEOP_MAX_VXY = 0.40
TELEOP_MAX_YAWRATE = 90.0

# Loop timing.
TELEOP_DT = 0.05

# Input activity threshold.
TELEOP_ACTIVITY_THRESHOLD = 0.12

# Sign conventions.
TELEOP_INVERT_ROLL = True
TELEOP_INVERT_YAW = True
