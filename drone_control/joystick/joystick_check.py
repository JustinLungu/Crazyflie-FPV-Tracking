import json
import time
import pygame
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().with_name("joystick_map.json")


def apply_deadband(x, db):
    return 0.0 if abs(x) < db else x


def read_axis_normalized(js, axis_cfg):
    raw = js.get_axis(int(axis_cfg["index"])) * float(axis_cfg.get("scale", 1.0))
    raw = apply_deadband(raw, float(axis_cfg.get("deadband", 0.08)))

    # Normalize so "moved direction" is positive
    if not bool(axis_cfg.get("positive_when_moved", True)):
        raw = -raw

    return raw


def main():
    with open(CONFIG_FILE, "r") as f:
        cfg = json.load(f)

    a = cfg["actions"]

    pygame.init()
    pygame.joystick.init()
    js = pygame.joystick.Joystick(0)
    js.init()

    print(
        "Move sticks to verify signs.\n"
        "Press TAKEOFF_LAND to see TOG=1.\n"
        "Press EMERGENCY_LAND to see EMERG=1.\n"
        "Ctrl+C to quit."
    )

    while True:
        pygame.event.pump()

        roll = read_axis_normalized(js, a["ROLL"])
        pitch = read_axis_normalized(js, a["PITCH"])
        yaw = read_axis_normalized(js, a["YAW"])
        height = read_axis_normalized(js, a["HEIGHT"])

        toggle = js.get_button(int(a["TAKEOFF_LAND"]["index"])) if "TAKEOFF_LAND" in a else 0
        emergency = js.get_button(int(a["EMERGENCY_LAND"]["index"]))

        print(
            f"\rROLL {roll:+.2f}  PITCH {pitch:+.2f}  YAW {yaw:+.2f}  HEIGHT {height:+.2f}  "
            f"TOG {toggle}  EMERG {emergency}",
            end=""
        )
        time.sleep(0.02)


if __name__ == "__main__":
    main()
