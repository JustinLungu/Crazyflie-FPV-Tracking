import time
import json
import pygame
import sys
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().with_name("joystick_map.json")

TIMEOUT = 12.0
AXIS_DEADBAND = 0.35
SETTLE_TIME = 0.25


def snapshot(js):
    pygame.event.pump()
    axes = [js.get_axis(i) for i in range(js.get_numaxes())]
    buttons = [js.get_button(i) for i in range(js.get_numbuttons())]
    return axes, buttons


def detect_button_press(js, timeout=TIMEOUT):
    _, base_buttons = snapshot(js)
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, buttons = snapshot(js)
        for i in range(len(buttons)):
            if base_buttons[i] == 0 and buttons[i] == 1:
                return {"type": "button", "index": i}
        time.sleep(0.01)
    return None


def detect_axis_by_single_direction(js, prompt, timeout=TIMEOUT):
    """
    Detects which axis moved the most from neutral after the user moves a stick once.

    Returns:
      {
        "type": "axis",
        "index": <int>,
        "positive_when_moved": <bool>,   # True if the moved direction yielded positive delta
        "deadband": 0.08,
        "scale": 1.0
      }
    """
    print("\nCenter all sticks and press ENTER")
    input()

    base_axes, _ = snapshot(js)

    print(prompt)
    t0 = time.time()
    max_dev = [0.0 for _ in base_axes]
    moved_sample = base_axes[:]

    while time.time() - t0 < timeout:
        axes, _ = snapshot(js)

        for i in range(len(axes)):
            dev = abs(axes[i] - base_axes[i])
            if dev > max_dev[i]:
                max_dev[i] = dev
                moved_sample[i] = axes[i]

        if max(max_dev) > AXIS_DEADBAND:
            break

        time.sleep(0.01)

    axis_index = max(range(len(max_dev)), key=lambda i: max_dev[i])
    delta = moved_sample[axis_index] - base_axes[axis_index]
    positive_when_moved = delta > 0.0

    print(f"Detected axis index: {axis_index}")
    print(f"Delta: {delta:+.3f}")
    print(f"Moved direction is {'positive' if positive_when_moved else 'negative'} on this axis")

    return {
        "type": "axis",
        "index": axis_index,
        "positive_when_moved": positive_when_moved,
        "deadband": 0.08,
        "scale": 1.0,
    }


def main():
    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick detected")
        sys.exit(1)

    js = pygame.joystick.Joystick(0)
    js.init()

    mapping = {
        "device": js.get_name(),
        "actions": {}
    }

    print(f"\nJoystick detected: {js.get_name()}")

    print("\nMapping ROLL (left stick RIGHT)")
    mapping["actions"]["ROLL"] = detect_axis_by_single_direction(
        js,
        "Move LEFT joystick fully RIGHT and hold"
    )
    time.sleep(SETTLE_TIME)

    print("\nMapping PITCH (left stick UP)")
    mapping["actions"]["PITCH"] = detect_axis_by_single_direction(
        js,
        "Move LEFT joystick fully UP and hold"
    )
    time.sleep(SETTLE_TIME)

    print("\nMapping YAW (right stick RIGHT)")
    mapping["actions"]["YAW"] = detect_axis_by_single_direction(
        js,
        "Move RIGHT joystick fully RIGHT and hold"
    )
    time.sleep(SETTLE_TIME)

    print("\nMapping HEIGHT (right stick UP)")
    mapping["actions"]["HEIGHT"] = detect_axis_by_single_direction(
        js,
        "Move RIGHT joystick fully UP and hold"
    )
    time.sleep(SETTLE_TIME)

    print("\nMapping TAKEOFF_LAND button (press once)")
    btn = detect_button_press(js)
    if btn is None:
        print("Failed to detect TAKEOFF_LAND button within timeout")
        sys.exit(1)
    mapping["actions"]["TAKEOFF_LAND"] = btn
    print(f"Mapped TAKEOFF_LAND -> {btn}")
    time.sleep(SETTLE_TIME)

    print("\nMapping EMERGENCY_LAND button (press once)")
    btn = detect_button_press(js)
    if btn is None:
        print("Failed to detect EMERGENCY_LAND button within timeout")
        sys.exit(1)
    mapping["actions"]["EMERGENCY_LAND"] = btn
    print(f"Mapped EMERGENCY_LAND -> {btn}")

    with open(CONFIG_FILE, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nSaved mapping to {CONFIG_FILE}")
    print(json.dumps(mapping, indent=2))


if __name__ == "__main__":
    main()
