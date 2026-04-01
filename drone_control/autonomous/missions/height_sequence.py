from drone_control.autonomous.missions.constants import (
    HEIGHT_SEQUENCE_HEIGHTS,
    HEIGHT_SEQUENCE_HOLD_S,
    HEIGHT_SEQUENCE_MODE,
    HEIGHT_SEQUENCE_TAKEOFF_HEIGHT,
)
from drone_control.autonomous.takeover_runner import AutonomousMission, TakeoverContext


class HeightSequenceMission(AutonomousMission):
    def __init__(
        self,
        heights: list[float] | tuple[float, ...] = HEIGHT_SEQUENCE_HEIGHTS,
        hold_s: float = HEIGHT_SEQUENCE_HOLD_S,
        mode: str = HEIGHT_SEQUENCE_MODE,
        takeoff_height: float = HEIGHT_SEQUENCE_TAKEOFF_HEIGHT,
    ):
        self.heights = list(heights)
        self.hold_s = hold_s
        self.mode = mode
        self.takeoff_height = takeoff_height

    def run(self, ctx: TakeoverContext) -> bool:
        print("Autonomous mission: height sequence")
        print("Touch any joystick or button to takeover")

        # takeoff always uses takeoff_height
        if ctx.ensure_takeoff(self.takeoff_height):
            return False

        for h in self.heights:
            if self.mode == "relative":
                z_target = self.takeoff_height + h
            else:
                z_target = h

            # clamp to teleop tuning limits
            z_target = max(ctx.teleop.tuning.z_min, min(ctx.teleop.tuning.z_max, z_target))

            print(f"Target height: {z_target:.2f} m")
            if ctx.goto_z(z_target, timeout_s=6.0):
                return False

            if ctx.wait(self.hold_s):
                return False

        print("Height sequence finished")
        return True
