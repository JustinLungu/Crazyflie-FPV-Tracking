from drone_control.autonomous.missions.constants import (
    ROLL_PITCH_YAW_ANGLE_DEG,
    ROLL_PITCH_YAW_HEIGHT_M,
    ROLL_PITCH_YAW_LEG_S,
    ROLL_PITCH_YAW_PAUSE_S,
    ROLL_PITCH_YAW_REPEATS,
    ROLL_PITCH_YAW_TEST,
    ROLL_PITCH_YAW_VXY,
    ROLL_PITCH_YAW_YAWRATE,
)
from drone_control.autonomous.takeover_runner import AutonomousMission, TakeoverContext


class RollPitchYawMission(AutonomousMission):
    def __init__(
        self,
        test: str = ROLL_PITCH_YAW_TEST,
        repeats: int = ROLL_PITCH_YAW_REPEATS,
        height_m: float = ROLL_PITCH_YAW_HEIGHT_M,
        vxy: float = ROLL_PITCH_YAW_VXY,
        yawrate: float = ROLL_PITCH_YAW_YAWRATE,
        yaw_angle_deg: float = ROLL_PITCH_YAW_ANGLE_DEG,
        leg_s: float = ROLL_PITCH_YAW_LEG_S,
        pause_s: float = ROLL_PITCH_YAW_PAUSE_S,
    ):
        self.test = test
        self.repeats = repeats
        self.height_m = height_m
        self.vxy = vxy
        self.yawrate = yawrate
        self.yaw_angle_deg = yaw_angle_deg
        self.leg_s = leg_s
        self.pause_s = pause_s

    def _do_roll(self, ctx: TakeoverContext) -> bool:
        print("Roll test: left/right")
        for _ in range(self.repeats):
            if ctx.command(vx=0.0, vy=-self.vxy, vz=0.0, yawrate=0.0, duration_s=self.leg_s):
                return True
            if ctx.stop(self.pause_s):
                return True
            if ctx.command(vx=0.0, vy=self.vxy, vz=0.0, yawrate=0.0, duration_s=self.leg_s):
                return True
            if ctx.stop(self.pause_s):
                return True
        return False

    def _do_pitch(self, ctx: TakeoverContext) -> bool:
        print("Pitch test: forward/back")
        for _ in range(self.repeats):
            if ctx.command(vx=self.vxy, vy=0.0, vz=0.0, yawrate=0.0, duration_s=self.leg_s):
                return True
            if ctx.stop(self.pause_s):
                return True
            if ctx.command(vx=-self.vxy, vy=0.0, vz=0.0, yawrate=0.0, duration_s=self.leg_s):
                return True
            if ctx.stop(self.pause_s):
                return True
        return False

    def _do_yaw(self, ctx: TakeoverContext) -> bool:
        print("Yaw test: left/right")
        if self.yawrate <= 0.0:
            raise ValueError("yawrate must be > 0")
        if self.yaw_angle_deg <= 0.0:
            raise ValueError("yaw_angle_deg must be > 0")
        yaw_leg_s = self.yaw_angle_deg / self.yawrate
        for _ in range(self.repeats):
            if ctx.command(vx=0.0, vy=0.0, vz=0.0, yawrate=-self.yawrate, duration_s=yaw_leg_s):
                return True
            if ctx.stop(self.pause_s):
                return True
            if ctx.command(vx=0.0, vy=0.0, vz=0.0, yawrate=self.yawrate, duration_s=yaw_leg_s):
                return True
            if ctx.stop(self.pause_s):
                return True
        return False

    def run(self, ctx: TakeoverContext) -> bool:
        print("Autonomous mission: roll/pitch/yaw test")
        print("Touch any joystick or button to takeover")

        if ctx.ensure_takeoff(self.height_m):
            return False

        test = self.test.lower()
        if test == "roll":
            if self._do_roll(ctx):
                return False
        elif test == "pitch":
            if self._do_pitch(ctx):
                return False
        elif test == "yaw":
            if self._do_yaw(ctx):
                return False
        elif test == "all":
            if self._do_roll(ctx):
                return False
            if ctx.wait(self.pause_s):
                return False
            if self._do_pitch(ctx):
                return False
            if ctx.wait(self.pause_s):
                return False
            if self._do_yaw(ctx):
                return False
        else:
            raise ValueError(f"Unknown test: {self.test}")

        print("Roll/pitch/yaw test finished")
        return True
