import math

from drone_control.autonomous.missions.constants import (
    ORIGIN_TO_POINT_DT,
    ORIGIN_TO_POINT_HEIGHT_M,
    ORIGIN_TO_POINT_KP,
    ORIGIN_TO_POINT_MAX_VEL,
    ORIGIN_TO_POINT_TARGET_X,
    ORIGIN_TO_POINT_TARGET_Y,
    ORIGIN_TO_POINT_TOL,
)
from drone_control.autonomous.takeover_runner import AutonomousMission, TakeoverContext


class OriginToPointMission(AutonomousMission):
    def __init__(
        self,
        target_x: float = ORIGIN_TO_POINT_TARGET_X,
        target_y: float = ORIGIN_TO_POINT_TARGET_Y,
        height_m: float = ORIGIN_TO_POINT_HEIGHT_M,
        kp: float = ORIGIN_TO_POINT_KP,
        max_vel: float = ORIGIN_TO_POINT_MAX_VEL,
        tol: float = ORIGIN_TO_POINT_TOL,
        dt: float = ORIGIN_TO_POINT_DT,
    ):
        self.target_x = target_x
        self.target_y = target_y
        self.height_m = height_m
        self.kp = kp
        self.max_vel = max_vel
        self.tol = tol
        self.dt = dt

    def _goto_xy(self, ctx: TakeoverContext, target_x: float, target_y: float) -> bool:
        while True:
            if ctx.teleop.joystick_activity():
                return True

            x = ctx.teleop.state_estimate["x"]
            y = ctx.teleop.state_estimate["y"]

            dx = target_x - x
            dy = target_y - y
            dist = math.hypot(dx, dy)

            if dist < self.tol:
                ctx.stop(0.2)
                return False

            vx = self.kp * dx
            vy = self.kp * dy

            speed = math.hypot(vx, vy)
            if speed > self.max_vel:
                scale = self.max_vel / speed
                vx *= scale
                vy *= scale

            if ctx.command(vx=vx, vy=vy, vz=0.0, yawrate=0.0, duration_s=self.dt):
                return True

    def run(self, ctx: TakeoverContext) -> bool:
        print("Autonomous mission: origin → point → origin")
        print("Touch any joystick or button to takeover")

        if ctx.ensure_takeoff(self.height_m):
            return False

        if self._goto_xy(ctx, self.target_x, self.target_y):
            return False

        if ctx.wait(0.5):
            return False

        if self._goto_xy(ctx, 0.0, 0.0):
            return False

        print("Origin-to-point mission finished")
        return True
