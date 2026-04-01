# autonomy/missions/square.py
from drone_control.autonomous.missions.constants import (
    SQUARE_FORWARD_SPEED,
    SQUARE_HEIGHT_M,
    SQUARE_PAUSE_S,
    SQUARE_SIDE_LENGTH,
    SQUARE_YAW_RATE,
)
from drone_control.autonomous.takeover_runner import AutonomousMission, TakeoverContext


class SquareMission(AutonomousMission):
    def __init__(
        self,
        height_m: float = SQUARE_HEIGHT_M,
        forward_speed: float = SQUARE_FORWARD_SPEED,
        yaw_rate: float = SQUARE_YAW_RATE,
        side_length: float = SQUARE_SIDE_LENGTH,
        pause_s: float = SQUARE_PAUSE_S,
    ):
        self.height_m = height_m
        self.forward_speed = forward_speed
        self.yaw_rate = yaw_rate
        self.side_length = side_length
        self.pause_s = pause_s

    def run(self, ctx: TakeoverContext) -> bool:
        print("Autonomous mission: square")
        print("Touch any joystick or button to takeover")

        # takeoff
        if ctx.ensure_takeoff(self.height_m):
            return False

        leg_time = self.side_length / self.forward_speed
        turn_time = 90.0 / self.yaw_rate

        for i in range(4):
            print(f"Square leg {i + 1}/4")

            # forward
            if ctx.command(vx=self.forward_speed, vy=0.0, vz=0.0, yawrate=0.0, duration_s=leg_time):
                return False
            if ctx.stop(self.pause_s):
                return False

            # yaw left 90 degrees
            if ctx.command(vx=0.0, vy=0.0, vz=0.0, yawrate=self.yaw_rate, duration_s=turn_time):
                return False
            if ctx.stop(self.pause_s):
                return False

        print("Autonomous square finished")
        return True
