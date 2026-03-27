from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drone_control.autonomous.missions.height_sequence import HeightSequenceMission
from drone_control.autonomous.missions.origin_to_point import OriginToPointMission
from drone_control.autonomous.missions.roll_pitch_yaw import RollPitchYawMission
from drone_control.autonomous.missions.square import SquareMission
from drone_control.autonomous.takeover_runner import AutonomousMission, TakeoverRunner
from drone_control.constants import (
    HEIGHT_SEQUENCE_HEIGHTS,
    HEIGHT_SEQUENCE_HOLD_S,
    HEIGHT_SEQUENCE_MODE,
    HEIGHT_SEQUENCE_TAKEOFF_HEIGHT,
    LAND_AFTER_MISSION_IF_NO_TAKEOVER,
    MISSION,
    ORIGIN_TO_POINT_HEIGHT_M,
    ORIGIN_TO_POINT_TARGET_X,
    ORIGIN_TO_POINT_TARGET_Y,
    ROLL_PITCH_YAW_ANGLE_DEG,
    ROLL_PITCH_YAW_HEIGHT_M,
    ROLL_PITCH_YAW_LEG_S,
    ROLL_PITCH_YAW_PAUSE_S,
    ROLL_PITCH_YAW_REPEATS,
    ROLL_PITCH_YAW_TEST,
    ROLL_PITCH_YAW_VXY,
    ROLL_PITCH_YAW_YAWRATE,
    SQUARE_MISSION_FORWARD_SPEED,
    SQUARE_MISSION_HEIGHT_M,
    SQUARE_MISSION_SIDE_LENGTH,
    SQUARE_MISSION_YAW_RATE,
    TAKEOVER_ON_ANY_INPUT,
    TELEOP_DEFAULT_TARGET_Z,
    TELEOP_INVERT_ROLL,
    TELEOP_INVERT_YAW,
)
from drone_control.joystick.teleoperation import TeleoperationController

MissionFactory = Callable[[], AutonomousMission]


class DroneControlApp:
    """
    Class-based drone control entrypoint.

    This can be embedded in a bigger runtime
    (for example, depth estimation + detection + flight control together).
    """

    def __init__(
        self,
        mission: str = MISSION,
        default_target_z: float = TELEOP_DEFAULT_TARGET_Z,
        invert_roll: bool = TELEOP_INVERT_ROLL,
        invert_yaw: bool = TELEOP_INVERT_YAW,
        takeover_on_any_input: bool = TAKEOVER_ON_ANY_INPUT,
        land_after_mission_if_no_takeover: bool = LAND_AFTER_MISSION_IF_NO_TAKEOVER,
        teleop: TeleoperationController | None = None,
        runner: TakeoverRunner | None = None,
    ) -> None:
        self.mission = mission
        self.default_target_z = default_target_z
        self.invert_roll = invert_roll
        self.invert_yaw = invert_yaw
        self.takeover_on_any_input = takeover_on_any_input
        self.land_after_mission_if_no_takeover = land_after_mission_if_no_takeover

        self._teleop = teleop
        self._runner = runner
        self._mission_factories: dict[str, MissionFactory] = {}
        self._register_default_missions()

    def _register_default_missions(self) -> None:
        self.register_mission("square", self._build_square_mission)
        self.register_mission("height", self._build_height_mission)
        self.register_mission("origin_to_point", self._build_origin_to_point_mission)
        self.register_mission("roll_pitch_yaw", self._build_roll_pitch_yaw_mission)

    def register_mission(self, name: str, factory: MissionFactory) -> None:
        self._mission_factories[name.strip().lower()] = factory

    def _build_teleop(self) -> TeleoperationController:
        if self._teleop is None:
            self._teleop = TeleoperationController(
                tuning={
                    "default_target_z": self.default_target_z,
                    "invert_roll": self.invert_roll,
                    "invert_yaw": self.invert_yaw,
                }
            )
        return self._teleop

    def _build_runner(self) -> TakeoverRunner:
        if self._runner is None:
            teleop = self._build_teleop()
            self._runner = TakeoverRunner(
                teleop=teleop,
                dt=teleop.tuning.dt,
                takeover_on_any_input=self.takeover_on_any_input,
                land_after_mission_if_no_takeover=self.land_after_mission_if_no_takeover,
            )
        return self._runner

    def _build_square_mission(self) -> AutonomousMission:
        return SquareMission(
            height_m=SQUARE_MISSION_HEIGHT_M,
            forward_speed=SQUARE_MISSION_FORWARD_SPEED,
            yaw_rate=SQUARE_MISSION_YAW_RATE,
            side_length=SQUARE_MISSION_SIDE_LENGTH,
        )

    def _build_height_mission(self) -> AutonomousMission:
        return HeightSequenceMission(
            heights=HEIGHT_SEQUENCE_HEIGHTS,
            hold_s=HEIGHT_SEQUENCE_HOLD_S,
            mode=HEIGHT_SEQUENCE_MODE,
            takeoff_height=HEIGHT_SEQUENCE_TAKEOFF_HEIGHT,
        )

    def _build_origin_to_point_mission(self) -> AutonomousMission:
        return OriginToPointMission(
            target_x=ORIGIN_TO_POINT_TARGET_X,
            target_y=ORIGIN_TO_POINT_TARGET_Y,
            height_m=ORIGIN_TO_POINT_HEIGHT_M,
        )

    def _build_roll_pitch_yaw_mission(self) -> AutonomousMission:
        return RollPitchYawMission(
            test=ROLL_PITCH_YAW_TEST,
            repeats=ROLL_PITCH_YAW_REPEATS,
            height_m=ROLL_PITCH_YAW_HEIGHT_M,
            vxy=ROLL_PITCH_YAW_VXY,
            yawrate=ROLL_PITCH_YAW_YAWRATE,
            yaw_angle_deg=ROLL_PITCH_YAW_ANGLE_DEG,
            leg_s=ROLL_PITCH_YAW_LEG_S,
            pause_s=ROLL_PITCH_YAW_PAUSE_S,
        )

    def build_mission(self, mission_name: str | None = None) -> AutonomousMission:
        name = (mission_name or self.mission).strip().lower()
        if name not in self._mission_factories:
            supported = ", ".join(sorted(self._mission_factories))
            raise ValueError(f"Unknown mission: {name}. Supported missions: {supported}")
        return self._mission_factories[name]()

    def run(self, mission_name: str | None = None) -> None:
        mission = self.build_mission(mission_name)
        runner = self._build_runner()
        runner.run(mission)


def main() -> None:
    app = DroneControlApp(mission=MISSION)
    app.run()


if __name__ == "__main__":
    main()
