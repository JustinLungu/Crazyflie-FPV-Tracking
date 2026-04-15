from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demos.drone_follower.constants import (
    DEMO_DEPTH_METHOD,
    DEMO_FOLLOW_CONTROL_DT,
    DEMO_FOLLOW_TARGET_DISTANCE_M,
    DEMO_FOLLOW_TAKEOFF_HEIGHT_M,
    DEMO_LAND_AFTER_MISSION_IF_NO_TAKEOVER,
    DEMO_TAKEOVER_ON_ANY_INPUT,
    DEMO_TELEOP_DEFAULT_TARGET_Z,
    DEMO_TELEOP_INVERT_ROLL,
    DEMO_TELEOP_INVERT_YAW,
)
from demos.drone_follower.mission import DroneFollowerMission
from depth_estimation.live_depth_review import PIPELINE_SPECS, build_pipeline
from depth_estimation.pipeline_base import LiveDepthPipeline
from drone_control.autonomous.takeover_runner import TakeoverRunner
from drone_control.joystick.teleoperation import TeleoperationController


PipelineFactory = Callable[[], LiveDepthPipeline]


def _build_pipeline_factory(depth_method: str) -> PipelineFactory:
    method = str(depth_method).strip().lower()
    if method not in PIPELINE_SPECS:
        supported = ", ".join(sorted(PIPELINE_SPECS))
        raise ValueError(f"Unsupported DEMO_DEPTH_METHOD='{depth_method}'. Supported: {supported}")

    def _factory() -> LiveDepthPipeline:
        return build_pipeline(method)

    return _factory


class DroneFollowerDemoApp:
    def __init__(
        self,
        depth_method: str = DEMO_DEPTH_METHOD,
    ) -> None:
        self.depth_method = str(depth_method).strip().lower()

    def _build_teleop(self) -> TeleoperationController:
        return TeleoperationController(
            tuning={
                "default_target_z": DEMO_TELEOP_DEFAULT_TARGET_Z,
                "invert_roll": DEMO_TELEOP_INVERT_ROLL,
                "invert_yaw": DEMO_TELEOP_INVERT_YAW,
            }
        )

    def _build_runner(self, teleop: TeleoperationController) -> TakeoverRunner:
        return TakeoverRunner(
            teleop=teleop,
            dt=DEMO_FOLLOW_CONTROL_DT,
            takeover_on_any_input=DEMO_TAKEOVER_ON_ANY_INPUT,
            land_after_mission_if_no_takeover=DEMO_LAND_AFTER_MISSION_IF_NO_TAKEOVER,
        )

    def _build_mission(self) -> DroneFollowerMission:
        return DroneFollowerMission(
            target_distance_m=DEMO_FOLLOW_TARGET_DISTANCE_M,
            takeoff_height_m=DEMO_FOLLOW_TAKEOFF_HEIGHT_M,
            dt=DEMO_FOLLOW_CONTROL_DT,
            pipeline_factory=_build_pipeline_factory(self.depth_method),
        )

    def run(self) -> None:
        print("Starting drone_follower demo")
        print(f"- depth method: {self.depth_method}")
        print(f"- target distance: {DEMO_FOLLOW_TARGET_DISTANCE_M:.2f} m")
        print(f"- takeoff height: {DEMO_FOLLOW_TAKEOFF_HEIGHT_M:.2f} m")
        print("- safety takeover: joystick/button activity")
        if self.depth_method != "naive":
            print(
                "- note: follow control currently consumes naive-style metrics "
                "(track_state + z_rel_m + yaw_error_deg)."
            )

        teleop = self._build_teleop()
        runner = self._build_runner(teleop)
        mission = self._build_mission()
        runner.run(mission)


def main() -> None:
    DroneFollowerDemoApp().run()


if __name__ == "__main__":
    main()
