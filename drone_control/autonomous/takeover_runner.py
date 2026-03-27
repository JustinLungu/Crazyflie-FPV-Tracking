# autonomy/takeover_runner.py
import time

from drone_control.autonomous.constants import (
    LAND_AFTER_MISSION_IF_NO_TAKEOVER,
    TAKEOVER_DT,
    TAKEOVER_ON_ANY_INPUT,
)
from drone_control.joystick.teleoperation import TeleoperationController


class TakeoverContext:
    """
    Context given to missions.
    All methods return True if takeover happened (meaning autonomy must stop).
    """

    def __init__(
        self,
        teleop: TeleoperationController,
        dt: float,
        takeover_on_any_input: bool,
    ):
        self.teleop = teleop
        self.dt = dt
        self.takeover_on_any_input = takeover_on_any_input

    def _takeover(self) -> bool:
        if not self.takeover_on_any_input:
            return False
        return self.teleop.joystick_activity()

    def _safety_abort(self) -> bool:
        guard = getattr(self.teleop, "battery_guard", None)
        if guard is None:
            return False
        if guard.should_land():
            if self.teleop.flying:
                print(f"Low battery detected. Landing now. {guard.status_text()}")
                self.teleop.land()
            return True
        return False

    def ensure_takeoff(self, height_m: float) -> bool:
        if self._takeover():
            return True
        if self._safety_abort():
            return True

        self.teleop.target_z = height_m
        if not self.teleop.flying:
            self.teleop.takeoff()
            if not self.teleop.flying:
                return True

        # quick settle, still takeover aware
        return self.wait(1.0)

    def wait(self, duration_s: float) -> bool:
        t0 = time.time()
        while time.time() - t0 < duration_s:
            if self._takeover():
                return True
            if self._safety_abort():
                return True
            time.sleep(self.dt)
        return False

    def command(self, vx: float, vy: float, vz: float, yawrate: float, duration_s: float) -> bool:
        """
        Sends constant body-frame velocities for duration_s.
        vx, vy, vz in m/s, yawrate in deg/s.
        """
        t0 = time.time()
        while time.time() - t0 < duration_s:
            if self._takeover():
                return True
            if self._safety_abort():
                return True

            # if we are not flying, treat as abort of autonomy
            if (not self.teleop.flying) or (self.teleop.mc is None):
                return True

            self.teleop.mc.start_linear_motion(vx, vy, vz, yawrate)
            time.sleep(self.dt)

        return False

    def stop(self, duration_s: float = 0.2) -> bool:
        return self.command(0.0, 0.0, 0.0, 0.0, duration_s)

    def handover_to_teleop_forever(self):
        """
        After takeover, we permanently stay in teleop.
        """
        print("Takeover detected. Switching to teleoperation. Autonomous disabled.")
        while self.teleop._running:
            self.teleop.step()


    def goto_z(self, z_target: float, timeout_s: float = 5.0, tol: float = 0.03) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self._takeover():
                return True
            if self._safety_abort():
                return True

            if (not self.teleop.flying) or (self.teleop.mc is None):
                return True

            z_now = self.teleop.state_estimate["z"]
            err = z_target - z_now

            if abs(err) < tol:
                return self.stop(0.2)

            # simple P controller on vz
            vz = 1.2 * err
            vz = max(-0.4, min(0.4, vz))

            self.teleop.mc.start_linear_motion(0.0, 0.0, vz, 0.0)
            time.sleep(self.dt)

        # timeout, stop but keep running
        return self.stop(0.2)



class AutonomousMission:
    """
    Base class for missions.
    Implement run(ctx) and return True if completed without takeover.
    If takeover happens, return False.
    """
    def run(self, ctx: TakeoverContext) -> bool:
        raise NotImplementedError


class TakeoverRunner:
    """
    Orchestrates:
      teleop.start()
      mission.run(ctx) until takeover or finish
      if takeover: teleop forever
      else: land and stop
    """

    def __init__(
        self,
        teleop: TeleoperationController,
        dt: float | None = TAKEOVER_DT,
        takeover_on_any_input: bool = TAKEOVER_ON_ANY_INPUT,
        land_after_mission_if_no_takeover: bool = LAND_AFTER_MISSION_IF_NO_TAKEOVER,
    ):
        self.teleop = teleop
        self.dt = TAKEOVER_DT if dt is None else dt
        self.takeover_on_any_input = takeover_on_any_input
        self.land_after_mission_if_no_takeover = land_after_mission_if_no_takeover

    def run(self, mission: AutonomousMission):
        self.teleop.start()
        ctx = TakeoverContext(
            teleop=self.teleop,
            dt=self.dt,
            takeover_on_any_input=self.takeover_on_any_input,
        )

        try:
            ok = mission.run(ctx)

            if not ok:
                ctx.handover_to_teleop_forever()
                return

            if self.land_after_mission_if_no_takeover:
                print("Mission finished. Landing.")
                self.teleop.land()
            else:
                print("Mission finished. Staying in hover. Press TAKEOFF_LAND to land or fly teleop.")
                while self.teleop._running:
                    self.teleop.step()

        except (KeyboardInterrupt, SystemExit):
            print("Interrupted. Landing if flying.")
        finally:
            self.teleop.stop()
