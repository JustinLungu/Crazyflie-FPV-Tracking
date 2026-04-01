import json
import logging
import time
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Optional, Dict, Any

import pygame

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from drone_control.joystick.constants import (
    MAPPING_FILE,
    TELEOP_ACTIVITY_THRESHOLD,
    TELEOP_DEFAULT_TARGET_Z,
    TELEOP_DT,
    TELEOP_INVERT_ROLL,
    TELEOP_INVERT_YAW,
    TELEOP_KP_Z,
    TELEOP_MAX_VXY,
    TELEOP_MAX_VZ,
    TELEOP_MAX_YAWRATE,
    TELEOP_TARGET_RATE_MPS,
    TELEOP_TOL_HOLD,
    TELEOP_Z_MAX,
    TELEOP_Z_MIN,
    URI,
)
from drone_control.safety.battery_guard import BatteryGuard

logging.basicConfig(level=logging.CRITICAL)

def build_teleop_tuning(tuning: Any | None = None) -> SimpleNamespace:
    cfg = {
        "z_min": TELEOP_Z_MIN,
        "z_max": TELEOP_Z_MAX,
        "default_target_z": TELEOP_DEFAULT_TARGET_Z,
        "target_rate_mps": TELEOP_TARGET_RATE_MPS,
        "kp_z": TELEOP_KP_Z,
        "max_vz": TELEOP_MAX_VZ,
        "tol_hold": TELEOP_TOL_HOLD,
        "max_vxy": TELEOP_MAX_VXY,
        "max_yawrate": TELEOP_MAX_YAWRATE,
        "dt": TELEOP_DT,
        "activity_threshold": TELEOP_ACTIVITY_THRESHOLD,
        "invert_roll": TELEOP_INVERT_ROLL,
        "invert_yaw": TELEOP_INVERT_YAW,
    }

    if tuning is None:
        return SimpleNamespace(**cfg)

    if isinstance(tuning, dict):
        source = tuning
    else:
        source = {k: getattr(tuning, k) for k in cfg if hasattr(tuning, k)}

    for key, value in source.items():
        if key in cfg:
            cfg[key] = value
    return SimpleNamespace(**cfg)


class TeleoperationController:
    """
    Teleop controller that can run standalone now, and later be embedded into an autonomous
    controller via:
      - joystick_activity(): detect takeover intent
      - step(): run one control cycle
    """

    def __init__(
        self,
        uri: str = URI,
        mapping_file: str | None = None,
        tuning: Any | None = None,
        battery_guard: Optional[BatteryGuard] = None,
    ):
        self.uri = uri
        self.mapping_file = mapping_file or MAPPING_FILE
        self.tuning = build_teleop_tuning(tuning)
        self.battery_guard = battery_guard or BatteryGuard()

        self.deck_attached_event = Event()
        self.state_estimate = {"x": 0.0, "y": 0.0, "z": 0.0}

        self.mapping: Dict[str, Any] = {}
        self.actions: Dict[str, Any] = {}

        self.js = None
        self.scf = None
        self.logconf = None

        self.flying = False
        self.mc: Optional[MotionCommander] = None
        self.target_z = self.tuning.default_target_z

        # rising edge detection for buttons
        self._prev_emergency = 0
        self._prev_toggle = 0

        self._running = False

    # -----------------------
    # Setup utilities
    # -----------------------
    def _param_deck_flow(self, _, value_str):
        value = int(value_str)
        if value:
            self.deck_attached_event.set()
            print("Flow deck is attached!")
        else:
            print("Flow deck is NOT attached!")

    def _log_pos_callback(self, timestamp, data, logconf):
        self.state_estimate["z"] = data["stateEstimate.z"]
        self.state_estimate["y"] = data["stateEstimate.y"]
        self.state_estimate["x"] = data["stateEstimate.x"]

    @staticmethod
    def _clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    @staticmethod
    def _apply_deadband(x, deadband):
        #A deadband is a small zone around zero where input is intentionally ignored.
        return 0.0 if abs(x) < deadband else x

    def _load_mapping(self):
        mapping_path = Path(self.mapping_file)
        if not mapping_path.is_absolute():
            mapping_path = Path(__file__).resolve().parent / mapping_path

        with open(mapping_path, "r") as f:
            cfg = json.load(f)

        actions = cfg["actions"]

        for key in ["ROLL", "PITCH", "YAW", "HEIGHT", "TAKEOFF_LAND", "EMERGENCY_LAND"]:
            if key not in actions:
                raise KeyError(f"Missing '{key}' in mapping file: {self.mapping_file}")

        self.mapping = {"device": cfg.get("device", "Unknown"), "actions": actions}
        self.actions = actions

    def _init_joystick(self):
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected")

        # Use the first joystick device detected
        js = pygame.joystick.Joystick(0)
        js.init()

        name = js.get_name()
        print(f"Joystick: {name}")
        expected = self.mapping.get("device")
        # This check catches: "you mapped controller X but plugged in controller Y"
        if expected and expected not in name:
            print("Warning: controller name does not match mapping device name")

        self.js = js

    def _read_axis_normalized(self, axis_cfg):
        #refresh controller readings
        pygame.event.pump()

        # The scale is in case we want to scale, but the 1.0 means no scaling.
        raw = self.js.get_axis(int(axis_cfg["index"])) * float(axis_cfg.get("scale", 1.0))
        # Any value between -0.08 and 0.08 becomes 0.0
        raw = self._apply_deadband(raw, float(axis_cfg.get("deadband", 0.08)))

        # normalize so that the direction used during mapping becomes positive
        if not bool(axis_cfg.get("positive_when_moved", True)):
            raw = -raw

        return raw

    def _read_button(self, button_cfg):
        pygame.event.pump()
        # For buttons we don't need scaling or deadband
        return self.js.get_button(int(button_cfg["index"]))

    # -----------------------
    # Public API for takeover
    # -----------------------
    def joystick_activity(self) -> bool:
        """
        Returns True if the user is *actively touching* any axis or pressing any relevant button.
        Intended for autonomous->teleop takeover detection.

        This intentionally ignores tiny noise via activity_threshold.
        """
        a = self.actions
        thr = self.tuning.activity_threshold

        roll = abs(self._read_axis_normalized(a["ROLL"]))
        pitch = abs(self._read_axis_normalized(a["PITCH"]))
        yaw = abs(self._read_axis_normalized(a["YAW"]))
        height = abs(self._read_axis_normalized(a["HEIGHT"]))

        if roll > thr or pitch > thr or yaw > thr or height > thr:
            return True

        # any button press counts as activity
        if self._read_button(a["TAKEOFF_LAND"]) == 1:
            return True
        if self._read_button(a["EMERGENCY_LAND"]) == 1:
            return True

        return False

    # -----------------------
    # Flight state transitions
    # -----------------------
    def takeoff(self):
        if self.flying:
            return
        print("Takeoff requested")

        if self.battery_guard is not None and not self.battery_guard.ok_to_takeoff():
            print(f"Takeoff blocked: {self.battery_guard.status_text()}")
            return

        self.mc = MotionCommander(self.scf, default_height=self.target_z)
        self.mc.__enter__()
        time.sleep(1.0)

        self.flying = True
        print("State: FLYING. Press TAKEOFF_LAND to land.")

    def land(self):
        if not self.flying or self.mc is None:
            return

        print("Land requested")
        self.mc.start_linear_motion(0.0, 0.0, 0.0, 0.0)
        self.mc.land()

        try:
            self.mc.__exit__(None, None, None)
        except Exception:
            pass

        self.mc = None
        self.flying = False
        print("State: GROUNDED. Press TAKEOFF_LAND to take off again.")

    def emergency_land_and_exit(self):
        print("Emergency land pressed")
        # land if flying
        if self.mc is not None:
            try:
                self.mc.start_linear_motion(0.0, 0.0, 0.0, 0.0)
                self.mc.land()
            except Exception:
                pass
            try:
                self.mc.__exit__(None, None, None)
            except Exception:
                pass
        self.mc = None
        self.flying = False
        self._running = False

    # -----------------------
    # Core control step
    # -----------------------
    def step(self):
        """
        One teleop tick. Safe to call from an outer loop.
        - handles button edges
        - if grounded: does nothing else
        - if flying: sends motion commands
        """
        a = self.actions
        t = self.tuning

        if self.battery_guard is not None and self.battery_guard.should_land():
            if self.flying:
                print(f"Low battery detected. Landing now. {self.battery_guard.status_text()}")
                self.land()
            time.sleep(t.dt)
            return

        emergency = self._read_button(a["EMERGENCY_LAND"])
        toggle = self._read_button(a["TAKEOFF_LAND"])

        # emergency rising edge
        if emergency == 1 and self._prev_emergency == 0:
            self.emergency_land_and_exit()
            self._prev_emergency = emergency
            self._prev_toggle = toggle
            return
        self._prev_emergency = emergency

        # toggle rising edge
        if toggle == 1 and self._prev_toggle == 0:
            if not self.flying:
                self.takeoff()
            else:
                self.land()
        self._prev_toggle = toggle

        # grounded: do not command anything
        if not self.flying or self.mc is None:
            time.sleep(t.dt)
            return

        # axes
        roll_cmd = self._read_axis_normalized(a["ROLL"])
        pitch_cmd = self._read_axis_normalized(a["PITCH"])
        yaw_cmd = self._read_axis_normalized(a["YAW"])
        height_cmd = self._read_axis_normalized(a["HEIGHT"])

        # sign fixes
        if t.invert_roll:
            roll_cmd = -roll_cmd
        if t.invert_yaw:
            yaw_cmd = -yaw_cmd

        # height target update
        self.target_z += height_cmd * t.target_rate_mps * t.dt
        self.target_z = self._clamp(self.target_z, t.z_min, t.z_max)

        # target_z -> vz
        err_z = self.target_z - self.state_estimate["z"]
        if abs(err_z) < t.tol_hold:
            vz = 0.0
        else:
            vz = self._clamp(t.kp_z * err_z, -t.max_vz, t.max_vz)

        # roll/pitch/yaw -> vx/vy/yawrate
        vx = self._clamp(pitch_cmd * t.max_vxy, -t.max_vxy, t.max_vxy)
        vy = self._clamp(roll_cmd * t.max_vxy, -t.max_vxy, t.max_vxy)
        yawrate = self._clamp(yaw_cmd * t.max_yawrate, -t.max_yawrate, t.max_yawrate)

        self.mc.start_linear_motion(vx, vy, vz, yawrate)
        time.sleep(t.dt)

    # -----------------------
    # Lifecycle management
    # -----------------------
    def start(self):
        """
        Connects to CF, starts logging, arms, initializes joystick and mapping.
        Does NOT take off.
        """
        self._load_mapping()
        self._init_joystick()

        cflib.crtp.init_drivers()

        self.scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache="./cache"))
        self.scf.__enter__()

        # flow deck check
        self.scf.cf.param.add_update_callback(group="deck", name="bcFlow2", cb=self._param_deck_flow)
        time.sleep(1.0)
        if not self.deck_attached_event.wait(timeout=5):
            raise RuntimeError("No flow deck detected. Exiting for safety.")

        # z estimate logging
        self.logconf = LogConfig(name="StateEstimate", period_in_ms=20)
        self.logconf.add_variable("stateEstimate.z", "float")
        self.logconf.add_variable("stateEstimate.y", "float")
        self.logconf.add_variable("stateEstimate.x", "float")
        self.scf.cf.log.add_config(self.logconf)
        self.logconf.data_received_cb.add_callback(self._log_pos_callback)
        self.logconf.start()

        try:
            if self.battery_guard is not None:
                self.battery_guard.start(self.scf)
                # Wait briefly for the first voltage sample so we can report it.
                t0 = time.time()
                while self.battery_guard.last_vbat is None and (time.time() - t0) < 2.0:
                    time.sleep(0.05)
                print(self.battery_guard.status_text())
        except Exception as exc:
            raise RuntimeError(f"Failed to start battery logging: {exc}") from exc

        # arm (does not take off)
        self.scf.cf.platform.send_arming_request(True)
        time.sleep(0.5)

        self._running = True
        print("State: GROUNDED. Press TAKEOFF_LAND to take off.")

    def stop(self):
        """
        Safe shutdown: land if needed, stop logging, close connection.
        """
        self._running = False

        try:
            if self.flying:
                self.land()
        except Exception:
            pass

        try:
            if self.logconf is not None:
                self.logconf.stop()
        except Exception:
            pass

        try:
            if self.battery_guard is not None:
                self.battery_guard.stop()
        except Exception:
            pass

        try:
            if self.scf is not None:
                self.scf.__exit__(None, None, None)
        except Exception:
            pass

        self.scf = None
        self.logconf = None

    def run(self):
        """
        Standalone run loop. Later you can embed this controller and call step()
        from your own loop instead.
        """
        try:
            self.start()
            while self._running:
                self.step()
        except (KeyboardInterrupt, SystemExit):
            print("Interrupted. Landing (if flying).")
        finally:
            self.stop()


if __name__ == "__main__":
    ctrl = TeleoperationController()
    ctrl.run()
