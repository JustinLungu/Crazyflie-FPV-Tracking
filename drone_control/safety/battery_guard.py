from typing import Optional

from cflib.crazyflie.log import LogConfig
from drone_control.safety.constants import (
    BATTERY_LOG_PERIOD_MS,
    BATTERY_MIN_FLIGHT_VOLTAGE,
    BATTERY_MIN_TAKEOFF_VOLTAGE,
)


class BatteryGuard:
    def __init__(
        self,
        min_takeoff_voltage: float = BATTERY_MIN_TAKEOFF_VOLTAGE,
        min_flight_voltage: float = BATTERY_MIN_FLIGHT_VOLTAGE,
        log_period_ms: int = BATTERY_LOG_PERIOD_MS,
    ):
        self.min_takeoff_voltage = min_takeoff_voltage
        self.min_flight_voltage = min_flight_voltage
        self.log_period_ms = log_period_ms
        self.last_vbat: Optional[float] = None
        self._logconf: Optional[LogConfig] = None

    def _log_vbat_callback(self, timestamp, data, logconf):
        vbat = data.get("pm.vbat")
        if vbat is None:
            return
        self.last_vbat = float(vbat)

    def start(self, scf):
        if self._logconf is not None:
            return

        logconf = LogConfig(name="Battery", period_in_ms=self.log_period_ms)
        logconf.add_variable("pm.vbat", "float")

        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(self._log_vbat_callback)
        logconf.start()

        self._logconf = logconf

    def stop(self):
        if self._logconf is None:
            return
        try:
            self._logconf.stop()
        except Exception:
            pass
        self._logconf = None

    def ok_to_takeoff(self) -> bool:
        if self.last_vbat is None:
            return False
        return self.last_vbat >= self.min_takeoff_voltage

    def should_land(self) -> bool:
        if self.last_vbat is None:
            return False
        return self.last_vbat < self.min_flight_voltage

    def status_text(self) -> str:
        if self.last_vbat is None:
            return "Battery voltage unknown"
        return (
            f"Battery {self.last_vbat:.2f}V "
            f"(takeoff min {self.min_takeoff_voltage:.2f}V, "
            f"flight min {self.min_flight_voltage:.2f}V)"
        )
