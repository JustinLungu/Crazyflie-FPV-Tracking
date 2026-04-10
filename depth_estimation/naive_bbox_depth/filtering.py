from __future__ import annotations


class ExponentialMovingAverage:
    def __init__(self, alpha: float):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        return self._value

    def reset(self) -> None:
        self._value = None

    def update(self, measurement: float) -> float:
        x = float(measurement)
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1.0 - self.alpha) * self._value
        return self._value


class ConstantVelocityKalman1D:
    def __init__(
        self,
        process_var: float,
        measurement_var: float,
    ):
        if process_var <= 0.0:
            raise ValueError(f"Kalman process variance must be > 0, got {process_var}")
        if measurement_var <= 0.0:
            raise ValueError(f"Kalman measurement variance must be > 0, got {measurement_var}")
        self.process_var = float(process_var)
        self.measurement_var = float(measurement_var)
        self.reset()

    @property
    def value(self) -> float | None:
        return self.x

    def reset(self) -> None:
        self.x: float | None = None
        self.v: float = 0.0
        self.p00: float = 1.0
        self.p01: float = 0.0
        self.p10: float = 0.0
        self.p11: float = 1.0

    def _predict(self, dt: float = 1.0) -> None:
        # State prediction (constant velocity).
        self.x = float(self.x + self.v * dt)

        q = self.process_var
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q00 = 0.25 * dt4 * q
        q01 = 0.5 * dt3 * q
        q11 = dt2 * q

        p00 = self.p00 + dt * (self.p10 + self.p01) + dt2 * self.p11 + q00
        p01 = self.p01 + dt * self.p11 + q01
        p10 = self.p10 + dt * self.p11 + q01
        p11 = self.p11 + q11

        self.p00, self.p01, self.p10, self.p11 = p00, p01, p10, p11

    def update(self, measurement: float) -> float:
        z = float(measurement)
        if self.x is None:
            self.x = z
            self.v = 0.0
            self.p00 = 1.0
            self.p01 = 0.0
            self.p10 = 0.0
            self.p11 = 1.0
            return self.x

        self._predict(dt=1.0)

        # Measurement update with H = [1, 0].
        y = z - self.x
        s = self.p00 + self.measurement_var
        k0 = self.p00 / s
        k1 = self.p10 / s

        self.x = self.x + k0 * y
        self.v = self.v + k1 * y

        p00 = (1.0 - k0) * self.p00
        p01 = (1.0 - k0) * self.p01
        p10 = self.p10 - k1 * self.p00
        p11 = self.p11 - k1 * self.p01

        self.p00, self.p01, self.p10, self.p11 = p00, p01, p10, p11
        return self.x


class ScalarSignalFilter:
    def __init__(
        self,
        mode: str,
        ema_alpha: float,
        kalman_process_var: float,
        kalman_measurement_var: float,
    ):
        self.mode = mode.strip().lower()
        if self.mode not in {"none", "ema", "kalman"}:
            raise ValueError(f"Unsupported scalar filter mode: {mode}")

        self._raw_value: float | None = None
        self._impl = None
        if self.mode == "ema":
            self._impl = ExponentialMovingAverage(alpha=ema_alpha)
        elif self.mode == "kalman":
            self._impl = ConstantVelocityKalman1D(
                process_var=kalman_process_var,
                measurement_var=kalman_measurement_var,
            )

    @property
    def value(self) -> float | None:
        if self.mode == "none":
            return self._raw_value
        return self._impl.value

    def reset(self) -> None:
        self._raw_value = None
        if self._impl is not None:
            self._impl.reset()

    def update(self, measurement: float) -> float:
        x = float(measurement)
        if self.mode == "none":
            self._raw_value = x
            return x
        return float(self._impl.update(x))
