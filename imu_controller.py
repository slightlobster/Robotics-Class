################################################################
# IMU Controller for ND Robotics Course
# 12-19-2025
# Professor McLaughlin
################################################################
# Minimal student-facing interface for BNO055 heading on a Raspberry Pi (I2C).
#
# Public API:
#  - zero(): set current heading as reference (0°)
#  - heading() -> float | None: absolute heading in degrees [0, 360)
#  - delta() -> float | None: signed degrees from reference in [-180, +180]
#  - close(): release resources (set CONFIG_MODE, deinit I2C)
#
# Implementation notes:
# - Uses Adafruit Blinka + Adafruit CircuitPython BNO055 driver.
# - Uses IMUPLUS_MODE (gyro + accel fusion; magnetometer disabled).
# - Synchronous reads (caller throttles in their control loop).
################################################################
from __future__ import annotations

import time
from typing import Optional

import board
import busio
import adafruit_bno055


class IMUDevice:
    # ---------------------------
    # Student-editable constants
    # ---------------------------

    # Typical BNO055 I2C address is 0x28; can be 0x29 if ADR pin changed.
    I2C_ADDRESS = 0x28

    # Fusion mode: IMU mode without magnetometer (gyro + accel).
    _FUSION_MODE = adafruit_bno055.IMUPLUS_MODE

    # SIGN controls whether turning right is positive or negative.
    # Flip to -1 if your robot reports the opposite direction.
    SIGN = +1

    # Small delay after changing modes (helps ensure stable reads).
    _MODE_SETTLE_SEC = 0.05

    def __init__(self) -> None:
        self._i2c: Optional[busio.I2C] = None
        self._sensor: Optional[adafruit_bno055.BNO055_I2C] = None
        self._zero_heading: Optional[float] = None

        # Initialize I2C bus
        self._i2c = busio.I2C(board.SCL, board.SDA)

        # Initialize BNO055 sensor
        self._sensor = adafruit_bno055.BNO055_I2C(
            self._i2c, address=self.I2C_ADDRESS
        )

        # Set fusion mode
        self._sensor.mode = self._FUSION_MODE
        time.sleep(self._MODE_SETTLE_SEC)

        # Optional convenience: zero immediately on startup
        self.zero()

    # ---------------------------
    # Public API (minimal)
    # ---------------------------

    def heading(self) -> Optional[float]:
        """
        Return absolute heading in degrees [0, 360), or None if unavailable.
        """
        if self._sensor is None:
            return None

        try:
            euler = self._sensor.euler
            if euler is None or euler[0] is None:
                return None

            return float(euler[0]) % 360.0
        except Exception:
            return None

    def zero(self) -> None:
        """
        Lock the current heading as the zero reference.
        """
        h = self.heading()
        if h is not None:
            self._zero_heading = h

    def delta(self) -> Optional[float]:
        """
        Signed heading delta from the zero reference in [-180, +180].
        """
        h = self.heading()
        z = self._zero_heading

        if h is None or z is None:
            return None

        raw = (h - z) * float(self.SIGN)
        return self._wrap_to_180(raw)

    def close(self) -> None:
        """
        Release resources so repeated runs of the control loop are clean.
        """
        try:
            if self._sensor is not None:
                self._sensor.mode = adafruit_bno055.CONFIG_MODE
                time.sleep(self._MODE_SETTLE_SEC)
        except Exception:
            pass

        try:
            if self._i2c is not None:
                self._i2c.deinit()
        except Exception:
            pass

        self._sensor = None
        self._i2c = None

    # ---------------------------
    # Internal helpers
    # ---------------------------

    @staticmethod
    def _wrap_to_180(deg: float) -> float:
        """
        Wrap angle to [-180, +180].
        """
        x = deg % 360.0
        if x > 180.0:
            x -= 360.0
        return x
