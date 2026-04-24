from __future__ import annotations

import math
from typing import Any, Dict

import cv2
import numpy as np

from robot_types import MotionMode


# ---------------------------------------------------------------------------
# Auto drive constants (preserved from original)
# ---------------------------------------------------------------------------

CAMERA_RES = (640, 480)

AUTO_DRIVE_SPEED  = 60
AUTO_TURN_SPEED   = 80
TURN_TARGET_DEG   = 75.0

_FRAME_W, _FRAME_H = CAMERA_RES

HEADING_CORRECTION_THRESHOLD = 1.0
HEADING_CORRECTION_GAIN      = 3.0
HEADING_CORRECTION_MAX       = 45

LANE_BOX_CX          = 392  # lane centering box (lower = further left; nudge until robot tracks centered)
LANE_BOX_CY          = 300
LANE_BOX_W           = 80
LANE_BOX_H           = 360
LANE_BOX_TILT_DEG    = -5.0
LANE_MIN_DASHES      = 4
LANE_DASH_MIN_AREA   = 150
LANE_DASH_MAX_AREA   = 6000
LANE_CORRECTION_GAIN = 0.30
LANE_CORRECTION_MAX  = 40

_YELLOW_LO = np.array([18, 100, 100])
_YELLOW_HI = np.array([38, 255, 255])

# Combined clamp after blending lane + IMU corrections
TOTAL_CORRECTION_MAX = 60


def _build_lane_box_pts() -> np.ndarray:
    """Build the rotated bounding-box polygon used for HUD overlay."""
    angle_rad      = math.radians(LANE_BOX_TILT_DEG)
    cos_a, sin_a   = math.cos(angle_rad), math.sin(angle_rad)
    half_w, half_h = LANE_BOX_W / 2.0, LANE_BOX_H / 2.0
    local_corners  = [
        (-half_w, -half_h), (half_w, -half_h),
        ( half_w,  half_h), (-half_w,  half_h),
    ]
    pts = []
    for lx, ly in local_corners:
        rx = int(cos_a * lx - sin_a * ly + LANE_BOX_CX)
        ry = int(sin_a * lx + cos_a * ly + LANE_BOX_CY)
        pts.append([rx, ry])
    return np.array(pts, dtype=np.int32)


_LANE_BOX_PTS = _build_lane_box_pts()


class Team1NavigationIMU:
    """
    Team 1 responsibilities:

    1. Use the latest camera frame to keep the robot centered in the lane.
    2. Use the latest IMU reading as an input to steering correction.
    3. Optionally request an IMU reset-to-zero when Team 1 decides that a new
       heading reference is needed.

    CONTROLLER CONTRACT
    -------------------
    Team 1 is the lane-following team only.

    Team 1 does NOT decide:
    - when sign reading starts
    - when pause/turn actions start
    - whether the robot is in the intersection workflow

    The controller enables Team 1 only when:
    - state.auto_mode is True
    - state.motion_mode == MotionMode.TEAM1_LANE_FOLLOW

    In all other cases, Team 1 should effectively be ignored by the controller.

    Team 1 does NOT send IMU readings back to the controller. The main
    controller reads the IMU and places the latest value in sensors.

    WHAT TEAM 1 RECEIVES
    --------------------
    The main controller calls:

        updates = team1.update(state, sensors)

    Useful values Team 1 will commonly read:

        state.auto_mode
        state.motion_mode
        state.intersection_stage
        state.auto_forward_speed
        sensors.frame
        sensors.imu_delta

    WHAT TEAM 1 SHOULD RETURN
    -------------------------
    Team 1 should return only these kinds of updates:

    - "drive_command"
    - "stop"
    - "reset_imu"
    - "debug"

    The motor command must always be sent as one paired structure so speed and
    turn are never separated.

    EXAMPLE: do nothing this cycle
    ------------------------------
        return {}

    EXAMPLE: drive straight
    -----------------------
        return {
            "drive_command": {
                "speed": state.auto_forward_speed,
                "turn": 0,
            },
            "stop": False,
            "debug": "Lane centered; driving straight"
        }

    EXAMPLE: steering correction right
    ----------------------------------
        return {
            "drive_command": {
                "speed": state.auto_forward_speed,
                "turn": 12,
            },
            "stop": False,
            "debug": "Lane is left of center; steering right"
        }

    EXAMPLE: steering correction left
    ---------------------------------
        return {
            "drive_command": {
                "speed": state.auto_forward_speed,
                "turn": -10,
            },
            "stop": False,
            "debug": "Lane is right of center; steering left"
        }

    EXAMPLE: request IMU reset
    --------------------------
        return {
            "reset_imu": True,
            "debug": "Requesting IMU zero after stable lane reacquisition"
        }

    EXAMPLE: combined drive command plus IMU reset
    ----------------------------------------------
        return {
            "drive_command": {
                "speed": state.auto_forward_speed,
                "turn": 0,
            },
            "stop": False,
            "reset_imu": True,
            "debug": "Centered in lane; resetting IMU heading reference"
        }
    """

    def __init__(self) -> None:
        self.last_debug = "Team 1 initialized"
        self.last_requested_reset_time = 0.0
        self.reset_cooldown_seconds = 1.0

        # Tracking for IMU-reset heuristic
        self._centered_streak = 0

    def update(self, state: Any, sensors: Any) -> Dict[str, Any]:
        """
        Main Team 1 entry point.

        Team 1 should:
        1. Check whether lane-centering should run in the current state.
        2. Read the latest frame and IMU value.
        3. Estimate lane position.
        4. Convert lane/heading error into a drive_command.
        5. Optionally request IMU reset when appropriate.

        Return {} when Team 1 has nothing to change.
        """
        if not self.should_run_lane_centering(state):
            return {}

        lane_updates = self.compute_lane_centering(state, sensors)
        reset_updates = self.maybe_request_imu_reset(state, sensors)

        return self.merge_updates(lane_updates, reset_updates)

    def should_run_lane_centering(self, state: Any) -> bool:
        """
        Lane centering should run only when the controller has explicitly given
        Team 1 motion authority.

        Keep this rule simple. Team 1 should check whether it currently owns
        lane-following motion, then compute a drive command.
        """
        if not getattr(state, "auto_mode", False):
            return False

        if getattr(state, "motion_mode", None) != MotionMode.TEAM1_LANE_FOLLOW:
            return False

        return True

    def compute_lane_centering(self, state: Any, sensors: Any) -> Dict[str, Any]:
        """
        Lane centering using yellow dash detection (camera) blended with IMU
        heading correction.

        Team 1 consumes:
        - sensors.frame      – BGR/XRGB numpy array from Picamera2
        - sensors.imu_delta  – heading drift in degrees since last IMU zero
        - state.auto_forward_speed

        Team 1 returns:
        - drive_command = {"speed": ..., "turn": ...}
        - stop
        - debug
        """
        frame        = getattr(sensors, "frame", None)
        imu_delta    = float(getattr(sensors, "imu_delta", 0.0))
        forward_speed = int(getattr(state, "auto_forward_speed", 80))

        if frame is None:
            return {
                "drive_command": {
                    "speed": forward_speed,
                    "turn": 0,
                },
                "stop": False,
                "debug": "Team 1: no frame available; holding straight",
            }

        lane_info  = self.estimate_lane_position(frame)
        lane_offset = float(lane_info.get("lane_offset", 0.0))
        dash_count  = int(lane_info.get("dash_count", 0))
        avg_x       = lane_info.get("avg_x", None)

        turn_correction = self.compute_turn_from_lane_and_imu(lane_offset, imu_delta)

        avg_x_str = f"{avg_x:.1f}" if avg_x is not None else "N/A"
        debug_msg = (
            f"Team 1: dashes={dash_count}, "
            f"avg_x={avg_x_str}, "
            f"offset={lane_offset:.2f}, "
            f"imu={imu_delta:.1f}deg, "
            f"turn={turn_correction}"
        )

        return {
            "drive_command": {
                "speed": forward_speed,
                "turn": turn_correction,
            },
            "stop": False,
            "debug": debug_msg,
        }

    def maybe_request_imu_reset(self, state: Any, sensors: Any) -> Dict[str, Any]:
        """
        Request an IMU reset when the robot has been well-centered for several
        consecutive frames, indicating the current heading is a good reference.

        Strategy:
        - Count consecutive frames where |imu_delta| is small (robot is
          driving straight without accumulated drift).
        - After a stable streak, request a reset so the IMU re-zeroes to the
          current true heading.
        - A cooldown prevents repeated resets within a short window.
        """
        import time

        imu_delta = float(getattr(sensors, "imu_delta", 0.0))
        now       = getattr(sensors, "now", 0.0)

        STABLE_THRESHOLD    = 1.5   # degrees – consider robot straight below this
        STREAK_NEEDED       = 30    # consecutive stable frames before reset
        COOLDOWN_SECONDS    = 5.0   # minimum gap between resets

        if abs(imu_delta) < STABLE_THRESHOLD:
            self._centered_streak += 1
        else:
            self._centered_streak = 0

        if (
            self._centered_streak >= STREAK_NEEDED
            and (now - self.last_requested_reset_time) >= COOLDOWN_SECONDS
        ):
            self._centered_streak = 0
            self.last_requested_reset_time = now
            return {
                "reset_imu": True,
                "debug": (
                    f"Team 1: stable for {STREAK_NEEDED} frames "
                    f"(imu={imu_delta:.2f}°); resetting IMU reference"
                ),
            }

        return {}

    def estimate_lane_position(self, frame: Any) -> Dict[str, float]:
        """
        Detect yellow dashes in the camera frame and compute a lane offset.

        Algorithm (from original DashCorrector):
        1. Convert BGR frame to HSV.
        2. Threshold for yellow (dashes on the road).
        3. Find contours; keep those within the area band
           [LANE_DASH_MIN_AREA, LANE_DASH_MAX_AREA].
        4. Collect horizontal centroids of valid contours.
        5. If at least LANE_MIN_DASHES are found, compute their mean x and
           derive the offset from the target center (LANE_BOX_CX).

        Returns a dict with:
          lane_offset  – positive means dashes are left of target (steer right),
                         negative means dashes are right of target (steer left),
                         0 if fewer than LANE_MIN_DASHES found.
          dash_count   – number of valid dash contours detected.
          avg_x        – mean centroid x, or None if insufficient dashes.

        Meaning of lane_offset sign:
          positive => robot is too far left  => steer right (positive turn)
          negative => robot is too far right => steer left  (negative turn)
          zero     => centered
        """
        # Picamera2 with XRGB8888 delivers a 4-channel array; drop alpha if present
        if frame.ndim == 3 and frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr = frame

        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _YELLOW_LO, _YELLOW_HI)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        centroids = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if LANE_DASH_MIN_AREA <= area <= LANE_DASH_MAX_AREA:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    centroids.append(cx)

        dash_count = len(centroids)

        if dash_count < LANE_MIN_DASHES:
            return {
                "lane_offset": 0.0,
                "dash_count":  dash_count,
                "avg_x":       None,
            }

        avg_x       = float(np.mean(centroids))
        # Matches original: error = LANE_BOX_CX - avg_x
        # positive → dashes are left of target → robot drifted right → steer left (negative turn)
        # negative → dashes are right of target → robot drifted left → steer right (positive turn)
        lane_offset = LANE_BOX_CX - avg_x

        return {
            "lane_offset": lane_offset,
            "dash_count":  dash_count,
            "avg_x":       avg_x,
        }

    def compute_turn_from_lane_and_imu(self, lane_offset: float, imu_delta: float) -> int:
        """
        Blend lane-position error and IMU heading drift into a single turn value.

        Lane correction (from original DashCorrector / AutoDriver):
          - lane_offset is raw pixel error (avg_x − LANE_BOX_CX).
          - Multiplied by LANE_CORRECTION_GAIN then clamped to
            ±LANE_CORRECTION_MAX.
          - Positive lane_offset → dashes are right of target → robot drifted
            left → need negative turn (steer left back to center).

        IMU correction (from original AutoDriver.update):
          - imu_delta is heading drift in degrees.
          - Only applied when |imu_delta| >= HEADING_CORRECTION_THRESHOLD.
          - Multiplied by HEADING_CORRECTION_GAIN then clamped to
            ±HEADING_CORRECTION_MAX.

        The two corrections are summed and the total is clamped to
        ±TOTAL_CORRECTION_MAX before being cast to int.
        """
        # Exact match of original AutoDriver.update() + DashCorrector logic:
        #
        # cam_correction = (LANE_BOX_CX - avg_x) * LANE_CORRECTION_GAIN
        #   lane_offset here IS (LANE_BOX_CX - avg_x), so:
        #   cam_correction = lane_offset * LANE_CORRECTION_GAIN
        #
        # imu_corr = heading_delta * HEADING_CORRECTION_GAIN  (no negation)
        #
        # total = imu_corr + cam_correction  (both added, same sign convention)

        # IMU correction — applied whenever drift exceeds threshold
        if abs(imu_delta) >= HEADING_CORRECTION_THRESHOLD:
            imu_corr = float(imu_delta) * HEADING_CORRECTION_GAIN
            imu_corr = max(-HEADING_CORRECTION_MAX, min(HEADING_CORRECTION_MAX, imu_corr))
        else:
            imu_corr = 0.0

        # Lane correction — only applied when dashes are visible
        # lane_offset = LANE_BOX_CX - avg_x, so this is error * gain exactly as original
        cam_correction = lane_offset * LANE_CORRECTION_GAIN
        cam_correction = max(-LANE_CORRECTION_MAX, min(LANE_CORRECTION_MAX, cam_correction))

        total = imu_corr + cam_correction
        total = max(-TOTAL_CORRECTION_MAX, min(TOTAL_CORRECTION_MAX, total))

        return int(total)

    def merge_updates(self, first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two update dictionaries.

        If both dictionaries contain the same key, the second dictionary wins.
        """
        merged = dict(first)
        merged.update(second)
        return merged

    @staticmethod
    def clamp(value: int, low: int, high: int) -> int:
        return max(low, min(high, value))