from __future__ import annotations

from typing import Any, Dict

from robot_types import MotionMode


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
        Starting framework for lane centering.

        Team 1 consumes:
        - sensors.frame
        - sensors.imu_delta
        - state.auto_forward_speed

        Team 1 returns:
        - drive_command = {"speed": ..., "turn": ...}
        - stop
        - debug
        """
        frame = getattr(sensors, "frame", None)
        imu_delta = float(getattr(sensors, "imu_delta", 0.0))
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

        lane_info = self.estimate_lane_position(frame)
        lane_offset = float(lane_info.get("lane_offset", 0.0))
        turn_correction = self.compute_turn_from_lane_and_imu(lane_offset, imu_delta)

        return {
            "drive_command": {
                "speed": forward_speed,
                "turn": turn_correction,
            },
            "stop": False,
            "debug": (
                f"Team 1 lane control: offset={lane_offset:.2f}, "
                f"imu={imu_delta:.1f}, turn={turn_correction}"
            ),
        }

    def maybe_request_imu_reset(self, state: Any, sensors: Any) -> Dict[str, Any]:
        """
        Optional starting framework for requesting an IMU reset.

        Team 1 consumes IMU information but does not publish IMU values back.
        The only IMU-related update Team 1 should request is reset_imu.

        The starter logic is intentionally conservative and does not
        request resets automatically. Students may replace this with their own
        strategy.
        """
        _ = state
        _ = sensors
        return {}

    def estimate_lane_position(self, frame: Any) -> Dict[str, float]:
        """
        Placeholder lane-position estimator.

        Students will replace this method with their OpenCV strategy.

        Suggested meaning of lane_offset:
        - negative => robot is too far right, should steer left
        - positive => robot is too far left, should steer right
        - zero     => centered
        """
        _ = frame
        return {
            "lane_offset": 0.0,
        }

    def compute_turn_from_lane_and_imu(self, lane_offset: float, imu_delta: float) -> int:
        """
        Convert lane-position error and IMU heading information into a turn value.

        This starting implementation is intentionally simple so students can
        understand and replace it.
        """
        lane_gain = 25.0
        imu_gain = 0.4
        turn = int((lane_offset * lane_gain) - (imu_delta * imu_gain))
        return self.clamp(turn, -40, 40)

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
