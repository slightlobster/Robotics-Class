from __future__ import annotations

import time
from typing import Any, Dict

from robot_types import AutoCurrentSign, AutoMotion, IntersectionStage


class Team4SignActions:
    """
    Team 4 responsibilities:

    1. Implement the action associated with the currently detected sign.
    2. Handle Pause, Left Turn, Right Turn, Go Straight, and Unknown sign cases.
    3. Report when the current action is complete.

    IMPORTANT DESIGN NOTE
    ---------------------
    Team 4 runs inside the main controller process.

    CONTROLLER CONTRACT
    -------------------
    Team 4 is the action-execution team only.

    Team 4 is expected to run when the controller places the robot in:
    - PAUSE_ACTION
    - TURN_ACTION

    Team 4 does NOT decide when those stages start. The controller decides that
    from Team 2 milestones plus Team 3 sign results.

    Team 4 does NOT talk directly to motors. Instead, it returns update
    dictionaries that the main controller applies.

    Team 4 also does NOT read the IMU hardware directly.
    It consumes the latest IMU value that the main controller already placed in:

        sensors.imu_delta

    WHAT TEAM 4 RECEIVES
    --------------------
    The main controller calls:

        updates = team4.update(state, sensors)

    Useful values Team 4 will commonly read:

        state.auto_mode
        state.auto_motion
        state.intersection_stage
        state.auto_current_sign
        state.auto_turn_speed
        state.auto_forward_speed
        sensors.imu_delta
        sensors.now

    WHAT TEAM 4 SHOULD RETURN
    -------------------------
    Team 4 may return these keys:

    - "drive_command"
    - "stop"
    - "action_complete"
    - "reset_imu"
    - "debug"

    EXAMPLE: start a pause action
    -----------------------------
        return {
            "auto_motion": AutoMotion.PAUSED,
            "drive_command": {"speed": 0, "turn": 0},
            "stop": True,
            "debug": "Pause sign action started"
        }

    EXAMPLE: finish a pause action
    ------------------------------
        return {
            "action_complete": True,
            "drive_command": {"speed": state.auto_forward_speed, "turn": 0},
            "stop": False,
            "debug": "Pause complete; returning to travel lane"
        }

    EXAMPLE: start a left turn
    --------------------------
        return {
            "auto_motion": AutoMotion.TURNING_LEFT,
            "drive_command": {"speed": 0, "turn": -state.auto_turn_speed},
            "stop": False,
            "debug": "Starting left turn"
        }

    EXAMPLE: finish a left turn
    ---------------------------
        return {
            "action_complete": True,
            "reset_imu": True,
            "drive_command": {"speed": state.auto_forward_speed, "turn": 0},
            "stop": False,
            "debug": "Left turn complete"
        }

    EXAMPLE: go straight / unknown handling
    ---------------------------------------
        return {
            "action_complete": True,
            "drive_command": {"speed": state.auto_forward_speed, "turn": 0},
            "stop": False,
            "debug": "Proceeding straight through intersection"
        }
    """

    def __init__(self) -> None:
        self.pause_duration_seconds = 5.0
        self.pause_start_time: float | None = None
        self.turn_tolerance_degrees = 5.0
        self.turn_target_degrees = 90.0
        self.turn_in_progress_sign: AutoCurrentSign | None = None

    def update(self, state: Any, sensors: Any) -> Dict[str, Any]:
        """
        Main Team 4 entry point.

        Team 4 should:
        1. Check the controller-owned intersection_stage.
        2. If the stage is an action stage, run the corresponding behavior.
        3. Return action_complete when the controller should hand off motion.
        """
        if not getattr(state, "auto_mode", False):
            self.reset_action_state()
            return {}

        sign = getattr(state, "auto_current_sign", AutoCurrentSign.UNKNOWN)
        stage = getattr(state, "intersection_stage", IntersectionStage.TRAVEL_LANE)

        if stage == IntersectionStage.PAUSE_ACTION and sign == AutoCurrentSign.PAUSE:
            return self.handle_pause(state, sensors)

        if stage == IntersectionStage.TURN_ACTION and sign == AutoCurrentSign.LEFT:
            return self.handle_left_turn(state, sensors)

        if stage == IntersectionStage.TURN_ACTION and sign == AutoCurrentSign.RIGHT:
            return self.handle_right_turn(state, sensors)

        if stage == IntersectionStage.TURN_ACTION and sign == AutoCurrentSign.GO_STRAIGHT:
            return self.handle_go_straight(state)

        if stage == IntersectionStage.TURN_ACTION and sign == AutoCurrentSign.UNKNOWN:
            return self.handle_unknown_sign(state)

        self.reset_action_state()
        return {}

    def reset_action_state(self) -> None:
        """
        Clear any in-progress pause/turn bookkeeping when auto mode is not active.
        """
        self.pause_start_time = None
        self.turn_in_progress_sign = None

    def handle_pause(self, state: Any, sensors: Any) -> Dict[str, Any]:
        """
        Implement the Pause sign action.

        First call:
        - stop the robot
        - remember the pause start time

        Later calls:
        - once 5 seconds pass, report action_complete to the controller
        """
        now = float(getattr(sensors, "now", time.time()))
        motion = getattr(state, "auto_motion", AutoMotion.PAUSED)

        if motion != AutoMotion.PAUSED or self.pause_start_time is None:
            self.pause_start_time = now
            return {
                "auto_motion": AutoMotion.PAUSED,
                "drive_command": {"speed": 0, "turn": 0},
                "stop": True,
                "debug": "Team 4 pause started",
            }

        if now - self.pause_start_time < self.pause_duration_seconds:
            remaining = self.pause_duration_seconds - (now - self.pause_start_time)
            return {
                "auto_motion": AutoMotion.PAUSED,
                "drive_command": {"speed": 0, "turn": 0},
                "stop": True,
                "debug": f"Team 4 pausing... {remaining:.1f}s remaining",
            }

        self.pause_start_time = None
        return {
            "action_complete": True,
            "drive_command": {"speed": int(getattr(state, 'auto_forward_speed', 80)), "turn": 0},
            "stop": False,
            "debug": "Team 4 pause complete",
        }

    def handle_left_turn(self, state: Any, sensors: Any) -> Dict[str, Any]:
        """
        Implement the LEFT sign action using IMU heading delta.

        The main controller already updates sensors.imu_delta.
        Team 4 only consumes that value.
        """
        imu_delta = float(getattr(sensors, "imu_delta", 0.0))
        turn_speed = int(getattr(state, "auto_turn_speed", 80))
        if self.turn_in_progress_sign is None:
            self.turn_in_progress_sign = AutoCurrentSign.LEFT
            return {
                "auto_motion": AutoMotion.TURNING_LEFT,
                "drive_command": {"speed": 0, "turn": -turn_speed},
                "stop": False,
                "debug": "Team 4 starting left turn",
            }

        if abs(abs(imu_delta) - self.turn_target_degrees) <= self.turn_tolerance_degrees or imu_delta <= -self.turn_target_degrees:
            self.turn_in_progress_sign = None
            return {
                "action_complete": True,
                "reset_imu": True,
                "drive_command": {"speed": int(getattr(state, 'auto_forward_speed', 80)), "turn": 0},
                "stop": False,
                "debug": f"Team 4 left turn complete at IMU {imu_delta:.1f}",
            }

        return {
            "auto_motion": AutoMotion.TURNING_LEFT,
            "drive_command": {"speed": 0, "turn": -turn_speed},
            "stop": False,
            "debug": f"Team 4 left turn in progress, IMU={imu_delta:.1f}",
        }

    def handle_right_turn(self, state: Any, sensors: Any) -> Dict[str, Any]:
        """
        Implement the RIGHT sign action using IMU heading delta.
        """
        imu_delta = float(getattr(sensors, "imu_delta", 0.0))
        turn_speed = int(getattr(state, "auto_turn_speed", 80))
        if self.turn_in_progress_sign is None:
            self.turn_in_progress_sign = AutoCurrentSign.RIGHT
            return {
                "auto_motion": AutoMotion.TURNING_RIGHT,
                "drive_command": {"speed": 0, "turn": turn_speed},
                "stop": False,
                "debug": "Team 4 starting right turn",
            }

        if abs(abs(imu_delta) - self.turn_target_degrees) <= self.turn_tolerance_degrees or imu_delta >= self.turn_target_degrees:
            self.turn_in_progress_sign = None
            return {
                "action_complete": True,
                "reset_imu": True,
                "drive_command": {"speed": int(getattr(state, 'auto_forward_speed', 80)), "turn": 0},
                "stop": False,
                "debug": f"Team 4 right turn complete at IMU {imu_delta:.1f}",
            }

        return {
            "auto_motion": AutoMotion.TURNING_RIGHT,
            "drive_command": {"speed": 0, "turn": turn_speed},
            "stop": False,
            "debug": f"Team 4 right turn in progress, IMU={imu_delta:.1f}",
        }

    def handle_go_straight(self, state: Any) -> Dict[str, Any]:
        """
        Implement the Go Straight sign action.
        """
        return {
            "action_complete": True,
            "drive_command": {"speed": int(getattr(state, 'auto_forward_speed', 80)), "turn": 0},
            "stop": False,
            "debug": "Team 4 go-straight action complete",
        }

    def handle_unknown_sign(self, state: Any) -> Dict[str, Any]:
        """
        Safe fallback when the sign is unknown.
        """
        return {
            "action_complete": True,
            "drive_command": {"speed": int(getattr(state, 'auto_forward_speed', 80)), "turn": 0},
            "stop": False,
            "debug": "Team 4 unknown-sign fallback: proceeding straight",
        }
