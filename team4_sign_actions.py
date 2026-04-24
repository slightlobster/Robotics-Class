from __future__ import annotations

import sys
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

    def __init__(self, saber: Any = None, imu: Any = None) -> None:
        self.pause_duration_seconds = 5.0
        self.pause_start_time: float | None = None
        # Turn completes when the IMU delta is within +/- tolerance of the
        # target. Target is shaved below 90 deg to compensate for the coast
        # between the stop command going out and the robot physically stopping.
        self.turn_tolerance_degrees = 2.0
        self.turn_target_degrees = 90
        # Empirical coast between the brake command going out and the robot
        # physically stopping. Completion triggers at (target - coast_offset)
        # so the robot actually lands on the target. Measure: run a turn, note
        # "braking (IMU=X)" vs "complete at IMU Y", difference is the offset.
        self.turn_coast_offset_degrees = 0.0
        # After a turn hits its (offset-adjusted) target, hold an explicit stop
        # for this many seconds so the Sabertooth can fully ramp the turn
        # channel to zero before the forward command starts pushing.
        self.turn_brake_seconds = 0.15
        self.turn_brake_start_time: float | None = None
        self.turn_in_progress_sign: AutoCurrentSign | None = None
        self.latched_sign: AutoCurrentSign | None = None
        self.unknown_start_time: float | None = None
        self.unknown_warning_printed = False
        self.saber = saber
        self.imu = imu
        self._last_log: str | None = None

    def _log(self, msg: str) -> None:
        """Print a [Team4] message, but skip consecutive duplicates."""
        if msg == self._last_log:
            return
        self._last_log = msg
        print(f"[Team4] {msg}")

    def _drive(self, speed: int, turn: int) -> None:
        if self.saber is not None:
            try:
                self.saber.drive(int(speed), int(turn))
            except Exception as exc:
                print(f"[Team4] saber.drive failed: {exc}", file=sys.stderr)

    def _stop_motors(self) -> None:
        if self.saber is not None:
            try:
                self.saber.stop()
            except Exception as exc:
                print(f"[Team4] saber.stop failed: {exc}", file=sys.stderr)

    def _imu_delta(self, sensors: Any) -> float:
        if self.imu is not None:
            try:
                raw = self.imu.get_delta()
                if raw is not None:
                    return float(raw)
            except Exception as exc:
                print(f"[Team4] imu.get_delta failed: {exc}", file=sys.stderr)
        return float(getattr(sensors, "imu_delta", 0.0))

    def _imu_reset(self) -> None:
        if self.imu is not None:
            try:
                self.imu.zero_reference()
            except Exception as exc:
                print(f"[Team4] imu.zero_reference failed: {exc}", file=sys.stderr)

    def _apply_hardware(self, updates: Dict[str, Any]) -> None:
        """
        If this instance was constructed with real hardware handles,
        push the computed updates to the motors / IMU directly.
        Safe to call either way: with no hardware, this is a no-op.
        """
        if updates.get("stop"):
            self._stop_motors()
        elif "drive_command" in updates:
            cmd = updates["drive_command"]
            self._drive(cmd.get("speed", 0), cmd.get("turn", 0))
        if updates.get("reset_imu"):
            self._imu_reset()

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

        stage = getattr(state, "intersection_stage", IntersectionStage.TRAVEL_LANE)
        live_sign = getattr(state, "auto_current_sign", AutoCurrentSign.UNKNOWN)

        if stage not in (IntersectionStage.PAUSE_ACTION, IntersectionStage.TURN_ACTION):
            self.reset_action_state()
            return {}

        if self.latched_sign is None:
            self.latched_sign = live_sign

        effective_sign = self.latched_sign

        if stage == IntersectionStage.PAUSE_ACTION and effective_sign == AutoCurrentSign.PAUSE:
            updates = self.handle_pause(state, sensors)
        elif stage == IntersectionStage.TURN_ACTION and effective_sign == AutoCurrentSign.LEFT:
            updates = self.handle_left_turn(state, sensors)
        elif stage == IntersectionStage.TURN_ACTION and effective_sign == AutoCurrentSign.RIGHT:
            updates = self.handle_right_turn(state, sensors)
        elif stage == IntersectionStage.TURN_ACTION and effective_sign == AutoCurrentSign.GO_STRAIGHT:
            updates = self.handle_go_straight(state)
        elif stage == IntersectionStage.TURN_ACTION and effective_sign == AutoCurrentSign.UNKNOWN:
            updates = self.handle_unknown_sign(state, sensors)
        else:
            self.latched_sign = None
            return {}

        debug_msg = updates.get("debug", "")
        if debug_msg:
            self._log(debug_msg)

        if updates.get("action_complete"):
            self.latched_sign = None
            self._last_log = None

        self._apply_hardware(updates)
        return updates

    def reset_action_state(self) -> None:
        """
        Clear any in-progress pause/turn bookkeeping when auto mode is not active.
        """
        self.pause_start_time = None
        self.turn_in_progress_sign = None
        self.turn_brake_start_time = None
        self.latched_sign = None
        self.unknown_start_time = None
        self.unknown_warning_printed = False
        self._last_log = None

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
                "debug": f"Team 4 pausing... {int(remaining) + 1}s remaining",
            }

        self.pause_start_time = None
        return {
            "action_complete": True,
            "reset_imu": True,
            "drive_command": {"speed": int(getattr(state, 'auto_forward_speed', 80)), "turn": 0},
            "stop": False,
            "debug": "Team 4 pause complete",
        }

    def handle_left_turn(self, state: Any, sensors: Any) -> Dict[str, Any]:
        """
        Implement the LEFT sign action using IMU heading delta.

        Reads IMU directly if this instance was constructed with an IMU handle,
        otherwise falls back to sensors.imu_delta supplied by the controller.
        """
        now = float(getattr(sensors, "now", time.time()))
        imu_delta = self._imu_delta(sensors)
        turn_speed = int(getattr(state, "auto_turn_speed", 80))
        if self.turn_in_progress_sign is None:
            self.turn_in_progress_sign = AutoCurrentSign.LEFT
            # Zero the IMU *before* the turn command goes out so the turn
            # measures a clean 90 deg from the reference, not from a drifted
            # starting angle. The inline call handles the case where Team 4
            # owns the IMU handle; reset_imu=True signals the main controller
            # to zero its own IMU + sensors.imu_delta.
            self._imu_reset()
            return {
                "auto_motion": AutoMotion.TURNING_LEFT,
                "reset_imu": True,
                "drive_command": {"speed": 0, "turn": turn_speed},
                "stop": False,
                "debug": "Team 4 starting left turn (IMU zeroed)",
            }

        if self.turn_brake_start_time is not None:
            if now - self.turn_brake_start_time < self.turn_brake_seconds:
                return {
                    "auto_motion": AutoMotion.PAUSED,
                    "drive_command": {"speed": 0, "turn": 0},
                    "stop": True,
                    "debug": f"Team 4 braking after left turn, IMU={imu_delta:.1f}",
                }
            self.turn_brake_start_time = None
            self.turn_in_progress_sign = None
            return {
                "action_complete": True,
                "reset_imu": True,
                "drive_command": {"speed": int(getattr(state, 'auto_forward_speed', 80)), "turn": 0},
                "stop": False,
                "debug": f"Team 4 left turn complete at IMU {imu_delta:.1f}",
            }

        effective_target = self.turn_target_degrees - self.turn_coast_offset_degrees
        if abs(abs(imu_delta) - effective_target) <= self.turn_tolerance_degrees or imu_delta >= effective_target:
            self.turn_brake_start_time = now
            return {
                "auto_motion": AutoMotion.PAUSED,
                "drive_command": {"speed": 0, "turn": 0},
                "stop": True,
                "debug": f"Team 4 hit left turn target, braking (IMU={imu_delta:.1f})",
            }

        return {
            "auto_motion": AutoMotion.TURNING_LEFT,
            "drive_command": {"speed": 0, "turn": turn_speed},
            "stop": False,
            "debug": f"Team 4 left turn in progress, IMU={imu_delta:.1f}",
        }

    def handle_right_turn(self, state: Any, sensors: Any) -> Dict[str, Any]:
        """
        Implement the RIGHT sign action using IMU heading delta.
        """
        now = float(getattr(sensors, "now", time.time()))
        imu_delta = self._imu_delta(sensors)
        turn_speed = int(getattr(state, "auto_turn_speed", 80))
        if self.turn_in_progress_sign is None:
            self.turn_in_progress_sign = AutoCurrentSign.RIGHT
            # Zero the IMU *before* the turn command goes out so the turn
            # measures a clean 90 deg from the reference, not from a drifted
            # starting angle. reset_imu=True tells the main controller to zero
            # its own IMU too (the inline _imu_reset only fires when Team 4
            # owns the IMU handle directly).
            self._imu_reset()
            return {
                "auto_motion": AutoMotion.TURNING_RIGHT,
                "reset_imu": True,
                "drive_command": {"speed": 0, "turn": -turn_speed},
                "stop": False,
                "debug": "Team 4 starting right turn (IMU zeroed)",
            }

        if self.turn_brake_start_time is not None:
            if now - self.turn_brake_start_time < self.turn_brake_seconds:
                return {
                    "auto_motion": AutoMotion.PAUSED,
                    "drive_command": {"speed": 0, "turn": 0},
                    "stop": True,
                    "debug": f"Team 4 braking after right turn, IMU={imu_delta:.1f}",
                }
            self.turn_brake_start_time = None
            self.turn_in_progress_sign = None
            return {
                "action_complete": True,
                "reset_imu": True,
                "drive_command": {"speed": int(getattr(state, 'auto_forward_speed', 80)), "turn": 0},
                "stop": False,
                "debug": f"Team 4 right turn complete at IMU {imu_delta:.1f}",
            }

        effective_target = self.turn_target_degrees - self.turn_coast_offset_degrees
        if abs(abs(imu_delta) - effective_target) <= self.turn_tolerance_degrees or imu_delta <= -effective_target:
            self.turn_brake_start_time = now
            return {
                "auto_motion": AutoMotion.PAUSED,
                "drive_command": {"speed": 0, "turn": 0},
                "stop": True,
                "debug": f"Team 4 hit right turn target, braking (IMU={imu_delta:.1f})",
            }

        return {
            "auto_motion": AutoMotion.TURNING_RIGHT,
            "drive_command": {"speed": 0, "turn": -turn_speed},
            "stop": False,
            "debug": f"Team 4 right turn in progress, IMU={imu_delta:.1f}",
        }

    def handle_go_straight(self, state: Any) -> Dict[str, Any]:
        """
        Implement the Go Straight sign action.
        """
        return {
            "action_complete": True,
            "reset_imu": True,
            "drive_command": {"speed": int(getattr(state, 'auto_forward_speed', 80)), "turn": 0},
            "stop": False,
            "debug": "Team 4 go-straight action complete",
        }

    def handle_unknown_sign(self, state: Any, sensors: Any) -> Dict[str, Any]:
        """
        Safe fallback when the sign is unknown or no sign was detected at the
        intersection. Pause for 5 seconds, print a terminal warning once, then
        resume driving straight.
        """
        now = float(getattr(sensors, "now", time.time()))
        forward_speed = int(getattr(state, 'auto_forward_speed', 80))

        if self.unknown_start_time is None:
            self.unknown_start_time = now
            if not self.unknown_warning_printed:
                print(
                    "[Team4] WARNING: unknown/undetected sign at intersection "
                    "-- pausing 5s before defaulting to straight",
                    file=sys.stderr,
                )
                self.unknown_warning_printed = True
            return {
                "auto_motion": AutoMotion.PAUSED,
                "drive_command": {"speed": 0, "turn": 0},
                "stop": True,
                "debug": "UNKNOWN SIGN -- pausing 5s before defaulting to straight",
            }

        if now - self.unknown_start_time < self.pause_duration_seconds:
            remaining = self.pause_duration_seconds - (now - self.unknown_start_time)
            return {
                "auto_motion": AutoMotion.PAUSED,
                "drive_command": {"speed": 0, "turn": 0},
                "stop": True,
                "debug": f"UNKNOWN SIGN -- resuming straight in {int(remaining) + 1}s",
            }

        self.unknown_start_time = None
        self.unknown_warning_printed = False
        return {
            "action_complete": True,
            "reset_imu": True,
            "drive_command": {"speed": forward_speed, "turn": 0},
            "stop": False,
            "debug": "Team 4 unknown-sign fallback: proceeding straight",
        }
