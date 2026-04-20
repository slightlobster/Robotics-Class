"""
Shared enums and dataclasses used by the autonomous controller and team files.

IMPORTANT CONTRACT
------------------
The controller is the orchestration authority.

Teams should treat these fields in two groups:

1. Controller-owned workflow fields
   - intersection_stage
   - motion_mode
   - auto_mode

   Teams may READ these to understand what part of the course is active.
   In general, teams should not directly redefine the workflow themselves.

2. Team-produced perception / action fields
   - auto_current_sign
   - sign_locked
   - requested_speed / requested_turn
   - sign / intersection annotation fields

The controller uses the workflow fields to decide which team currently owns
motion:
- TEAM1_LANE_FOLLOW => Team 1 drive command is used
- CREEP_FORWARD     => controller drives straight forward at a fixed low speed
- TEAM4_ACTION      => Team 4 drive command is used

Future AI threads should preserve this separation. Keep workflow control in the
controller and keep the team files focused on their specific responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class AutoMotion(Enum):
    MOVING_FORWARD = "Moving Forward"
    PAUSED = "Paused"
    TURNING_LEFT = "Turning Left"
    TURNING_RIGHT = "Turning Right"


class AutoRoadLocation(Enum):
    AT_START = "At Start"
    IN_TRAVEL_LANE = "In Travel Lane"
    AT_SIGN_READ_LOCATION = "At Sign Read Location"
    AT_PAUSE_LOCATION = "At Pause Location"
    AT_TURN_LOCATION = "At Turn Location"
    AT_END = "At End"


class AutoCurrentSign(Enum):
    UNKNOWN = "Unknown"
    LEFT = "Left"
    RIGHT = "Right"
    PAUSE = "Pause"
    GO_STRAIGHT = "Go Straight"


class IntersectionStage(Enum):
    """Controller-owned workflow stage for one intersection cycle."""
    TRAVEL_LANE = "Travel Lane"
    SIGN_READ = "Sign Read"
    POST_SIGN_TRAVEL = "Post Sign Travel"
    PAUSE_ACTION = "Pause Action"
    PRE_TURN_TRAVEL = "Pre Turn Travel"
    TURN_ACTION = "Turn Action"
    END_REACHED = "End Reached"


class MotionMode(Enum):
    """Controller-owned choice of who currently owns motor commands."""
    TEAM1_LANE_FOLLOW = "Team 1 Lane Follow"
    CREEP_FORWARD = "Creep Forward"
    TEAM4_ACTION = "Team 4 Action"


@dataclass
class RobotState:
    """
    Shared state snapshot consumed by all teams.

    Ownership notes:
    - intersection_stage / motion_mode are controller-owned.
    - auto_motion / auto_road_location are derived display/debug fields.
      They are useful for overlays and student understanding, while
      intersection_stage + motion_mode define the workflow contract.
    - requested_speed / requested_turn are the controller's currently selected
      motor command inputs after applying the active team's output.
    """
    auto_mode: bool = False
    auto_motion: AutoMotion = AutoMotion.PAUSED
    auto_road_location: AutoRoadLocation = AutoRoadLocation.AT_START
    auto_current_sign: AutoCurrentSign = AutoCurrentSign.UNKNOWN
    intersection_stage: IntersectionStage = IntersectionStage.TRAVEL_LANE
    motion_mode: MotionMode = MotionMode.TEAM1_LANE_FOLLOW
    auto_current_imu_direction: float = 0.0
    auto_forward_speed: int = 80
    auto_turn_speed: int = 80
    auto_sign_implemented: bool = False
    sign_locked: bool = False
    requested_speed: int = 0
    requested_turn: int = 0
    stop_requested: bool = True
    debug_message: str = "System initialized"
    sign_roi: Optional[tuple[int, int, int, int]] = None
    sign_bbox: Optional[tuple[int, int, int, int]] = None
    sign_label: str = ""
    sign_confidence: float = 0.0
    intersection_roi: Optional[tuple[int, int, int, int]] = None
    intersection_boxes: list = field(default_factory=list)
    intersection_lines: list = field(default_factory=list)
    intersection_points: list = field(default_factory=list)
    intersection_label: str = ""


@dataclass
class SensorData:
    """Live sensor snapshot passed into team update functions."""
    now: float = 0.0
    controller_buttons: Dict[str, bool] = field(default_factory=dict)
    controller_left_y: int = 0
    controller_left_x: int = 0
    imu_delta: float = 0.0
    frame: Any = None
