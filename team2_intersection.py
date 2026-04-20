from __future__ import annotations

import queue
import signal
from multiprocessing import shared_memory
from typing import Any, Dict, Optional

import cv2
import numpy as np

from robot_types import AutoRoadLocation
from robot_utils import replace_latest_queue_item


class Team2Intersection:
    """
    Team 2 responsibilities:

    1. Detect when the robot is approaching / entering an intersection.
    2. Report milestone cues that describe where the robot is relative to the
       intersection workflow.
    3. Return optional display annotation data so the live screen shows HOW the
       team believes the intersection is being detected.

    CONTROLLER CONTRACT
    -------------------
    Team 2 owns perception of intersection position, but not final workflow
    authority.

    Team 2 should report milestone-style answers such as:
    - sign_read_ready
    - pause_point_ready
    - turn_point_ready

    The controller then decides whether those cues cause stage transitions.
    This keeps Team 2 focused on the hard perception problem while the
    controller handles the handoff to Team 3 and Team 4.

    IMPORTANT PROCESS MODEL
    -----------------------
    Team 2 does NOT run in the main controller process.
    It runs in its own worker process.

    The main controller sends Team 2:
    - the latest camera image through shared memory
    - a small job dictionary through a queue

    Team 2 sends back:
    - a small result dictionary through a queue

    Team 2 should never directly modify the main controller state.
    It only returns results for the main controller to apply.

    WHAT TEAM 2 RECEIVES IN EACH JOB
    --------------------------------
    The worker receives a job dictionary such as:

        {
            "type": "PROCESS",
            "frame_id": 25,
            "timestamp": 123.45,
            "roi_xywh": (180, 220, 280, 180),
            "road_location": "In Travel Lane",
            "auto_mode": True,
        }

    The image itself is read from shared memory.

    WHAT TEAM 2 RETURNS
    -------------------
    Team 2 may return these fields:

    - "sign_read_ready"
    - "pause_point_ready"
    - "turn_point_ready"
    - "intersection_cleared"
    - "intersection_roi"
    - "intersection_boxes"
    - "intersection_lines"
    - "intersection_points"
    - "intersection_label"
    - "debug"

    EXAMPLE RESULT: no state change, only show the ROI being searched
    ---------------------------------------------------------------
        {
            "frame_id": frame_id,
            "intersection_roi": (180, 220, 280, 180),
            "intersection_label": "Searching for intersection cues",
            "debug": "Scanning lower road ROI"
        }

    EXAMPLE RESULT: sign-read milestone candidate
    -------------------------------------------------
        {
            "frame_id": frame_id,
            "sign_read_ready": True,
            "intersection_roi": (180, 220, 280, 180),
            "intersection_lines": [(200, 400, 260, 290), (430, 400, 370, 290)],
            "intersection_points": [(320, 290)],
            "intersection_label": "Road edges opening into intersection",
            "debug": "Intersection candidate detected from lane-edge geometry"
        }

    EXAMPLE RESULT: pause-point milestone candidate
    ---------------------------------------------------------
        {
            "frame_id": frame_id,
            "pause_point_ready": True,
            "intersection_roi": (220, 240, 180, 120),
            "intersection_boxes": [(260, 260, 60, 40)],
            "intersection_label": "Intersection road marker detected",
            "debug": "Center intersection marker found"
        }

    DESIGN NOTES
    ------------
    Team 2 may choose different strategies, for example:
    - road edges start curving outward
    - specific road pattern / marker appears
    - intersection center texture appears
    - combined cue approach

    The framework is intentionally generic so students can choose their own
    detection strategy without changing the controller.
    """

    def __init__(self) -> None:
        self.last_debug = "Team 2 initialized"
        self.current_phase = "SEEKING_INTERSECTION"
        self.phase_started_at = 0.0
        self.phase_confirmation_seconds = 0.20

    def process(self, frame_bgr: np.ndarray, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main Team 2 processing entry point.

        INPUTS
        ------
        frame_bgr:
            Latest BGR image copied from shared memory.

        job:
            Lightweight metadata from the main controller.

        RETURNS
        -------
        A result dictionary to send back to the main controller.

        Team 2 should return lightweight metadata only.
        It should not return the whole image.
        """
        frame_id = int(job.get("frame_id", 0))
        auto_mode = bool(job.get("auto_mode", False))
        timestamp = float(job.get("timestamp", 0.0))
        road_location_name = job.get("road_location", AutoRoadLocation.IN_TRAVEL_LANE.value)
        roi_xywh = job.get("roi_xywh")

        result: Dict[str, Any] = {
            "frame_id": frame_id,
            "intersection_roi": roi_xywh,
            "intersection_boxes": [],
            "intersection_lines": [],
            "intersection_points": [],
            "intersection_label": "",
            "debug": "",
        }

        if not auto_mode:
            self.reset_cycle("Auto mode is off")
            result["debug"] = "Team 2 idle: auto mode is off"
            return result

        road_location = self.parse_road_location(road_location_name)

        if road_location in (AutoRoadLocation.AT_START, AutoRoadLocation.AT_END):
            self.reset_cycle(f"Road location is {road_location.value}")
            result["debug"] = f"Team 2 idle: road location is {road_location.value}"
            return result

        if frame_bgr is None or frame_bgr.size == 0:
            result["debug"] = "Team 2: no frame available"
            return result

        work_frame, roi_xywh = self.extract_intersection_roi(frame_bgr, roi_xywh)
        result["intersection_roi"] = roi_xywh

        detection = self.detect_intersection_features(work_frame, roi_xywh)
        result.update(detection)

        result["sign_read_ready"] = self.sign_read_ready(detection)
        result["pause_point_ready"] = self.pause_ready(detection)
        result["turn_point_ready"] = self.turn_ready(detection)
        result["intersection_cleared"] = self.cleared_intersection(detection)

        if not result.get("debug"):
            result["debug"] = "Team 2 processed frame"

        return result

    def extract_intersection_roi(
        self,
        frame_bgr: np.ndarray,
        roi_xywh: Optional[tuple[int, int, int, int]],
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """
        Determine the region Team 2 will inspect for intersection cues.

        If the controller has already provided an ROI, use it.
        Otherwise build a default ROI in the lower-middle part of the image.
        """
        h, w = frame_bgr.shape[:2]

        if roi_xywh is None:
            roi_xywh = self.default_intersection_roi(w, h)

        x, y, rw, rh = map(int, roi_xywh)
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))

        roi = frame_bgr[y:y + rh, x:x + rw].copy()
        return roi, (x, y, rw, rh)

    def default_intersection_roi(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        """
        Starting ROI suggestion for Team 2.

        This ROI emphasizes the lower central part of the image where road-edge
        changes or an intersection marker may become visible.
        """
        roi_w = int(frame_width * 0.45)
        roi_h = int(frame_height * 0.30)
        roi_x = int((frame_width - roi_w) / 2)
        roi_y = int(frame_height * 0.50)
        return roi_x, roi_y, roi_w, roi_h

    def detect_intersection_features(
        self,
        roi_bgr: np.ndarray,
        roi_xywh: tuple[int, int, int, int],
    ) -> Dict[str, Any]:
        """
        Starting framework for Team 2 intersection detection.

        Students should replace the placeholder logic here.

        The framework supports several annotation styles:
        - intersection_lines
        - intersection_points
        - intersection_boxes
        - intersection_label

        This starter implementation performs a very simple edge + Hough-lines
        pass so students can see how data should flow back.
        """
        x0, y0, _, _ = roi_xywh

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=25,
            maxLineGap=15,
        )

        intersection_lines = []
        if raw_lines is not None:
            for line in raw_lines[:6]:
                x1, y1, x2, y2 = line[0]
                intersection_lines.append((x0 + x1, y0 + y1, x0 + x2, y0 + y2))

        center_point = (x0 + roi_bgr.shape[1] // 2, y0 + roi_bgr.shape[0] // 2)

        result: Dict[str, Any] = {
            "intersection_lines": intersection_lines,
            "intersection_points": [center_point],
            "intersection_boxes": [],
            "intersection_label": "Scanning intersection ROI",
            "debug": f"Team 2 phase={self.current_phase} lines={len(intersection_lines)}",
        }

        return result

    def advance_phase_if_ready(
        self,
        current_location: AutoRoadLocation,
        detection: Dict[str, Any],
        now: float,
    ) -> Optional[AutoRoadLocation]:
        """
        Optional internal staging helper for Team 2's own perception logic.

        The controller interface for Team 2 is the milestone booleans returned
        from process(). This method is useful if Team 2 wants internal memory
        while deciding when those milestones should become true.
        """
        self.sync_phase_with_controller(current_location)

        if self.current_phase == "SEEKING_INTERSECTION":
            if self.confirm_transition(self.sign_read_ready(detection), now):
                self.set_phase("SIGN_READ_READY", now)
                return AutoRoadLocation.AT_SIGN_READ_LOCATION
            return None

        if self.current_phase == "SIGN_READ_READY":
            if self.confirm_transition(self.pause_ready(detection), now):
                self.set_phase("PAUSE_READY", now)
                return AutoRoadLocation.AT_PAUSE_LOCATION
            return None

        if self.current_phase == "PAUSE_READY":
            if self.confirm_transition(self.turn_ready(detection), now):
                self.set_phase("TURN_READY", now)
                return AutoRoadLocation.AT_TURN_LOCATION
            return None

        if self.current_phase == "TURN_READY":
            if self.confirm_transition(self.cleared_intersection(detection), now):
                self.set_phase("SEEKING_INTERSECTION", now)
                return AutoRoadLocation.IN_TRAVEL_LANE
            return None

        return None

    def sync_phase_with_controller(self, current_location: AutoRoadLocation) -> None:
        """
        Keep Team 2's internal phase aligned with the controller's accepted state.

        This avoids drifting if the controller resets auto mode or restarts a run.
        """
        if current_location == AutoRoadLocation.IN_TRAVEL_LANE and self.current_phase == "TURN_READY":
            self.current_phase = "SEEKING_INTERSECTION"
        elif current_location == AutoRoadLocation.AT_SIGN_READ_LOCATION and self.current_phase == "SEEKING_INTERSECTION":
            self.current_phase = "SIGN_READ_READY"
        elif current_location == AutoRoadLocation.AT_PAUSE_LOCATION and self.current_phase in {"SEEKING_INTERSECTION", "SIGN_READ_READY"}:
            self.current_phase = "PAUSE_READY"
        elif current_location == AutoRoadLocation.AT_TURN_LOCATION and self.current_phase != "TURN_READY":
            self.current_phase = "TURN_READY"

    def confirm_transition(self, condition_met: bool, now: float) -> bool:
        """
        Require a cue to remain true briefly before advancing phases.
        """
        if not condition_met:
            self.phase_started_at = now
            return False
        if self.phase_started_at == 0.0:
            self.phase_started_at = now
            return False
        return (now - self.phase_started_at) >= self.phase_confirmation_seconds

    def set_phase(self, phase_name: str, now: float) -> None:
        self.current_phase = phase_name
        self.phase_started_at = now

    def reset_cycle(self, debug_message: str) -> None:
        self.current_phase = "SEEKING_INTERSECTION"
        self.phase_started_at = 0.0
        self.last_debug = debug_message

    def sign_read_ready(self, detection: Dict[str, Any]) -> bool:
        """
        Placeholder cue for "close enough to start sign reading".
        """
        return len(detection.get("intersection_lines", []) or []) >= 2

    def pause_ready(self, detection: Dict[str, Any]) -> bool:
        """
        Placeholder cue for "robot has reached the pause point".
        """
        return len(detection.get("intersection_lines", []) or []) >= 3

    def turn_ready(self, detection: Dict[str, Any]) -> bool:
        """
        Placeholder cue for "robot is centered enough in the intersection to turn".
        """
        return len(detection.get("intersection_lines", []) or []) >= 4

    def cleared_intersection(self, detection: Dict[str, Any]) -> bool:
        """
        Placeholder cue for "robot has exited the intersection and is back in lane".
        """
        return len(detection.get("intersection_lines", []) or []) <= 1

    @staticmethod
    def parse_road_location(name: str) -> AutoRoadLocation:
        for value in AutoRoadLocation:
            if value.value == name:
                return value
        return AutoRoadLocation.IN_TRAVEL_LANE


# -----------------------------------------------------------------------------
# Worker entry point used by main_auto_controller.py
# -----------------------------------------------------------------------------

def run_intersection_worker(
    shm_name: str,
    image_shape: tuple[int, int, int],
    frame_counter: Any,
    canvas_lock: Any,
    input_q: Any,
    result_q: Any,
) -> None:
    """
    Team 2 worker-process entry point.

    This function is started by the main controller in a separate process.

    HOW IT WORKS
    ------------
    1. Attach to the shared-memory image buffer.
    2. Wait for a lightweight job dict from input_q.
    3. Copy the latest image from shared memory.
    4. Run Team 2 processing.
    5. Send a lightweight result dict back to result_q.

    The queue uses latest-item semantics, so stale work can be replaced by newer
    work. This keeps the perception path responsive.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    shm = shared_memory.SharedMemory(name=shm_name, create=False)
    image = np.ndarray(image_shape, dtype=np.uint8, buffer=shm.buf)
    team = Team2Intersection()

    try:
        while True:
            try:
                job = input_q.get(timeout=0.05)
            except queue.Empty:
                continue

            if isinstance(job, dict) and job.get("type") == "QUIT":
                return

            with canvas_lock:
                worker_frame = image.copy()

            result = team.process(worker_frame, job)
            replace_latest_queue_item(result_q, result)
    except KeyboardInterrupt:
        return

    finally:
        try:
            shm.close()
        except Exception:
            pass
