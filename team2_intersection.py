from __future__ import annotations

import queue
import signal
import threading
from multiprocessing import shared_memory
from typing import Any, Dict, Optional

import numpy as np

from robot_types import AutoRoadLocation, IntersectionStage

try:
    from ultralytics import YOLO, settings as _yolo_settings
    import ultralytics.utils as _yolo_utils
    import ultralytics.utils.downloads as _yolo_downloads
    _yolo_settings.update({"sync": False})
    _yolo_utils.ONLINE = False
    _yolo_downloads.get_github_assets = lambda *_, **__: ("", [])  # suppress GitHub API calls
    YOLO_AVAILABLE = True
except ImportError:
    YOLO = None
    YOLO_AVAILABLE = False

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
            "intersection_lines": [(200, 400, 260, 290), (430, 400, 370, 290)],
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

    # --- Detection constants ---
    MODEL_PATH            = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "best.pt")
    SIGN_DETECT_CONF      = 0.3   # YOLO confidence threshold
    SIGN_READ_MIN_HEIGHT  = 100    # px — lower bound of reading window; sign_read_ready fires here
    SIGN_HEIGHT_THRESHOLD = 170   # px — upper bound; robot is at intersection, pause_point_ready fires here
    FORWARD_DRIVE_SECONDS = 3   # seconds in PRE_TURN_TRAVEL before turn_point_ready fires

    # assumes sign is 10 inches from the road
    def __init__(self) -> None:
        self.last_debug = "Team 2 initialized"
        self.current_phase = "SEEKING_INTERSECTION"
        self.phase_started_at = 0.0
        self.phase_confirmation_seconds = 0.20

        # YOLO model (lazy-loaded on first frame to avoid fork issues)
        self.model = None

        # Sign detection state
        self.last_sign_height = 0       # bounding-box height of best detection this frame
        self.sign_visible = False        # True if YOLO found a sign this frame
        self.max_sign_height_seen = 0   # peak height since last reset (for drove-past check)

        # Turn-point timer — starts when controller enters PRE_TURN_TRAVEL
        self.forward_drive_started_at = 0.0

        # Per-frame context set at the top of process() for use in milestone methods
        self._current_timestamp = 0.0
        self._current_stage = ""

        # Background YOLO thread — keeps the main loop unblocked while inference runs
        self._yolo_lock = threading.Lock()
        self._frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._pending_frame: Optional[np.ndarray] = None
        self._latest_yolo_result: tuple = (0, "", 0.0, None)
        self._yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self._yolo_thread.start()

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
        # Store per-frame context for milestone methods
        self._current_timestamp = timestamp
        self._current_stage = job.get("intersection_stage", "")

        result: Dict[str, Any] = {
            "frame_id": frame_id,
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

        # Keep a reference to the full frame so _run_yolo can use it regardless of ROI
        self._full_frame = frame_bgr

        detection = self.detect_intersection_features()
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

    def _yolo_worker(self) -> None:
        """Background thread: wait for a new frame, run inference, store result."""
        while not self._stop_event.is_set():
            if not self._frame_event.wait(timeout=0.1):
                continue
            self._frame_event.clear()
            with self._yolo_lock:
                frame = self._pending_frame
            if frame is None:
                continue
            result = self._run_yolo_sync(frame)
            with self._yolo_lock:
                self._latest_yolo_result = result

    def stop(self) -> None:
        """Signal the background YOLO thread to exit and wait for it."""
        self._stop_event.set()
        self._yolo_thread.join(timeout=1.0)

    def _run_yolo_sync(self, frame_bgr: np.ndarray) -> tuple:
        """
        Pure YOLO inference — only called from the background thread.

        Returns (sign_height, label, conf, bbox_xywh) or (0, "", 0.0, None).
        No side effects on instance state.
        """
        if not YOLO_AVAILABLE:
            return 0, "", 0.0, None

        if self.model is None:
            try:
                self.model = YOLO(self.MODEL_PATH)
            except Exception:
                return 0, "", 0.0, None

        try:
            results = self.model(frame_bgr, conf=self.SIGN_DETECT_CONF, verbose=False)[0]
        except Exception:
            return 0, "", 0.0, None

        best_height = 0
        best_label = ""
        best_conf = 0.0
        best_bbox = None

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h = y2 - y1
            if h > best_height:
                best_height = h
                best_conf = float(box.conf)
                best_label = results.names[int(box.cls)]
                best_bbox = (x1, y1, x2 - x1, y2 - y1)

        return best_height, best_label, best_conf, best_bbox

    def _run_yolo(self, frame_bgr: np.ndarray) -> tuple:
        """
        Submit frame to the background thread and return the latest cached result.

        The main loop never blocks on inference; it reads whatever the worker
        finished most recently.  Side effects on sign state are applied here in
        the main thread so all callers see a consistent snapshot.
        """
        with self._yolo_lock:
            self._pending_frame = frame_bgr
        self._frame_event.set()

        with self._yolo_lock:
            best_height, best_label, best_conf, best_bbox = self._latest_yolo_result

        self.last_sign_height = best_height
        self.sign_visible = best_height > 0
        if best_height > self.max_sign_height_seen:
            self.max_sign_height_seen = best_height

        return best_height, best_label, best_conf, best_bbox

    def detect_intersection_features(self) -> Dict[str, Any]:
        """
        Run YOLO sign detection and build the annotation payload.

        YOLO runs on the full frame (stored in self._full_frame) rather than
        the cropped ROI so the sign is visible at any distance.
        """
        sign_height, label, conf, bbox_xywh = self._run_yolo(self._full_frame)

        boxes = [bbox_xywh] if bbox_xywh is not None else []

        if sign_height >= self.SIGN_HEIGHT_THRESHOLD:
            intersection_label = f"STOP — sign h={sign_height}px"
        elif sign_height >= self.SIGN_READ_MIN_HEIGHT:
            intersection_label = f"Reading sign — h={sign_height}px ({label} {conf:.2f})"
        elif sign_height > 0:
            intersection_label = f"Sign approaching — h={sign_height}px"
        else:
            intersection_label = "No sign detected"

        return {
            "intersection_lines": [],
            "intersection_points": [],
            "intersection_boxes": boxes,
            "intersection_label": intersection_label,
            "debug": f"Team 2 phase={self.current_phase} sign_h={sign_height} stage={self._current_stage}",
        }

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
        self.last_sign_height = 0
        self.sign_visible = False
        self.max_sign_height_seen = 0
        self.forward_drive_started_at = 0.0

    def sign_read_ready(self, _detection: Dict[str, Any]) -> bool:
        """
        True while the sign is in the reading window:
            SIGN_READ_MIN_HEIGHT <= height < SIGN_HEIGHT_THRESHOLD

        This gives Team 3 a clean window to classify the sign before the robot
        reaches the stop point.  Once height crosses SIGN_HEIGHT_THRESHOLD the
        robot is too close to read and this drops back to False.
        """
        return self.SIGN_READ_MIN_HEIGHT <= self.last_sign_height < self.SIGN_HEIGHT_THRESHOLD

    def pause_ready(self, _detection: Dict[str, Any]) -> bool:
        """
        True when the sign bounding-box height reaches SIGN_HEIGHT_THRESHOLD,
        meaning the robot has arrived at the intersection stop point.
        """
        return self.last_sign_height >= self.SIGN_HEIGHT_THRESHOLD

    def turn_ready(self, _detection: Dict[str, Any]) -> bool:
        """
        True after the robot has been driving forward inside the intersection
        for FORWARD_DRIVE_SECONDS.

        The timer starts on the first frame where the controller reports
        PRE_TURN_TRAVEL, which is when Team 1 actually begins driving forward
        into the intersection after the pause action completes.  This mirrors
        the standalone script's constant-time forward drive.
        """
        in_pre_turn = self._current_stage == IntersectionStage.PRE_TURN_TRAVEL.value

        if in_pre_turn:
            if self.forward_drive_started_at == 0.0:
                self.forward_drive_started_at = self._current_timestamp
        else:
            # Reset timer whenever we leave PRE_TURN_TRAVEL so it starts fresh
            # if the stage is re-entered.
            self.forward_drive_started_at = 0.0

        return (
            self.forward_drive_started_at > 0.0
            and (self._current_timestamp - self.forward_drive_started_at) >= self.FORWARD_DRIVE_SECONDS
        )

    def cleared_intersection(self, _detection: Dict[str, Any]) -> bool:
        """
        True when the sign disappears after having been tracked.

        Two sub-cases — the intersection_label set in detect_intersection_features
        distinguishes them on the display:

        1. Normal exit: sign gone after turn_point_ready was reached.
        2. Drove past: sign gone but the robot never entered PRE_TURN_TRAVEL,
           meaning it passed the intersection without stopping.
        """
        sign_gone = not self.sign_visible
        was_tracking = self.max_sign_height_seen >= self.SIGN_READ_MIN_HEIGHT
        return sign_gone and was_tracking

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
    _frame_counter: Any,
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
