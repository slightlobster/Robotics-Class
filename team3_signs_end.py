from __future__ import annotations

import queue
import signal
from multiprocessing import shared_memory
from typing import Any, Dict, Optional

import cv2
import numpy as np

from robot_types import AutoCurrentSign, IntersectionStage
from robot_utils import replace_latest_queue_item


class Team3SignsEnd:
    """
    Team 3 responsibilities:

    1. Detect and classify the sign when the robot is at the sign-read portion
       of the intersection flow.
    2. Return bounding-box / ROI data so the display can show what image region
       the robot is using for sign recognition.
    3. Detect the end of the course and request the proper stop/end state.

    IMPORTANT PROCESS MODEL
    -----------------------
    Team 3 runs in its own worker process.

    The main controller sends Team 3:
    - the latest camera image through shared memory
    - a small job dictionary through a queue

    Team 3 sends back:
    - a small result dictionary through a queue

    CONTROLLER CONTRACT
    -------------------
    Team 3 should actively classify signs only during the controller-owned
    SIGN_READ stage.

    Team 3 does not own motion. The controller decides whether the robot is:
    - lane following
    - creeping forward for sign read
    - executing a Team 4 action

    Team 3 may also detect end-of-course. If the end is detected, Team 3
    should return an end-detection result and let the controller stop the run.

    WHAT TEAM 3 RECEIVES IN EACH JOB
    --------------------------------
    Example:

        {
            "type": "PROCESS",
            "frame_id": 51,
            "timestamp": 123.45,
            "roi_xywh": (40, 70, 280, 220),
            "road_location": "At Sign Read Location",
            "auto_mode": True,
        }

    WHAT TEAM 3 RETURNS
    -------------------
    Team 3 may return these fields:

    - "auto_current_sign"
    - "sign_locked"
    - "sign_roi"
    - "sign_bbox"
    - "sign_label"
    - "sign_confidence"
    - "debug"

    EXAMPLE RESULT: searching sign ROI but nothing found yet
    --------------------------------------------------------
        {
            "frame_id": frame_id,
            "sign_roi": (40, 70, 280, 220),
            "sign_bbox": None,
            "sign_label": "",
            "sign_confidence": 0.0,
            "debug": "Scanning sign ROI"
        }

    EXAMPLE RESULT: LEFT sign found
    -------------------------------
        {
            "frame_id": frame_id,
            "auto_current_sign": AutoCurrentSign.LEFT,
            "sign_roi": (40, 70, 280, 220),
            "sign_bbox": (210, 95, 88, 92),
            "sign_label": "LEFT",
            "sign_confidence": 0.91,
            "debug": "LEFT sign detected"
        }

    EXAMPLE RESULT: unknown sign candidate
    --------------------------------------
        {
            "frame_id": frame_id,
            "auto_current_sign": AutoCurrentSign.UNKNOWN,
            "sign_roi": (40, 70, 280, 220),
            "sign_bbox": (205, 92, 84, 88),
            "sign_label": "UNKNOWN",
            "sign_confidence": 0.42,
            "debug": "Candidate found but confidence too low"
        }

    EXAMPLE RESULT: end of course detected
    --------------------------------------
        {
            "frame_id": frame_id,
            "end_detected": True,
            "debug": "End of course detected"
        }

    DESIGN NOTES
    ------------
    Team 3 may use either:
    - OpenCV-based sign recognition
    - model-based sign recognition
    - hybrid approaches

    The framework is generic so students can choose their strategy without
    changing the main controller.
    """

    def __init__(self) -> None:
        self.last_debug = "Team 3 initialized"
        self.sign_lock_threshold = 0.65

    def process(self, frame_bgr: np.ndarray, job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main Team 3 processing entry point.

        INPUTS
        ------
        frame_bgr:
            Latest BGR image copied from shared memory.

        job:
            Lightweight metadata from the main controller.

        RETURNS
        -------
        A lightweight result dictionary for the main controller.
        """
        frame_id = int(job.get("frame_id", 0))
        auto_mode = bool(job.get("auto_mode", False))
        stage_name = job.get("intersection_stage", IntersectionStage.TRAVEL_LANE.value)
        roi_xywh = job.get("roi_xywh")

        result: Dict[str, Any] = {
            "frame_id": frame_id,
            "sign_roi": roi_xywh,
            "sign_bbox": None,
            "sign_label": "",
            "sign_confidence": 0.0,
            "debug": "",
        }

        if frame_bgr is None or frame_bgr.size == 0:
            result["debug"] = "Team 3: no frame available"
            return result

        if not auto_mode:
            result["debug"] = "Team 3 idle: auto mode is off"
            return result

        stage = self.parse_stage(stage_name)

        end_result = self.detect_end_of_course(frame_bgr, frame_id)
        if end_result is not None:
            return end_result

        if stage != IntersectionStage.SIGN_READ:
            result["debug"] = f"Team 3 idle: stage is {stage.value}"
            return result

        work_frame, roi_xywh = self.extract_sign_roi(frame_bgr, roi_xywh)
        result["sign_roi"] = roi_xywh

        detection = self.detect_sign(work_frame, roi_xywh)
        result.update(detection)

        if not result.get("debug"):
            result["debug"] = "Team 3 processed sign frame"

        return result

    def extract_sign_roi(
        self,
        frame_bgr: np.ndarray,
        roi_xywh: Optional[tuple[int, int, int, int]],
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """
        Determine the region Team 3 will inspect for sign recognition.

        If the controller has already provided an ROI, use it.
        Otherwise create a default ROI on the left side of the road, matching
        the project expectation that the sign is placed to the left.
        """
        h, w = frame_bgr.shape[:2]

        if roi_xywh is None:
            roi_xywh = self.default_sign_roi(w, h)

        x, y, rw, rh = map(int, roi_xywh)
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        rw = max(1, min(rw, w - x))
        rh = max(1, min(rh, h - y))

        roi = frame_bgr[y:y + rh, x:x + rw].copy()
        return roi, (x, y, rw, rh)

    def default_sign_roi(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        """
        Starting ROI suggestion for Team 3.

        This ROI emphasizes the left portion of the image where the course sign
        is expected to appear.
        """
        roi_w = int(frame_width * 0.40)
        roi_h = int(frame_height * 0.45)
        roi_x = int(frame_width * 0.02)
        roi_y = int(frame_height * 0.15)
        return roi_x, roi_y, roi_w, roi_h

    def detect_sign(
        self,
        roi_bgr: np.ndarray,
        roi_xywh: tuple[int, int, int, int],
    ) -> Dict[str, Any]:
        """
        Starting framework for Team 3 sign recognition.

        Students should replace the placeholder logic here with either:
        - OpenCV processing
        - model inference
        - a hybrid approach

        This starter code performs a very simple contour-based candidate search
        so students can see how ROI, bounding-box, label, confidence, and
        sign_locked should flow back to the main controller.

        sign_locked is the important controller contract:
        - False => keep creeping / keep trying to classify
        - True  => controller may leave SIGN_READ and continue the workflow
        """
        x0, y0, _, _ = roi_xywh

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_box = None
        best_area = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > best_area and area >= 400:
                best_area = area
                best_box = (x0 + x, y0 + y, w, h)

        if best_box is None:
            return {
                "sign_bbox": None,
                "sign_label": "",
                "sign_confidence": 0.0,
                "debug": "Team 3 scanning sign ROI",
            }

        sign_guess, confidence = self.classify_sign_candidate(roi_bgr, best_box, roi_xywh)

        result: Dict[str, Any] = {
            "auto_current_sign": sign_guess,
            "sign_bbox": best_box,
            "sign_label": sign_guess.value.upper() if sign_guess != AutoCurrentSign.GO_STRAIGHT else "GO STRAIGHT",
            "sign_confidence": confidence,
            "sign_locked": confidence >= self.sign_lock_threshold and sign_guess != AutoCurrentSign.UNKNOWN,
            "debug": f"Team 3 sign detection: {sign_guess.value} ({confidence:.2f})",
        }
        return result

    def classify_sign_candidate(
        self,
        roi_bgr: np.ndarray,
        global_bbox: tuple[int, int, int, int],
        roi_xywh: tuple[int, int, int, int],
    ) -> tuple[AutoCurrentSign, float]:
        """
        Placeholder sign classifier.

        Students should replace this with real sign-classification logic.

        This starter version uses the candidate box shape only, purely so the
        framework demonstrates how classification results should be returned.

        The assignment requires support for LEFT, RIGHT, PAUSE, GO_STRAIGHT,
        and UNKNOWN. This placeholder only makes coarse guesses so students
        can focus on replacing it with a real recognizer.
        """
        xg, yg, bw, bh = global_bbox
        x0, y0, _, _ = roi_xywh
        _ = roi_bgr

        x_local = xg - x0
        y_local = yg - y0
        _ = x_local, y_local

        aspect = bw / max(1, bh)

        if 0.8 <= aspect <= 1.2:
            return AutoCurrentSign.UNKNOWN, 0.40
        if aspect < 0.8:
            return AutoCurrentSign.LEFT, 0.55
        return AutoCurrentSign.RIGHT, 0.55

    def detect_end_of_course(self, frame_bgr: np.ndarray, frame_id: int) -> Optional[Dict[str, Any]]:
        """
        Starting framework for end-of-course detection.

        Students should replace this with their own end-detection logic.

        This starter implementation never triggers the end condition, but shows
        the exact structure Team 3 should return when the end is found.
        """
        _ = frame_bgr
        _ = frame_id
        return None

    @staticmethod
    def parse_stage(name: str) -> IntersectionStage:
        for value in IntersectionStage:
            if value.value == name:
                return value
        return IntersectionStage.TRAVEL_LANE


# -----------------------------------------------------------------------------
# Worker entry point used by main_auto_controller.py
# -----------------------------------------------------------------------------

def run_sign_worker(
    shm_name: str,
    image_shape: tuple[int, int, int],
    frame_counter: Any,
    canvas_lock: Any,
    input_q: Any,
    result_q: Any,
) -> None:
    """
    Team 3 worker-process entry point.

    This function is started by the main controller in a separate process.

    HOW IT WORKS
    ------------
    1. Attach to the shared-memory image buffer.
    2. Wait for a lightweight job dict from input_q.
    3. Copy the latest image from shared memory.
    4. Run Team 3 processing.
    5. Send a lightweight result dict back to result_q.
    """
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    shm = shared_memory.SharedMemory(name=shm_name, create=False)
    image = np.ndarray(image_shape, dtype=np.uint8, buffer=shm.buf)
    team = Team3SignsEnd()

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
