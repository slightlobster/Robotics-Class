from __future__ import annotations

import os
import queue
import signal
import time
from collections.abc import Sequence
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict
import cv2
import numpy as np

from robot_types import AutoCurrentSign, IntersectionStage
from robot_utils import replace_latest_queue_item

# YOLO weights: load best-v2.pt from this directory (unless TEAM3_YOLO_MODEL is set)
TEAM3_WEIGHTS_BASENAME = "best-v2.pt"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.environ.get("TEAM3_YOLO_MODEL", os.path.join(_SCRIPT_DIR, TEAM3_WEIGHTS_BASENAME))
YOLO_CONF = 0.45
SPEECH_COOLDOWN_SEC = 1.25
END_STOP_DELAY_SEC = 5.0
# Stop the run when the end-sign box is this large (px²) OR END_STOP_DELAY_SEC has passed since first seen.
END_STOP_AREA_PX2 = 15000

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore[misc, assignment]

try:
    from usb_sound_controller import USB_SoundController
except ImportError:  # pragma: no cover
    USB_SoundController = None  # type: ignore[misc, assignment]


# --- YOLO sign helpers ---


class SignDetection(TypedDict):
    """One detection from :func:`detect_signs_in_frame`."""

    sign: str
    confidence: float
    xyxy: list[float]


def load_model(weights_path: str | Path) -> Any:
    """Load a YOLO ``.pt`` checkpoint and return the model."""
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")
    return YOLO(str(weights_path))


def detect_signs_in_frame(
    model: Any,
    frame: np.ndarray[Any, Any],
    *,
    conf: float = 0.25, # adjust if necessary
    iou: float = 0.7, # adjust if necessary
    device: str | None = None,
    imgsz: int | None = None,
    verbose: bool = False,
) -> list[SignDetection]:
    """Run one BGR image (``H x W x 3``, ``uint8``) through the model.

    Returns every retained box after NMS: class name (lowercase), confidence, and
    ``xyxy`` pixel coordinates ``[x1, y1, x2, y2]``.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR uint8 array of shape (H, W, 3)")

    kwargs: dict[str, Any] = {"conf": conf, "iou": iou, "verbose": verbose}
    if device is not None:
        kwargs["device"] = device
    if imgsz is not None:
        kwargs["imgsz"] = imgsz

    results = model.predict(frame, **kwargs)
    if not results:
        return []

    r = results[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return []

    names: dict[int, str] = dict(model.names)
    out: list[SignDetection] = []

    xyxy_t = boxes.xyxy.detach().float().cpu().numpy()
    conf_t = boxes.conf.detach().float().cpu().numpy()
    cls_t = boxes.cls.detach().long().cpu().numpy()

    for i in range(len(boxes)):
        cid = int(cls_t[i])
        raw = names.get(cid, str(cid))
        sign = raw.strip().lower()
        xyxy = [float(x) for x in xyxy_t[i].tolist()]
        out.append(SignDetection(sign=sign, confidence=float(conf_t[i]), xyxy=xyxy))

    return out


def _area_xyxy_sign(xyxy: Sequence[float]) -> float:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bounding_box_area_xyxy(xyxy: Sequence[float]) -> float:
    """Pixel area of an axis-aligned box from ``[x1, y1, x2, y2]`` (YOLO ``xyxy`` style)."""
    return _area_xyxy_sign(xyxy)


def bounding_box_area_xywh(xywh: tuple[int, int, int, int] | Sequence[int]) -> float:
    """Pixel area from ``(x, y, width, height)`` (OpenCV ``boundingRect`` / controller ``sign_bbox`` style)."""
    _x, _y, w, h = (int(v) for v in xywh)
    return float(max(0, w) * max(0, h))


def _normalize_sign_for_closest(raw: str) -> str | None:
    """Map a YOLO class string to one of ``right|left|stop|straight|end``, or ``None``."""
    r = raw.replace("_", " ").strip().lower()
    compact = r.replace(" ", "")
    if "end" in r and ("course" in r or r in ("end", "finish", "done")):
        return "end"
    if compact == "endcourse" or r == "end":
        return "end"
    if "stop" in r or r == "pause":
        return "stop"
    if "left" in r:
        return "left"
    if "right" in r:
        return "right"
    if "straight" in r or "forward" in r or r in ("go", "go straight", "gostraight"):
        return "straight"
    return None


def closest_sign_detection(detections: Sequence[SignDetection]) -> SignDetection | None:
    """Return the detection with the LARGEST box among navigable sign classes."""
    allowed = frozenset({"right", "left", "stop", "straight", "end"})
    best_key: tuple[float, float, int] | None = None
    best_det: SignDetection | None = None

    for idx, det in enumerate(detections):
        sign = _normalize_sign_for_closest(det["sign"])
        if sign is None or sign not in allowed:
            continue
        area = _area_xyxy_sign(det["xyxy"])
        conf = float(det["confidence"])
        key = (area, conf, -idx)
        if best_key is None or key > best_key:
            best_key = key
            best_det = det

    return best_det


def closest_sign_name(detections: Sequence[SignDetection]) -> str | None:
    """Return the canonical sign name for :func:`closest_sign_detection`, or ``None``."""
    det = closest_sign_detection(detections)
    if det is None:
        return None
    return _normalize_sign_for_closest(det["sign"])


def _picamera_array_to_bgr(arr: np.ndarray) -> np.ndarray:
    """Convert Picamera2 main-stream arrays to BGR for YOLO (layout varies by firmware)."""
    if arr is None or arr.size == 0 or arr.ndim != 3:
        return arr
    if arr.shape[2] == 3:
        return arr
    if arr.shape[2] != 4:
        return arr
    a = arr
    for conv in (cv2.COLOR_BGRA2BGR, cv2.COLOR_RGBA2BGR):
        try:
            out = cv2.cvtColor(a, conv)
            if out is not None and out.size:
                return out
        except Exception:
            continue
    rgb = a[:, :, 1:4] if a.shape[2] >= 4 else a[:, :, :3]
    try:
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return a[:, :, :3].copy()


def _xyxy_global_from_roi_local(
    xyxy: tuple[float, float, float, float], roi_origin_xy: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Convert ROI-local ``xyxy`` to full-frame ``(x, y, width, height)`` integers."""
    x0, y0 = roi_origin_xy
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    gx1 = int(round(x1)) + x0
    gy1 = int(round(y1)) + y0
    gx2 = int(round(x2)) + x0
    gy2 = int(round(y2)) + y0
    gw = max(1, gx2 - gx1)
    gh = max(1, gy2 - gy1)
    return gx1, gy1, gw, gh


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
        # YOLO + speech
        self._yolo_model = None
        self._yolo_load_error: Optional[str] = None
        self._yolo_cache_frame_id: Optional[int] = None
        self._yolo_results = None
        self._sound = None
        self._last_speech_time = 0.0
        self._last_speech_key = ""
        self._end_first_seen_mono: Optional[float] = None
        self._end_spoken = False
        self._cached_end_sign_area: Optional[float] = None
        self._team3_setup_yolo_and_speech()

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
        job = dict(job) if isinstance(job, dict) else {}
        if job.get("type") == "SIGN_READ":
            job["type"] = "PROCESS"
            job.setdefault("auto_mode", True)
            job.setdefault("intersection_stage", IntersectionStage.SIGN_READ.value)
            seq = int(getattr(self, "_sign_read_frame_seq", 0)) + 1
            self._sign_read_frame_seq = seq
            job["frame_id"] = seq

        if frame_bgr is not None and frame_bgr.size and frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
            frame_bgr = _picamera_array_to_bgr(frame_bgr)

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
            self._end_first_seen_mono = None
            self._end_spoken = False
            self._cached_end_sign_area = None
            result["debug"] = "Team 3 idle: auto mode is off"
            return result

        self._ensure_yolo_detections(frame_bgr, frame_id)

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
        # w and h are percent of width/height covered
        # x = 0 is left edge x = 1 is right edge, y = 0 is top y = 1 is bottom
        roi_w = int(frame_width * 0.50)
        roi_h = int(frame_height * 0.98)
        roi_x = int(frame_width * 0.01)
        roi_y = int(frame_height * 0.01)
        return roi_x, roi_y, roi_w, roi_h

    def detect_sign(
        self,
        roi_bgr: np.ndarray,
        roi_xywh: tuple[int, int, int, int],
    ) -> Dict[str, Any]:
        """
        YOLO sign recognition (best-with-end.pt): stop, go straight, turn left, turn right.

        sign_locked is the important controller contract:
        - False => keep creeping / keep trying to classify
        - True  => controller may leave SIGN_READ and continue the workflow
        """
        if self._yolo_load_error:
            return {
                "sign_bbox": None,
                "sign_label": "",
                "sign_confidence": 0.0,
                "debug": self._yolo_load_error,
            }
        if self._yolo_model is None:
            return {
                "sign_bbox": None,
                "sign_label": "",
                "sign_confidence": 0.0,
                "debug": "Team 3: no YOLO model",
            }

        x0, y0, rw, rh = roi_xywh

        # ROI inference + largest-area sign among right/left/stop/straight/end.
        if YOLO is not None:
            try:
                dets = detect_signs_in_frame(
                    self._yolo_model,
                    roi_bgr,
                    conf=YOLO_CONF,
                    verbose=False,
                )
            except Exception as exc:
                return {
                    "sign_bbox": None,
                    "sign_label": "",
                    "sign_confidence": 0.0,
                    "debug": f"Team 3 YOLO ROI predict failed ({exc})",
                }

            chosen = closest_sign_detection(dets)
            if chosen is None:
                return {
                    "sign_bbox": None,
                    "sign_label": "",
                    "sign_confidence": 0.0,
                    "debug": "Team 3 scanning sign ROI (YOLO, no eligible sign)",
                }

            raw = chosen["sign"]
            mapped = self._map_yolo_sign_to_auto(raw)
            conf = float(chosen["confidence"])
            best_box = _xyxy_global_from_roi_local(
                (chosen["xyxy"][0], chosen["xyxy"][1], chosen["xyxy"][2], chosen["xyxy"][3]),
                (x0, y0),
            )

            if mapped is None and self._yolo_name_is_end(raw):
                self._speak_command("END", key="END")
                return {
                    "sign_bbox": best_box,
                    "sign_label": "END",
                    "sign_confidence": conf,
                    "sign_locked": False,
                    "debug": f"Team 3 YOLO (closest/largest): end ({conf:.2f})",
                }

            if mapped is None:
                return {
                    "sign_bbox": None,
                    "sign_label": "",
                    "sign_confidence": 0.0,
                    "debug": f"Team 3: closest sign {raw!r} not mapped to AutoCurrentSign",
                }

            label_str = self._command_phrase_for_sign(mapped)
            self._speak_command(label_str, key=label_str)

            return {
                "auto_current_sign": mapped,
                "sign_bbox": best_box,
                "sign_label": label_str,
                "sign_confidence": conf,
                "sign_locked": conf >= self.sign_lock_threshold and mapped != AutoCurrentSign.UNKNOWN,
                "debug": f"Team 3 YOLO (closest/largest): {mapped.value} ({conf:.2f})",
            }

        if self._yolo_results is None:
            return {
                "sign_bbox": None,
                "sign_label": "",
                "sign_confidence": 0.0,
                "debug": "Team 3: no YOLO results",
            }

        best: Optional[tuple[float, tuple[int, int, int, int], AutoCurrentSign, str]] = None

        boxes = self._yolo_results.boxes
        if boxes is None or len(boxes) == 0:
            return {
                "sign_bbox": None,
                "sign_label": "",
                "sign_confidence": 0.0,
                "debug": "Team 3 scanning sign ROI (YOLO)",
            }

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            raw = str(self._yolo_model.names[cls_id]).lower()
            if self._yolo_name_is_end(raw):
                continue
            mapped = self._map_yolo_sign_to_auto(raw)
            if mapped is None:
                continue
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            gx1, gy1 = int(round(x1)), int(round(y1))
            gx2, gy2 = int(round(x2)), int(round(y2))
            cx = (gx1 + gx2) // 2
            cy = (gy1 + gy2) // 2
            if not (x0 <= cx <= x0 + rw and y0 <= cy <= y0 + rh):
                continue
            gw, gh = max(1, gx2 - gx1), max(1, gy2 - gy1)
            if best is None or conf > best[0]:
                label_str = self._command_phrase_for_sign(mapped)
                best = (conf, (gx1, gy1, gw, gh), mapped, label_str)

        if best is None:
            return {
                "sign_bbox": None,
                "sign_label": "",
                "sign_confidence": 0.0,
                "debug": "Team 3 scanning sign ROI (YOLO)",
            }

        conf, best_box, sign_guess, label_str = best
        self._speak_command(label_str, key=label_str)

        result: Dict[str, Any] = {
            "auto_current_sign": sign_guess,
            "sign_bbox": best_box,
            "sign_label": label_str,
            "sign_confidence": conf,
            "sign_locked": conf >= self.sign_lock_threshold and sign_guess != AutoCurrentSign.UNKNOWN,
            "debug": f"Team 3 YOLO: {sign_guess.value} ({conf:.2f})",
        }
        return result

    def detect_end_of_course(self, frame_bgr: np.ndarray, frame_id: int) -> Optional[Dict[str, Any]]:
        """
        End-of-course via YOLO class (name contains 'end' / end-of-course).
        Speaks "end of course" once when first seen; returns end_detected when
        either END_STOP_DELAY_SEC has passed since first seen or the end-sign
        box area exceeds END_STOP_AREA_PX2 (whichever comes first).
        """
        _ = frame_bgr
        if self._yolo_load_error:
            return None
        if self._yolo_results is None:
            return None

        end_snap = self._best_end_of_course_detection()
        end_conf = end_snap[0] if end_snap else None
        end_area: Optional[float] = _area_xyxy_sign(end_snap[1]) if end_snap else None

        now = time.monotonic()
        thr = float(YOLO_CONF)
        active_end = end_conf is not None and end_conf >= thr

        if active_end and end_area is not None:
            self._cached_end_sign_area = end_area

        def _end_area_debug_suffix() -> str:
            a = end_area if end_area is not None else self._cached_end_sign_area
            return f" area={a:.0f}" if a is not None else ""

        if self._end_first_seen_mono is None:
            if active_end:
                self._end_first_seen_mono = now
                if not self._end_spoken:
                    self._speak_command("end of course", key="end_course")
                    self._end_spoken = True

        if self._end_first_seen_mono is not None:
            elapsed = now - float(self._end_first_seen_mono)
            area_now = end_area if (active_end and end_area is not None) else None
            area_best = area_now if area_now is not None else self._cached_end_sign_area
            hit_time = elapsed >= END_STOP_DELAY_SEC
            hit_area = area_best is not None and float(area_best) > float(END_STOP_AREA_PX2)
            if not hit_time and not hit_area:
                t_left = max(0.0, END_STOP_DELAY_SEC - elapsed)
                a_note = f" area={area_best:.0f}/{END_STOP_AREA_PX2}" if area_best is not None else ""
                return {
                    "frame_id": frame_id,
                    "debug": f"End of course sign — {t_left:.1f}s left or need area>{END_STOP_AREA_PX2}{a_note}",
                }
            # Report end once, then clear latch so the next Square press can read other signs
            final_area_suffix = _end_area_debug_suffix()
            reason = "area threshold" if hit_area else "time elapsed"
            self._end_first_seen_mono = None
            self._end_spoken = False
            self._cached_end_sign_area = None
            return {
                "frame_id": frame_id,
                "end_detected": True,
                "debug": f"End of course — stop ({reason}){final_area_suffix}",
            }

        return None

    def _team3_setup_yolo_and_speech(self) -> None:
        if USB_SoundController is not None:
            try:
                self._sound = USB_SoundController(volume=0.75)
            except Exception as exc:
                self._sound = None
                self.last_debug = f"Team 3: sound unavailable ({exc})"
        if YOLO is None:
            self._yolo_load_error = "Team 3: ultralytics not installed (pip install ultralytics)"
            return
        if not os.path.isfile(YOLO_MODEL_PATH):
            self._yolo_load_error = (
                f"Team 3: model not found at {YOLO_MODEL_PATH} "
                f"(place {TEAM3_WEIGHTS_BASENAME} next to team3_signs_end.py or set TEAM3_YOLO_MODEL)"
            )
            return
        try:
            # load the actual yolo model 
            self._yolo_model = YOLO(YOLO_MODEL_PATH)
        except Exception as exc:
            self._yolo_load_error = f"Team 3: failed to load YOLO model ({exc})"
            self._yolo_model = None

    def _ensure_yolo_detections(self, frame_bgr: np.ndarray, frame_id: int) -> None:
        if self._yolo_model is None:
            self._yolo_results = None
            return
        if self._yolo_cache_frame_id == frame_id:
            return
        self._yolo_cache_frame_id = frame_id
        try:
            self._yolo_results = self._yolo_model(frame_bgr, conf=YOLO_CONF, verbose=False)[0]
        except Exception:
            self._yolo_results = None

    def _speak_command(self, phrase: str, *, key: str) -> None:
        now = time.monotonic()
        if key == self._last_speech_key and (now - self._last_speech_time) < SPEECH_COOLDOWN_SEC:
            return
        self._last_speech_time = now
        self._last_speech_key = key
        if self._sound is not None:
            try:
                self._sound.play_text_to_speech(phrase)
            except Exception:
                print(f"[Team3 TTS] {phrase}")
        else:
            print(f"[Team3 TTS] {phrase}")

    @staticmethod
    def _yolo_name_is_end(raw: str) -> bool:
        r = raw.replace("_", " ").strip().lower()
        if "end" in r and "course" in r:
            return True
        if r in ("end", "end_of_course", "end of course", "finish", "done"):
            return True
        if "endcourse" in r.replace(" ", ""):
            return True
        return False

    def _best_end_of_course_detection(self) -> Optional[tuple[float, list[float]]]:
        """Highest-confidence end-of-course box: ``(confidence, [x1,y1,x2,y2])``."""
        if self._yolo_results is None or self._yolo_model is None:
            return None
        boxes = self._yolo_results.boxes
        if boxes is None or len(boxes) == 0:
            return None
        best_conf: Optional[float] = None
        best_xyxy: Optional[list[float]] = None
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            raw = str(self._yolo_model.names[cls_id])
            if not self._yolo_name_is_end(raw.lower()):
                continue
            c = float(boxes.conf[i])
            if best_conf is None or c > best_conf:
                best_conf = c
                best_xyxy = [float(v) for v in boxes.xyxy[i].tolist()]
        if best_conf is None or best_xyxy is None:
            return None
        return (best_conf, best_xyxy)

    @staticmethod
    def _map_yolo_sign_to_auto(raw: str) -> Optional[AutoCurrentSign]:
        r = raw.replace("_", " ").strip().lower()
        if "stop" in r or r == "pause":
            return AutoCurrentSign.PAUSE
        if "left" in r:
            return AutoCurrentSign.LEFT
        if "right" in r:
            return AutoCurrentSign.RIGHT
        if "straight" in r or "forward" in r or r in ("go", "go straight", "gostraight"):
            return AutoCurrentSign.GO_STRAIGHT
        return None

    @staticmethod
    def _command_phrase_for_sign(sign: AutoCurrentSign) -> str:
        if sign == AutoCurrentSign.PAUSE:
            return "Stop"
        if sign == AutoCurrentSign.LEFT:
            return "Turn left"
        if sign == AutoCurrentSign.RIGHT:
            return "Turn right"
        if sign == AutoCurrentSign.GO_STRAIGHT:
            return "Go straight"
        return ""

    @staticmethod
    def parse_stage(name: str) -> IntersectionStage:
        for value in IntersectionStage:
            if value.value == name:
                return value
        return IntersectionStage.TRAVEL_LANE


def detect_end_sign_only(
    model: Any,
    frame_bgr: np.ndarray,
    *,
    conf: float = 0.45,
    iou: float = 0.7,
    device: str | None = None,
    imgsz: int | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Run full-frame YOLO inference and keep only **end-of-course** class detections.

    Returns a small dict: ``end_sign`` (bool), ``confidence`` (best end box),
    ``xyxy`` (``[x1,y1,x2,y2]`` or ``None``).
    """
    dets = detect_signs_in_frame(
        model,
        frame_bgr,
        conf=conf,
        iou=iou,
        device=device,
        imgsz=imgsz,
        verbose=verbose,
    )
    best_conf = 0.0
    best_xyxy: list[float] | None = None
    for d in dets:
        raw = str(d["sign"]).lower()
        if not Team3SignsEnd._yolo_name_is_end(raw):
            continue
        c = float(d["confidence"])
        if c > best_conf:
            best_conf = c
            best_xyxy = [float(v) for v in d["xyxy"]]
    return {
        "end_sign": best_xyxy is not None,
        "confidence": best_conf,
        "xyxy": best_xyxy,
    }


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


if __name__ == "__main__":
    print(
        "team 3 file running\n",
        flush=True,
    )