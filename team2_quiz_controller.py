"""
Team 2 Controller v1.0

PURPOSE
-------
This controller exists solely to test Team 2 (intersection perception).

All Team 1, Team 3, and Team 4 logic has been removed or stubbed out.
The robot drives forward at a fixed creep speed the entire time so that
Team 2 receives live frames and can fire its milestone cues.

The only thing to watch on the display is the intersection_stage value
advancing as Team 2 reports:
  - sign_read_ready      => TRAVEL_LANE     -> SIGN_READ
  - pause_point_ready    => POST_SIGN_TRAVEL -> PAUSE_ACTION  (simulated)
  - turn_point_ready     => PRE_TURN_TRAVEL  -> TURN_ACTION   (simulated)

The full intersection sequence is exercised:
  pause_point_ready  => stop, hold STOP_HOLD_SECONDS, enter PRE_TURN_TRAVEL
  turn_point_ready   => stop, hold STOP_HOLD_SECONDS, reset to TRAVEL_LANE

Teams 1, 3, and 4 are NOT imported or called.
Sign-read timeout is disabled so SIGN_READ stays visible until Team 2
fires the next milestone.
"""

from __future__ import annotations

import signal
import time
import queue
import multiprocessing as mp
from multiprocessing import shared_memory
from typing import Any, Dict

import cv2
import numpy as np
import pygame

from ps5_controller import PS5_Controller
from sabertooth import Sabertooth
from robot_types import (
    AutoCurrentSign,
    AutoMotion,
    AutoRoadLocation,
    IntersectionStage,
    MotionMode,
    RobotState,
    SensorData,
)
from robot_utils import replace_latest_queue_item

# Team 2 only — Teams 1, 3, and 4 are intentionally excluded in this test controller.
from team2_intersection import run_intersection_worker

try:
    from display import RobotDisplay
    DISPLAY_AVAILABLE = True
except Exception:
    RobotDisplay = None
    DISPLAY_AVAILABLE = False

try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except Exception:
    Picamera2 = None
    CAMERA_AVAILABLE = False

try:
    from imu_controller import IMUDevice
    IMU_AVAILABLE = True
except Exception:
    IMUDevice = None
    IMU_AVAILABLE = False


class SharedImageBuffer:
    """Shared-memory image container used to feed worker processes efficiently."""
    def __init__(self, width: int, height: int, channels: int = 3):
        self.width = int(width)
        self.height = int(height)
        self.channels = int(channels)
        self.nbytes = self.width * self.height * self.channels
        self.lock = mp.Lock()
        self.frame_counter = mp.Value("I", 0)
        self.shm = shared_memory.SharedMemory(create=True, size=self.nbytes)
        self.array = np.ndarray((self.height, self.width, self.channels), dtype=np.uint8, buffer=self.shm.buf)
        self.array[:, :] = 0

    def write(self, image_bgr: np.ndarray) -> int:
        with self.lock:
            self.array[:, :] = image_bgr
        with self.frame_counter.get_lock():
            self.frame_counter.value += 1
            return int(self.frame_counter.value)

    def close(self) -> None:
        try:
            self.shm.close()
        except Exception:
            pass
        try:
            self.shm.unlink()
        except Exception:
            pass


class IMUInterface:
    """Small safety wrapper around the IMU device so hardware failures degrade cleanly."""
    def __init__(self) -> None:
        self.device = None
        self.enabled = False
        self.last_error = ""

        if not IMU_AVAILABLE:
            self.last_error = "IMU driver not available"
            return

        try:
            self.device = IMUDevice()
            self.enabled = self.device is not None
        except Exception as exc:
            self.device = None
            self.enabled = False
            self.last_error = f"IMU unavailable: {exc}"

    def zero_reference(self) -> None:
        if self.device is not None:
            try:
                self.device.zero()
            except Exception as exc:
                self.last_error = f"IMU zero failed: {exc}"
                self.close()

    def get_delta(self) -> float:
        if self.device is not None:
            try:
                return float(self.device.delta())
            except Exception as exc:
                self.last_error = f"IMU read failed: {exc}"
                self.close()
        return 0.0

    def close(self) -> None:
        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
        self.device = None
        self.enabled = False
def draw_decision_banner(frame: np.ndarray, text: str, *, bg=(0, 255, 255), fg=(0, 0, 0)) -> None:
    """Draw the top-of-screen action banner used on the classroom display."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 42), bg, -1)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fg, 2, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Draw a simple FPS readout so students can see runtime performance."""
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)


def draw_crosshair(frame: np.ndarray) -> None:
    """Draw a center reference marker to help judge alignment in the frame."""
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 0), 1)


def draw_state_overlay(frame: np.ndarray, state: RobotState) -> None:
    """Overlay a compact summary of the controller state on the live frame."""
    h, _ = frame.shape[:2]
    lines = [
        f"Mode: {'AUTO' if state.auto_mode else 'MANUAL'}",
        f"Stage: {state.intersection_stage.value}",
        f"Motion Owner: {state.motion_mode.value}",
        f"Motion: {state.auto_motion.value}",
        f"Road: {state.auto_road_location.value}",
        f"Sign: {state.auto_current_sign.value}",
        f"IMU: {state.auto_current_imu_direction:.1f}",
    ]
    y = 105
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26
    cv2.putText(frame, state.debug_message, (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255) if state.auto_mode else (255, 255, 255), 2, cv2.LINE_AA)


def draw_roi(frame: np.ndarray, roi_xywh, *, label="ROI", color=(255, 0, 255)) -> None:
    """Draw a region of interest supplied by one of the perception teams."""
    if roi_xywh is None:
        return
    x, y, w2, h2 = map(int, roi_xywh)
    cv2.rectangle(frame, (x, y), (x + w2, y + h2), color, 2)
    cv2.putText(frame, label, (x + 6, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, bbox_xywh, label: str, conf: float, *, color=(0, 255, 255)) -> None:
    """Draw a labeled detection box with confidence text."""
    if bbox_xywh is None:
        return
    x, y, bw, bh = map(int, bbox_xywh)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
    text = f"{label}  {conf:.2f}" if label else f"{conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    y0 = max(0, y - th - 10)
    cv2.rectangle(frame, (x, y0), (x + tw + 10, y), color, -1)
    cv2.putText(frame, text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)


def draw_intersection_lines(frame: np.ndarray, lines) -> None:
    """Draw line annotations reported by Team 2."""
    for line in lines or []:
        if len(line) != 4:
            continue
        x1, y1, x2, y2 = map(int, line)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)


def draw_intersection_points(frame: np.ndarray, points) -> None:
    """Draw point annotations reported by Team 2."""
    for point in points or []:
        if len(point) != 2:
            continue
        x, y = map(int, point)
        cv2.circle(frame, (x, y), 6, (0, 165, 255), -1)


def draw_intersection_boxes(frame: np.ndarray, boxes) -> None:
    """Draw box annotations reported by Team 2."""
    for box in boxes or []:
        if len(box) != 4:
            continue
        x, y, w2, h2 = map(int, box)
        cv2.rectangle(frame, (x, y), (x + w2, y + h2), (0, 165, 255), 2)


def draw_team_annotations(frame: np.ndarray, state: RobotState) -> None:
    """Render the latest Team 2 and Team 3 visual annotations."""
    draw_roi(frame, state.sign_roi, label="Sign Search ROI", color=(255, 0, 255))
    if state.sign_bbox is not None:
        draw_detection(frame, state.sign_bbox, state.sign_label or state.auto_current_sign.value, state.sign_confidence)
    draw_roi(frame, state.intersection_roi, label="Intersection ROI", color=(0, 165, 255))
    draw_intersection_boxes(frame, state.intersection_boxes)
    draw_intersection_lines(frame, state.intersection_lines)
    draw_intersection_points(frame, state.intersection_points)
    if state.intersection_label:
        cv2.putText(frame, state.intersection_label, (10, frame.shape[0] - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 165, 255), 2, cv2.LINE_AA)
class AutonomousRobotController:
    """
    Main runtime object.

    The controller's job is not to "solve" each perception task itself.
    Its job is to:
    - gather sensor inputs
    - run each team module on schedule
    - apply a clear controller-owned workflow
    - choose the active motor command source
    - keep shutdown and hardware lifecycle stable
    """
    def __init__(self, use_display: bool = True, display_fullscreen: bool = True) -> None:
        self.state = RobotState()
        self.sensors = SensorData()
        self._shutdown_started = False

        self.display = None
        if use_display and DISPLAY_AVAILABLE:
            try:
                # The display runs in its own process. After construction, the
                # controller only needs to submit frames and poll for button events.
                self.display = RobotDisplay(fullscreen=display_fullscreen, fps=30, size=(800, 480), bg_color=(0, 0, 0))
                self.display.set_buttons(
                    [
                        {"handle": "auto", "text": "AUTO", "bg_color": (30, 80, 200), "text_color": (255, 255, 255)},
                        {"handle": "stop", "text": "STOP", "bg_color": (200, 60, 60), "text_color": (255, 255, 255)},
                        {"handle": "snap", "text": "SNAP", "bg_color": (60, 160, 90), "text_color": (255, 255, 255)},
                        {"handle": "quit", "text": "QUIT", "bg_color": (80, 80, 80), "text_color": (255, 255, 255)},
                    ],
                    position="left",
                )
                self.display.set_status("Mode: MANUAL | Robot controller starting...", position="top", bg_color=(15, 15, 15), text_color=(255, 255, 255))
                print(f"Display enabled (fullscreen={display_fullscreen})")
            except Exception as exc:
                print(f"Display init failed: {exc}")
                self.display = None

        self.ps5 = PS5_Controller()
        self.saber = Sabertooth()
        self.imu = IMUInterface()

        self.picam2 = None
        self.camera_enabled = False
        self.snapshot_count = 0
        self.last_frame_time = time.time()
        self.fps_smooth = 0.0
        self.fps_alpha = 0.15
        self.first_camera_frame_logged = False
        self.first_display_frame_logged = False

        self.saber.set_ramping(21)

        # Team 2 only — Teams 1, 3, and 4 are not instantiated in this test controller.
        self.perception_width = 640
        self.perception_height = 480
        # Only Team 2 gets a shared frame buffer and queue pair.
        self.team2_buffer = SharedImageBuffer(self.perception_width, self.perception_height)
        self.team2_input_q: mp.Queue = mp.Queue(maxsize=1)
        self.team2_result_q: mp.Queue = mp.Queue(maxsize=1)
        self.team2_proc = None
        self.last_team2_result_frame_id = 0

        self.main_sleep = 0.001
        self.motor_interval = 1.0 / 25.0
        self.imu_interval = 1.0 / 20.0
        self.display_interval = 1.0 / 30.0
        self.team2_submit_interval = 1.0 / 20.0
        self.perception_poll_interval = 1.0 / 50.0
        # Sign-read timeout is DISABLED in this test controller so SIGN_READ
        # stays visible on the display until Team 2 fires the next milestone.
        self.sign_read_timeout = float("inf")
        self.creep_forward_speed = 35
        self.stop_hold_seconds = 1.5
        self.stage_started_at = time.time()

        now = time.time()
        self.last_team2_time = now
        self.last_motor_time = now
        self.last_imu_time = now
        self.last_display_time = now
        self.last_perception_poll_time = now
        self.running = True
        self.use_display = use_display

    def initialize(self) -> None:
        """Bring up controller input, worker processes, camera, and IMU.
        Auto mode is started immediately — no button press needed for Team 2 testing.
        """
        self.ps5.initialize_controller()
        self.saber.stop()
        self.start_perception_processes()
        self.initialize_camera()
        self.imu.zero_reference()
        imu_status = "enabled" if self.imu.enabled else self.imu.last_error or "disabled"
        camera_status = "enabled" if self.camera_enabled else self.state.debug_message
        display_status = "enabled" if self.display is not None else "disabled"
        self.state.debug_message = "Initialization complete"
        print(f"Startup status: camera={camera_status}, display={display_status}, imu={imu_status}")
        self._update_display_status()
        # Auto-start immediately for Team 2 testing — no button press needed.
        self.start_autonomous_mode()

    def _sync_stage_derived_state(self) -> None:
        """
        Maintain display/debug-friendly fields from the controller-owned stage.

        `auto_motion` and `auto_road_location` are useful for display and for
        helping students understand the robot state at a glance.

        Treat these as derived state. The real
        workflow source of truth is:
        - state.intersection_stage
        - state.motion_mode
        """
        stage = self.state.intersection_stage

        if stage == IntersectionStage.TRAVEL_LANE:
            self.state.auto_road_location = AutoRoadLocation.IN_TRAVEL_LANE
            self.state.auto_motion = AutoMotion.MOVING_FORWARD if self.state.auto_mode else AutoMotion.PAUSED
        elif stage == IntersectionStage.SIGN_READ:
            self.state.auto_road_location = AutoRoadLocation.AT_SIGN_READ_LOCATION
            self.state.auto_motion = AutoMotion.MOVING_FORWARD if self.state.auto_mode else AutoMotion.PAUSED
        elif stage == IntersectionStage.POST_SIGN_TRAVEL:
            self.state.auto_road_location = AutoRoadLocation.AT_PAUSE_LOCATION
            self.state.auto_motion = AutoMotion.MOVING_FORWARD if self.state.auto_mode else AutoMotion.PAUSED
        elif stage == IntersectionStage.PAUSE_ACTION:
            self.state.auto_road_location = AutoRoadLocation.AT_PAUSE_LOCATION
            self.state.auto_motion = AutoMotion.PAUSED
        elif stage == IntersectionStage.PRE_TURN_TRAVEL:
            self.state.auto_road_location = AutoRoadLocation.AT_TURN_LOCATION
            self.state.auto_motion = AutoMotion.MOVING_FORWARD if self.state.auto_mode else AutoMotion.PAUSED
        elif stage == IntersectionStage.TURN_ACTION:
            self.state.auto_road_location = AutoRoadLocation.AT_TURN_LOCATION
            if self.state.auto_current_sign == AutoCurrentSign.LEFT:
                self.state.auto_motion = AutoMotion.TURNING_LEFT
            elif self.state.auto_current_sign == AutoCurrentSign.RIGHT:
                self.state.auto_motion = AutoMotion.TURNING_RIGHT
            else:
                self.state.auto_motion = AutoMotion.MOVING_FORWARD
        elif stage == IntersectionStage.END_REACHED:
            self.state.auto_road_location = AutoRoadLocation.AT_END
            self.state.auto_motion = AutoMotion.PAUSED

    def _enter_stage(self, stage: IntersectionStage, *, motion_mode: MotionMode, debug_message: str | None = None) -> None:
        """
        Single helper for stage transitions.

        All controller-owned workflow changes should go through this helper so:
        - stage timing stays consistent
        - motion ownership is updated in one place
        - derived display fields remain synchronized
        """
        self.state.intersection_stage = stage
        self.state.motion_mode = motion_mode
        self.stage_started_at = self.sensors.now or time.time()
        self._sync_stage_derived_state()
        if debug_message:
            self.state.debug_message = debug_message

    def _finish_intersection_cycle(self, message: str) -> None:
        """
        Reset intersection-specific state after Team 4 completes the action.

        This is the normal "back to lane travel" handoff point for the next
        intersection cycle.
        """
        self.state.auto_current_sign = AutoCurrentSign.UNKNOWN
        self.state.sign_locked = False
        self.state.auto_sign_implemented = False
        self.state.sign_label = ""
        self.state.sign_confidence = 0.0
        self.state.sign_bbox = None
        self._enter_stage(IntersectionStage.TRAVEL_LANE, motion_mode=MotionMode.CREEP_FORWARD, debug_message=message)

    def shutdown(self) -> None:
        """Best-effort shutdown that stops hardware first and then tears down helpers."""
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self.running = False
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            pass
        try:
            self.saber.stop()
        except Exception:
            pass
        try:
            self.imu.close()
        except Exception:
            pass
        # Stop worker processes before closing their shared memory and queues.
        self.stop_perception_processes()
        try:
            if self.picam2 is not None:
                self.picam2.stop()
        except Exception:
            pass
        try:
            if self.picam2 is not None:
                self.picam2.close()
        except Exception:
            pass
        self.picam2 = None
        self.camera_enabled = False
        try:
            if self.display is not None:
                self.display.close()
        except Exception:
            pass
        self.display = None
        try:
            self.saber.close()
        except Exception:
            pass
        try:
            self.team2_buffer.close()
        except Exception:
            pass
        for qobj in (self.team2_input_q, self.team2_result_q):
            try:
                qobj.cancel_join_thread()
            except Exception:
                pass
            try:
                qobj.close()
            except Exception:
                pass
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass

    def initialize_camera(self) -> None:
        """Configure the video stream used by both display and perception."""
        if not CAMERA_AVAILABLE:
            self.camera_enabled = False
            self.state.debug_message = "Picamera2 not available"
            return
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_video_configuration(main={"size": (640, 480), "format": "XRGB8888"})
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(1.5)
            try:
                meta = self.picam2.capture_metadata()
                if "ColourGains" in meta:
                    self.picam2.set_controls({"AwbEnable": False, "ColourGains": meta["ColourGains"]})
            except Exception:
                pass
            self.camera_enabled = True
        except Exception as exc:
            self.camera_enabled = False
            self.state.debug_message = f"Camera init failed: {exc}"

    def update_controller_inputs(self) -> None:
        """Poll pygame/joystick state and copy a normalized snapshot into SensorData."""
        pygame.event.pump()
        self.ps5.check_controls()
        self.sensors.controller_buttons = {
            "cross": self.ps5.control_request.get("reqCross", False),
            "circle": self.ps5.control_request.get("reqCircle", False),
            "triangle": self.ps5.control_request.get("reqTriangle", False),
            "square": self.ps5.control_request.get("reqSquare", False),
            "options": self.ps5.control_request.get("reqOptions", False),
            "ps": self.ps5.control_request.get("reqPS", False),
            "arrow_up": self.ps5.control_request.get("reqArrowUp", False),
            "arrow_down": self.ps5.control_request.get("reqArrowDown", False),
            "arrow_left": self.ps5.control_request.get("reqArrowLeft", False),
            "arrow_right": self.ps5.control_request.get("reqArrowRight", False),
        }
        self.sensors.controller_left_y = self.ps5.control_request.get("reqLeftJoyYValue", 0)
        self.sensors.controller_left_x = self.ps5.control_request.get("reqLeftJoyXValue", 0)

    def update_imu(self) -> None:
        """Refresh the latest IMU delta for both controller logic and team modules."""
        self.sensors.imu_delta = self.imu.get_delta()
        self.state.auto_current_imu_direction = self.sensors.imu_delta

    def update_camera_frame(self) -> None:
        """Capture the newest camera frame and update the smoothed FPS estimate."""
        if not self.camera_enabled or self.picam2 is None:
            self.sensors.frame = None
            return
        try:
            frame_bgra = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            self.sensors.frame = frame_bgr
            if not self.first_camera_frame_logged:
                print(f"First camera frame captured: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}")
                self.first_camera_frame_logged = True
            now = self.sensors.now
            dt = now - self.last_frame_time
            self.last_frame_time = now
            if dt > 0:
                fps_inst = 1.0 / dt
                self.fps_smooth = fps_inst if self.fps_smooth == 0 else (self.fps_alpha * fps_inst + (1 - self.fps_alpha) * self.fps_smooth)
        except Exception as exc:
            self.sensors.frame = None
            self.state.debug_message = f"Camera capture failed: {exc}"

    def start_perception_processes(self) -> None:
        """Launch only the Team 2 worker process."""
        self.team2_proc = mp.Process(
            target=run_intersection_worker,
            args=(self.team2_buffer.shm.name, (self.perception_height, self.perception_width, 3), self.team2_buffer.frame_counter, self.team2_buffer.lock, self.team2_input_q, self.team2_result_q),
            daemon=True,
        )
        self.team2_proc.start()

    def stop_perception_processes(self) -> None:
        """Request Team 2 worker shutdown, then escalate to terminate/kill if needed."""
        replace_latest_queue_item(self.team2_input_q, {"type": "QUIT"})
        try:
            if self.team2_proc is not None and self.team2_proc.is_alive():
                self.team2_proc.join(timeout=1.0)
            if self.team2_proc is not None and self.team2_proc.is_alive():
                self.team2_proc.terminate()
                self.team2_proc.join(timeout=1.0)
            if self.team2_proc is not None and self.team2_proc.is_alive():
                self.team2_proc.kill()
                self.team2_proc.join(timeout=1.0)
            if self.team2_proc is not None:
                self.team2_proc.close()
        except Exception:
            pass
        self.team2_proc = None

    def submit_perception_jobs_if_due(self) -> None:
        """Feed Team 2 with the latest frame plus minimal controller context."""
        now = self.sensors.now
        if self.sensors.frame is None:
            return
        if now - self.last_team2_time >= self.team2_submit_interval:
            frame_id = self.team2_buffer.write(self.sensors.frame)
            replace_latest_queue_item(self.team2_input_q, {"type": "PROCESS", "frame_id": frame_id, "timestamp": now, "roi_xywh": self.state.intersection_roi, "road_location": self.state.auto_road_location.value, "intersection_stage": self.state.intersection_stage.value, "auto_mode": self.state.auto_mode})
            self.last_team2_time = now

    def poll_perception_results_if_due(self) -> None:
        """Drain Team 2 result queue and apply the newest result."""
        now = self.sensors.now
        if now - self.last_perception_poll_time < self.perception_poll_interval:
            return
        self.last_perception_poll_time = now
        while True:
            try:
                result = self.team2_result_q.get_nowait()
            except Exception:
                break
            if result.get("frame_id", 0) >= self.last_team2_result_frame_id:
                self.last_team2_result_frame_id = result.get("frame_id", 0)
                self.apply_team2_updates(result)

    def call_team_modules_if_due(self) -> None:
        """
        Team 2 runs in its own process — no in-process team modules to call.
        Teams 1 and 4 are excluded from this test controller.
        The controller stage logic still runs to process Team 2 milestones.
        """
        self.handle_controller_stage_logic()

    def handle_controller_stage_logic(self) -> None:
        """
        Controller-owned stage rules for Team 2 testing.

        Mirrors test_team2.py sequence:
        - PAUSE_ACTION: stop, hold stop_hold_seconds, then enter PRE_TURN_TRAVEL
        - TURN_ACTION: stop, hold stop_hold_seconds, then reset to TRAVEL_LANE
        - Sign-read timeout is DISABLED so SIGN_READ stays until Team 2 advances.
        """
        if not self.state.auto_mode:
            return

        now = self.sensors.now or time.time()
        elapsed = now - self.stage_started_at
        stage = self.state.intersection_stage

        if stage == IntersectionStage.PAUSE_ACTION:
            if elapsed >= self.stop_hold_seconds:
                self._enter_stage(IntersectionStage.PRE_TURN_TRAVEL, motion_mode=MotionMode.CREEP_FORWARD, debug_message="Pause complete => PRE_TURN_TRAVEL")

        elif stage == IntersectionStage.TURN_ACTION:
            if elapsed >= self.stop_hold_seconds:
                self._finish_intersection_cycle("Turn complete => TRAVEL_LANE")

    def apply_team2_updates(self, updates: Dict[str, Any]) -> None:
        """
        Apply Team 2's perception results.

        In this test controller the robot always uses CREEP_FORWARD.
        Stage transitions still fire when Team 2 reports milestones so the
        intersection_stage value on the display reflects Team 2's output.
        """
        if not updates:
            return
        if "intersection_roi" in updates:
            self.state.intersection_roi = updates["intersection_roi"]
        if "intersection_boxes" in updates:
            self.state.intersection_boxes = list(updates["intersection_boxes"] or [])
        if "intersection_lines" in updates:
            self.state.intersection_lines = list(updates["intersection_lines"] or [])
        if "intersection_points" in updates:
            self.state.intersection_points = list(updates["intersection_points"] or [])
        if "intersection_label" in updates:
            self.state.intersection_label = str(updates["intersection_label"])
        if self.state.auto_mode:
            sign_read_ready = bool(updates.get("sign_read_ready", False))
            pause_point_ready = bool(updates.get("pause_point_ready", False))
            turn_point_ready = bool(updates.get("turn_point_ready", False))

            if self.state.intersection_stage == IntersectionStage.TRAVEL_LANE and sign_read_ready:
                self._enter_stage(IntersectionStage.SIGN_READ, motion_mode=MotionMode.CREEP_FORWARD, debug_message="Team 2: sign_read_ready => SIGN_READ")
            elif self.state.intersection_stage == IntersectionStage.SIGN_READ and pause_point_ready:
                self._enter_stage(IntersectionStage.POST_SIGN_TRAVEL, motion_mode=MotionMode.CREEP_FORWARD, debug_message="Team 2: pause_point_ready => POST_SIGN_TRAVEL")
            elif self.state.intersection_stage == IntersectionStage.POST_SIGN_TRAVEL and pause_point_ready:
                self._enter_stage(IntersectionStage.PAUSE_ACTION, motion_mode=MotionMode.CREEP_FORWARD, debug_message="Team 2: pause_point_ready => PAUSE_ACTION")
            elif self.state.intersection_stage in (IntersectionStage.SIGN_READ, IntersectionStage.POST_SIGN_TRAVEL, IntersectionStage.PRE_TURN_TRAVEL) and turn_point_ready:
                self._enter_stage(IntersectionStage.TURN_ACTION, motion_mode=MotionMode.CREEP_FORWARD, debug_message="Team 2: turn_point_ready => TURN_ACTION")
        if "debug" in updates:
            self.state.debug_message = str(updates["debug"])

    def update_motor_output_if_due(self) -> None:
        """
        Final motor arbitration point.

        This is the most important controller-owned safety rule in the file:
        exactly one motion source is allowed to own the motors at a time.

        Priority:
        1. auto_mode off or END_REACHED => stop
        2. CREEP_FORWARD => fixed low-speed straight motion
        3. stop_requested => stop
        4. otherwise use the selected team's requested_speed/requested_turn
        """
        now = self.sensors.now
        if now - self.last_motor_time < self.motor_interval:
            return
        if not self.state.auto_mode:
            self.saber.stop()
            self.last_motor_time = now
            return
        if self.state.intersection_stage == IntersectionStage.END_REACHED:
            self.saber.stop()
            self.last_motor_time = now
            return
        if self.state.intersection_stage in (IntersectionStage.PAUSE_ACTION, IntersectionStage.TURN_ACTION):
            self.saber.stop()
            self.last_motor_time = now
            return
        if self.state.motion_mode == MotionMode.CREEP_FORWARD:
            self.saber.drive(self.creep_forward_speed, 0)
            self.last_motor_time = now
            return
        if self.state.stop_requested:
            self.saber.stop()
            self.last_motor_time = now
            return
        self.saber.drive(self.state.requested_speed, self.state.requested_turn)
        self.last_motor_time = now

    def _build_status_line(self) -> str:
        """Build the short status string shown in the display's top bar."""
        return f"Mode: {'AUTO' if self.state.auto_mode else 'MANUAL'} | Stage: {self.state.intersection_stage.value} | Motion Owner: {self.state.motion_mode.value} | Sign: {self.state.auto_current_sign.value}"

    def _update_display_status(self) -> None:
        """Send the latest status string to the display process."""
        if self.display is None:
            return
        try:
            self.display.set_status(self._build_status_line(), position="top", bg_color=(15, 15, 15), text_color=(255, 255, 255))
        except Exception:
            pass

    def start_autonomous_mode(self) -> None:
        """
        Initialize a fresh autonomous run for Team 2 testing.
        The robot will creep forward the entire time so Team 2 receives live frames.
        """
        self.imu.zero_reference()
        self.state.auto_mode = True
        self.state.intersection_stage = IntersectionStage.TRAVEL_LANE
        self.state.motion_mode = MotionMode.CREEP_FORWARD
        self.state.auto_current_sign = AutoCurrentSign.UNKNOWN
        self.state.sign_locked = False
        self.state.auto_current_imu_direction = 0.0
        self.state.auto_sign_implemented = False
        self.state.requested_speed = self.creep_forward_speed
        self.state.requested_turn = 0
        self.state.stop_requested = False
        self.state.sign_roi = None
        self.state.sign_bbox = None
        self.state.sign_label = ""
        self.state.sign_confidence = 0.0
        self.state.intersection_roi = None
        self.state.intersection_boxes = []
        self.state.intersection_lines = []
        self.state.intersection_points = []
        self.state.intersection_label = ""
        self.stage_started_at = self.sensors.now or time.time()
        self._sync_stage_derived_state()
        self.state.debug_message = "Team 2 test: autonomous mode started"

    def stop_autonomous_mode(self, message: str = "Autonomous mode stopped") -> None:
        """Immediate operator stop of autonomous workflow."""
        self.state.auto_mode = False
        self.state.intersection_stage = IntersectionStage.TRAVEL_LANE
        self.state.motion_mode = MotionMode.CREEP_FORWARD
        self.state.auto_motion = AutoMotion.PAUSED
        self.state.requested_speed = 0
        self.state.requested_turn = 0
        self.state.stop_requested = True
        self.state.sign_locked = False
        self.state.debug_message = message

    def handle_controller_start_stop(self) -> None:
        """Handle physical PS5 button presses that start or stop autonomous mode."""
        buttons = self.sensors.controller_buttons
        if buttons.get("triangle", False) and not self.state.auto_mode:
            self.start_autonomous_mode()
        if buttons.get("cross", False):
            self.stop_autonomous_mode("Stopped from PS5 X button")

    def handle_display_events(self) -> None:
        """Handle touchscreen UI button presses coming back from the display process."""
        if self.display is None:
            return
        for ev in self.display.poll_events():
            if ev == "auto":
                if self.state.auto_mode:
                    self.stop_autonomous_mode("AUTO toggled off from display")
                else:
                    self.start_autonomous_mode()
                    self.state.debug_message = "AUTO toggled on from display"
            elif ev == "stop":
                self.stop_autonomous_mode("Stopped from display")
            elif ev == "snap":
                if self.sensors.frame is not None:
                    self.snapshot_count += 1
                    fname = f"snapshot_{self.snapshot_count:03d}.jpg"
                    try:
                        cv2.imwrite(fname, self.sensors.frame)
                        self.state.debug_message = f"Saved {fname}"
                    except Exception as exc:
                        self.state.debug_message = f"Snapshot failed: {exc}"
            elif ev == "quit":
                self.running = False
                self.state.debug_message = "Quit requested from display"

    def update_display_if_due(self) -> None:
        """Compose the annotated display frame and submit it to the UI process."""
        now = self.sensors.now
        if now - self.last_display_time >= self.display_interval:
            self._update_display_status()
            if self.display is not None and self.sensors.frame is not None:
                # Use a copy so display annotations never modify the frame used by
                # the perception pipeline.
                frame = self.sensors.frame.copy()
                draw_decision_banner(frame, "Action: AUTONOMOUS" if self.state.auto_mode else "Action: MANUAL")
                draw_fps(frame, self.fps_smooth)
                draw_crosshair(frame)
                draw_team_annotations(frame, self.state)
                draw_state_overlay(frame, self.state)
                try:
                    self.display.set_frame(frame)
                    if not self.first_display_frame_logged:
                        print("First display frame submitted")
                        self.first_display_frame_logged = True
                except Exception as exc:
                    self.state.debug_message = f"Display frame update failed: {exc}"
                    print(self.state.debug_message)
                    try:
                        self.display.close()
                    except Exception:
                        pass
                    self.display = None
            self.last_display_time = now

    def reset_one_shot_controller_buttons(self) -> None:
        """Clear button-edge events after the current loop iteration consumes them."""
        if self.ps5.control_request.get("reqMade", False):
            self.ps5.reset_controller_state()

    def run(self) -> None:
        """
        Main loop.

        Ordering matters:
        - sensors are updated first
        - perception workers receive fresh frames
        - team outputs are applied
        - controller stage logic runs
        - final motor arbitration happens near the end of the cycle
        """
        self.initialize()
        try:
            while self.running:
                now = time.time()
                self.sensors.now = now
                self.update_controller_inputs()
                if now - self.last_imu_time >= self.imu_interval:
                    self.update_imu()
                    self.last_imu_time = now
                self.update_camera_frame()
                self.handle_controller_start_stop()
                self.handle_display_events()
                self.submit_perception_jobs_if_due()
                self.poll_perception_results_if_due()
                self.call_team_modules_if_due()
                self.update_motor_output_if_due()
                self.update_display_if_due()
                self.reset_one_shot_controller_buttons()
                time.sleep(self.main_sleep)
        except KeyboardInterrupt:
            self.running = False
            self.state.debug_message = "KeyboardInterrupt received"
            print("Exiting program...")
        finally:
            self.shutdown()


if __name__ == "__main__":
    controller = AutonomousRobotController(use_display=True, display_fullscreen=True)
    controller.run()