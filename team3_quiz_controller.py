from __future__ import annotations

import queue
import threading
import time

import cv2
import numpy as np
import pygame

from ps5_controller import PS5_Controller
from sabertooth import Sabertooth

from team3_signs_end import Team3SignsEnd, _picamera_array_to_bgr, bounding_box_area_xywh

try:
    from display import RobotDisplay
    DISPLAY_AVAILABLE = True
except Exception:
    RobotDisplay = None
    DISPLAY_AVAILABLE = False

try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    Picamera2 = None
    CAMERA_AVAILABLE = False


def draw_decision_banner(frame, text: str, *, bg=(0, 255, 255), fg=(0, 0, 0)) -> None:
    """same as robot_auto_controller."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 42), bg, -1)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, fg, 2, cv2.LINE_AA)


def draw_fps(frame, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)


def draw_crosshair(frame) -> None:
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 0), 1)


def _picamera_raw_to_bgr_for_display(raw: np.ndarray) -> np.ndarray:
    """same as robot_auto_controller.update_camera_frame: BGRA -> BGR for Picamera2 main stream."""
    if raw is None or raw.size == 0 or raw.ndim != 3:
        return raw
    if raw.shape[2] == 4:
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    return _picamera_array_to_bgr(raw)


class MinimalTeam3Controller:
    def __init__(self, use_display: bool = True, display_fullscreen: bool = True):
        # 1. Display
        self.display = None
        self.use_display = use_display
        if use_display and DISPLAY_AVAILABLE:
            try:
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
                self.display.set_status("Team 3 quiz controller starting...", position="top", bg_color=(15, 15, 15), text_color=(255, 255, 255))
                print(f"Display enabled (fullscreen={display_fullscreen})")
            except Exception as exc:
                print(f"Display init failed: {exc}")
                self.display = None

        # 2. Controller (must be after display for display to work)
        self.ps5 = PS5_Controller()
        self.ps5.initialize_controller()

        self.picam2 = None
        if CAMERA_AVAILABLE:
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
                print("Camera enabled.")
            except Exception as e:
                print(f"Camera init failed: {e}")
                self.picam2 = None

        # 3. load YOLO
        self.team3 = Team3SignsEnd()

        self.running = True
        self.latest_frame = None
        self.snapshot_count = 0
        self.display_interval = 1.0 / 30.0
        self.last_display_time = time.time()
        self.last_frame_time = time.time()
        # same as robot_auto_controller
        self.main_sleep = 0.001
        self.fps_smooth = 0.0
        self.fps_alpha = 0.15
        self.first_display_frame_logged = False
        self.first_camera_frame_logged = False

        self.ps5_interval = 0.02
        self.ps5_last_check = time.time()

        self._prev_square = False
        self._prev_circle = False
        self._circle_drive_active = False
        self._drive_frame_n = 0
        self._end_check_frame_id = 0

        # Straight-line driving
        self.straight_drive_speed = 70
        self.motor_interval = 1.0 / 25.0
        self.last_motor_time = 0.0
        self.saber: Sabertooth | None = None
        try:
            sab = Sabertooth()
            if getattr(sab, "ser", None) is None:
                print("Sabertooth serial not available (straight drive disabled).")
            else:
                sab.set_ramping(21)
                sab.stop()
                self.saber = sab
                print("Sabertooth motors ready (straight drive uses drive(speed, 0)).")
        except Exception as exc:
            print(f"Sabertooth init failed (straight drive disabled): {exc}")
            self.saber = None

        # Background end-of-course check
        self._team3_yolo_lock = threading.Lock()
        self._end_job_queue: queue.Queue[dict | None] = queue.Queue(maxsize=1)
        self._end_result_lock = threading.Lock()
        self._end_async_latest: dict | None = None
        self._end_drive_generation = 0
        self._end_async_worker: threading.Thread | None = None

    def _submit_end_check_job(self, frame_id: int, frame_bgr: np.ndarray, gen: int) -> None:
        job = {"frame_id": frame_id, "frame_bgr": frame_bgr, "gen": gen}
        try:
            self._end_job_queue.put_nowait(job)
        except queue.Full:
            try:
                _ = self._end_job_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._end_job_queue.put_nowait(job)
            except queue.Full:
                pass

    def _end_detection_worker_loop(self) -> None:
        while True:
            job = self._end_job_queue.get()
            if job is None:
                return
            gen = int(job["gen"])
            frame_id = int(job["frame_id"])
            frame_bgr = job["frame_bgr"]
            end_result = None
            try:
                with self._team3_yolo_lock:
                    self.team3._ensure_yolo_detections(frame_bgr, frame_id)
                    end_result = self.team3.detect_end_of_course(frame_bgr, frame_id)
            except Exception as exc:
                end_result = {"debug": f"end worker: {exc}", "end_detected": False}
            with self._end_result_lock:
                self._end_async_latest = {"gen": gen, "frame_id": frame_id, "end_result": end_result}

    def _consume_end_async_results(self) -> None:
        with self._end_result_lock:
            pack = self._end_async_latest
            self._end_async_latest = None
        if not pack or not self._circle_drive_active:
            return
        if int(pack.get("gen", -1)) != self._end_drive_generation:
            return
        end_result = pack.get("end_result")
        if end_result is None:
            return
        if end_result.get("end_detected"):
            print(f"detect_end_of_course (FINAL): {end_result}")
            self._circle_drive_active = False
            if self.saber is not None:
                self.saber.stop()
        else:
            print(f"detect_end_of_course: {end_result.get('debug', end_result)}")

    def _build_display_status_line(self) -> str:
        drive = "drive ON" if self._circle_drive_active else "drive OFF"
        return f"Team 3 Quiz | {drive} | Square=read Circle=toggle Cross=quit"

    def _update_display_status(self) -> None:
        if self.display is None:
            return
        try:
            self.display.set_status(self._build_display_status_line(), position="top", bg_color=(15, 15, 15), text_color=(255, 255, 255))
        except Exception:
            pass

    def handle_display_events(self) -> None:
        if self.display is None:
            return
        for ev in self.display.poll_events():
            if ev == "auto":
                print("Display AUTO: not used in Team 3 quiz mode (use PS5 controls).")
            elif ev == "stop":
                self._circle_drive_active = False
                if self.saber is not None:
                    self.saber.stop()
                print("Display STOP: straight drive off, motors stopped.")
            elif ev == "snap":
                if self.latest_frame is None:
                    print("Snapshot: no camera frame yet.")
                else:
                    self.snapshot_count += 1
                    fname = f"snapshot_team3_quiz_{self.snapshot_count:03d}.jpg"
                    try:
                        frame_bgr = _picamera_array_to_bgr(self.latest_frame.copy())
                        cv2.imwrite(fname, frame_bgr)
                        print(f"Saved {fname}")
                    except Exception as exc:
                        print(f"Snapshot failed: {exc}")
            elif ev == "quit":
                print("Quit requested from display.")
                self.running = False
                if self.saber is not None:
                    self.saber.stop()

    def update_display_if_due(self, now: float) -> None:
        if self.display is None:
            return
        if now - self.last_display_time < self.display_interval:
            return
        self._update_display_status()
        if self.latest_frame is not None:
            frame = np.ascontiguousarray(_picamera_raw_to_bgr_for_display(self.latest_frame))
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            draw_decision_banner(frame, "Team 3 Quiz (signs + end check)")
            draw_fps(frame, self.fps_smooth)
            draw_crosshair(frame)
            try:
                self.display.set_frame(frame)
                if not self.first_display_frame_logged:
                    print("First display frame submitted")
                    self.first_display_frame_logged = True
            except Exception as exc:
                print(f"Display frame update failed: {exc}")
                try:
                    self.display.close()
                except Exception:
                    pass
                self.display = None
        self.last_display_time = now

    def run(self):
        print("Starting Minimal Team 3 Controller (Sign Detection Only)...")
        print("Press CROSS to quit.")
        print("Press SQUARE to capture a frame and perform a sign read.")
        print("Press CIRCLE (tap) to toggle straight driving + end-of-course check every 5 frames (YOLO in background thread).")

        self._end_async_worker = threading.Thread(target=self._end_detection_worker_loop, name="Team3EndDetect", daemon=True)
        self._end_async_worker.start()

        try:
            while self.running:
                loop_start = time.time()
                now = loop_start

                # Controller checks
                if now - self.ps5_last_check >= self.ps5_interval:
                    pygame.event.pump()
                    self.ps5.check_controls()
                    self.ps5_last_check = now
                
                # --- A. READ SENSORS ---
                # Continuously pull frames so the camera buffer doesn't go stale
                if self.picam2 is not None:
                    try:
                        self.latest_frame = self.picam2.capture_array()
                        if not self.first_camera_frame_logged and self.latest_frame is not None:
                            shp = self.latest_frame.shape
                            print(f"First camera frame captured: {shp[1]}x{shp[0]} channels={shp[2] if len(shp) > 2 else 'n/a'}")
                            self.first_camera_frame_logged = True
                        dt = now - self.last_frame_time
                        self.last_frame_time = now
                        if dt > 0:
                            fps_inst = 1.0 / dt
                            self.fps_smooth = fps_inst if self.fps_smooth == 0 else (self.fps_alpha * fps_inst + (1 - self.fps_alpha) * self.fps_smooth)
                    except Exception as e:
                        print(f"Frame capture error: {e}")
                        self.latest_frame = None

                # Check PS5 controller inputs
                self.ps5.check_controls()
                self.handle_display_events()
                
                # Exit condition
                if self.ps5.control_request["reqCross"]:
                    print("Cross button pressed. Exiting...")
                    self.running = False
                    if self.saber is not None:
                        self.saber.stop()
                    break
                
                # --- B. PROCESS TEAM 3 LOGIC ON DEMAND ---
                req_square = bool(self.ps5.control_request.get("reqSquare", False))
                req_circle = bool(self.ps5.control_request.get("reqCircle", False))

                # SQUARE: rising edge → classify sign via Team3SignsEnd.detect_sign
                if req_square and not self._prev_square:
                    if self.latest_frame is None:
                        print("Cannot read sign: Waiting for camera frame...")
                    else:
                        print("\n--- SQUARE: SIGN READ (detect_sign) ---")
                        frame = _picamera_array_to_bgr(self.latest_frame.copy())
                        work_frame, roi_xywh = self.team3.extract_sign_roi(frame, None)
                        with self._team3_yolo_lock:
                            sign_result = self.team3.detect_sign(work_frame, roi_xywh)
                        print(f"detect_sign result: {sign_result}")
                        bb = sign_result.get("sign_bbox")
                        if bb is not None:
                            print(f"sign bbox (x,y,w,h) area (px^2): {bounding_box_area_xywh(bb):.0f}")
                self._prev_square = req_square

                # CIRCLE: rising edge → toggle straight-drive + periodic end-of-course checks
                if req_circle and not self._prev_circle:
                    self._circle_drive_active = not self._circle_drive_active
                    self._drive_frame_n = 0
                    self._end_drive_generation += 1
                    with self._end_result_lock:
                        self._end_async_latest = None
                    if self._circle_drive_active:
                        print("\n--- CIRCLE: straight drive ON + end checks every 5 frames (background) ---")
                    else:
                        print("\n--- CIRCLE: straight drive OFF ---")
                        if self.saber is not None:
                            self.saber.stop()
                self._prev_circle = req_circle

                now_loop = time.time()
                if self._circle_drive_active and self.latest_frame is not None:
                    # Same pattern as AutonomousRobotController.update_motor_output_if_due: fixed forward, zero turn.
                    if self.saber is not None and (now_loop - self.last_motor_time) >= self.motor_interval:
                        self.saber.drive(self.straight_drive_speed, 0)
                        self.last_motor_time = now_loop

                    self._drive_frame_n += 1
                    if self._drive_frame_n % 5 == 0:
                        frame = _picamera_array_to_bgr(self.latest_frame.copy())
                        self._end_check_frame_id += 1
                        self._submit_end_check_job(self._end_check_frame_id, frame, self._end_drive_generation)

                self._consume_end_async_results()

                if self.ps5.control_request.get("reqMade", False):
                    self.ps5.reset_controller_state()

                self.update_display_if_due(time.time())

                # --- C. LOOP TIMING ---
                # Same as robot_auto_controller so display frame_counter keeps up
                time.sleep(self.main_sleep)
                
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught. Stopping...")
        finally:
            self.shutdown()

    def shutdown(self):
        print("Shutting down hardware...")
        if self._end_async_worker is not None and self._end_async_worker.is_alive():
            try:
                self._end_job_queue.put_nowait(None)
            except queue.Full:
                try:
                    _ = self._end_job_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._end_job_queue.put_nowait(None)
                except queue.Full:
                    pass
            self._end_async_worker.join(timeout=5.0)
        self._end_async_worker = None

        if self.saber is not None:
            try:
                self.saber.stop()
            except Exception:
                pass
        if self.picam2 is not None:
            self.picam2.stop()
        try:
            if self.picam2 is not None:
                self.picam2.close()
        except Exception:
            pass
        self.picam2 = None
        try:
            if self.display is not None:
                self.display.close()
        except Exception:
            pass
        self.display = None
        print("Done.")


if __name__ == "__main__":
    tester = MinimalTeam3Controller(use_display=True, display_fullscreen=True)
    tester.run()