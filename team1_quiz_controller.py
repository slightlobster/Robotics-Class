import time
import cv2
import pygame
import numpy as np

# Hardware imports from the actual project structure
from ps5_controller import PS5_Controller
from sabertooth import Sabertooth

try:
    from imu_controller import IMUDevice
    IMU_AVAILABLE = True
except ImportError:
    IMUDevice = None
    IMU_AVAILABLE = False

try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    print("Failed to grab camera")
    Picamera2 = None
    CAMERA_AVAILABLE = False

try:
    from display import RobotDisplay
    DISPLAY_AVAILABLE = True
except Exception:
    RobotDisplay = None
    DISPLAY_AVAILABLE = False

from robot_types import RobotState, SensorData, MotionMode
from team1_navigation_imu import Team1NavigationIMU


# ---------------------------------------------------------------------------
# Display helpers (copied from main autonomous controller)
# ---------------------------------------------------------------------------

def draw_decision_banner(frame: np.ndarray, text: str, *,
                         bg=(0, 255, 255), fg=(0, 0, 0)) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 42), bg, -1)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, fg, 2, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)


def draw_crosshair(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)
    cv2.line(frame, (0, h // 2), (w, h // 2), (255, 255, 0), 1)


def draw_state_overlay(frame: np.ndarray, state: RobotState,
                       imu_delta: float, paused: bool) -> None:
    lines = [
        f"Mode: {'PAUSED' if paused else 'RUNNING'}",
        f"IMU delta: {imu_delta:.1f} deg",
        f"Speed: {state.requested_speed}  Turn: {state.requested_turn}",
        state.debug_message,
    ]
    y = 105
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += 26


# ---------------------------------------------------------------------------
# IMU wrapper (unchanged from original)
# ---------------------------------------------------------------------------

class IMUInterface:
    """Small safety wrapper around the IMU device extracted from the main controller."""
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


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

class MinimalTeam1Controller:
    def __init__(self, use_display: bool = True, display_fullscreen: bool = True):
        # 1. Initialize State and Sensors
        self.state = RobotState()
        self.sensors = SensorData(now=time.time(), frame=None, imu_delta=0.0)

        # 2. Instantiate Team 1 module
        self.team1 = Team1NavigationIMU()

        # 3. Lock the state for continuous lane following
        self.state.auto_mode = True
        self.state.motion_mode = MotionMode.TEAM1_LANE_FOLLOW
        self.state.auto_forward_speed = 80  # Base testing speed

        # 4. Initialize Hardware
        self.ps5 = PS5_Controller()
        self.ps5.initialize_controller()

        self.saber = Sabertooth()
        self.saber.set_ramping(21)
        self.saber.stop()

        self.imu = IMUInterface()
        self.imu.zero_reference()

        self.ps5_interval = 0.02
        self.ps5_last_check = time.time()

        self.paused = True  # Start paused — circle resumes (unchanged)

        # Camera (unchanged from original)
        self.picam2 = None
        if CAMERA_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": (640, 480), "format": "XRGB8888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(1.5)
                print("Camera enabled.")
            except Exception as e:
                print(f"Camera init failed: {e}")
                self.picam2 = None

        # Display (new — gracefully disabled if unavailable)
        self.display = None
        if use_display and DISPLAY_AVAILABLE:
            try:
                self.display = RobotDisplay(
                    fullscreen=display_fullscreen,
                    fps=30,
                    size=(800, 480),
                    bg_color=(0, 0, 0),
                )
                self.display.set_buttons(
                    [
                        {"handle": "resume", "text": "RESUME",
                         "bg_color": (30, 80, 200), "text_color": (255, 255, 255)},
                        {"handle": "pause",  "text": "PAUSE",
                         "bg_color": (200, 60, 60), "text_color": (255, 255, 255)},
                        {"handle": "quit",   "text": "QUIT",
                         "bg_color": (80, 80, 80),  "text_color": (255, 255, 255)},
                    ],
                    position="left",
                )
                self._update_display_status()
                print(f"Display enabled (fullscreen={display_fullscreen})")
            except Exception as exc:
                print(f"Display init failed: {exc}")
                self.display = None

        # FPS + display timing (new)
        self.fps_smooth      = 0.0
        self.fps_alpha       = 0.15
        self.last_frame_time  = time.time()
        self.display_interval = 1.0 / 30.0
        self.last_display_time = time.time()

        self.is_moving_manual = False
        self.running = True

    # ------------------------------------------------------------------
    # Display helpers (new)
    # ------------------------------------------------------------------

    def _update_display_status(self) -> None:
        """Push the current status string to the display's top bar."""
        if self.display is None:
            return
        try:
            status = (
                "PAUSED | Circle = resume + reset IMU, Square = pause, Cross = quit"
                if self.paused
                else "RUNNING | Square = pause, Cross = quit"
            )
            self.display.set_status(
                status, position="top",
                bg_color=(15, 15, 15), text_color=(255, 255, 255),
            )
        except Exception:
            pass

    def _poll_display_events(self) -> None:
        """
        Handle touchscreen button presses.
        Each button mirrors the exact same logic as its PS5 equivalent.
        """
        if self.display is None:
            return
        for ev in self.display.poll_events():
            if ev == "resume":
                # Mirrors circle button exactly
                print("Display RESUME pressed. Resuming...")
                self.paused = False
                self.imu.zero_reference()
                self.sensors.imu_delta = 0.0
                print("IMU reset to zero.")
                self._update_display_status()
            elif ev == "pause":
                # Mirrors square button exactly
                print("Display PAUSE pressed. Pausing...")
                self.paused = True
                self.saber.stop()
                self._update_display_status()
            elif ev == "quit":
                print("Display QUIT pressed. Exiting...")
                self.running = False

    def _submit_display_frame(self) -> None:
        """Annotate and push the current frame to the display process."""
        if self.display is None or self.sensors.frame is None:
            return
        now = self.sensors.now
        if now - self.last_display_time < self.display_interval:
            return
        self.last_display_time = now

        frame = self.sensors.frame.copy()
        banner_text = "PAUSED" if self.paused else "RUNNING — LANE FOLLOW"
        banner_bg   = (80, 80, 80) if self.paused else (0, 150, 0)
        draw_decision_banner(frame, banner_text, bg=banner_bg)
        draw_fps(frame, self.fps_smooth)
        draw_crosshair(frame)
        draw_state_overlay(frame, self.state, self.sensors.imu_delta, self.paused)
        try:
            self.display.set_frame(frame)
        except Exception as exc:
            print(f"Display frame error: {exc}")
            try:
                self.display.close()
            except Exception:
                pass
            self.display = None

    # ------------------------------------------------------------------
    # Main loop — original structure preserved line-for-line
    # ------------------------------------------------------------------

    def run(self):
        print("Starting Minimal Team 1 Controller (Lane Follow Only)...")
        print("Press PS5 Cross button (or Ctrl+C) to quit.")
        print("Press PS5 Circle button to start/resume (also resets IMU).")
        print("Press PS5 Square button to stop/pause.")

        try:
            while self.running:
                loop_start = time.time()
                self.sensors.now = loop_start

                # Controller checks (unchanged)
                now = time.time()
                if now - self.ps5_last_check >= self.ps5_interval:
                    pygame.event.pump()
                    self.ps5.check_controls()
                    self.ps5_last_check = now

                # --- A. READ INPUT & SENSORS (original button logic unchanged) ---
                if self.ps5.control_request["reqCross"]:
                    print("Cross button pressed. Exiting...")
                    self.running = False
                    break

                if self.ps5.control_request["reqSquare"]:
                    print("Square button pressed. Pausing...")
                    self.paused = True
                    self.saber.stop()
                    self._update_display_status()

                if self.ps5.control_request["reqCircle"]:
                    print("Circle button pressed. Resuming...")
                    self.paused = False
                    self.imu.zero_reference()
                    self.sensors.imu_delta = 0.0
                    print("IMU reset to zero.")
                    self._update_display_status()

                if self.ps5.control_request.get("reqMade", False):
                    self.ps5.reset_controller_state()

                # Poll display touchscreen events (new)
                self._poll_display_events()

                if not self.paused:
                    # Read IMU (unchanged)
                    self.sensors.imu_delta = self.imu.get_delta()

                    # Grab the latest camera frame (unchanged)
                    if self.picam2 is not None:
                        try:
                            raw = self.picam2.capture_array()
                            # Convert XRGB8888 → BGR so OpenCV and Team 1 both work
                            self.sensors.frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

                            # FPS tracking (new)
                            dt = loop_start - self.last_frame_time
                            self.last_frame_time = loop_start
                            if dt > 0:
                                fps_inst = 1.0 / dt
                                self.fps_smooth = (
                                    fps_inst if self.fps_smooth == 0
                                    else (self.fps_alpha * fps_inst
                                          + (1 - self.fps_alpha) * self.fps_smooth)
                                )
                        except Exception as e:
                            print(f"Frame capture error: {e}")
                            self.sensors.frame = None
                    else:
                        print("self.picam2 is None")

                    if self.sensors.frame is None:
                        time.sleep(0.01)
                        print("here 3")
                        continue

                    # --- B. PROCESS TEAM 1 LOGIC (unchanged) ---
                    updates = self.team1.update(self.state, self.sensors)
                    print(updates)

                    # --- C. APPLY MOTOR COMMANDS (unchanged) ---
                    if "drive_command" in updates:
                        speed = updates["drive_command"].get("speed", 0)
                        turn  = updates["drive_command"].get("turn", 0)
                        self.state.requested_speed = speed
                        self.state.requested_turn  = turn
                        self.saber.drive(speed, turn)

                    if updates.get("reset_imu", False):
                        print("Team 1 requested IMU reset.")
                        self.imu.zero_reference()
                        self.sensors.imu_delta = 0.0

                    if "debug" in updates:
                        self.state.debug_message = updates["debug"]

                    # Submit display frame (new — after motor commands so
                    # debug_message is already updated)
                    self._submit_display_frame()

                    # --- D. LOOP TIMING (unchanged) ---
                    elapsed = time.time() - loop_start
                    sleep_time = max(0.0, 0.033 - elapsed)
                    time.sleep(sleep_time)

                else:
                    # Paused: manual joystick driving + keep display alive
                    if self.ps5.control_request.get("reqLeftJoyMade", False):
                        speed = -self.ps5.control_request.get("reqLeftJoyYValue", 0)
                        turn  = -self.ps5.control_request.get("reqLeftJoyXValue", 0)
                        self.saber.drive(speed, turn)
                        self.state.requested_speed = speed
                        self.state.requested_turn  = turn
                        self.is_moving_manual = True
                    else:
                        if self.is_moving_manual:
                            self.saber.stop()
                            self.state.requested_speed = 0
                            self.state.requested_turn  = 0
                            self.is_moving_manual = False

                    # Still grab frames so the display stays live while paused
                    if self.picam2 is not None:
                        try:
                            raw = self.picam2.capture_array()
                            self.sensors.frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                        except Exception:
                            pass

                    self._submit_display_frame()
                    time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught. Stopping robot...")
        finally:
            self.shutdown()

    def shutdown(self):
        print("Shutting down hardware...")
        self.saber.stop()
        if self.picam2 is not None:
            self.picam2.stop()
        print("here man.")
        self.imu.close()
        if self.display is not None:
            try:
                self.display.close()
            except Exception:
                pass
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass
        print("Done.")


if __name__ == "__main__":
    tester = MinimalTeam1Controller(use_display=True, display_fullscreen=True)
    tester.run()