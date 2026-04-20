import time
import cv2

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

from robot_types import RobotState, SensorData, MotionMode
from team1_navigation_imu import Team1NavigationIMU

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


class MinimalTeam1Controller:
    def __init__(self):
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
        
        self.picam2 = None
        if CAMERA_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(main={"size": (640, 480), "format": "XRGB8888"})
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(1.5)
                print("Camera enabled.")
            except Exception as e:
                print(f"Camera init failed: {e}")
                self.picam2 = None

        self.running = True

    def run(self):
        print("Starting Minimal Team 1 Controller (Lane Follow Only)...")
        print("Press PS5 Cross button (or Ctrl+C) to quit.")
        
        try:
            while self.running:
                loop_start = time.time()
                self.sensors.now = loop_start
                
                # --- A. READ INPUT & SENSORS ---
                # Check PS5 controller for emergency stop or exit
                self.ps5.check_controls()
                if self.ps5.control_request["reqCross"]:
                    print("Cross button pressed. Exiting...")
                    self.running = False
                    break
                
                # Read IMU
                self.sensors.imu_delta = self.imu.get_delta()
                
                # Grab the latest camera frame
                if self.picam2 is not None:
                    try:
                        # Capture as a BGR numpy array compatible with OpenCV
                        self.sensors.frame = self.picam2.capture_array()
                    except Exception as e:
                        print(f"Frame capture error: {e}")
                        self.sensors.frame = None
                else:
                    print("self.picam2 is None")
                
                if self.sensors.frame is None:
                    time.sleep(0.01)
                    print("here 3")
                    continue

                # --- B. PROCESS TEAM 1 LOGIC ---
                updates = self.team1.update(self.state, self.sensors)
                print(updates)
                
                # --- C. APPLY MOTOR COMMANDS ---
                if "drive_command" in updates:
                    speed = updates["drive_command"].get("speed", 0)
                    turn = updates["drive_command"].get("turn", 0)
                    
                    # Send to Sabertooth motors
                    self.saber.drive(speed, turn)
                    
                # Check for an IMU reset request
                if updates.get("reset_imu", False):
                    print("Team 1 requested IMU reset.")
                    self.imu.zero_reference()
                    self.sensors.imu_delta = 0.0

                # --- D. LOOP TIMING ---
                # Maintain roughly 30Hz loop rate (0.033s)
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, 0.033 - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught. Stopping robot...")
        finally:
            self.shutdown()

    def shutdown(self):
        print("Shutting down hardware...")
        self.saber.stop()
        if self.picam2 is not None:
            self.picam2.stop()
        self.imu.close()
        print("Done.")


if __name__ == "__main__":
    tester = MinimalTeam1Controller()
    tester.run()