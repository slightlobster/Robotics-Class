import time
import cv2
import pygame

# Hardware and Team modules
from ps5_controller import PS5_Controller
from sabertooth import Sabertooth
from team4_sign_actions import Team4SignActions
from robot_types import AutoCurrentSign, RobotState, SensorData, IntersectionStage, MotionMode

try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    Picamera2 = None
    CAMERA_AVAILABLE = False

class DpadTurnController:
    def __init__(self):
        # 1. Logic Modules
        self.state = RobotState()
        self.sensors = SensorData()
        self.team4 = Team4SignActions()
        
        # 2. Hardware Initialization
        self.ps5 = PS5_Controller()
        self.ps5.initialize_controller()
        self.saber = Sabertooth()
        
        self.picam2 = None
        if CAMERA_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(main={"size": (640, 480), "format": "XRGB8888"})
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(1.5)
                print("Camera active for visual feedback.")
            except Exception as e:
                print(f"Camera init failed: {e}")

        self.running = True

    def trigger_action(self, sign_type):
        """Helper to lock in the initial sign and start the action immediately."""
        print(f"\n--- ACTION TRIGGERED: {sign_type.value} ---")
        self.state.auto_mode = True
        self.state.auto_current_sign = sign_type
        
        # Set stage based on sign type
        if sign_type == AutoCurrentSign.PAUSE:
            self.state.intersection_stage = IntersectionStage.PAUSE_ACTION
        else:
            self.state.intersection_stage = IntersectionStage.TURN_ACTION
            
        self.team4.reset_action_state()

    def run(self):
        print("--- D-PAD TURN TESTER ---")
        print("D-PAD LEFT : 90° Left Turn")
        print("D-PAD RIGHT: 90° Right Turn")
        print("D-PAD UP   : Pause/Stop-and-Go")
        print("CROSS      : Emergency Stop / Quit")
        
        try:
            while self.running:
                now = time.time()
                self.sensors.now = now
                pygame.event.pump()
                self.ps5.check_controls()

                # --- 1. D-PAD INPUTS (Goal: Just take the initial sign) ---
                if not self.state.auto_mode:
                    if self.ps5.control_request["reqArrowLeft"]:
                        self.trigger_action(AutoCurrentSign.LEFT)
                    elif self.ps5.control_request["reqArrowRight"]:
                        self.trigger_action(AutoCurrentSign.RIGHT)
                    elif self.ps5.control_request["reqArrowUp"]:
                        self.trigger_action(AutoCurrentSign.PAUSE)
                    elif self.ps5.control_request["reqArrowDown"]:
                        print("MESSAGE: Down button pressed - No action assigned.")

                # --- 2. EXECUTE TEAM 4 (Goal: Stop/Go and 90° Turns) ---
                if self.state.auto_mode:
                    updates = self.team4.update(self.state, self.sensors)
                    
                    if updates.get("action_complete"):
                        print("Task Complete. Returning to standby.")
                        self.state.auto_mode = False
                        self.saber.stop()
                    else:
                        cmd = updates.get("drive_command", {"speed": 0, "turn": 0})
                        self.saber.drive(cmd["speed"], cmd["turn"])

                # --- 3. SAFETY STOP ---
                if self.ps5.control_request["reqCross"]:
                    if self.state.auto_mode:
                        print("Emergency Stop Triggered.")
                        self.state.auto_mode = False
                        self.saber.stop()
                    else:
                        print("Exiting...")
                        self.running = False
                
                if self.ps5.control_request.get("reqMade", False):
                    self.ps5.reset_controller_state()

                time.sleep(0.02)

        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        self.saber.stop()
        if self.picam2: self.picam2.stop()
        print("Hardware safely closed.")

if __name__ == "__main__":
    tester = DpadTurnController()
    tester.run()