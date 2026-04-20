import time
import cv2
import pygame

# Hardware imports
from ps5_controller import PS5_Controller
from team3_signs_end import Team3SignsEnd

try:
    from picamera2 import Picamera2
    CAMERA_AVAILABLE = True
except ImportError:
    Picamera2 = None
    CAMERA_AVAILABLE = False


class MinimalTeam3Controller:
    def __init__(self):
        # 1. Instantiate Team 3 module
        self.team3 = Team3SignsEnd()
        
        # 2. Initialize Hardware using your fixed code
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
                print("Camera enabled.")
            except Exception as e:
                print(f"Camera init failed: {e}")
                self.picam2 = None

        self.running = True
        self.latest_frame = None

        self.ps5_interval = 0.02
        self.ps5_last_check = time.time()

    def run(self):
        print("Starting Minimal Team 3 Controller (Sign Detection Only)...")
        print("Press CROSS to quit.")
        print("Press SQUARE to capture a frame and perform a sign read.")
        
        try:
            while self.running:
                loop_start = time.time()

                # Controller checks
                now = time.time()
                if now - self.ps5_last_check >= self.ps5_interval:
                    pygame.event.pump()
                    self.ps5.check_controls()
                    self.ps5_last_check = now
                
                # --- A. READ SENSORS ---
                # Continuously pull frames so the camera buffer doesn't go stale
                if self.picam2 is not None:
                    try:
                        self.latest_frame = self.picam2.capture_array()
                    except Exception as e:
                        print(f"Frame capture error: {e}")
                        self.latest_frame = None

                # Check PS5 controller inputs
                self.ps5.check_controls()
                
                # Exit condition
                if self.ps5.control_request["reqCross"]:
                    print("Cross button pressed. Exiting...")
                    self.running = False
                    break
                
                # --- B. PROCESS TEAM 3 LOGIC ON DEMAND ---
                # Trigger a read when Square is pressed
                if self.ps5.control_request["reqSquare"]:
                    if self.latest_frame is None:
                        print("Cannot read sign: Waiting for camera frame...")
                    else:
                        print("\n--- TRIGGERING SIGN READ ---")
                        # Create the lightweight job dict that the worker normally gets
                        job = {"type": "SIGN_READ"}
                        
                        # Pass a copy of the frame to Team 3 so they don't mutate our buffer
                        result = self.team3.process(self.latest_frame.copy(), job)
                        
                        # Output the result
                        print(f"Team 3 Result: {result}")
                        
                        # OPTIONAL: Save the frame to disk so Team 3 can review what the camera saw
                        # filename = f"team3_debug_{int(time.time())}.jpg"
                        # cv2.imwrite(filename, self.latest_frame)
                        # print(f"Saved debug frame to {filename}")
                        
                        # Simple debounce to prevent reading 30 times from a single button press
                        time.sleep(0.5)

                if self.ps5.control_request.get("reqMade", False):
                    self.ps5.reset_controller_state()

                # --- C. LOOP TIMING ---
                # Maintain roughly 30Hz loop rate (0.033s) to keep camera buffer happy
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, 0.033 - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt caught. Stopping...")
        finally:
            self.shutdown()

    def shutdown(self):
        print("Shutting down hardware...")
        if self.picam2 is not None:
            self.picam2.stop()
        print("Done.")


if __name__ == "__main__":
    tester = MinimalTeam3Controller()
    tester.run()