#!/usr/bin/env python3
import os
os.environ["ULTRALYTICS_OFFLINE"] = "1"  # must be set before any ultralytics import

"""
test_team2.py — Standalone test harness for Team2Intersection.

Mocks the robot_auto_controller.py job inputs and drives the robot through
a full intersection sequence:
  1. Drive forward until pause_point_ready (sign reaches stop threshold)  → STOP
  2. Hold briefly, then advance to PRE_TURN_TRAVEL
  3. Drive forward until turn_point_ready (FORWARD_DRIVE_SECONDS elapsed) → STOP
  4. Hold briefly, then drive forward past the intersection
  5. Stop when intersection_cleared (sign gone after tracking)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import cv2
from picamera2 import Picamera2
from display import RobotDisplay
from sabertooth import Sabertooth
from robot_types import AutoRoadLocation, IntersectionStage
from team2_intersection import Team2Intersection

CRUISE_SPEED      = 50   # forward speed during approach and drive-past
CREEP_SPEED       = 30   # forward speed during pre-turn travel
STOP_HOLD_SECONDS = 1.5  # pause duration at each stop point
DRIVE_PAST_SECONDS = 3.0 # how long to drive forward after the turn point

# Test phase names
PHASE_APPROACH          = "APPROACH"
PHASE_STOPPED_AT_PAUSE  = "STOPPED AT PAUSE"
PHASE_PRE_TURN          = "PRE-TURN TRAVEL"
PHASE_STOPPED_AT_TURN   = "STOPPED AT TURN"
PHASE_DRIVE_PAST        = "DRIVE PAST"
PHASE_DONE              = "DONE"


# ---- phase → (IntersectionStage, AutoRoadLocation) mock values ----
PHASE_JOB_CONTEXT = {
    PHASE_APPROACH:         (IntersectionStage.TRAVEL_LANE,     AutoRoadLocation.IN_TRAVEL_LANE),
    PHASE_STOPPED_AT_PAUSE: (IntersectionStage.PAUSE_ACTION,    AutoRoadLocation.AT_PAUSE_LOCATION),
    PHASE_PRE_TURN:         (IntersectionStage.PRE_TURN_TRAVEL, AutoRoadLocation.AT_TURN_LOCATION),
    PHASE_STOPPED_AT_TURN:  (IntersectionStage.PRE_TURN_TRAVEL, AutoRoadLocation.AT_TURN_LOCATION),
    PHASE_DRIVE_PAST:       (IntersectionStage.TRAVEL_LANE,     AutoRoadLocation.IN_TRAVEL_LANE),
    PHASE_DONE:             (IntersectionStage.TRAVEL_LANE,     AutoRoadLocation.IN_TRAVEL_LANE),
}


def draw_annotations(frame, result):
    """Draw Team 2 ROI, boxes, lines, and points onto the frame."""
    roi = result.get("intersection_roi")
    if roi:
        x, y, rw, rh = map(int, roi)
        cv2.rectangle(frame, (x, y), (x + rw, y + rh), (0, 165, 255), 2)
        cv2.putText(frame, "T2 ROI", (x + 4, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    for box in result.get("intersection_boxes", []):
        if box and len(box) == 4:
            bx, by, bw, bh = map(int, box)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

    for line in result.get("intersection_lines", []):
        if line and len(line) == 4:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)

    for pt in result.get("intersection_points", []):
        if pt and len(pt) == 2:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 6, (0, 165, 255), -1)


def draw_overlay(frame, result, phase, sign_height, fps):
    """Draw the phase banner, 4 state indicators, label, and debug text."""
    h, w = frame.shape[:2]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 42), (0, 60, 120), -1)
    cv2.putText(frame, f"Phase: {phase}  |  sign_h={sign_height}px  |  FPS:{fps:4.1f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # 4 state indicators (right side panel)
    states = [
        ("sign_read_ready",      result.get("sign_read_ready",      False)),
        ("pause_point_ready",    result.get("pause_point_ready",    False)),
        ("turn_point_ready",     result.get("turn_point_ready",     False)),
        ("intersection_cleared", result.get("intersection_cleared", False)),
    ]

    panel_x = w - 270
    panel_top = 50
    panel_h = len(states) * 34 + 12
    cv2.rectangle(frame, (panel_x - 8, panel_top), (w - 4, panel_top + panel_h), (20, 20, 20), -1)

    for i, (name, value) in enumerate(states):
        y = panel_top + 28 + i * 34
        color = (0, 220, 60) if value else (80, 80, 200)
        marker = "[YES]" if value else "[ NO]"
        cv2.putText(frame, f"{marker} {name}", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)

    # Intersection label (bottom)
    label = result.get("intersection_label", "")
    if label:
        cv2.putText(frame, label, (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)

    # Debug line (bottom-most)
    debug = result.get("debug", "")
    if debug:
        cv2.putText(frame, debug, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": (640, 480), "format": "XRGB8888"}
    ))
    picam2.start()
    time.sleep(1.5)

    saber = Sabertooth()
    saber.set_ramping(5)

    disp = RobotDisplay(fullscreen=True, fps=30, size=(800, 480), bg_color=(0, 0, 0))
    disp.set_buttons(
        [
            {"handle": "quit",  "text": "QUIT",  "bg_color": (200, 0, 0),   "text_color": (255, 255, 255)},
            {"handle": "reset", "text": "RESET", "bg_color": (0, 150, 200), "text_color": (255, 255, 255)},
        ],
        position="bottom",
    )

    team2 = Team2Intersection()
    phase = PHASE_APPROACH
    phase_changed_at = time.time()
    frame_id = 0
    last_frame_time = time.time()
    fps_smooth = 0.0

    # Speed for each phase — sent every loop to keep Sabertooth auto-stop fed
    PHASE_SPEED = {
        PHASE_APPROACH:         CRUISE_SPEED,
        PHASE_STOPPED_AT_PAUSE: 0,
        PHASE_PRE_TURN:         CREEP_SPEED,
        PHASE_STOPPED_AT_TURN:  0,
        PHASE_DRIVE_PAST:       CRUISE_SPEED,
        PHASE_DONE:             0,
    }

    print(f"[test_team2] Starting in phase: {phase}")

    try:
        while True:
            now = time.time()

            # FPS smoothing
            dt = now - last_frame_time
            last_frame_time = now
            if dt > 0:
                fps_smooth = 0.15 * (1.0 / dt) + 0.85 * fps_smooth

            frame_bgra = picam2.capture_array()
            frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            frame_id += 1

            # Build mock job dict to simulate controller context
            stage_enum, loc_enum = PHASE_JOB_CONTEXT.get(phase, PHASE_JOB_CONTEXT[PHASE_APPROACH])
            job = {
                "type": "PROCESS",
                "frame_id": frame_id,
                "timestamp": now,
                "roi_xywh": None,
                "road_location": loc_enum.value,
                "intersection_stage": stage_enum.value,
                "auto_mode": True,
            }

            result = team2.process(frame, job)
            sign_height = team2.last_sign_height

            print(f"[test_team2] frame={frame_id:4d} phase={phase:<22s} sign_h={sign_height:3d}px  "
                  f"srr={result.get('sign_read_ready', False)!s:<5} "
                  f"ppr={result.get('pause_point_ready', False)!s:<5} "
                  f"tpr={result.get('turn_point_ready', False)!s:<5} "
                  f"fps={fps_smooth:.1f}")

            # --- Phase state machine ---
            prev_phase = phase

            if phase == PHASE_APPROACH:
                if result.get("pause_point_ready"):
                    phase = PHASE_STOPPED_AT_PAUSE
                    phase_changed_at = now

            elif phase == PHASE_STOPPED_AT_PAUSE:
                if now - phase_changed_at >= STOP_HOLD_SECONDS:
                    phase = PHASE_PRE_TURN
                    phase_changed_at = now

            elif phase == PHASE_PRE_TURN:
                if result.get("turn_point_ready"):
                    phase = PHASE_STOPPED_AT_TURN
                    phase_changed_at = now

            elif phase == PHASE_STOPPED_AT_TURN:
                if now - phase_changed_at >= STOP_HOLD_SECONDS:
                    phase = PHASE_DRIVE_PAST
                    phase_changed_at = now

            elif phase == PHASE_DRIVE_PAST:
                if now - phase_changed_at >= DRIVE_PAST_SECONDS:
                    phase = PHASE_DONE

            if phase != prev_phase:
                print(f"[test_team2] Phase transition: {prev_phase} -> {phase}  "
                      f"sign_h={sign_height} frame={frame_id}")

            # Send drive command every loop so Sabertooth auto-stop timer never expires
            saber.drive(PHASE_SPEED.get(phase, 0), 0)

            # Annotate and display
            draw_annotations(frame, result)
            draw_overlay(frame, result, phase, sign_height, fps_smooth)
            disp.set_frame(frame)

            # Button events
            for ev in disp.poll_events():
                if ev == "quit":
                    raise KeyboardInterrupt
                elif ev == "reset":
                    team2.reset_cycle("Manual reset")
                    phase = PHASE_APPROACH
                    phase_changed_at = now
                    saber.drive(CRUISE_SPEED, 0)

    except KeyboardInterrupt:
        pass
    finally:
        saber.stop()
        saber.close()
        picam2.stop()
        disp.close()
        time.sleep(0.3)


if __name__ == "__main__":
    main()
