"""
display.py - Student-facing display client (simplified).

Usage:
    from display import RobotDisplay
    disp = RobotDisplay()
    disp.set_buttons([...], position="bottom")
    disp.set_status("Auto: ON", position="top", bg_color=(0,0,0), text_color=(255,255,255))
    disp.set_frame(frame_bgr)   # OpenCV BGR uint8 frame
    events = disp.poll_events()
    disp.close()

Notes:
- set_frame expects OpenCV-native BGR uint8 ndarray (H,W,3)
- All annotation should be done in OpenCV before calling set_frame()
- This display scales the input frame to FIT within the active area (no cropping).
  The full frame is always visible; black bars may appear (letterbox/pillarbox).
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import queue
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ui import run_ui  # ui.py must be in the same directory


RGB = Tuple[int, int, int]
Rect = Tuple[int, int, int, int]  # x, y, w, h


# ----------------------------
# Uniform UI style (internal)
# ----------------------------
BUTTON_STRIP_THICKNESS_PX = 110   # touch-friendly on 800x480
STATUS_BAR_HEIGHT_PX = 60
PADDING_PX = 10
BUTTON_GAP_PX = 8

DEFAULT_BUTTON_FONT_SIZE = 26
DEFAULT_STATUS_FONT_SIZE = 24

PRESSED_HIGHLIGHT_MS = 140


def _validate_rgb(color: RGB, name: str = "color") -> None:
    if (
        not isinstance(color, tuple)
        or len(color) != 3
        or any((not isinstance(c, int) or c < 0 or c > 255) for c in color)
    ):
        raise ValueError(f"{name} must be an RGB tuple of ints 0..255, got: {color!r}")


def _sanitize_single_line(text: str) -> str:
    if text is None:
        return ""
    return str(text).replace("\r", " ").replace("\n", " ").strip()


@dataclass
class _ButtonsState:
    position: str  # "top"|"bottom"|"left"|"right"
    buttons: List[dict]  # each: handle,text,bg_color,text_color


@dataclass
class _StatusState:
    position: str  # "top"|"bottom"
    text: str
    bg_color: RGB
    text_color: RGB


class RobotDisplay:
    """
    Main class students use.

    - set_frame(frame_bgr): updates the live image in the active area using FIT-TO-WINDOW.
      (No cropping; full frame visible; may add bars.)
    - set_buttons(...): show up to 4 overlay buttons and receive events via poll_events().
    - set_status(...): show a single-line status bar with ellipsis truncation (handled in ui.py).
    """

    def __init__(
        self,
        *,
        fullscreen: bool = True,
        fps: int = 30,
        size: Tuple[int, int] = (800, 480),
        bg_color: RGB = (0, 0, 0),
    ):
        self.fullscreen = bool(fullscreen)
        self.fps = int(fps) if int(fps) > 0 else 30
        self.width, self.height = int(size[0]), int(size[1])

        self.bg_color = bg_color
        _validate_rgb(self.bg_color, "bg_color")

        self._closed = False
        self._canvas_lock = mp.Lock()

        # Overlay state
        self._buttons_state: Optional[_ButtonsState] = None
        self._status_state: Optional[_StatusState] = None

        # Layout computed locally (screen coords)
        self._layout: Dict[str, object] = {}
        self._compute_layout()

        # Shared memory: full-screen BGR canvas
        nbytes = self.width * self.height * 3  # uint8 BGR
        self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
        self._frame_counter = mp.Value("I", 0)

        self._canvas = np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=self._shm.buf)
        self._canvas[:, :] = self.bg_color

        # Queues
        self._cmd_q: mp.Queue = mp.Queue()
        self._event_q: mp.Queue = mp.Queue()

        # Start UI process
        initial_state = self._build_state_payload()
        self._ui_proc = mp.Process(
            target=run_ui,
            args=(
                self._shm.name,
                self._frame_counter,
                (self.width, self.height),
                self.fps,
                self.fullscreen,
                self._cmd_q,
                self._event_q,
                initial_state,
                self._canvas_lock,
            ),
            daemon=True,
        )
        self._ui_proc.start()

        self._send_overlay_update()
        atexit.register(self.close)

    # -------------------------
    # Lifecycle
    # -------------------------
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            self._cmd_q.put_nowait({"type": "QUIT"})
        except Exception:
            pass

        try:
            if self._ui_proc.is_alive():
                self._ui_proc.join(timeout=1.0)
            if self._ui_proc.is_alive():
                self._ui_proc.terminate()
                self._ui_proc.join(timeout=0.5)
            if self._ui_proc.is_alive():
                self._ui_proc.kill()
                self._ui_proc.join(timeout=0.5)
            self._ui_proc.close()
        except Exception:
            pass
        self._ui_proc = None

        try:
            self._cmd_q.cancel_join_thread()
        except Exception:
            pass
        try:
            self._cmd_q.close()
        except Exception:
            pass
        try:
            self._event_q.cancel_join_thread()
        except Exception:
            pass
        try:
            self._event_q.close()
        except Exception:
            pass

        try:
            self._shm.close()
        except Exception:
            pass
        try:
            self._shm.unlink()
        except Exception:
            pass

    # -------------------------
    # Overlay API
    # -------------------------
    def set_buttons(self, buttons: List[dict], *, position: str) -> None:
        pos = str(position).lower().strip()
        if pos not in ("top", "bottom", "left", "right"):
            raise ValueError("buttons position must be one of: top, bottom, left, right")

        if buttons is None:
            buttons = []
        if not isinstance(buttons, list):
            raise ValueError("buttons must be a list of dicts")
        if len(buttons) > 4:
            raise ValueError("buttons must contain at most 4 items")

        seen = set()
        norm_buttons: List[dict] = []
        for b in buttons:
            if not isinstance(b, dict):
                raise ValueError("each button must be a dict")
            handle = str(b.get("handle", "")).strip()
            text = str(b.get("text", "")).strip()
            if not handle:
                raise ValueError("each button must have a non-empty 'handle'")
            if handle in seen:
                raise ValueError(f"duplicate button handle: {handle}")
            seen.add(handle)

            bg = b.get("bg_color", (40, 40, 40))
            tc = b.get("text_color", (255, 255, 255))
            _validate_rgb(bg, f"bg_color for {handle}")
            _validate_rgb(tc, f"text_color for {handle}")

            norm_buttons.append({"handle": handle, "text": text, "bg_color": bg, "text_color": tc})

        self._buttons_state = _ButtonsState(position=pos, buttons=norm_buttons) if norm_buttons else None
        self._compute_layout()
        self._send_overlay_update()

    def clear_buttons(self) -> None:
        self._buttons_state = None
        self._compute_layout()
        self._send_overlay_update()

    def set_status(
        self,
        text: str,
        *,
        position: str,
        bg_color: RGB,
        text_color: RGB,
    ) -> None:
        pos = str(position).lower().strip()
        if pos not in ("top", "bottom"):
            raise ValueError("status position must be one of: top, bottom")

        _validate_rgb(bg_color, "status bg_color")
        _validate_rgb(text_color, "status text_color")

        clean_text = _sanitize_single_line(text)

        self._status_state = _StatusState(position=pos, text=clean_text, bg_color=bg_color, text_color=text_color)
        self._compute_layout()
        self._send_overlay_update()

    def clear_status(self) -> None:
        self._status_state = None
        self._compute_layout()
        self._send_overlay_update()

    # -------------------------
    # Events
    # -------------------------
    def poll_events(self) -> List[str]:
        events: List[str] = []
        while True:
            try:
                ev = self._event_q.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            if isinstance(ev, str):
                events.append(ev)
        return events

    # -------------------------
    # Frames
    # -------------------------
    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Display an OpenCV-native BGR uint8 frame into the active area.
        Behavior: FIT-TO-WINDOW (no cropping). Full frame always visible.
        """
        if self._closed:
            return

        if not isinstance(frame_bgr, np.ndarray):
            raise ValueError("set_frame expects a numpy ndarray (OpenCV image)")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("set_frame expects an image shaped (H,W,3)")
        if frame_bgr.dtype != np.uint8:
            raise ValueError("set_frame expects dtype uint8 (OpenCV native)")

        ax, ay, aw, ah = self._layout["active_rect"]  # screen coords
        aw, ah = int(aw), int(ah)
        if aw <= 0 or ah <= 0:
            return

        src_h, src_w = frame_bgr.shape[:2]
        if src_w <= 0 or src_h <= 0:
            return

        # Fill active area with background (creates letterbox/pillarbox bars)
 
        # Fit-to-window scaling (preserve aspect ratio)
        scale = min(aw / src_w, ah / src_h)
        dw = max(1, int(round(src_w * scale)))
        dh = max(1, int(round(src_h * scale)))

        resized = cv2.resize(frame_bgr, (dw, dh), interpolation=cv2.INTER_LINEAR)

        # Center inside active rect
        dx = ax + (aw - dw) // 2
        dy = ay + (ah - dh) // 2

        with self._canvas_lock:
            self._canvas[ay : ay + ah, ax : ax + aw, :] = self.bg_color
            self._canvas[dy : dy + dh, dx : dx + dw, :] = resized

        with self._frame_counter.get_lock():
            self._frame_counter.value += 1

    def clear_active(self, *, color: Optional[RGB] = None) -> None:
        """
        Fill only the active area with a solid color.
        Note: If your main loop continues calling set_frame(), the next frame will overwrite this.
        """
        if self._closed:
            return
        c = self.bg_color if color is None else color
        _validate_rgb(c, "clear_active color")

        ax, ay, aw, ah = self._layout["active_rect"]
        aw, ah = int(aw), int(ah)
        if aw <= 0 or ah <= 0:
            return

        with self._canvas_lock:
            self._canvas[ay : ay + ah, ax : ax + aw, :] = c
        with self._frame_counter.get_lock():
            self._frame_counter.value += 1

    # -------------------------
    # Private helpers
    # -------------------------
    def _send_cmd(self, cmd: dict) -> None:
        if self._closed:
            return
        try:
            self._cmd_q.put_nowait(cmd)
        except Exception:
            pass  # robot should keep running if UI is down

    def _build_state_payload(self) -> dict:
        return {
            "buttons": self._buttons_state.buttons if self._buttons_state else [],
            "buttons_position": self._buttons_state.position if self._buttons_state else None,
            "status": (
                {
                    "text": self._status_state.text,
                    "position": self._status_state.position,
                    "bg_color": self._status_state.bg_color,
                    "text_color": self._status_state.text_color,
                }
                if self._status_state
                else None
            ),
            "layout": {
                "status_rect": self._layout.get("status_rect"),
                "button_rects": self._layout.get("button_rects", {}),
            },
            "style": {
                "button_strip_thickness": BUTTON_STRIP_THICKNESS_PX,
                "status_bar_height": STATUS_BAR_HEIGHT_PX,
                "padding": PADDING_PX,
                "button_gap": BUTTON_GAP_PX,
                "button_font_size": DEFAULT_BUTTON_FONT_SIZE,
                "status_font_size": DEFAULT_STATUS_FONT_SIZE,
                "pressed_highlight_ms": PRESSED_HIGHLIGHT_MS,
            },
        }

    def _send_overlay_update(self) -> None:
        payload = self._build_state_payload()
        self._send_cmd({"type": "SET_OVERLAYS", **payload})

    def _compute_layout(self) -> None:
        """
        Compute active rect, status rect, and button rects in SCREEN coords.

        Active rect is the remaining area after reserving:
        - status bar (top or bottom)
        - button strip (top/bottom/left/right)
        """
        W, H = self.width, self.height

        top_used = 0
        bottom_used = 0
        left_used = 0
        right_used = 0

        status_rect = None
        button_strip_rect = None
        button_rects: Dict[str, Rect] = {}

        buttons_pos = self._buttons_state.position if self._buttons_state else None
        status_pos = self._status_state.position if self._status_state else None

        def reserve_top(height_px: int) -> Rect:
            nonlocal top_used
            r = (0, top_used, W, height_px)
            top_used += height_px
            return r

        def reserve_bottom(height_px: int) -> Rect:
            nonlocal bottom_used
            r = (0, H - bottom_used - height_px, W, height_px)
            bottom_used += height_px
            return r

        def reserve_left(width_px: int) -> Rect:
            nonlocal left_used
            r = (left_used, top_used, width_px, H - top_used - bottom_used)
            left_used += width_px
            return r

        def reserve_right(width_px: int) -> Rect:
            nonlocal right_used
            r = (W - right_used - width_px, top_used, width_px, H - top_used - bottom_used)
            right_used += width_px
            return r

        # TOP stack: status above buttons if both top
        if status_pos == "top":
            status_rect = reserve_top(STATUS_BAR_HEIGHT_PX)
        if buttons_pos == "top":
            button_strip_rect = reserve_top(BUTTON_STRIP_THICKNESS_PX)

        # BOTTOM stack: buttons above status if both bottom
        if buttons_pos == "bottom":
            button_strip_rect = reserve_bottom(BUTTON_STRIP_THICKNESS_PX)
        if status_pos == "bottom":
            status_rect = reserve_bottom(STATUS_BAR_HEIGHT_PX)

        # LEFT/RIGHT: buttons only
        if buttons_pos == "left":
            button_strip_rect = reserve_left(BUTTON_STRIP_THICKNESS_PX)
        if buttons_pos == "right":
            button_strip_rect = reserve_right(BUTTON_STRIP_THICKNESS_PX)

        # Active area after reservations
        ax = left_used
        ay = top_used
        aw = W - left_used - right_used
        ah = H - top_used - bottom_used
        active_rect = (ax, ay, max(0, aw), max(0, ah))

        # Button rects
        if self._buttons_state and button_strip_rect:
            bx, by, bw, bh = button_strip_rect
            btns = self._buttons_state.buttons
            n = len(btns)
            if n > 0:
                if buttons_pos in ("top", "bottom"):
                    each_w = bw // n
                    for i, b in enumerate(btns):
                        x = bx + i * each_w
                        w2 = each_w if i < n - 1 else (bx + bw - x)
                        button_rects[b["handle"]] = (x, by, w2, bh)
                else:
                    each_h = bh // n
                    for i, b in enumerate(btns):
                        y = by + i * each_h
                        h2 = each_h if i < n - 1 else (by + bh - y)
                        button_rects[b["handle"]] = (bx, y, bw, h2)

        self._layout = {
            "active_rect": active_rect,
            "status_rect": status_rect,
            "button_strip_rect": button_strip_rect,
            "button_rects": button_rects,
        }

