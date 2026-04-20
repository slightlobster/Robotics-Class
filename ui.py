"""
ui.py - UI renderer process (pygame).

Simple renderer:
- Attaches to shared memory BGR canvas (display-sized, e.g. 800x480).
- Blits latest canvas when frame_counter changes.
- Draws overlays (status + buttons) on top.
- Emits button press events via event_queue (strings: handle).
- Handles only:
    - SET_OVERLAYS (layout/buttons/status/style update)
    - QUIT
"""

from __future__ import annotations

import queue
import signal
import time
from multiprocessing import shared_memory
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pygame

RGB = Tuple[int, int, int]
Rect = Tuple[int, int, int, int]


def _truncate_with_ellipsis(font: pygame.font.Font, text: str, max_width_px: int) -> str:
    """Single-line truncation with '...' if needed."""
    if max_width_px <= 0:
        return ""
    s = str(text)
    w, _ = font.size(s)
    if w <= max_width_px:
        return s

    ell = "..."
    ell_w, _ = font.size(ell)
    if ell_w >= max_width_px:
        return ell

    lo, hi = 0, len(s)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = s[:mid] + ell
        cw, _ = font.size(cand)
        if cw <= max_width_px:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    return best if best else ell


def _bgr_canvas_to_surface(canvas_bgr: np.ndarray) -> pygame.Surface:
    """
    Convert (H,W,3) BGR uint8 to pygame Surface.
    Use a tightly packed RGB buffer to avoid surfarray stride/orientation issues.
    """
    rgb = np.ascontiguousarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    return pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB").copy()


def run_ui(
    shm_name: str,
    frame_counter,
    size: Tuple[int, int],
    fps: int,
    fullscreen: bool,
    cmd_queue,
    event_queue,
    initial_state: dict,
    canvas_lock,
) -> None:
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:
        pass

    W, H = int(size[0]), int(size[1])
    fps = int(fps) if int(fps) > 0 else 30

    shm = shared_memory.SharedMemory(name=shm_name, create=False)
    canvas = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm.buf)

    pygame.init()
    flags = pygame.FULLSCREEN if fullscreen else 0
    screen = pygame.display.set_mode((W, H), flags)
    pygame.display.set_caption("Robot Display")
    clock = pygame.time.Clock()

    # -------------------------
    # State (overlays/layout)
    # -------------------------
    layout = (initial_state or {}).get("layout", {}) or {}
    buttons = (initial_state or {}).get("buttons", []) or []
    status = (initial_state or {}).get("status", None)
    style = (initial_state or {}).get("style", {}) or {}

    button_font_size = int(style.get("button_font_size", 26))
    status_font_size = int(style.get("status_font_size", 24))
    pressed_highlight_ms = int(style.get("pressed_highlight_ms", 140))
    padding = int(style.get("padding", 10))

    button_font = pygame.font.Font(None, button_font_size)
    status_font = pygame.font.Font(None, status_font_size)

    status_rect: Optional[Rect] = layout.get("status_rect", None)
    button_rects: Dict[str, Rect] = layout.get("button_rects", {}) or {}

    # Pressed feedback
    pressed_handle: Optional[str] = None
    pressed_until: float = 0.0

    last_frame_id = -1
    base_surface: Optional[pygame.Surface] = None

    def update_overlays(payload: dict) -> None:
        nonlocal layout, buttons, status, style
        nonlocal button_font_size, status_font_size, pressed_highlight_ms, padding
        nonlocal button_font, status_font
        nonlocal status_rect, button_rects

        layout = payload.get("layout", layout) or {}
        buttons = payload.get("buttons", buttons) or []
        status = payload.get("status", status)
        style = payload.get("style", style) or style

        button_font_size = int(style.get("button_font_size", button_font_size))
        status_font_size = int(style.get("status_font_size", status_font_size))
        pressed_highlight_ms = int(style.get("pressed_highlight_ms", pressed_highlight_ms))
        padding = int(style.get("padding", padding))

        button_font = pygame.font.Font(None, button_font_size)
        status_font = pygame.font.Font(None, status_font_size)

        status_rect = layout.get("status_rect", None)
        button_rects = layout.get("button_rects", {}) or {}

    def handle_command(cmd: dict) -> bool:
        """Return True if should quit."""
        t = cmd.get("type")
        if t == "QUIT":
            return True
        if t == "SET_OVERLAYS":
            update_overlays(cmd)
        return False

    # -------------------------
    # Overlay drawing
    # -------------------------
    def draw_status_overlay() -> None:
        if not status_rect or not status:
            return
        x, y, w, h = map(int, status_rect)
        bg = tuple(status.get("bg_color", (0, 0, 0)))
        tc = tuple(status.get("text_color", (255, 255, 255)))
        text = str(status.get("text", ""))

        pygame.draw.rect(screen, bg, pygame.Rect(x, y, w, h), 0)

        max_text_w = max(0, w - 2 * padding)
        line = _truncate_with_ellipsis(status_font, text, max_text_w)
        surf = status_font.render(line, True, tc)

        tx = x + padding
        ty = y + (h - surf.get_height()) // 2
        screen.blit(surf, (tx, ty))

    def draw_buttons_overlay(now: float) -> None:
        nonlocal pressed_handle, pressed_until
        if not button_rects or not buttons:
            return

        spec_by_handle = {b["handle"]: b for b in buttons if "handle" in b}

        for handle, r in button_rects.items():
            b = spec_by_handle.get(handle)
            if not b:
                continue

            x, y, w, h = map(int, r)
            bg = tuple(b.get("bg_color", (40, 40, 40)))
            tc = tuple(b.get("text_color", (255, 255, 255)))
            label = str(b.get("text", handle))

            rect = pygame.Rect(x, y, w, h)

            pygame.draw.rect(screen, bg, rect, 0)
            pygame.draw.rect(screen, (0, 0, 0), rect, 2)

            label_s = _truncate_with_ellipsis(button_font, label, max(0, w - 2 * padding))
            surf = button_font.render(label_s, True, tc)
            tx = x + (w - surf.get_width()) // 2
            ty = y + (h - surf.get_height()) // 2
            screen.blit(surf, (tx, ty))

            if pressed_handle == handle and now < pressed_until:
                overlay = pygame.Surface((w, h), pygame.SRCALPHA)
                overlay.fill((255, 255, 255, 70))
                screen.blit(overlay, (x, y))
                pygame.draw.rect(screen, (255, 255, 255), rect, 3)

        if pressed_handle is not None and now >= pressed_until:
            pressed_handle = None

    def hit_test_button(pos: Tuple[int, int]) -> Optional[str]:
        px, py = pos
        for handle, r in button_rects.items():
            x, y, w, h = r
            if px >= x and px < x + w and py >= y and py < y + h:
                return handle
        return None

    try:
        while True:
            now = time.time()

            # -------------------------
            # Pygame window events
            # -------------------------
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    return
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    handle = hit_test_button(ev.pos)
                    if handle:
                        try:
                            event_queue.put_nowait(handle)
                        except Exception:
                            pass
                        pressed_handle = handle
                        pressed_until = now + (pressed_highlight_ms / 1000.0)

            # -------------------------
            # Drain commands
            # -------------------------
            should_quit = False
            while True:
                try:
                    cmd = cmd_queue.get_nowait()
                except queue.Empty:
                    break
                except Exception:
                    break
                if isinstance(cmd, dict) and handle_command(cmd):
                    should_quit = True
                    break
            if should_quit:
                return

            # -------------------------
            # Update base frame if changed
            # -------------------------
            fid = frame_counter.value
            if fid != last_frame_id:
                last_frame_id = fid
                with canvas_lock:
                    base_bgr = canvas.copy()  # avoid tearing
                base_surface = _bgr_canvas_to_surface(base_bgr)

            # If we never received a frame yet, just clear to black
            if base_surface is None:
                screen.fill((0, 0, 0))
            else:
                screen.blit(base_surface, (0, 0))

            # -------------------------
            # Overlays
            # -------------------------
            draw_status_overlay()
            draw_buttons_overlay(now)

            pygame.display.flip()
            clock.tick(fps)

    except KeyboardInterrupt:
        return
    finally:
        try:
            shm.close()
        except Exception:
            pass
        pygame.quit()
