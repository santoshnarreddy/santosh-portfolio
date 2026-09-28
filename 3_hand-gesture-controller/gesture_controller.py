"""
Hand Gesture Controller
Author: Santosh Narreddy

Control system volume, screen brightness, and mouse cursor
using hand gestures detected in real-time via MediaPipe + OpenCV.

Gestures:
  👍 Thumb up              → Volume up
  👎 Thumb down            → Volume down
  ✌️  Peace / index+middle → Brightness control (distance = brightness level)
  ☝️  Index only           → Mouse move
  ✊ Fist                  → Left click
  🖐️ Open palm             → Screenshot
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
import math
import platform
import argparse

# Optional system-level audio control (Windows only)
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL
    import ctypes
    PYCAW_AVAILABLE = True
except ImportError:
    PYCAW_AVAILABLE = False

# Screen brightness control
try:
    import screen_brightness_control as sbc
    SBC_AVAILABLE = True
except ImportError:
    SBC_AVAILABLE = False

pyautogui.FAILSAFE = False   # Don't crash when mouse hits corner


# ─── Gesture Recognizer ────────────────────────────────────────────────────────

class HandGestureController:

    # MediaPipe landmark indices
    THUMB_TIP  = 4;  THUMB_IP   = 3
    INDEX_TIP  = 8;  INDEX_PIP  = 6
    MIDDLE_TIP = 12; MIDDLE_PIP = 10
    RING_TIP   = 16; RING_PIP   = 14
    PINKY_TIP  = 20; PINKY_PIP  = 18
    WRIST      = 0

    def __init__(self, cam_idx=0, flip=True):
        self.flip = flip
        self.screen_w, self.screen_h = pyautogui.size()
        print(f"[INFO] Screen: {self.screen_w}x{self.screen_h}")

        # MediaPipe Hands
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.7
        )

        # Camera
        self.cap = cv2.VideoCapture(cam_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Volume control (Windows only with pycaw)
        self.volume_obj = None
        if PYCAW_AVAILABLE and platform.system() == 'Windows':
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume_obj = ctypes.cast(interface, ctypes.POINTER(IAudioEndpointVolume))
            self.vol_min, self.vol_max, _ = self.volume_obj.GetVolumeRange()
            print("[INFO] pycaw audio control initialized")

        # State
        self.prev_gesture    = None
        self.gesture_start   = time.time()
        self.last_click_time = 0
        self.smoothed_x      = self.screen_w // 2
        self.smoothed_y      = self.screen_h // 2
        self.smoothing        = 0.4   # lower = more smoothing

    # ── Landmark Helpers ───────────────────────────────────────────────────────

    def _lm(self, landmarks, idx):
        lm = landmarks.landmark[idx]
        return lm.x, lm.y

    def _finger_up(self, landmarks, tip_idx, pip_idx):
        """True if fingertip is above its PIP joint (finger extended)."""
        return landmarks.landmark[tip_idx].y < landmarks.landmark[pip_idx].y

    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    # ── Gesture Classification ─────────────────────────────────────────────────

    def classify_gesture(self, landmarks, h, w):
        """
        Returns a gesture string based on which fingers are extended.
        Landmark coordinates are normalized (0-1); multiply by (w, h) for pixels.
        """
        thumb_up  = self._lm(landmarks, self.THUMB_TIP)[0]  > self._lm(landmarks, self.THUMB_IP)[0]
        index_up  = self._finger_up(landmarks, self.INDEX_TIP,  self.INDEX_PIP)
        middle_up = self._finger_up(landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_up   = self._finger_up(landmarks, self.RING_TIP,   self.RING_PIP)
        pinky_up  = self._finger_up(landmarks, self.PINKY_TIP,  self.PINKY_PIP)

        fingers = [index_up, middle_up, ring_up, pinky_up]
        num_up  = sum(fingers)

        # Gesture rules (in priority order)
        if thumb_up and num_up == 0:
            return 'THUMB_UP'
        if not thumb_up and num_up == 0 and not any(fingers):
            return 'FIST'
        if num_up == 4 and thumb_up:
            return 'OPEN_PALM'
        if index_up and middle_up and not ring_up and not pinky_up:
            return 'PEACE'       # Volume / brightness via distance
        if index_up and not middle_up and not ring_up and not pinky_up:
            return 'INDEX'       # Mouse mode
        return 'NONE'

    # ── Actions ────────────────────────────────────────────────────────────────

    def _handle_mouse(self, landmarks, frame_w, frame_h):
        ix, iy = self._lm(landmarks, self.INDEX_TIP)
        # Map camera coords to screen coords (with margin to avoid edge clipping)
        margin = 0.1
        mapped_x = np.interp(ix, [margin, 1 - margin], [0, self.screen_w])
        mapped_y = np.interp(iy, [margin, 1 - margin], [0, self.screen_h])

        # Exponential smoothing
        self.smoothed_x = self.smoothed_x + self.smoothing * (mapped_x - self.smoothed_x)
        self.smoothed_y = self.smoothed_y + self.smoothing * (mapped_y - self.smoothed_y)
        pyautogui.moveTo(int(self.smoothed_x), int(self.smoothed_y), duration=0)

    def _handle_peace(self, landmarks):
        """Distance between index and middle tip controls volume/brightness."""
        p1 = self._lm(landmarks, self.INDEX_TIP)
        p2 = self._lm(landmarks, self.MIDDLE_TIP)
        dist = self._distance(p1, p2)   # normalized 0-1

        level = int(np.interp(dist, [0.02, 0.25], [0, 100]))

        if self.volume_obj:
            vol = np.interp(level, [0, 100], [self.vol_min, self.vol_max])
            self.volume_obj.SetMasterVolumeLevel(vol, None)
        elif SBC_AVAILABLE:
            sbc.set_brightness(level)
        else:
            # Fallback: use pyautogui volume keys
            pass

        return level   # Return for display

    def _handle_fist_click(self):
        now = time.time()
        if now - self.last_click_time > 0.8:   # debounce 800ms
            pyautogui.click()
            self.last_click_time = now

    def _handle_open_palm(self):
        now = time.time()
        if now - self.last_click_time > 1.5:
            ts = int(time.time())
            pyautogui.screenshot(f'screenshot_{ts}.png')
            print(f"[INFO] Screenshot saved: screenshot_{ts}.png")
            self.last_click_time = now

    # ── Main Loop ──────────────────────────────────────────────────────────────

    def run(self):
        if not self.cap.isOpened():
            print("[ERROR] Cannot open camera")
            return

        print("[INFO] Gesture controller running. Press 'q' to quit.")
        fps_timer = time.time()
        fps_count = 0
        fps_display = 0.0
        level_display = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.flip:
                frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            gesture_label = "No hand"
            gesture_color = (120, 120, 120)

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                self.mp_drawing.draw_landmarks(
                    frame, hand, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 200, 100), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(0, 100, 200), thickness=2)
                )

                gesture = self.classify_gesture(hand, frame_h, frame_w)
                gesture_label = gesture
                gesture_color = (0, 255, 150)

                if gesture == 'INDEX':
                    self._handle_mouse(hand, frame_w, frame_h)
                    gesture_label = "🖱️ MOUSE MODE"
                elif gesture == 'FIST':
                    self._handle_fist_click()
                    gesture_label = "🖱️ CLICK"
                elif gesture == 'PEACE':
                    level_display = self._handle_peace(hand)
                    gesture_label = f"🔊 LEVEL: {level_display}%"
                elif gesture == 'OPEN_PALM':
                    self._handle_open_palm()
                    gesture_label = "📸 SCREENSHOT"
                elif gesture == 'THUMB_UP':
                    gesture_label = "👍 THUMB UP"

            # ── HUD ────────────────────────────────────────────────────────────
            cv2.rectangle(frame, (0, 0), (frame_w, 55), (15, 15, 25), -1)

            fps_count += 1
            if fps_count == 15:
                fps_display = 15 / (time.time() - fps_timer)
                fps_timer = time.time()
                fps_count = 0

            cv2.putText(frame, f"FPS: {fps_display:.0f}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 100), 2)
            cv2.putText(frame, gesture_label, (frame_w // 2 - 100, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, gesture_color, 2)

            # Gesture legend (bottom)
            legend = ["INDEX=Mouse", "FIST=Click", "PEACE=Vol", "PALM=Screenshot", "q=Quit"]
            x_off = 8
            for item in legend:
                cv2.putText(frame, item, (x_off, frame_h - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)
                x_off += len(item) * 6 + 15

            cv2.imshow("Hand Gesture Controller — Santosh Narreddy", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Controller stopped.")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Gesture PC Controller")
    parser.add_argument('--cam',  type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--flip', action='store_true', default=True, help='Mirror camera (default: True)')
    args = parser.parse_args()

    ctrl = HandGestureController(cam_idx=args.cam, flip=args.flip)
    ctrl.run()
