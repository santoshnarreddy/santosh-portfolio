"""
Face Recognition Attendance System — Main Script
Author: Santosh Narreddy

Real-time face recognition from webcam that:
  1. Detects and recognizes enrolled faces
  2. Logs attendance to a CSV with timestamp (once per person per session)
  3. Shows recognized names + confidence live on webcam feed

Usage:
    # First enroll some people:
    python enroll.py --name "Santosh" --samples 30

    # Then run attendance:
    python attendance_system.py
    python attendance_system.py --log_dir logs/ --cam 0
"""

import os
import cv2
import pickle
import argparse
import numpy as np
import time
from datetime import datetime

from utils.face_utils import get_face_encoding, detect_faces, cosine_similarity
from utils.attendance_utils import AttendanceLogger

DATABASE_PATH  = "database/encodings.pkl"
UNKNOWN_LABEL  = "Unknown"
MATCH_THRESHOLD = 0.55   # Cosine similarity — tune this per your use case


# ── Database Loader ────────────────────────────────────────────────────────────

def load_database(db_path: str) -> dict:
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        print("[INFO]  Run `python enroll.py --name YourName` first.")
        return {}
    with open(db_path, 'rb') as f:
        db = pickle.load(f)
    print(f"[INFO] Loaded database: {len(db)} people enrolled")
    for name, encs in db.items():
        print(f"       • {name} ({len(encs)} encodings)")
    return db


# ── Recognition ───────────────────────────────────────────────────────────────

def recognize_face(encoding: np.ndarray, database: dict) -> tuple[str, float]:
    """
    Find best match for an encoding in the database.
    Uses average cosine similarity across all stored encodings per person.
    Returns (name, confidence) — name is UNKNOWN_LABEL if below threshold.
    """
    best_name  = UNKNOWN_LABEL
    best_score = 0.0

    for name, stored_encodings in database.items():
        # Average similarity across all stored encodings for this person
        similarities = [cosine_similarity(encoding, se) for se in stored_encodings]
        avg_sim = float(np.mean(similarities))

        if avg_sim > best_score:
            best_score = avg_sim
            best_name  = name

    if best_score < MATCH_THRESHOLD:
        return UNKNOWN_LABEL, best_score

    return best_name, best_score


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_recognition(frame, results: list) -> None:
    """Draw bounding boxes and name labels on frame (in-place)."""
    for res in results:
        x, y, w, h = res['bbox']
        name  = res['name']
        conf  = res['confidence']
        color = (0, 220, 100) if name != UNKNOWN_LABEL else (0, 80, 220)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = f"{name}  {conf*100:.0f}%" if name != UNKNOWN_LABEL else "Unknown"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - th - 14), (x + tw + 8, y), color, -1)
        cv2.putText(frame, label, (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # "MARKED" badge if attendance was just logged
        if res.get('just_marked'):
            cv2.putText(frame, "✓ MARKED", (x, y + h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 100), 2)


# ── Main Attendance Loop ───────────────────────────────────────────────────────

def run_attendance(db_path=DATABASE_PATH, log_dir='logs', cam_idx=0):
    database = load_database(db_path)
    if not database:
        return

    logger = AttendanceLogger(log_dir=log_dir)
    logger.start_session()

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # Track who was already marked this session (avoid duplicate entries)
    marked_this_session = set()

    fps_counter = 0
    fps_timer   = time.time()
    fps_display = 0.0
    just_marked_flash = {}   # name → timestamp of when they were marked

    print("\n[INFO] Attendance system running. Press 'q' to quit | 'r' for report.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        faces = detect_faces(frame)
        results = []

        for (x, y, w, h) in faces:
            face_roi  = frame[y:y + h, x:x + w]
            encoding  = get_face_encoding(face_roi)
            if encoding is None:
                continue

            name, conf = recognize_face(encoding, database)

            just_marked = False
            if name != UNKNOWN_LABEL and name not in marked_this_session:
                logger.mark_attendance(name)
                marked_this_session.add(name)
                just_marked_flash[name] = time.time()
                just_marked = True
                print(f"  ✅ Marked: {name}  ({datetime.now().strftime('%H:%M:%S')})")

            # Show flash for 3 seconds after marking
            if name in just_marked_flash and (time.time() - just_marked_flash[name]) < 3:
                just_marked = True

            results.append({
                'bbox': (x, y, w, h),
                'name': name,
                'confidence': conf,
                'just_marked': just_marked
            })

        draw_recognition(frame, results)

        # ── HUD ────────────────────────────────────────────────────────────────
        fps_counter += 1
        if fps_counter == 20:
            fps_display = 20 / (time.time() - fps_timer)
            fps_timer   = time.time()
            fps_counter = 0

        # Top bar
        cv2.rectangle(frame, (0, 0), (frame_w, 48), (12, 12, 20), -1)
        cv2.putText(frame, "Attendance System — Santosh Narreddy", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 100), 2)
        cv2.putText(frame, f"FPS: {fps_display:.0f}  |  Marked: {len(marked_this_session)}",
                    (frame_w - 230, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        # Marked attendance sidebar
        sidebar_x = frame_w - 180
        cv2.rectangle(frame, (sidebar_x, 48), (frame_w, frame_h), (18, 18, 30), -1)
        cv2.putText(frame, "Today's Attendance", (sidebar_x + 6, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 180, 100), 1)
        for i, name in enumerate(sorted(marked_this_session)):
            y_pos = 92 + i * 22
            if y_pos < frame_h - 10:
                cv2.putText(frame, f"✓ {name}", (sidebar_x + 6, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 100), 1)

        cv2.putText(frame, "q: quit | r: print report", (10, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

        cv2.imshow("Face Recognition Attendance — Santosh Narreddy", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            logger.print_report()

    cap.release()
    cv2.destroyAllWindows()
    logger.save_session()
    logger.print_report()
    print(f"\n[DONE] Session ended. Attendance saved to {log_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
    parser.add_argument('--db',      default=DATABASE_PATH, help='Path to encodings database')
    parser.add_argument('--log_dir', default='logs',        help='Directory to save attendance logs')
    parser.add_argument('--cam',     type=int, default=0,   help='Camera index')
    parser.add_argument('--threshold', type=float, default=0.55,
                        help='Face match confidence threshold (0-1, default: 0.55)')
    args = parser.parse_args()

    MATCH_THRESHOLD = args.threshold
    run_attendance(db_path=args.db, log_dir=args.log_dir, cam_idx=args.cam)
