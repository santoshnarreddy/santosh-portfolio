"""
Face Recognition Attendance System — Enrollment
Author: Santosh Narreddy

Register a new person by capturing their face from webcam.
We take multiple frames at different angles to build a robust encoding.

Usage:
    python enroll.py --name "Santosh" --samples 30
    python enroll.py --name "Priya" --image priya.jpg
"""

import os
import cv2
import pickle
import argparse
import time
import numpy as np
from utils.face_utils import get_face_encoding, detect_faces

DATABASE_PATH = "database/encodings.pkl"
SAMPLES_DIR   = "database/samples"


def load_database():
    """Load existing face database or return empty dict."""
    if os.path.exists(DATABASE_PATH):
        with open(DATABASE_PATH, 'rb') as f:
            return pickle.load(f)
    return {}   # {name: [encoding1, encoding2, ...]}


def save_database(db: dict):
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    with open(DATABASE_PATH, 'wb') as f:
        pickle.dump(db, f)
    print(f"[INFO] Database saved → {DATABASE_PATH} ({len(db)} people)")


def enroll_from_webcam(name: str, num_samples: int = 30, cam_idx: int = 0):
    """
    Capture `num_samples` face frames from webcam.
    We space them out so we get different micro-expressions and head angles.
    """
    db = load_database()

    if name in db:
        print(f"[WARN] '{name}' already exists in database ({len(db[name])} encodings). "
              f"Adding more samples will improve accuracy.")
    else:
        db[name] = []

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    os.makedirs(os.path.join(SAMPLES_DIR, name), exist_ok=True)

    collected  = 0
    start_time = time.time()
    print(f"\n[INFO] Enrolling: {name}")
    print(f"[INFO] Look at the camera — slowly move your head slightly for varied angles.")
    print(f"[INFO] Collecting {num_samples} samples. Press 'q' to stop early.\n")

    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        faces = detect_faces(frame)

        display = frame.copy()
        status_color = (0, 200, 100)

        if len(faces) == 1:
            x, y, w, h = faces[0]
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 100), 2)

            # Capture every ~0.2 seconds to space out samples
            if time.time() - start_time > 0.2 * (collected + 1):
                face_roi = frame[y:y + h, x:x + w]
                encoding = get_face_encoding(face_roi)
                if encoding is not None:
                    db[name].append(encoding)
                    sample_path = os.path.join(SAMPLES_DIR, name, f"{collected:03d}.jpg")
                    cv2.imwrite(sample_path, face_roi)
                    collected += 1
                    status_color = (0, 120, 255)

            status_text = f"Collecting: {collected}/{num_samples}"
        elif len(faces) == 0:
            status_text = "No face detected — move closer"
            status_color = (0, 100, 255)
        else:
            status_text = "Multiple faces — only 1 person please"
            status_color = (0, 0, 255)

        # Progress bar
        progress = int((collected / num_samples) * frame.shape[1])
        cv2.rectangle(display, (0, frame.shape[0] - 8), (progress, frame.shape[0]), (0, 200, 100), -1)

        cv2.putText(display, f"Enrolling: {name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(display, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(display, "Press 'q' to stop", (10, display.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

        cv2.imshow(f"Enrolling: {name} — Santosh Narreddy", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if db[name]:
        save_database(db)
        print(f"\n[DONE] Enrolled '{name}' with {len(db[name])} face encodings.")
        print(f"       Samples saved to {SAMPLES_DIR}/{name}/")
    else:
        print(f"\n[WARN] No encodings collected for '{name}'.")


def enroll_from_image(name: str, image_path: str):
    """Enroll a person from a single image file."""
    db = load_database()

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    faces = detect_faces(img)
    if not faces:
        print("[ERROR] No face found in image.")
        return
    if len(faces) > 1:
        print(f"[WARN] Found {len(faces)} faces. Using the largest one.")

    # Use largest face
    faces.sort(key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    face_roi = img[y:y + h, x:x + w]

    encoding = get_face_encoding(face_roi)
    if encoding is None:
        print("[ERROR] Could not compute face encoding.")
        return

    if name not in db:
        db[name] = []
    db[name].append(encoding)
    save_database(db)
    print(f"[DONE] Enrolled '{name}' from {image_path} (1 encoding).")
    print(f"       Tip: Run webcam enrollment for more samples → better accuracy.")


def list_enrolled():
    """Print all enrolled people."""
    db = load_database()
    if not db:
        print("[INFO] Database is empty. Run enroll.py to add people.")
        return
    print(f"\n{'Name':<25} {'Encodings':>10}")
    print("-" * 38)
    for name, encodings in sorted(db.items()):
        print(f"{name:<25} {len(encodings):>10}")
    print(f"\nTotal: {len(db)} people enrolled.")


def remove_person(name: str):
    """Remove a person from the database."""
    db = load_database()
    if name not in db:
        print(f"[ERROR] '{name}' not found in database.")
        return
    del db[name]
    save_database(db)
    print(f"[DONE] Removed '{name}' from database.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Enrollment Tool")
    parser.add_argument('--name',    help='Person name to enroll')
    parser.add_argument('--samples', type=int, default=30, help='Number of webcam samples')
    parser.add_argument('--image',   help='Enroll from image file instead of webcam')
    parser.add_argument('--cam',     type=int, default=0)
    parser.add_argument('--list',    action='store_true', help='List enrolled people')
    parser.add_argument('--remove',  help='Remove a person from database')
    args = parser.parse_args()

    if args.list:
        list_enrolled()
    elif args.remove:
        remove_person(args.remove)
    elif args.name:
        if args.image:
            enroll_from_image(args.name, args.image)
        else:
            enroll_from_webcam(args.name, num_samples=args.samples, cam_idx=args.cam)
    else:
        parser.print_help()
