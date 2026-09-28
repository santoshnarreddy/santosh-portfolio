"""
Face detection and encoding utilities.
Author: Santosh Narreddy

Uses OpenCV Haar cascade for detection and a lightweight CNN (FaceNet-style)
for generating 128-d face embeddings via the face_recognition library.
"""

import cv2
import numpy as np

try:
    import face_recognition
    FR_AVAILABLE = True
except ImportError:
    FR_AVAILABLE = False
    print("[WARN] face_recognition not installed. "
          "Run: pip install face-recognition  (requires cmake + dlib)")


# Haar cascade — fast enough for real-time, good for frontal faces
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def detect_faces(frame_bgr: np.ndarray) -> list:
    """
    Detect faces using Haar cascade.
    Returns list of (x, y, w, h) bounding boxes.
    """
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return list(faces) if len(faces) > 0 else []


def get_face_encoding(face_bgr: np.ndarray) -> np.ndarray | None:
    """
    Compute a 128-d face embedding for a single face crop.
    Returns None if no face found or library not available.
    """
    if not FR_AVAILABLE:
        # Fallback: flat normalized pixel vector (very basic, poor accuracy)
        gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64)).astype('float32') / 255.0
        return resized.flatten()

    # face_recognition expects RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    # Resize to 160x160 for better encoding quality
    face_rgb = cv2.resize(face_rgb, (160, 160))

    encodings = face_recognition.face_encodings(face_rgb)
    if not encodings:
        return None
    return encodings[0]   # 128-d numpy array


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two face encoding vectors.
    Returns value in [0, 1] — higher means more similar.

    I use cosine similarity instead of Euclidean distance because it's
    more robust to slight variation in embedding magnitudes.
    """
    a = a.astype('float32')
    b = b.astype('float32')
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
