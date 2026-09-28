"""
Real-time Face Emotion Detection
Author: Santosh Narreddy
Description: Detects facial emotions from webcam feed using a trained CNN model.
             Supports 7 emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised
"""

import cv2
import numpy as np
import time
import argparse
from tensorflow.keras.models import load_model

# ─── Constants ────────────────────────────────────────────────────────────────

EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

EMOTION_COLORS = {
    'Happy':     (0, 255, 100),
    'Sad':       (255, 80, 80),
    'Angry':     (0, 0, 255),
    'Surprised': (0, 220, 255),
    'Neutral':   (180, 180, 180),
    'Fearful':   (128, 0, 200),
    'Disgusted': (0, 160, 80),
}


# ─── Detector Class ───────────────────────────────────────────────────────────

class EmotionDetector:

    def __init__(self, model_path='model/emotion_cnn.h5'):
        print("[INFO] Loading model... ", end="", flush=True)
        self.model = load_model(model_path)
        print("done.")

        # Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.emotions = EMOTIONS

    def preprocess_face(self, face_img):
        """
        Resize to 48x48, convert to grayscale, normalize pixel values.
        FER2013 images are 48x48 grayscale — so we match that format here.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (48, 48))
        normalized = resized.astype('float32') / 255.0
        # Model expects shape (1, 48, 48, 1)
        return np.expand_dims(normalized, axis=(0, -1))

    def detect_faces(self, frame):
        """Detect all faces in a frame using Haar cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces if len(faces) > 0 else []

    def predict_emotion(self, face_roi):
        """Run CNN inference on a single face crop."""
        input_tensor = self.preprocess_face(face_roi)
        predictions = self.model.predict(input_tensor, verbose=0)[0]
        emotion_idx = int(np.argmax(predictions))
        return self.emotions[emotion_idx], float(predictions[emotion_idx]), predictions

    def process_frame(self, frame):
        """Full pipeline: detect faces → predict emotion → return annotated frame."""
        faces = self.detect_faces(frame)
        results = []

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            try:
                label, confidence, all_preds = self.predict_emotion(face_roi)
            except Exception as e:
                print(f"[WARN] Prediction failed for face at ({x},{y}): {e}")
                continue

            results.append({
                'bbox': (x, y, w, h),
                'emotion': label,
                'confidence': confidence * 100,
                'all_probs': dict(zip(self.emotions, all_preds.tolist()))
            })

        return self._draw(frame, results), results

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def _draw(self, frame, results):
        for r in results:
            x, y, w, h = r['bbox']
            emotion = r['emotion']
            conf = r['confidence']
            color = EMOTION_COLORS.get(emotion, (200, 200, 200))

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label background + text
            label_text = f"{emotion}  {conf:.1f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color, -1)
            cv2.putText(frame, label_text, (x + 4, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

            # Side bar chart (only if there's space to the right)
            if x + w + 120 < frame.shape[1]:
                self._draw_prob_bars(frame, r['all_probs'], x + w + 8, y)

        return frame

    def _draw_prob_bars(self, frame, probs, sx, sy):
        bar_w, bar_h, gap = 90, 11, 14
        for i, (emo, prob) in enumerate(probs.items()):
            ry = sy + i * gap
            if ry + bar_h > frame.shape[0]:
                break
            color = EMOTION_COLORS.get(emo, (200, 200, 200))
            cv2.rectangle(frame, (sx, ry), (sx + bar_w, ry + bar_h), (40, 40, 40), -1)
            filled = int(bar_w * prob)
            cv2.rectangle(frame, (sx, ry), (sx + filled, ry + bar_h), color, -1)
            cv2.putText(frame, emo[:3], (sx - 28, ry + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (220, 220, 220), 1)


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run(source=0, model_path='model/emotion_cnn.h5', save_output=False):
    detector = EmotionDetector(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('output.mp4', fourcc, 20.0, (720, 540))

    fps_timer = time.time()
    fps_count = 0
    fps_display = 0.0

    print("[INFO] Running. Press 'q' to quit | 's' to save screenshot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # Mirror — feels more natural on webcam

        annotated, results = detector.process_frame(frame)

        # FPS overlay
        fps_count += 1
        if fps_count == 20:
            fps_display = 20 / (time.time() - fps_timer)
            fps_timer = time.time()
            fps_count = 0

        cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)
        cv2.putText(annotated, f"Faces: {len(results)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.putText(annotated, "q: quit | s: screenshot",
                    (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        if writer:
            writer.write(annotated)

        cv2.imshow("Emotion Detector — Santosh Narreddy", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = int(time.time())
            fname = f"screenshot_{ts}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"[INFO] Saved {fname}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Emotion Detector")
    parser.add_argument('--source', default=0, help='Camera index or video file path (default: 0)')
    parser.add_argument('--model', default='model/emotion_cnn.h5', help='Path to trained .h5 model')
    parser.add_argument('--save', action='store_true', help='Save output to output.mp4')
    args = parser.parse_args()

    run(source=args.source, model_path=args.model, save_output=args.save)
