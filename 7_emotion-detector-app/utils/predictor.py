"""
Emotion prediction utility for the Streamlit app.
Author: Santosh Narreddy

Thin wrapper around the CNN model so app.py stays clean.
"""

import cv2
import numpy as np

EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']


class EmotionPredictor:

    def __init__(self, model_path='model/emotion_cnn.h5'):
        # Lazy import so Streamlit doesn't crash if TF isn't installed
        from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def predict(self, frame_bgr, min_confidence=0.3):
        """
        Detect all faces and return emotion predictions.
        Returns a list of dicts: {bbox, emotion, confidence, all_probs}
        """
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        detections = []
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48)).astype('float32') / 255.0
            inp  = np.expand_dims(face, axis=(0, -1))

            probs = self.model.predict(inp, verbose=0)[0]
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx]) * 100

            if conf / 100 >= min_confidence:
                detections.append({
                    'bbox':      (x, y, w, h),
                    'emotion':   EMOTIONS[idx],
                    'confidence': conf,
                    'all_probs': {e: float(p) for e, p in zip(EMOTIONS, probs)}
                })

        return detections
