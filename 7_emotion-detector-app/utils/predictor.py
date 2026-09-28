"""
Emotion prediction utility for the Streamlit app.
Author: Santosh Narreddy

Loads pre-trained CNN model and runs real-time face emotion inference.
"""

import os
import cv2
import numpy as np
import urllib.request

EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']
DEFAULT_URL = "https://raw.githubusercontent.com/oarriaga/face_classification/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5"

class EmotionPredictor:

    def __init__(self, model_path='model/emotion_cnn.h5'):
        # Auto-download if model file is missing
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
            print(f"[INFO] Downloading pretrained emotion model to {model_path}...")
            try:
                urllib.request.urlretrieve(DEFAULT_URL, model_path)
                print("[INFO] Download completed successfully.")
            except Exception as e:
                print(f"[WARN] Auto-download failed: {e}")

        from tensorflow.keras.models import load_model
        self.model = load_model(model_path, compile=False)

        # Determine target size dynamically from model architecture
        try:
            inp_shape = self.model.input_shape
            self.target_size = (inp_shape[1], inp_shape[2]) if inp_shape and inp_shape[1] else (64, 64)
        except Exception:
            self.target_size = (64, 64)

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def predict(self, frame_bgr, min_confidence=0.3):
        """
        Detect all faces and return emotion predictions.
        Returns a list of dicts: {bbox, emotion, confidence, all_probs}
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        detections = []
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, self.target_size).astype('float32') / 255.0
            face = (face - 0.5) * 2.0
            inp = np.expand_dims(face, axis=(0, -1))

            probs = self.model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx]) * 100

            if conf / 100 >= min_confidence:
                detections.append({
                    'bbox': (x, y, w, h),
                    'emotion': EMOTIONS[idx],
                    'confidence': conf,
                    'all_probs': {e: float(p) for e, p in zip(EMOTIONS, probs)}
                })

        return detections
