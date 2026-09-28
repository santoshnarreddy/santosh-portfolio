"""
Emotion Detector — Streamlit Web App
Author: Santosh Narreddy

A clean web interface for the face emotion detection model.
Upload an image or use your webcam — no installation needed for visitors.

Deploy free on Streamlit Community Cloud:
  1. Push this repo to GitHub
  2. Go to share.streamlit.io → New app → select repo → main file: app.py
  3. Hit Deploy — you get a public URL to share!
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time

from utils.predictor import EmotionPredictor
from utils.visualize import draw_emotion_overlay, plot_emotion_bars

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Emotion Detector · Santosh Narreddy",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .block-container { padding-top: 1.5rem; }
  .metric-box {
    background: #1e2130;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    border: 1px solid #2d3748;
    margin-bottom: 8px;
  }
  .metric-box .label { color: #8b949e; font-size: 0.78rem; }
  .metric-box .value { color: #e6edf3; font-size: 1.4rem; font-weight: 700; }
  .emotion-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.9rem;
  }
  h1 { color: #00c7f7 !important; }
  .stProgress > div > div { background-color: #00c7f7; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎭 Emotion Detector")
    st.markdown("**Author:** [Santosh Narreddy](https://github.com/santoshnarreddy)")
    st.divider()

    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider(
        "Min confidence to display", 0.1, 0.9, 0.3, step=0.05,
        help="Predictions below this threshold are filtered out"
    )
    show_all_emotions = st.toggle("Show all 7 emotion scores", value=True)
    show_landmarks = st.toggle("Show face bounding box", value=True)

    st.divider()
    st.markdown("### 🧠 Model Info")
    st.markdown("""
    - **Architecture:** Custom CNN (3 conv blocks)
    - **Dataset:** FER2013 (35,887 images)
    - **Classes:** 7 emotions
    - **Val Accuracy:** ~63%
    - **Input size:** 48×48 grayscale
    """)

    st.divider()
    st.markdown("### 😊 Emotions Detected")
    emotions_info = {
        "😠 Angry": "#ff4444",
        "🤢 Disgusted": "#44aa44",
        "😨 Fearful": "#9944cc",
        "😊 Happy": "#44cc44",
        "😐 Neutral": "#aaaaaa",
        "😢 Sad": "#4488ff",
        "😲 Surprised": "#ffaa00",
    }
    for emotion, color in emotions_info.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{color};"></div>'
            f'<span style="font-size:0.85rem;color:#c9d1d9;">{emotion}</span></div>',
            unsafe_allow_html=True
        )

# ── Main Area ─────────────────────────────────────────────────────────────────

st.title("🎭 Real-Time Face Emotion Detector")
st.markdown("Upload a photo or use your webcam to detect facial emotions using a CNN trained on FER2013.")

# Load model (cached so it only loads once)
@st.cache_resource(show_spinner="Loading emotion model...")
def load_predictor():
    model_path = os.environ.get("MODEL_PATH", "model/emotion_cnn.h5")
    return EmotionPredictor(model_path=model_path)

try:
    predictor = load_predictor()
    model_loaded = True
except Exception as e:
    st.warning(f"⚠️ Model not found at `model/emotion_cnn.h5`. Running in demo mode. "
               f"[Download pretrained weights from Releases](https://github.com/santoshnarreddy/face-emotion-detection/releases)")
    model_loaded = False

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📷 Upload Image", "🎥 Webcam (local only)", "ℹ️ About"])

# ─── Tab 1: Image Upload ──────────────────────────────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "Upload a photo (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Best results: clear frontal face, good lighting"
    )

    if uploaded:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col_img, col_results = st.columns([1.4, 1])

        with col_img:
            st.markdown("##### Input Image")
            if model_loaded:
                annotated, detections = draw_emotion_overlay(
                    img_bgr.copy(), predictor,
                    show_bbox=show_landmarks,
                    min_confidence=confidence_threshold
                )
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         use_container_width=True, caption="Annotated output")
            else:
                st.image(uploaded, use_container_width=True)
                detections = []

        with col_results:
            st.markdown("##### Detection Results")
            if not model_loaded:
                st.info("Load model weights to see predictions.")
            elif not detections:
                st.warning("No faces detected. Try a clearer frontal photo.")
            else:
                for i, det in enumerate(detections, 1):
                    st.markdown(f"**Face #{i}**")
                    dominant = det['emotion']
                    conf     = det['confidence']

                    # Color per emotion
                    color_map = {
                        'Happy': '#44cc44', 'Sad': '#4488ff', 'Angry': '#ff4444',
                        'Surprised': '#ffaa00', 'Neutral': '#aaaaaa',
                        'Fearful': '#9944cc', 'Disgusted': '#44aa44'
                    }
                    color = color_map.get(dominant, '#00c7f7')
                    st.markdown(
                        f'<div class="metric-box">'
                        f'<div class="label">Dominant Emotion</div>'
                        f'<div class="value" style="color:{color};">{dominant}</div>'
                        f'<div class="label">{conf:.1f}% confidence</div>'
                        f'</div>', unsafe_allow_html=True
                    )

                    if show_all_emotions and 'all_probs' in det:
                        st.markdown("**All scores:**")
                        fig = plot_emotion_bars(det['all_probs'])
                        st.plotly_chart(fig, use_container_width=True,
                                        config={'displayModeBar': False})

                    if i < len(detections):
                        st.divider()

# ─── Tab 2: Webcam ────────────────────────────────────────────────────────────
with tab2:
    st.info("""
    **Note:** Webcam streaming requires running this app **locally** (not on Streamlit Cloud).

    ```bash
    git clone https://github.com/santoshnarreddy/emotion-detector-app
    cd emotion-detector-app
    pip install -r requirements.txt
    streamlit run app.py
    ```
    Then open http://localhost:8501 — the webcam tab will work there.
    """)

    st.markdown("#### Or run the standalone OpenCV script for best performance:")
    st.code("python ../face-emotion-detection/emotion_detector.py", language="bash")

# ─── Tab 3: About ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("""
    ## About This App

    This web app wraps the [face-emotion-detection](https://github.com/santoshnarreddy/face-emotion-detection)
    CNN model in a Streamlit UI so anyone can try it without installing Python or OpenCV.

    ### How it works
    1. **Face Detection** — OpenCV Haar cascade locates faces in the image
    2. **Preprocessing** — Each face is resized to 48×48 and converted to grayscale
    3. **Emotion Classification** — A 3-block CNN outputs probabilities for 7 emotions
    4. **Visualization** — Bounding boxes + emotion labels are drawn on the original image

    ### Model Architecture
    ```
    Input (48×48×1)
    → Conv2D(32) → BN → ReLU → Conv2D(32) → BN → MaxPool → Dropout(0.25)
    → Conv2D(64) → BN → ReLU → Conv2D(64) → BN → MaxPool → Dropout(0.25)
    → Conv2D(128) → BN → ReLU → Conv2D(128) → BN → MaxPool → Dropout(0.4)
    → Flatten → Dense(256) → BN → Dropout(0.5) → Dense(7) → Softmax
    ```

    ### Limitations
    - Accuracy on FER2013 is ~63% (human accuracy is ~65% — it's a hard dataset)
    - Works best on well-lit, frontal faces
    - May struggle with occlusion (masks, glasses, extreme angles)

    ---
    Built by **Santosh Narreddy** | [GitHub](https://github.com/santoshnarreddy) |
    [face-emotion-detection repo](https://github.com/santoshnarreddy/face-emotion-detection)
    """)
