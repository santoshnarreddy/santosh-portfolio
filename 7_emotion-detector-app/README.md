# 🎭 Emotion Detector — Web App

A **Streamlit web app** that lets anyone detect facial emotions from uploaded photos — no Python or OpenCV installation required. Powered by the CNN from [face-emotion-detection](https://github.com/santoshnarreddy/face-emotion-detection).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://santoshnarreddy-emotion-detector.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)

---

## ✨ Features

- 📸 **Image upload** — JPG, PNG, WEBP
- 📊 **Interactive confidence bars** for all 7 emotions (Plotly)
- 🎨 **Dark-themed UI** matching GitHub's color scheme
- ⚙️ **Configurable** — adjust confidence threshold, toggle labels
- 🔗 **Live demo** on Streamlit Community Cloud (free hosting)

---

## 🚀 Run Locally

```bash
git clone https://github.com/santoshnarreddy/emotion-detector-app
cd emotion-detector-app
pip install -r requirements.txt

# Download model weights (from face-emotion-detection Releases)
mkdir model && mv emotion_cnn.h5 model/

streamlit run app.py
```
Open http://localhost:8501

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. **Fork or push** this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select this repo, branch `main`, main file `app.py`
4. Click **Deploy** — get a public URL in ~2 minutes!

> Add the live URL to your GitHub profile and LinkedIn for maximum impact.

---

## 📂 Project Structure

```
emotion-detector-app/
├── app.py                   # Main Streamlit app
├── utils/
│   ├── predictor.py         # CNN inference wrapper
│   └── visualize.py         # OpenCV + Plotly visualization
├── model/
│   └── emotion_cnn.h5       # Pretrained weights (download separately)
├── .streamlit/
│   └── config.toml          # Dark theme config
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
streamlit>=1.28
tensorflow>=2.10
opencv-python-headless>=4.7  # headless = no GUI needed for server deploy
numpy>=1.23
Pillow>=9.0
plotly>=5.15
```

---

## 📄 License

MIT
