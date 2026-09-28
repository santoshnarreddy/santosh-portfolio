# Quick Start — Emotion Detector Streamlit App

## Install
```bash
pip install -r requirements.txt
```

## Add model weights
Copy `emotion_cnn.h5` (trained in project 1) to the `model/` folder:
```bash
cp ../1_face-emotion-detection/model/emotion_cnn.h5 model/
```

## Run locally
```bash
streamlit run app.py
```
Opens at http://localhost:8501

## Deploy FREE on Streamlit Cloud (get a public URL)
1. Push this folder as a GitHub repo
2. Go to https://share.streamlit.io → New app
3. Select your repo, branch `main`, main file `app.py`
4. Click Deploy — live in ~2 minutes!

## Features
- Upload image → get emotion prediction with confidence bars
- Adjustable confidence threshold (sidebar)
- Dark theme matching GitHub
- Webcam tab works locally (not on cloud deploy)
