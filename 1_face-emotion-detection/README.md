# 🎭 Face Emotion Detection

Real-time facial emotion recognition from webcam using a custom CNN trained on the FER2013 dataset. Detects **7 emotions** live with per-class confidence bars displayed alongside each detected face.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## ✨ Features

- 🔴 **Real-time detection** — runs at ~25–30 FPS on a standard laptop
- 😠😢😲😊😐😱🤢 — Angry, Sad, Surprised, Happy, Neutral, Fearful, Disgusted
- 📊 **Live probability bars** drawn next to each detected face
- 💾 Screenshot capture with `s` key
- 🎥 Optional video output saving
- 🖥️ Works with webcam or pre-recorded video files

---

## 🛠️ Setup

```bash
git clone https://github.com/santoshnarreddy/face-emotion-detection
cd face-emotion-detection
pip install -r requirements.txt
```

> **Model**: Download the pre-trained `emotion_cnn.h5` from [Releases](https://github.com/santoshnarreddy/face-emotion-detection/releases) and place it in the `model/` folder.

---

## 🚀 Usage

**Real-time webcam:**
```bash
python emotion_detector.py
```

**Video file input:**
```bash
python emotion_detector.py --source path/to/video.mp4
```

**Save output video:**
```bash
python emotion_detector.py --save
```

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |

---

## 🏋️ Train From Scratch

1. Download the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) from Kaggle
2. Unzip to `data/fer2013/` (it should have `train/` and `test/` subfolders)
3. Run training:

```bash
python train_model.py --data_dir ./data/fer2013 --epochs 60 --batch 64
```

Training takes ~2 hours on a GPU. The best model (by val accuracy) is auto-saved to `model/emotion_cnn.h5`.

---

## 🧠 Model Architecture

Custom CNN with 3 convolutional blocks + batch normalization:

```
Input (48×48×1)
  → Conv2D(32) → BN → ReLU → Conv2D(32) → BN → MaxPool → Dropout(0.25)
  → Conv2D(64) → BN → ReLU → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  → Conv2D(128) → BN → ReLU → Conv2D(128) → BN → MaxPool → Dropout(0.4)
  → Flatten → Dense(256) → BN → Dropout(0.5)
  → Dense(7) → Softmax
```

**Training details:**
- Dataset: FER2013 (35,887 images)
- Optimizer: Adam (lr=0.001, ReduceLROnPlateau)
- Augmentation: rotation, flips, zoom, shifts
- Best val accuracy: **~63%** (FER2013 is a tough benchmark — human accuracy is ~65%)

---

## 📂 Project Structure

```
face-emotion-detection/
├── emotion_detector.py     # Real-time detection (main script)
├── train_model.py          # CNN training pipeline
├── model/
│   └── emotion_cnn.h5      # Pre-trained weights (download from Releases)
├── assets/
│   └── demo.gif
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
tensorflow>=2.10
opencv-python>=4.7
numpy>=1.23
matplotlib>=3.6
```

---

## 📌 Notes & Learnings

- FER2013 is notoriously noisy — some labels are clearly wrong. Getting above 65% is considered good.
- Batch normalization made a huge difference vs training without it (4–5% accuracy gain).
- The probability bars were trickier to implement than the detection itself — coordinate math with OpenCV is always a bit painful 😅

---

## 📄 License

MIT — feel free to use, modify, and share.
