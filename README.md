# 🤖 Santosh Narreddy — AI/ML Project Portfolio

Complete source code for all 7 AI/ML, Computer Vision, and Deep Learning projects.

**Author:** Santosh Narreddy · [github.com/santoshnarreddy](https://github.com/santoshnarreddy)

---

## 📦 Projects

| # | Project | Tech | What it does |
|---|---------|------|--------------|
| 1 | [face-emotion-detection](./1_face-emotion-detection/) | OpenCV · TensorFlow · CNN | Real-time 7-emotion detection from webcam |
| 2 | [crop-disease-classifier](./2_crop-disease-classifier/) | PyTorch · ResNet50 · Transfer Learning | 38-class plant disease detection from leaf images |
| 3 | [hand-gesture-controller](./3_hand-gesture-controller/) | OpenCV · MediaPipe | Control PC volume/mouse/brightness with gestures |
| 4 | [spam-mail-detector](./4_spam-mail-detector/) | scikit-learn · NLTK · TF-IDF | 97.3% accuracy email spam classifier |
| 5 | [yolo-object-detection](./5_yolo-object-detection/) | YOLOv8 · OpenCV | Real-time object detection + custom training |
| 6 | [face-recognition-attendance](./6_face-recognition-attendance/) | face_recognition · OpenCV | Auto attendance marking with face recognition |
| 7 | [emotion-detector-app](./7_emotion-detector-app/) | Streamlit · TensorFlow · Plotly | Web UI for the emotion detector — deploy free on Streamlit Cloud |

---

## ⚡ Quick Start

Each project has its own `README.md` with full setup instructions.
The general pattern for every project:

```bash
cd <project-folder>
pip install -r requirements.txt
python <main_script>.py
```

See `QUICKSTART.md` in each folder for the exact command.

---

## 🛠️ System Requirements

- Python 3.9 or 3.10 recommended
- pip 23+
- Webcam (for projects 1, 3, 5, 6)
- 4 GB RAM minimum; 8 GB recommended for training
- GPU optional but speeds up training (projects 2, 5)

---

## 📁 Folder Structure

```
santosh-github-portfolio/
├── README.md                          ← This file
├── 1_face-emotion-detection/          ← CNN emotion detection
├── 2_crop-disease-classifier/         ← ResNet50 plant disease
├── 3_hand-gesture-controller/         ← MediaPipe gesture control
├── 4_spam-mail-detector/              ← NLP spam classifier
├── 5_yolo-object-detection/           ← YOLOv8 detector
├── 6_face-recognition-attendance/     ← Face recognition + CSV log
└── 7_emotion-detector-app/            ← Streamlit web app
```
