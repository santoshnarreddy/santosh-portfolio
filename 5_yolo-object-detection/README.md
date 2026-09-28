# 🔍 YOLOv8 Object Detection

Real-time object detection using **YOLOv8** (Ultralytics) — the current state of the art for single-stage detectors. Supports webcam, images, and videos out of the box, plus a full pipeline for **training on custom datasets**.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## ✨ Features

- 🎥 **Real-time webcam detection** at 30+ FPS (nano model, modern CPU)
- 🖼️ **Image + video file** detection
- 🏋️ **Custom dataset training** with one command
- 🎨 **Per-class color coding** for all 80 COCO classes
- ⏸️ **Pause/resume** and screenshot capture during live detection
- 💾 **Output saving** to `runs/detect/`
- 🔧 **5 model sizes** — trade speed vs accuracy

---

## 🛠️ Setup

```bash
git clone https://github.com/santoshnarreddy/yolo-object-detection
cd yolo-object-detection
pip install -r requirements.txt
```

Models download automatically on first run (~6 MB for nano).

---

## 🚀 Usage

**Webcam (default):**
```bash
python detect.py
```

**Image:**
```bash
python detect.py --source photo.jpg
```

**Video:**
```bash
python detect.py --source video.mp4 --save
```

**Choose model size:**
```bash
python detect.py --source 0 --model small    # more accurate than nano
python detect.py --source 0 --model medium   # even better
```

**Controls (webcam):**
| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |
| `p` | Pause / resume |

---

## 🏋️ Train on a Custom Dataset

**1. Annotate your images**

Use [LabelImg](https://github.com/HumanSignal/labelImg) (free) or [Roboflow](https://roboflow.com) to annotate and export in YOLO format.

**2. Edit `data/custom_data.yaml`**
```yaml
path: data/my_dataset
train: images/train
val: images/val
nc: 2
names:
  0: cat
  1: dog
```

**3. Train:**
```bash
python train_custom.py --data data/custom_data.yaml --epochs 50 --model yolov8n.pt
```

**4. Detect with your model:**
```bash
python detect.py --source 0 --model runs/detect/custom_run/weights/best.pt
```

---

## 📊 YOLOv8 Model Comparison

| Model | Size | mAP@50-95 (COCO) | Speed (CPU, ms) |
|---|---|---|---|
| YOLOv8n (nano) | 6.2 MB | 37.3 | ~80 |
| YOLOv8s (small) | 21.5 MB | 44.9 | ~150 |
| YOLOv8m (medium) | 49.7 MB | 50.2 | ~234 |
| YOLOv8l (large) | 83.7 MB | 52.9 | ~375 |
| YOLOv8x (xlarge) | 130 MB | 53.9 | ~479 |

> For real-time on CPU, use `nano` or `small`. For GPU or offline processing, `medium`+ gives much better accuracy.

---

## 📂 Project Structure

```
yolo-object-detection/
├── detect.py            # Detection: webcam / image / video
├── train_custom.py      # Custom dataset training
├── data/
│   └── custom_data.yaml # Dataset config template
├── runs/
│   └── detect/          # Output directory (auto-created)
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
ultralytics>=8.0
opencv-python>=4.7
numpy>=1.23
```

---

## 📌 Notes

- YOLOv8 is a huge improvement over v5 in terms of API ergonomics — `model(frame)` returns everything you need in one call.
- The nano model is impressive for its size; I was surprised it handles real-time on a laptop CPU this well.
- For custom training, 50 epochs with mosaic augmentation was enough to get solid results on a small dataset (~300 images per class).

---

## 📄 License

MIT
