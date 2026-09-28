# ✋ Hand Gesture Controller

Control your PC — **volume, brightness, mouse movement, and clicks** — using nothing but hand gestures detected in real time from your webcam. Built with **MediaPipe** and **OpenCV**, no hardware required.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-teal?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=flat-square)

---

## 🤙 Gesture Map

| Gesture | Action |
|---|---|
| ☝️ Index finger only | **Mouse move** — fingertip controls cursor position |
| ✊ Fist | **Left click** |
| ✌️ Index + middle (peace) | **Volume / Brightness** — finger spread = level |
| 🖐️ Open palm (5 fingers) | **Screenshot** — saved to current directory |
| 👍 Thumb up | Visual feedback only (coming soon: app switch) |

---

## 🛠️ Setup

```bash
git clone https://github.com/santoshnarreddy/hand-gesture-controller
cd hand-gesture-controller
pip install -r requirements.txt
```

**Windows only (for audio control):**
```bash
pip install pycaw comtypes
```

**For brightness control:**
```bash
pip install screen-brightness-control
```

---

## 🚀 Usage

```bash
python gesture_controller.py
```

**Custom camera:**
```bash
python gesture_controller.py --cam 1
```

Press `q` to quit.

---

## ⚙️ How It Works

1. **Hand Detection** — MediaPipe Hands detects 21 3D landmarks per hand at 25+ FPS
2. **Gesture Classification** — Custom rule-based classifier checks which fingers are extended based on landmark y-coordinates
3. **Action Mapping** — Each gesture triggers an OS-level action via `pyautogui`, `pycaw`, or `screen_brightness_control`
4. **Mouse Smoothing** — Exponential moving average prevents jitter from small detection noise

---

## 📂 Project Structure

```
hand-gesture-controller/
├── gesture_controller.py   # Main script — detection + control
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
mediapipe>=0.10
opencv-python>=4.7
pyautogui>=0.9
numpy>=1.23
```

Optional:
```
pycaw          # Windows audio control
comtypes       # Windows COM interface (required by pycaw)
screen-brightness-control   # Cross-platform brightness
```

---

## 📌 Notes & Learnings

- MediaPipe's landmark coordinates are normalized (0–1) relative to frame size — I spent a while figuring out the coordinate mapping for accurate mouse control 😅
- The exponential smoothing on mouse position was essential — without it the cursor was completely unusable.
- Gesture classification is purely geometric (no ML) — this made it very snappy but sensitive to hand orientation. A proper ML classifier would handle edge cases better.

---

## 🔮 Planned Features

- [ ] Double-click gesture
- [ ] Scroll with pinch
- [ ] Multi-hand support (two-hand gestures)
- [ ] Gesture training mode (record custom gestures)

---

## 📄 License

MIT
