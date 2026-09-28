# 👤 Face Recognition Attendance System

Automated attendance marking using **real-time face recognition** from a webcam. Enroll people once, then the system recognizes them and logs their attendance to a CSV — no manual entry needed.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![face_recognition](https://img.shields.io/badge/face__recognition-dlib-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## ✨ Features

- 📸 **Enroll people** via webcam (30 samples) or from a photo
- 🧠 **128-d face embeddings** via `face_recognition` (built on dlib/FaceNet)
- 📋 **Auto-logs to CSV** — Name, Date, Time, Status
- 🔁 **Deduplication** — marks each person only once per session
- 📊 **Live sidebar** showing today's attendance on the webcam feed
- 🔧 **Configurable threshold** for match sensitivity
- 💾 **Persistent database** — add new people any time, old ones stay

---

## 🛠️ Setup

```bash
git clone https://github.com/santoshnarreddy/face-recognition-attendance
cd face-recognition-attendance
pip install -r requirements.txt
```

> **Note:** `face-recognition` requires `cmake` and `dlib`. On Ubuntu: `sudo apt install cmake`. On Windows: see [dlib install guide](http://dlib.net/compile.html).

---

## 🚀 Usage

**Step 1 — Enroll people:**
```bash
# Webcam enrollment (recommended — captures 30 varied frames)
python enroll.py --name "Santosh" --samples 30

# Or from a photo
python enroll.py --name "Priya" --image priya.jpg

# List enrolled people
python enroll.py --list
```

**Step 2 — Run attendance:**
```bash
python attendance_system.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit + save log |
| `r` | Print attendance report to terminal |

**View logs:**
```bash
cat logs/attendance_2026-09-28.csv
```

---

## 📋 Attendance Log Format

```csv
Name,Date,Time,Status
Santosh,2026-09-28,09:14:32,Present
Priya,2026-09-28,09:15:01,Present
```

---

## 🧠 How It Works

```
Webcam frame
    ↓
Haar Cascade → detect face bounding boxes
    ↓
face_recognition library → compute 128-d embedding per face
    ↓
Cosine similarity against all stored embeddings per person
    ↓
Best match above threshold → mark attendance in CSV
```

**Why cosine similarity?**
More robust than Euclidean distance when face embeddings have slight magnitude variation across lighting conditions.

---

## ⚙️ Configuration

| Parameter | Default | Description |
|---|---|---|
| `--threshold` | 0.55 | Cosine similarity cutoff (higher = stricter) |
| `--samples` | 30 | Enrollment frames per person |
| `--cam` | 0 | Camera index |
| `--log_dir` | `logs/` | Output directory |

Tune `--threshold` for your environment: busy/bright rooms need slightly higher (0.6), low-light rooms work better lower (0.5).

---

## 📂 Project Structure

```
face-recognition-attendance/
├── attendance_system.py      # Main recognition + logging loop
├── enroll.py                 # Enrollment (webcam or image)
├── utils/
│   ├── face_utils.py         # Detection, embedding, similarity
│   └── attendance_utils.py   # CSV logging, reports
├── database/
│   └── encodings.pkl         # Stored face embeddings (auto-created)
├── logs/
│   └── attendance_YYYY-MM-DD.csv   # Daily logs (auto-created)
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
face-recognition>=1.3.0
opencv-python>=4.7.0
numpy>=1.23.0
cmake>=3.25.0  # Required by dlib
```

---

## 🔮 Planned Improvements

- [ ] Web dashboard to view attendance history
- [ ] Support for anti-spoofing (detect printed photos)
- [ ] Multi-camera support
- [ ] Email report at end of day

---

## 📄 License

MIT
