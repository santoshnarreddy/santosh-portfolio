# Quick Start — Face Recognition Attendance System

## Install
### Step 1 — Install cmake (required for dlib)
- Ubuntu/Debian: `sudo apt install cmake build-essential`
- macOS: `brew install cmake`
- Windows: Download from https://cmake.org/download/ and add to PATH

### Step 2 — Install Python packages
```bash
pip install -r requirements.txt
```

## Enroll a person (webcam — recommended, 30 samples)
```bash
python enroll.py --name "YourName" --samples 30
```

## Enroll from a photo
```bash
python enroll.py --name "YourName" --image photo.jpg
```

## List enrolled people
```bash
python enroll.py --list
```

## Run attendance
```bash
python attendance_system.py
```

## View today's attendance log
```bash
cat logs/attendance_$(date +%Y-%m-%d).csv
```

## Controls (during attendance)
- `q` → quit + save log
- `r` → print attendance report to terminal
