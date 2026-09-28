# Quick Start — YOLOv8 Object Detection

## Install
```bash
pip install -r requirements.txt
```
YOLOv8 model weights download automatically on first run (~6 MB for nano).

## Run on webcam (default)
```bash
python detect.py
```

## Run on image
```bash
python detect.py --source photo.jpg
```

## Run on video
```bash
python detect.py --source video.mp4 --save
```

## Choose model size (speed vs accuracy)
```bash
python detect.py --model nano    # fastest (default)
python detect.py --model small   # better accuracy
python detect.py --model medium  # even better
```

## Train on custom dataset
1. Annotate images with LabelImg or Roboflow (YOLO format)
2. Edit `data/custom_data.yaml` with your class names and paths
3. Run: `python train_custom.py --data data/custom_data.yaml --epochs 50`
4. Use: `python detect.py --model runs/detect/custom_run/weights/best.pt`

## Controls (webcam)
- `q` → quit
- `s` → save screenshot
- `p` → pause / resume
