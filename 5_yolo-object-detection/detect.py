"""
YOLOv8 Object Detection
Author: Santosh Narreddy

Real-time object detection using YOLOv8 (Ultralytics).
Supports webcam, image files, and video files.
Includes custom dataset training support.

YOLOv8 is the current state of the art for real-time object detection —
much faster and more accurate than older YOLO versions.

Usage:
    python detect.py --source 0                        # webcam
    python detect.py --source image.jpg                # image
    python detect.py --source video.mp4                # video
    python detect.py --source 0 --model yolov8s.pt    # use small model
    python detect.py --source 0 --conf 0.4 --save     # save output
"""

import cv2
import argparse
import time
import os
import numpy as np
from pathlib import Path

from ultralytics import YOLO

# Available pretrained models (download automatically on first run)
MODELS = {
    'nano':   'yolov8n.pt',   # Fastest, least accurate — good for weak GPUs
    'small':  'yolov8s.pt',   # Good balance
    'medium': 'yolov8m.pt',   # Better accuracy
    'large':  'yolov8l.pt',   # High accuracy
    'xlarge': 'yolov8x.pt',   # Best accuracy, slowest
}

# COCO class colors — one per class, cycling
np.random.seed(42)
CLASS_COLORS = np.random.randint(50, 220, (80, 3), dtype=np.uint8)


# ── Visualization ──────────────────────────────────────────────────────────────

def draw_detections(frame, results, conf_threshold=0.3, show_labels=True):
    """
    Draw bounding boxes and labels for YOLOv8 detection results.
    results is a single Ultralytics Results object.
    """
    if results.boxes is None or len(results.boxes) == 0:
        return frame

    boxes   = results.boxes
    names   = results.names   # {class_id: class_name}

    for box in boxes:
        conf    = float(box.conf[0])
        cls_id  = int(box.cls[0])
        cls_name = names.get(cls_id, str(cls_id))

        if conf < conf_threshold:
            continue

        # Bounding box coords
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = tuple(int(c) for c in CLASS_COLORS[cls_id % 80])

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame


def draw_hud(frame, fps, num_detections, model_name, source_label):
    """Draw a clean info bar at the top of the frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 44), (12, 12, 20), -1)
    cv2.putText(frame, f"YOLOv8 | {model_name} | {source_label}",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 200), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}  |  Objects: {num_detections}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 120), 1)
    cv2.putText(frame, "q: quit | s: screenshot | p: pause",
                (w - 320, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)
    return frame


# ── Detection Modes ────────────────────────────────────────────────────────────

def detect_webcam(model, conf=0.4, model_name='yolov8n', save=False, save_dir='runs/detect'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    writer = None
    if save:
        os.makedirs(save_dir, exist_ok=True)
        ts   = int(time.time())
        path = os.path.join(save_dir, f"webcam_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, 20.0, (1280, 720))
        print(f"[INFO] Saving output to {path}")

    fps_timer   = time.time()
    fps_count   = 0
    fps_display = 0.0
    paused      = False

    print("[INFO] Webcam detection running. Press 'q' to quit.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # YOLOv8 inference
            results = model(frame, verbose=False, conf=conf)[0]
            draw_detections(frame, results, conf_threshold=conf)
            num_det = len(results.boxes) if results.boxes else 0

            fps_count += 1
            if fps_count == 15:
                fps_display = 15 / (time.time() - fps_timer)
                fps_timer   = time.time()
                fps_count   = 0

            draw_hud(frame, fps_display, num_det, model_name, "Webcam")

            if writer:
                writer.write(frame)

        cv2.imshow("YOLOv8 Object Detection — Santosh Narreddy", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("[INFO] Paused" if paused else "[INFO] Resumed")
        elif key == ord('s'):
            ts = int(time.time())
            fname = f"screenshot_{ts}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot saved: {fname}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def detect_image(model, image_path: str, conf=0.4, save=False, save_dir='runs/detect'):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return

    results = model(img, verbose=False, conf=conf)[0]
    draw_detections(img, results, conf_threshold=conf)

    num_det = len(results.boxes) if results.boxes else 0
    names   = results.names or {}

    print(f"\n[RESULTS] {image_path}")
    print(f"  Objects detected: {num_det}")
    if results.boxes:
        from collections import Counter
        class_counts = Counter(names.get(int(b.cls[0]), '?') for b in results.boxes)
        for cls, cnt in sorted(class_counts.items()):
            print(f"    {cls}: {cnt}")

    if save:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, f"det_{Path(image_path).name}")
        cv2.imwrite(out, img)
        print(f"  Saved: {out}")

    cv2.imshow("YOLOv8 Detection — Santosh Narreddy", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_video(model, video_path: str, conf=0.4, model_name='yolov8n',
                 save=False, save_dir='runs/detect'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if save:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"det_{Path(video_path).name}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps_src, (w, h))
        print(f"[INFO] Saving to {out_path}")

    frame_idx = 0
    t_start   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False, conf=conf)[0]
        draw_detections(frame, results, conf_threshold=conf)
        num_det = len(results.boxes) if results.boxes else 0

        elapsed = time.time() - t_start
        fps_cur = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        draw_hud(frame, fps_cur, num_det, model_name, Path(video_path).name)

        if writer:
            writer.write(frame)

        cv2.imshow("YOLOv8 Video Detection — Santosh Narreddy", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = (frame_idx / total * 100) if total > 0 else 0
            print(f"[INFO] Progress: {frame_idx}/{total} frames ({pct:.1f}%)  FPS: {fps_cur:.1f}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Processed {frame_idx} frames in {time.time()-t_start:.1f}s")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument('--source', default='0',
                        help='0=webcam, path to image/video (default: webcam)')
    parser.add_argument('--model', default='nano',
                        choices=list(MODELS.keys()) + list(MODELS.values()),
                        help='Model size: nano/small/medium/large/xlarge (default: nano)')
    parser.add_argument('--conf', type=float, default=0.4,
                        help='Confidence threshold (default: 0.4)')
    parser.add_argument('--save', action='store_true',
                        help='Save output to runs/detect/')
    parser.add_argument('--save_dir', default='runs/detect')
    args = parser.parse_args()

    # Resolve model name
    model_file = MODELS.get(args.model, args.model)
    print(f"[INFO] Loading model: {model_file}")
    model = YOLO(model_file)   # Downloads automatically on first run
    print(f"[INFO] Model loaded. Classes: {len(model.names)}")

    source = args.source
    is_webcam = (source == '0' or source.isdigit())

    if is_webcam:
        detect_webcam(model, conf=args.conf, model_name=model_file,
                      save=args.save, save_dir=args.save_dir)
    elif Path(source).suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
        detect_image(model, source, conf=args.conf,
                     save=args.save, save_dir=args.save_dir)
    else:
        detect_video(model, source, conf=args.conf, model_name=model_file,
                     save=args.save, save_dir=args.save_dir)
