"""
Visualization utilities for YOLOv8 detection results.
Author: Santosh Narreddy

Shared helpers used by detect.py — kept separate so they can
be imported in notebooks or other scripts without pulling in
the full detection loop.
"""

import cv2
import numpy as np
from pathlib import Path


def draw_boxes(frame: np.ndarray, results, conf_threshold: float = 0.4,
               show_labels: bool = True, line_thickness: int = 2) -> np.ndarray:
    """
    Draw YOLOv8 bounding boxes and labels on a frame.

    Args:
        frame:           BGR image (modified in-place)
        results:         Single Ultralytics Results object
        conf_threshold:  Min confidence to draw
        show_labels:     Whether to draw class + confidence text
        line_thickness:  Box border thickness

    Returns:
        Annotated frame (same array as input)
    """
    if results.boxes is None or len(results.boxes) == 0:
        return frame

    np.random.seed(42)
    colors = np.random.randint(50, 220, (80, 3), dtype=np.uint8)
    names  = results.names or {}

    for box in results.boxes:
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = tuple(int(c) for c in colors[cls_id % 80])

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

        if show_labels:
            cls_name = names.get(cls_id, str(cls_id))
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def count_detections(results) -> dict:
    """
    Return a {class_name: count} dict from a Results object.
    Useful for logging or building dashboards.
    """
    from collections import Counter
    if results.boxes is None:
        return {}
    names = results.names or {}
    return dict(Counter(names.get(int(b.cls[0]), '?') for b in results.boxes))


def save_annotated(frame: np.ndarray, output_path: str) -> str:
    """Save an annotated frame to disk and return the path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame)
    return output_path
