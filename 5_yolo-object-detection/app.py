"""
YOLOv8 Object Detection — Gradio Web Application
Author: Santosh Narreddy
Deployable directly on Hugging Face Spaces or local browser.
"""

import os
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import gradio as gr
from ultralytics import YOLO

# Cache models
models = {}

def get_model(model_name):
    if model_name not in models:
        models[model_name] = YOLO(model_name)
    return models[model_name]

def detect_objects(image, model_choice, conf_threshold, iou_threshold):
    if image is None:
        return None, "Please upload an image to run object detection."

    model = get_model(model_choice)
    img_np = np.array(image.convert('RGB'))

    results = model.predict(
        source=img_np,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )[0]

    # Render bounding boxes onto image
    annotated_bgr = results.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Class distribution statistics
    detected_classes = [results.names[int(box.cls[0])] for box in results.boxes]
    counts = Counter(detected_classes)

    if counts:
        summary_lines = [f"### 🎯 Detected {len(detected_classes)} Objects:"]
        for cls_name, count in counts.most_common():
            summary_lines.append(f"- **{cls_name}**: {count}")
        summary_md = "\n".join(summary_lines)
    else:
        summary_md = "### ℹ️ No objects detected above the confidence threshold."

    return Image.fromarray(annotated_rgb), summary_md

demo = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Dropdown(
            choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            value="yolov8n.pt",
            label="YOLOv8 Model Variant"
        ),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.35, step=0.05, label="Confidence Threshold"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.45, step=0.05, label="IoU Threshold")
    ],
    outputs=[
        gr.Image(type="pil", label="Detected Objects"),
        gr.Markdown(label="Detections Summary")
    ],
    title="🔍 YOLOv8 Real-Time Object Detection",
    description="State-of-the-art real-time object detection with YOLOv8. Upload any image to detect 80 COCO object classes with customizable confidence and IoU thresholds.",
    article="**Author:** [Santosh Narreddy](https://github.com/santoshnarreddy) | **GitHub Repository:** [yolo-object-detection](https://github.com/santoshnarreddy/yolo-object-detection)",
    theme="default"
)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
