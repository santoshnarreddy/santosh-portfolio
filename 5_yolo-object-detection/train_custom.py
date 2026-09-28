"""
YOLOv8 Custom Dataset Training
Author: Santosh Narreddy

Fine-tune YOLOv8 on your own dataset for custom object detection.
For example: detecting specific products, vehicles, defects, etc.

Dataset format: YOLO format (one .txt per image with normalized bbox coords)
  Each line in the .txt: class_id cx cy width height (all normalized 0-1)

Usage:
    python train_custom.py --data data/custom_data.yaml --epochs 50
    python train_custom.py --data data/custom_data.yaml --model yolov8s.pt --epochs 100
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def train(data_yaml: str, model_size: str = 'yolov8n.pt',
          epochs: int = 50, img_size: int = 640,
          batch: int = 16, name: str = 'custom_run'):
    """
    Fine-tune YOLOv8 on a custom dataset.

    Args:
        data_yaml:  Path to your dataset YAML (see data/custom_data.yaml)
        model_size: Pretrained weights to start from (n/s/m/l/x)
        epochs:     Training epochs (50 often enough for fine-tuning)
        img_size:   Input image size (640 is standard)
        batch:      Batch size — reduce to 8 if GPU OOM
        name:       Run name (outputs saved to runs/detect/<name>/)
    """
    print(f"\n{'='*60}")
    print(f"YOLOv8 Custom Training — Santosh Narreddy")
    print(f"{'='*60}")
    print(f"  Model:    {model_size}")
    print(f"  Data:     {data_yaml}")
    print(f"  Epochs:   {epochs}")
    print(f"  Img size: {img_size}")
    print(f"  Batch:    {batch}")
    print(f"{'='*60}\n")

    model = YOLO(model_size)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        name=name,

        # Augmentation — YOLOv8 has great built-in augmentation
        mosaic=1.0,       # Mosaic augmentation (4 images stitched)
        mixup=0.1,        # MixUp augmentation
        degrees=5.0,      # Random rotation
        translate=0.1,    # Random translation
        scale=0.5,        # Random scale
        fliplr=0.5,       # Horizontal flip
        hsv_h=0.015,      # Hue shift
        hsv_s=0.7,        # Saturation shift
        hsv_v=0.4,        # Value (brightness) shift

        # Training settings
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,          # Final LR = lr0 * lrf (cosine decay)
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=20,       # Early stopping
        save=True,
        plots=True,        # Save training plots automatically

        device='',         # Auto-select GPU or CPU
        workers=4,
        verbose=True,
    )

    # Print results
    best_map = results.results_dict.get('metrics/mAP50-95(B)', 0)
    save_dir = results.save_dir
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best mAP@50-95: {best_map:.4f}")
    print(f"  Weights saved:  {save_dir}/weights/best.pt")
    print(f"  To run inference with your custom model:")
    print(f"  python detect.py --source 0 --model {save_dir}/weights/best.pt")
    print(f"{'='*60}\n")

    return results


def evaluate(weights_path: str, data_yaml: str, img_size: int = 640):
    """Run validation on test set with trained weights."""
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, imgsz=img_size)

    print(f"\n[EVALUATION RESULTS]")
    print(f"  mAP@50:    {metrics.box.map50:.4f}")
    print(f"  mAP@50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Custom Training")
    parser.add_argument('--data',   required=True, help='Path to dataset YAML')
    parser.add_argument('--model',  default='yolov8n.pt',
                        help='Base model (yolov8n/s/m/l/x.pt, default: yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz',  type=int, default=640)
    parser.add_argument('--batch',  type=int, default=16)
    parser.add_argument('--name',   default='custom_run', help='Run name')
    parser.add_argument('--eval',   help='Evaluate saved weights (skip training)')
    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, args.data, args.imgsz)
    else:
        train(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            img_size=args.imgsz,
            batch=args.batch,
            name=args.name
        )
