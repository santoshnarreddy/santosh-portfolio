"""
Crop Disease Classifier — Inference Script
Author: Santosh Narreddy

Run predictions on single images or entire folders.
Outputs top-5 predictions with confidence scores.

Usage:
    python predict.py --image leaf.jpg --model checkpoints/best_model_phase2.pth
    python predict.py --folder ./test_images/ --model checkpoints/best_model_phase2.pth
"""

import os
import argparse
import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from model import build_resnet50, DISEASE_CLASSES, NUM_CLASSES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Friendly display names (strip the plant prefix for readability)
DISPLAY_NAMES = {cls: cls.replace('___', ' — ').replace('_', ' ') for cls in DISEASE_CLASSES}

# Preprocessing (same as validation transforms during training)
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(checkpoint_path):
    model = build_resnet50(num_classes=NUM_CLASSES, freeze_backbone=False)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(DEVICE)
    print(f"[INFO] Model loaded from {checkpoint_path} "
          f"(val_acc={ckpt.get('val_acc', 'N/A'):.4f})")
    return model


def predict_image(model, image_path, top_k=5):
    """
    Returns top-k (class_name, confidence) tuples for a single image.
    """
    img = Image.open(image_path).convert('RGB')
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=min(top_k, NUM_CLASSES))
    results = []
    for prob, idx in zip(top_probs.cpu().tolist(), top_idxs.cpu().tolist()):
        cls_name = DISEASE_CLASSES[idx]
        results.append((DISPLAY_NAMES[cls_name], float(prob) * 100))
    return results


def visualize_prediction(image_path, predictions, save_path=None):
    """
    Draw prediction results on the image using OpenCV.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[WARN] Could not read {image_path} with OpenCV")
        return

    # Resize for display
    h, w = img.shape[:2]
    target_h = 480
    scale = target_h / h
    img = cv2.resize(img, (int(w * scale), target_h))

    # Draw panel on the right side
    panel_w = 380
    panel = np.zeros((img.shape[0], panel_w, 3), dtype=np.uint8)
    panel[:] = (25, 25, 35)

    top_name, top_conf = predictions[0]
    plant, disease = top_name.split(' — ') if ' — ' in top_name else (top_name, 'Healthy')

    y_off = 30
    cv2.putText(panel, "PREDICTION", (12, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)
    y_off += 28
    cv2.putText(panel, plant, (12, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    y_off += 28
    cv2.putText(panel, disease, (12, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (80, 255, 120) if 'healthy' in disease.lower() else (80, 120, 255), 2)

    y_off += 25
    cv2.line(panel, (12, y_off), (panel_w - 12, y_off), (60, 60, 80), 1)
    y_off += 20

    cv2.putText(panel, "Top-5 Confidence:", (12, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)
    y_off += 20

    bar_colors = [(80, 255, 120), (120, 200, 255), (255, 200, 80), (255, 120, 80), (180, 80, 255)]
    bar_max_w = panel_w - 24
    for i, (name, conf) in enumerate(predictions):
        label = name if len(name) < 35 else name[:32] + "..."
        cv2.putText(panel, f"{label}  {conf:.1f}%", (12, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        y_off += 14
        bar_len = int(bar_max_w * conf / 100)
        color = bar_colors[i % len(bar_colors)]
        cv2.rectangle(panel, (12, y_off), (12 + bar_len, y_off + 8), color, -1)
        cv2.rectangle(panel, (12 + bar_len, y_off), (12 + bar_max_w, y_off + 8), (50, 50, 65), -1)
        y_off += 18

    combined = np.hstack([img, panel])

    cv2.imshow("Crop Disease Classifier — Santosh Narreddy", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if save_path:
        cv2.imwrite(save_path, combined)
        print(f"[INFO] Result saved to {save_path}")


def predict_folder(model, folder_path, top_k=1):
    """Batch prediction for a folder of images."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(exts)]

    if not image_files:
        print(f"[WARN] No images found in {folder_path}")
        return

    print(f"\n{'Image':<40} {'Prediction':<45} {'Confidence':>10}")
    print("-" * 100)
    for fname in sorted(image_files):
        fpath = os.path.join(folder_path, fname)
        preds = predict_image(model, fpath, top_k=top_k)
        name, conf = preds[0]
        print(f"{fname:<40} {name:<45} {conf:>9.1f}%")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Disease Prediction")
    parser.add_argument('--model',   required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--image',   help='Single image path')
    parser.add_argument('--folder',  help='Folder of images for batch prediction')
    parser.add_argument('--top_k',   type=int, default=5)
    parser.add_argument('--save',    help='Save visualization to this path')
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image:
        preds = predict_image(model, args.image, top_k=args.top_k)
        print(f"\n🌿 Results for: {args.image}")
        for rank, (name, conf) in enumerate(preds, 1):
            print(f"  #{rank}  {name:<50}  {conf:.2f}%")
        visualize_prediction(args.image, preds, save_path=args.save)

    elif args.folder:
        predict_folder(model, args.folder, top_k=args.top_k)
    else:
        parser.print_help()
