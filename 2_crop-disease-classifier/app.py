"""
Crop Disease Classifier — Gradio Web Application
Author: Santosh Narreddy
Deployable directly on Hugging Face Spaces or local browser.
"""

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr

from model import build_resnet50, DISEASE_CLASSES, NUM_CLASSES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Format user-friendly labels
DISPLAY_NAMES = {cls: cls.replace('___', ' — ').replace('_', ' ') for cls in DISEASE_CLASSES}

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Initialize model
model = build_resnet50(num_classes=NUM_CLASSES, freeze_backbone=False)

checkpoint_paths = [
    'checkpoints/best_model_phase2.pth',
    'checkpoints/best_model.pth',
    'checkpoints/model.pt',
    'model.pt'
]

loaded_ckpt = False
for cp in checkpoint_paths:
    if os.path.exists(cp):
        try:
            ckpt = torch.load(cp, map_location=DEVICE)
            state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
            model.load_state_dict(state_dict)
            loaded_ckpt = True
            print(f'[INFO] Loaded checkpoint from {cp}')
            break
        except Exception as e:
            print(f'[WARN] Could not load {cp}: {e}')

model.eval()
model.to(DEVICE)

def classify_crop_disease(image):
    if image is None:
        return {}, "Please upload an image of a plant leaf."

    img = image.convert('RGB')
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top_indices = probs.argsort()[::-1][:5]
    confidences = {DISPLAY_NAMES[DISEASE_CLASSES[i]]: float(probs[i]) for i in top_indices}

    top_class = DISEASE_CLASSES[top_indices[0]]
    plant, condition = top_class.split('___')
    plant_clean = plant.replace('_', ' ')
    condition_clean = condition.replace('_', ' ')

    if 'healthy' in condition.lower():
        status_md = f"### ✅ Healthy Leaf Detected\n**Plant:** {plant_clean}\n**Condition:** Healthy (No disease detected)"
    else:
        status_md = f"### ⚠️ Disease Detected: {condition_clean}\n**Plant:** {plant_clean}\n**Condition:** {condition_clean}"

    if not loaded_ckpt:
        status_md += "\n\n*(Demo mode: feature extractor initialized. Drop trained weights in `checkpoints/` for production accuracy)*"

    return confidences, status_md

demo = gr.Interface(
    fn=classify_crop_disease,
    inputs=gr.Image(type="pil", label="Upload Leaf Image"),
    outputs=[
        gr.Label(num_top_classes=5, label="Top 5 Predictions"),
        gr.Markdown(label="Analysis Result")
    ],
    title="🌿 Plant Disease Classifier — ResNet50",
    description="Upload a clear photo of a crop or plant leaf to identify 38 disease categories across 14 plant species. Powered by PyTorch & ResNet50 Transfer Learning.",
    article="**Author:** [Santosh Narreddy](https://github.com/santoshnarreddy) | **GitHub Repository:** [crop-disease-classifier](https://github.com/santoshnarreddy/crop-disease-classifier)",
    theme="default"
)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860)
