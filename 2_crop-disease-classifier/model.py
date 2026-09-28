"""
Crop Disease Classifier — Model Definition
Author: Santosh Narreddy

Uses ResNet50 pretrained on ImageNet as a feature extractor,
with a custom classification head for plant disease detection.

Dataset: PlantVillage (38 disease classes across 14 plant species)
"""

import torch
import torch.nn as nn
from torchvision import models


# 38 classes from PlantVillage dataset
DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_Black_Measles',
    'Grape___Leaf_blight_Isariopsis_Leaf_Spot', 'Grape___healthy',
    'Orange___Haunglongbing_Citrus_greening',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two_spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

NUM_CLASSES = len(DISEASE_CLASSES)   # 38


def build_resnet50(num_classes=NUM_CLASSES, freeze_backbone=True):
    """
    ResNet50 with pretrained ImageNet weights.
    We freeze the backbone and train only the custom head — fine-tune later.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the original 1000-class FC with our custom head
    in_features = model.fc.in_features  # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )

    return model


def unfreeze_last_n_layers(model, n=3):
    """
    After initial training with frozen backbone, unfreeze last n ResNet blocks
    for fine-tuning at a lower learning rate.
    """
    # ResNet50 layer groups: layer1, layer2, layer3, layer4
    layers_to_unfreeze = list(model.children())[-n - 1:]
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
    print(f"[INFO] Unfroze last {n} layer groups for fine-tuning.")
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}  ({100*trainable/total:.1f}%)")
