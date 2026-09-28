# 🌿 Crop Disease Classifier

Detects **38 plant diseases** across 14 crop species from leaf images using **transfer learning (ResNet50)**. Trained on the PlantVillage dataset — achieves **87%+ validation accuracy** with a two-phase fine-tuning strategy.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?style=flat-square&logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![Accuracy](https://img.shields.io/badge/Accuracy-87%25+-brightgreen?style=flat-square)

---

## 🌱 Supported Crops & Diseases

Covers diseases in: **Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato**

38 total classes (healthy + diseased variants per plant).

---

## ✨ Features

- 🔬 **38-class disease classification** with confidence scores
- 🖼️ **Visual output** — OpenCV overlay with top-5 predictions + bar chart
- 📁 **Batch prediction** for folders of images
- 🏋️ **Two-phase transfer learning** — head training then backbone fine-tuning
- 🚀 Works on CPU & GPU

---

## 🛠️ Setup

```bash
git clone https://github.com/santoshnarreddy/crop-disease-classifier
cd crop-disease-classifier
pip install -r requirements.txt
```

---

## 🚀 Usage

**Single image prediction:**
```bash
python predict.py --model checkpoints/best_model_phase2.pth --image leaf.jpg
```

**Batch prediction on a folder:**
```bash
python predict.py --model checkpoints/best_model_phase2.pth --folder ./test_images/
```

**Save visualized result:**
```bash
python predict.py --model checkpoints/best_model_phase2.pth --image leaf.jpg --save result.jpg
```

---

## 🏋️ Train From Scratch

**1. Download the dataset:**
```bash
# PlantVillage is available on Kaggle:
# https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
```

**2. Split into train/val:**
```bash
python utils/split_dataset.py --src ./data/raw --out ./data/PlantVillage --split 0.8
```

**3. Train:**
```bash
python train.py --data_dir ./data/PlantVillage --epochs_phase1 10 --epochs_phase2 20 --batch_size 32
```

---

## 🧠 Architecture & Approach

**Why ResNet50?**
- Pretrained on ImageNet — general visual features transfer well to leaf textures
- Deep enough to capture fine-grained disease patterns
- Faster to train than from scratch (crucial when GPU time is limited)

**Two-Phase Training:**

| Phase | What's trained | LR | Epochs |
|---|---|---|---|
| Phase 1 | Custom head only | 1e-3 | 10 |
| Phase 2 | Head + last 3 ResNet blocks | 1e-4 | 20 |

Phase 2 fine-tuning gave ~4% accuracy improvement over phase 1 alone.

---

## 📊 Results

| Metric | Value |
|---|---|
| Val Accuracy | **87.3%** |
| Val Loss | 0.38 |
| Best Epoch | 26 |

> Training on a single GPU takes ~45 min for both phases combined.

---

## 📂 Project Structure

```
crop-disease-classifier/
├── model.py          # ResNet50 architecture definition
├── train.py          # Two-phase training pipeline
├── predict.py        # Single image / batch inference + OpenCV visualization
├── utils/
│   └── split_dataset.py  # Train/val split utility
├── checkpoints/      # Saved model weights (created during training)
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
torch>=2.0
torchvision>=0.15
opencv-python>=4.7
Pillow>=9.0
numpy>=1.23
```

---

## 📌 Notes

- The dataset has class imbalance (tomato diseases are overrepresented). I experimented with weighted sampling but standard augmentation worked nearly as well.
- Predictions on blurry or heavily shadowed images tend to drop accuracy — something to fix with better augmentation or a segmentation preprocessing step.

---

## 📄 License

MIT
