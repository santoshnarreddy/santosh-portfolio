# Quick Start — Crop Disease Classifier

## Install
```bash
pip install -r requirements.txt
```

## Predict a single leaf image
```bash
python predict.py --model checkpoints/best_model_phase2.pth --image leaf.jpg
```

## Batch predict a folder
```bash
python predict.py --model checkpoints/best_model_phase2.pth --folder ./test_images/
```

## Train from scratch
1. Download PlantVillage dataset from https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Prepare train/val split: `python utils/split_dataset.py --src ./data/raw --out ./data/PlantVillage`
3. Train: `python train.py --data_dir ./data/PlantVillage --epochs_phase1 10 --epochs_phase2 20`
4. Best weights saved to `checkpoints/`

## Note
Model weights are NOT included (too large). Train using steps above.
