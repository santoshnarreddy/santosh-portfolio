"""
Train/Val Split Utility
Author: Santosh Narreddy

Splits PlantVillage dataset (or any flat class-folder structure) into
train/ and val/ subfolders, preserving class distribution.

Usage:
    python utils/split_dataset.py --src ./data/raw --out ./data/PlantVillage --split 0.8
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def split_dataset(src_dir, out_dir, train_ratio=0.8, seed=42):
    random.seed(seed)
    src = Path(src_dir)
    out = Path(out_dir)

    class_dirs = [d for d in src.iterdir() if d.is_dir()]
    print(f"[INFO] Found {len(class_dirs)} classes in {src_dir}")

    total_train = 0
    total_val   = 0

    for cls_dir in sorted(class_dirs):
        cls_name = cls_dir.name
        images = list(cls_dir.glob('*'))
        images = [f for f in images if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

        random.shuffle(images)
        split_idx  = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        val_imgs   = images[split_idx:]

        for split_name, img_list in [('train', train_imgs), ('val', val_imgs)]:
            dest = out / split_name / cls_name
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in img_list:
                shutil.copy2(img_path, dest / img_path.name)

        print(f"  {cls_name:<55} train={len(train_imgs):>4}  val={len(val_imgs):>4}")
        total_train += len(train_imgs)
        total_val   += len(val_imgs)

    print(f"\n[DONE] Total — train: {total_train} | val: {total_val}")
    print(f"       Output: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',   required=True, help='Source folder with class subfolders')
    parser.add_argument('--out',   required=True, help='Output folder for train/val split')
    parser.add_argument('--split', type=float, default=0.8, help='Train ratio (default: 0.8)')
    parser.add_argument('--seed',  type=int,   default=42)
    args = parser.parse_args()

    split_dataset(args.src, args.out, train_ratio=args.split, seed=args.seed)
