"""
Train the Emotion Detection CNN on FER2013 dataset.
Author: Santosh Narreddy

Dataset: FER2013 (Facial Expression Recognition 2013)
  - 35,887 grayscale 48x48 images
  - 7 classes: Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised
  - Download from: https://www.kaggle.com/datasets/msambare/fer2013

Usage:
    python train_model.py --data_dir ./data/fer2013 --epochs 60 --batch 64
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Reproducibility ────────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
IMG_SIZE = 48
NUM_CLASSES = 7


# ── Model Architecture ─────────────────────────────────────────────────────────
# I designed this after reading a few papers — went through several iterations
# before settling on this structure. Batch norm helps a lot here.

def build_cnn(input_shape=(48, 48, 1), num_classes=7):
    """Custom CNN for emotion classification."""
    model = models.Sequential(name="EmotionCNN")

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.25))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Dropout(0.4))

    # Classifier head
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


# ── Data Generators ────────────────────────────────────────────────────────────

def get_generators(data_dir, batch_size=64):
    """
    Expects folder structure:
        data_dir/train/angry/  ...
        data_dir/test/angry/   ...
    """
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    val_data = val_gen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )
    return train_data, val_data


# ── Training ───────────────────────────────────────────────────────────────────

def train(data_dir, epochs=60, batch_size=64, lr=0.001, output_dir='model'):
    os.makedirs(output_dir, exist_ok=True)

    train_data, val_data = get_generators(data_dir, batch_size)
    model = build_cnn()
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'emotion_cnn.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
    ]

    print(f"\n[INFO] Training for up to {epochs} epochs | batch={batch_size} | lr={lr}")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=cb_list,
    )

    _plot_history(history, output_dir)
    print(f"\n[INFO] Best model saved to {output_dir}/emotion_cnn.h5")
    return history


def _plot_history(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150)
    print(f"[INFO] Training curves saved to {save_path}")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Path to FER2013 dataset folder')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output', default='model')
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        output_dir=args.output
    )
