"""
preprocessing.py - Data preprocessing for Weather Image Classification
Dataset: Multi-class Weather Dataset (Cloudy, Rain, Shine, Sunrise)
"""

import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASSES = ["Cloudy", "Rain", "Shine", "Sunrise"]
SEED = 42


def split_dataset(source_dir: str, train_dir: str, test_dir: str, test_ratio: float = 0.2):
    """Split raw dataset into train/test directories."""
    np.random.seed(SEED)
    for cls in CLASSES:
        src = Path(source_dir) / cls
        if not src.exists():
            print(f"[WARN] Class folder not found: {src}")
            continue
        images = list(src.glob("*.jpg")) + list(src.glob("*.png")) + list(src.glob("*.jpeg"))
        np.random.shuffle(images)
        split = int(len(images) * (1 - test_ratio))
        train_cls = Path(train_dir) / cls
        test_cls = Path(test_dir) / cls
        train_cls.mkdir(parents=True, exist_ok=True)
        test_cls.mkdir(parents=True, exist_ok=True)
        for img in images[:split]:
            shutil.copy(str(img), str(train_cls / img.name))
        for img in images[split:]:
            shutil.copy(str(img), str(test_cls / img.name))
        print(f"{cls}: {split} train, {len(images)-split} test")


def get_train_generator(train_dir: str):
    """Return augmented training data generator."""
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    return datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
    )


def get_test_generator(test_dir: str):
    """Return test data generator (no augmentation)."""
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    return datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )


def preprocess_single_image(image_path: str) -> np.ndarray:
    """Preprocess a single image for inference."""
    img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Preprocess image from raw bytes for inference."""
    import io
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def get_class_counts(directory: str) -> dict:
    """Return image count per class in a directory."""
    counts = {}
    for cls in CLASSES:
        cls_path = Path(directory) / cls
        if cls_path.exists():
            counts[cls] = len(list(cls_path.glob("*")))
        else:
            counts[cls] = 0
    return counts
