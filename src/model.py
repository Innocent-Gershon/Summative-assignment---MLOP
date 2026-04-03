"""
model.py - Model creation and training for Weather Image Classification
Architecture: VGG16 transfer learning (fine-tuned)
"""

import os
import json
import numpy as np
import tensorflow as tf
try:
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
except Exception as e:
    raise ImportError(
        "Could not import TensorFlow Keras layers/callbacks. "
        "Please install `tensorflow-macos` or `tensorflow` and ensure versions are compatible. "
        f"Original error: {e}"
    )
from pathlib import Path

from preprocessing import get_train_generator, get_test_generator, CLASSES, IMG_SIZE

MODEL_PATH = Path(__file__).parent.parent / "models" / "weather_model_final.h5"
CLASS_NAMES_PATH = Path(__file__).parent.parent / "models" / "class_names.json"
EPOCHS = 25
NUM_CLASSES = len(CLASSES)


def build_model() -> Model:
    """Build VGG16-based transfer learning model."""
    base = VGG16(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    # Freeze base layers
    for layer in base.layers:
        layer.trainable = False
    # Unfreeze last 4 conv layers for fine-tuning
    for layer in base.layers[-4:]:
        layer.trainable = True

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(train_dir: str, test_dir: str, epochs: int = EPOCHS) -> dict:
    """Train the model and save weights. Returns training history."""
    train_gen = get_train_generator(train_dir)
    val_gen = get_test_generator(test_dir)

    model = build_model()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
    )

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(CLASSES, f)

    print(f"Model saved to {MODEL_PATH}")
    return hist_dict


def retrain_model(train_dir: str, test_dir: str, epochs: int = 10) -> dict:
    """Retrain existing model on new data (fine-tune all layers)."""
    if not MODEL_PATH.exists():
        print("No existing model found. Training from scratch.")
        return train_model(train_dir, test_dir, epochs)

    model = tf.keras.models.load_model(str(MODEL_PATH))
    # Unfreeze all layers for retraining
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    train_gen = get_train_generator(train_dir)
    val_gen = get_test_generator(test_dir)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
    )

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(CLASSES, f)

    print(f"Retrained model saved to {MODEL_PATH}")
    return hist_dict


def load_model() -> tf.keras.Model:
    """Load saved model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")
    try:
        return tf.keras.models.load_model(str(MODEL_PATH))
    except Exception:
        return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def get_class_names() -> list:
    """Load saved class names from disk."""
    if not CLASS_NAMES_PATH.exists():
        return CLASSES
    with open(CLASS_NAMES_PATH) as f:
        return json.load(f)


def get_training_history() -> dict:
    """Return empty dict — history not persisted separately."""
    return {}
