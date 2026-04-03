"""
prediction.py - Inference utilities for Weather Image Classification
"""

import numpy as np
import tensorflow as tf
from preprocessing import preprocess_single_image, preprocess_image_bytes, CLASSES
from model import MODEL_PATH

_model = None


def _load_model_safe():
    """Load the .h5 model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def get_model():
    """Lazy-load model singleton."""
    global _model
    if _model is None:
        _model = _load_model_safe()
    return _model


def reload_model():
    """Force reload model from disk (after retraining)."""
    global _model
    _model = _load_model_safe()
    return _model


def predict_from_path(image_path: str) -> dict:
    """Predict weather class from image file path."""
    model = get_model()
    img = preprocess_single_image(image_path)
    probs = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
    }


def predict_from_bytes(image_bytes: bytes) -> dict:
    """Predict weather class from raw image bytes."""
    model = get_model()
    img = preprocess_image_bytes(image_bytes)
    probs = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASSES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {cls: float(p) for cls, p in zip(CLASSES, probs)},
    }


def batch_predict(image_paths: list) -> list:
    """Predict on a list of image paths. Returns list of result dicts."""
    return [predict_from_path(p) for p in image_paths]
