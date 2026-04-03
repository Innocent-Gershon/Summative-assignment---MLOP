"""
prediction.py - Inference utilities for Weather Image Classification
"""

import io
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from preprocessing import preprocess_single_image, preprocess_image_bytes, CLASSES
from model import MODEL_PATH

_model = None


class _CompatInputLayer(tf.keras.layers.InputLayer):
    """InputLayer that silently drops unknown kwargs from newer Keras saves."""
    def __init__(self, **kwargs):
        kwargs.pop("batch_shape", None)
        kwargs.pop("optional", None)
        # convert batch_input_shape if present
        super().__init__(**kwargs)


def _load_model_safe():
    """Load model with compatibility shim for Keras version mismatches."""
    custom_objects = {"InputLayer": _CompatInputLayer}
    try:
        return tf.keras.models.load_model(
            str(MODEL_PATH),
            custom_objects=custom_objects,
            compile=False,
        )
    except Exception:
        # Last resort: rebuild model and load weights only
        import h5py
        raise RuntimeError(
            "Model file is incompatible with the installed Keras version. "
            "Please re-export the model from the same environment."
        )


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
    """
    Predict weather class from image file path.
    Returns dict with predicted class, confidence, and all class probabilities.
    """
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
    """
    Predict weather class from raw image bytes.
    Returns dict with predicted class, confidence, and all class probabilities.
    """
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
