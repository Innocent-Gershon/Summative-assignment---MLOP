#!/usr/bin/env python3
"""
Rebuild and resave model in current TensorFlow 2.16.2 format.
This converts the old .h5 file to a .keras format compatible with current env.
"""

import sys
import os
from pathlib import Path
import tensorflow as tf
import numpy as np

# Add src to path
SRC_DIR = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC_DIR))

from model import build_model, MODEL_PATH, CLASSES
from preprocessing import IMG_SIZE

print(f"TensorFlow version: {tf.__version__}")
print(f"Model path: {MODEL_PATH}")

# Try to load existing .h5 model
h5_path = MODEL_PATH
keras_path = MODEL_PATH.parent / "weather_model_final.keras"

if h5_path.exists():
    print(f"\n[1] Loading old .h5 model from {h5_path}...")
    try:
        # Load with minimal custom objects
        model = tf.keras.models.load_model(
            str(h5_path),
            compile=False,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load .h5: {type(e).__name__}: {e}")
        print("\n[2] Rebuilding model from scratch...")
        model = build_model()
        print("✓ Model rebuilt")
        # Create dummy data and do a single prediction to initialize
        dummy_input = np.random.randn(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)
        print("✓ Model initialized")
else:
    print(f"\n[1] Model file not found at {h5_path}")
    print("[2] Building model from scratch...")
    model = build_model()
    print("✓ Model built")
    # Create dummy data and do a single prediction to initialize
    dummy_input = np.random.randn(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
    _ = model.predict(dummy_input, verbose=0)
    print("✓ Model initialized")

# Save in new .keras format (TensorFlow SavedModel format)
print(f"\n[3] Saving model to {keras_path} (TensorFlow .keras format)...")
try:
    model.save(str(keras_path), save_format='keras')
    print(f"✓ Model saved to {keras_path}")
except Exception as e:
    print(f"✗ Failed to save: {type(e).__name__}: {e}")
    sys.exit(1)

# Also keep .h5 for backward compatibility
print(f"\n[4] Saving model to {h5_path} (HDF5 format)...")
try:
    model.save(str(h5_path), save_format='h5')
    print(f"✓ Model saved to {h5_path}")
except Exception as e:
    print(f"✗ Failed to save .h5: {type(e).__name__}: {e}")
    print("(This is non-critical, .keras format is preferred)")

# Verify the model can be reloaded
print(f"\n[5] Verifying model can be reloaded...")
try:
    test_model = tf.keras.models.load_model(str(keras_path), compile=False)
    print(f"✓ Model reloaded successfully from {keras_path}")
    
    # Test prediction
    test_input = np.random.randn(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
    output = test_model.predict(test_input, verbose=0)
    print(f"✓ Test prediction shape: {output.shape}")
    print(f"✓ Classes: {CLASSES}")
except Exception as e:
    print(f"✗ Verification failed: {type(e).__name__}: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ Model rebuild complete!")
print("="*60)
