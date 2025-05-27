#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'api'))

import tensorflow as tf
from pathlib import Path

# Import custom classes from process.py
from process import SpecAugment, GroupNorm, L2NormalizationLayer, l2_normalize_fn

MODEL_DIR = Path('models')
classifier_path = MODEL_DIR / "optimized_cnn_model.keras"

print("Testing model loading...")
print(f"Model path exists: {classifier_path.exists()}")

if classifier_path.exists():
    custom_objects = {
        "SpecAugment": SpecAugment,
        "GroupNorm": GroupNorm,
        "L2NormalizationLayer": L2NormalizationLayer,
        "l2_normalize_fn": l2_normalize_fn
    }
    
    try:
        model = tf.keras.models.load_model(classifier_path, custom_objects=custom_objects)
        print("✅ SUCCESS: Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        
        # Try without compilation
        print("Trying without compilation...")
        try:
            model = tf.keras.models.load_model(classifier_path, compile=False, custom_objects=custom_objects)
            print("✅ SUCCESS: Model loaded without compilation!")
        except Exception as e2:
            print(f"❌ ERROR (no compile): {e2}")
else:
    print("❌ Model file not found!")
