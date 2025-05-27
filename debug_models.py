#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app', 'api'))

from process import load_models, DEPENDENCIES, DEPENDENCIES_AVAILABLE, ML_READY

print("=== DEBUGGING MODEL LOADING ===")
print(f"Dependencies: {DEPENDENCIES}")
print(f"DEPENDENCIES_AVAILABLE: {DEPENDENCIES_AVAILABLE}")
print(f"ML_READY: {ML_READY}")

print("\n=== CALLING load_models() ===")
result = load_models()
classifier_model, encoder_model, triplet_encoder, index, metadata = result

print(f"\n=== RESULTS ===")
print(f"Classifier model loaded: {classifier_model is not None}")
print(f"Encoder model loaded: {encoder_model is not None}")
print(f"Triplet encoder loaded: {triplet_encoder is not None}")
print(f"Index loaded: {index is not None}")
print(f"Metadata loaded: {metadata is not None}")

if classifier_model is not None:
    print(f"Classifier model input shape: {classifier_model.input_shape}")
    print(f"Classifier model output shape: {classifier_model.output_shape}")
