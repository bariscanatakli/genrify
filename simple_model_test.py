#!/usr/bin/env python3
import os
import sys
import tensorflow as tf
from pathlib import Path

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define custom layers exactly as they would be in the model
@tf.keras.utils.register_keras_serializable()
class SpecAugment(tf.keras.layers.Layer):
    def __init__(self, max_f=25, max_t=40, **kwargs):
        super().__init__(**kwargs)
        self.max_f, self.max_t = max_f, max_t
        self.h, self.w = 128, 128
        
    def build(self, input_shape):
        super().build(input_shape)
        self.freq_axis = tf.reshape(tf.range(self.h), (self.h, 1, 1))
        self.time_axis = tf.reshape(tf.range(self.w), (1, self.w, 1))

    def call(self, x, training=None):
        if not training:
            return x
        f = tf.random.uniform([], 0, self.max_f + 1, tf.int32)
        f0 = tf.random.uniform([], 0, self.h - f + 1, tf.int32)
        t = tf.random.uniform([], 0, self.max_t + 1, tf.int32)
        t0 = tf.random.uniform([], 0, self.w - t + 1, tf.int32)
        mask_f = tf.cast(~tf.logical_and(self.freq_axis >= f0,
                                        self.freq_axis < f0 + f), x.dtype)
        mask_t = tf.cast(~tf.logical_and(self.time_axis >= t0,
                                        self.time_axis < t0 + t), x.dtype)
        return x * mask_f * mask_t
        
    def get_config(self):
        config = super().get_config()
        config.update({"max_f": self.max_f, "max_t": self.max_t})
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

@tf.keras.utils.register_keras_serializable()
class GroupNorm(tf.keras.layers.Layer):
    def __init__(self, groups=8, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups, self.eps = groups, epsilon

    def build(self, inp_shape):
        super().build(inp_shape)
        self.gamma = self.add_weight(name="gamma", shape=(inp_shape[-1],),
                                    initializer="ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=(inp_shape[-1],),
                                   initializer="zeros", trainable=True)

    def call(self, x):
        N, H, W, C = tf.shape(x)[0], *x.shape[1:]
        G = self.groups
        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)
        x = tf.reshape(x, [N, H, W, C])
        return x * self.gamma + self.beta
        
    def get_config(self):
        config = super().get_config()
        config.update({"groups": self.groups, "epsilon": self.eps})
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Try loading with the custom objects
model_path = Path('models/optimized_cnn_model.keras')
print(f"Model file exists: {model_path.exists()}")

if model_path.exists():
    custom_objects = {
        "SpecAugment": SpecAugment,
        "GroupNorm": GroupNorm,
    }
    
    try:
        print("Loading model with custom objects...")
        model = tf.keras.models.load_model(str(model_path), compile=False, custom_objects=custom_objects)
        print("✅ SUCCESS: Model loaded!")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
        
        # Try a prediction
        import numpy as np
        dummy_input = np.random.random((1, 128, 128, 1)).astype(np.float32)
        output = model.predict(dummy_input, verbose=0)
        print(f"Prediction output shape: {output.shape}")
        print("Model is working correctly!")
        
    except Exception as e:
        print(f"❌ Failed to load: {e}")
        print("\nTrying to load with safe mode...")
        
        try:
            # Enable unsafe deserialization
            import keras
            keras.config.enable_unsafe_deserialization()
            model = tf.keras.models.load_model(str(model_path), compile=False, custom_objects=custom_objects)
            print("✅ SUCCESS with unsafe deserialization!")
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
else:
    print("❌ Model file not found")
