#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs

print("Testing basic imports...")
try:
    import tensorflow as tf
    print("✅ TensorFlow imported")
except Exception as e:
    print(f"❌ TensorFlow import failed: {e}")
    exit(1)

print("Testing custom layer creation...")
try:
    @tf.keras.utils.register_keras_serializable()
    class SpecAugment(tf.keras.layers.Layer):
        def __init__(self, max_f=25, max_t=40, **kwargs):
            super().__init__(**kwargs)
            self.max_f = max_f
            self.max_t = max_t
            
        def get_config(self):
            config = super().get_config()
            config.update({"max_f": self.max_f, "max_t": self.max_t})
            return config
            
        @classmethod
        def from_config(cls, config):
            return cls(**config)
    
    print("✅ SpecAugment layer defined")
    
    # Test creating an instance
    layer = SpecAugment()
    print("✅ SpecAugment instance created")
    
    # Test serialization
    config = layer.get_config()
    print("✅ SpecAugment config generated")
    
    # Test deserialization
    new_layer = SpecAugment.from_config(config)
    print("✅ SpecAugment deserialization works")
    
except Exception as e:
    print(f"❌ Custom layer test failed: {e}")
    exit(1)

print("All basic tests passed! The issue is likely with the actual model file.")
