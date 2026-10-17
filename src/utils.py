"""
Shared Utilities for NBA Predictor
===================================
Common functions used across multiple modules.
"""

import os

# Retain warnings/errors and respect an explicitly configured logging level.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import tensorflow as tf


def configure_gpu():
    """Configure TensorFlow to use GPU if available, with memory growth.
    
    Returns:
        bool: True if GPU is available and configured, False otherwise.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU ENABLED: {len(gpus)} GPU(s) detected")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("No GPU detected, using CPU")
        return False
