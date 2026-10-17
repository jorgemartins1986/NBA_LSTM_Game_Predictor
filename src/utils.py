"""
Shared Utilities for NBA Predictor
===================================
Common functions used across multiple modules.
"""

import os
import logging
import warnings

# Retain warnings/errors and respect an explicitly configured logging level.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

# Keras' optional ONNX-export shim probes a removed NumPy alias. It does not
# affect loading or predicting with our Keras model. Keep unrelated warnings.
warnings.filterwarnings(
    'ignore', category=FutureWarning,
    message=r'In the future `np\.object` will be defined as the corresponding NumPy scalar\.',
    module=r'keras\.src\.export\.tf2onnx_lib',
)


class _KerasPlaceholderNotice(logging.Filter):
    def filter(self, record):
        return not (record.levelno == logging.WARNING and
                    'The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.'
                    in record.getMessage())


logging.getLogger('tensorflow').addFilter(_KerasPlaceholderNotice())

import tensorflow as tf


def configure_gpu(verbose=True):
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
            if verbose:
                print(f"GPU ENABLED: {len(gpus)} GPU(s) detected")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu.name}")
            return True
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        if verbose:
            if not tf.test.is_built_with_cuda():
                print('TensorFlow is using CPU (this build has no CUDA support).')
            else:
                print('TensorFlow cannot access a supported GPU; using CPU.')
        return False
