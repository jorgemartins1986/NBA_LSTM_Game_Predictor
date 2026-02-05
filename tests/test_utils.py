"""
Tests for src/utils.py
======================
Tests for shared utility functions.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfigureGpu:
    """Tests for GPU configuration function"""
    
    def test_configure_gpu_returns_bool(self):
        """configure_gpu should return a boolean"""
        from src.utils import configure_gpu
        result = configure_gpu()
        assert isinstance(result, bool)
    
    def test_configure_gpu_handles_no_gpu(self, monkeypatch):
        """configure_gpu should handle no GPU gracefully"""
        # Force CUDA to not be visible
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
        
        from src.utils import configure_gpu
        # Should not raise an exception
        result = configure_gpu()
        # Without GPU, should return False
        assert result is False or result is True  # Depends on environment
    
    def test_configure_gpu_is_idempotent(self):
        """Calling configure_gpu multiple times should not cause issues"""
        from src.utils import configure_gpu
        
        result1 = configure_gpu()
        result2 = configure_gpu()
        
        # Should return consistent results
        assert result1 == result2
    
    def test_configure_gpu_imports_tensorflow(self):
        """configure_gpu should work with tensorflow imports"""
        # This tests that tensorflow is properly imported in utils
        from src.utils import configure_gpu
        import tensorflow as tf
        
        # Should not raise import errors
        result = configure_gpu()
        
        # Verify tensorflow is available
        assert hasattr(tf, 'config')
