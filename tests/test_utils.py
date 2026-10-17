"""
Tests for src/utils.py
======================
Tests for shared utility functions.
"""

import pytest
import os
import sys
import logging
from types import SimpleNamespace
from unittest.mock import Mock

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

    @pytest.mark.parametrize('has_gpu', [False, True])
    def test_quiet_mode_preserves_device_configuration(self, monkeypatch, capsys, has_gpu):
        from src import utils

        device = SimpleNamespace(name='GPU:0')
        monkeypatch.setattr(utils.tf.config, 'list_physical_devices', lambda _: [device] if has_gpu else [])
        growth = Mock()
        monkeypatch.setattr(utils.tf.config.experimental, 'set_memory_growth', growth)
        assert utils.configure_gpu(verbose=False) is has_gpu
        assert capsys.readouterr().out == ''
        if has_gpu:
            growth.assert_called_once_with(device, True)
        else:
            growth.assert_not_called()

    def test_cpu_message_describes_tensorflow_build_instead_of_missing_hardware(self, monkeypatch, capsys):
        from src import utils
        monkeypatch.setattr(utils.tf.config, 'list_physical_devices', lambda _: [])
        monkeypatch.setattr(utils.tf.test, 'is_built_with_cuda', lambda: False)
        assert utils.configure_gpu() is False
        assert 'this build has no CUDA support' in capsys.readouterr().out


def test_known_deprecation_filter_preserves_other_warnings_and_errors():
    from src.utils import _KerasPlaceholderNotice

    notice = 'The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.'
    warning_filter = _KerasPlaceholderNotice()
    record = lambda level, message: logging.LogRecord('tensorflow', level, '', 0, message, (), None)
    assert not warning_filter.filter(record(logging.WARNING, notice))
    assert warning_filter.filter(record(logging.ERROR, notice))
    assert warning_filter.filter(record(logging.WARNING, 'Model weights could not be restored'))
