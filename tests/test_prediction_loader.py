"""
Tests for Model Loader Module
=============================
Tests for src/prediction/loader.py
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, mock_open
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.prediction.loader import (
    LoadedEnsemble,
    ModelLoader
)


class TestLoadedEnsemble:
    """Tests for LoadedEnsemble dataclass"""
    
    def test_basic_creation(self):
        """Test basic creation"""
        models = [Mock(), Mock()]
        scalers = [StandardScaler(), StandardScaler()]
        
        ensemble = LoadedEnsemble(
            models=models,
            scalers=scalers,
            feature_cols=['feat1', 'feat2'],
            model_types=['xgboost', 'lstm']
        )
        
        assert ensemble.n_models == 2
        assert len(ensemble.feature_cols) == 2
    
    def test_get_model(self):
        """Test getting model by index"""
        models = [Mock(name='m1'), Mock(name='m2')]
        scalers = [Mock(name='s1'), Mock(name='s2')]
        
        ensemble = LoadedEnsemble(
            models=models,
            scalers=scalers,
            feature_cols=[],
            model_types=['xgboost', 'lstm']
        )
        
        model, scaler, mtype = ensemble.get_model(1)
        
        assert model is models[1]
        assert scaler is scalers[1]
        assert mtype == 'lstm'
    
    def test_optional_fields(self):
        """Test optional stacking fields"""
        ensemble = LoadedEnsemble(
            models=[Mock()],
            scalers=[StandardScaler()],
            feature_cols=['feat'],
            model_types=['xgboost'],
            meta_clf=Mock(),
            platt=Mock(),
            ensemble_weights=[0.5],
            ensemble_threshold=0.52
        )
        
        assert ensemble.meta_clf is not None
        assert ensemble.platt is not None
        assert ensemble.ensemble_weights == [0.5]
        assert ensemble.ensemble_threshold == 0.52


class TestModelLoader:
    """Tests for ModelLoader class"""
    
    @pytest.fixture
    def loader(self):
        """Create a loader with mocked path resolver"""
        mock_resolver = Mock(side_effect=lambda name: f"/models/{name}")
        return ModelLoader(path_resolver=mock_resolver)
    
    def test_init_with_custom_resolver(self):
        """Test initialization with custom path resolver"""
        custom_resolver = Mock(return_value="/custom/path")
        loader = ModelLoader(path_resolver=custom_resolver)
        
        assert loader._path_resolver is custom_resolver
    
    def test_init_with_custom_pickle_loader(self):
        """Test initialization with custom pickle loader"""
        custom_loader = Mock()
        loader = ModelLoader(pickle_loader=custom_loader)
        
        assert loader._pickle_loader is custom_loader
    
    @patch('builtins.open', mock_open(read_data=b''))
    def test_load_pickle(self, loader):
        """Test pickle loading"""
        expected_data = {'test': 'data'}
        loader._pickle_loader = Mock(return_value=expected_data)
        
        result = loader._load_pickle('/some/path.pkl')
        
        assert result == expected_data
    
    @patch('builtins.open', mock_open(read_data=b''))
    def test_load_model_types(self, loader):
        """Test loading model types"""
        expected_types = ['xgboost', 'keras']
        loader._pickle_loader = Mock(return_value=expected_types)
        
        # Directly test _load_pickle
        result = loader._load_pickle('/test/path.pkl')
        
        assert result == expected_types
    
    @patch('builtins.open', mock_open(read_data=b''))
    def test_load_scalers(self, loader):
        """Test loading scalers"""
        mock_scalers = [StandardScaler(), StandardScaler()]
        loader._pickle_loader = Mock(return_value=mock_scalers)
        
        # Directly test _load_pickle
        result = loader._load_pickle('/test/path.pkl')
        
        assert len(result) == 2
    
    @patch('builtins.open', mock_open(read_data=b''))
    def test_load_feature_cols(self, loader):
        """Test loading feature columns"""
        expected_cols = ['feat1', 'feat2', 'feat3']
        loader._pickle_loader = Mock(return_value=expected_cols)
        
        # Directly test _load_pickle
        result = loader._load_pickle('/test/path.pkl')
        
        assert result == expected_cols
    
    def test_load_xgboost_model(self, loader):
        """Test XGBoost model loading with mock"""
        with patch('xgboost.XGBClassifier') as mock_xgb:
            mock_model = Mock()
            mock_xgb.return_value = mock_model
            
            model = loader._load_xgboost_model('/models/xgb.json')
            
            mock_model.load_model.assert_called_once_with('/models/xgb.json')
    
    def test_load_keras_model_structure(self, loader):
        """Test Keras model loading creates proper structure"""
        # This test verifies the method exists and has correct interface
        assert hasattr(loader, '_load_keras_model')
        assert callable(loader._load_keras_model)
    
    def test_load_ensemble_handles_missing_models(self, loader):
        """Test that load_ensemble handles missing models gracefully"""
        # Setup mocks
        with patch.object(loader, 'load_model_types', return_value=['xgboost', 'lstm']):
            with patch.object(loader, '_load_xgboost_model', return_value=Mock()):
                with patch.object(loader, '_load_keras_model', side_effect=Exception("Model not found")):
                    with patch.object(loader, 'load_scalers', return_value=[Mock(), Mock()]):
                        with patch.object(loader, 'load_feature_cols', return_value=['feat1']):
                            with patch.object(loader, '_load_pickle', return_value=None):
                                ensemble = loader.load_ensemble(verbose=False)
        
        # Should have loaded 1 model (XGBoost), skipped Keras
        assert ensemble.n_models == 1
        assert ensemble.model_types == ['xgboost']
    
    def test_load_ensemble_loads_all_artifacts(self, loader):
        """Test complete ensemble loading"""
        mock_model = Mock()
        mock_scaler = StandardScaler()
        mock_meta = Mock()
        mock_platt = Mock()
        
        with patch.object(loader, 'load_model_types', return_value=['xgboost']):
            with patch.object(loader, '_load_xgboost_model', return_value=mock_model):
                with patch.object(loader, 'load_scalers', return_value=[mock_scaler]):
                    with patch.object(loader, 'load_feature_cols', return_value=['feat1', 'feat2']):
                        with patch.object(loader, '_load_pickle') as mock_pickle:
                            # Return different values for different calls
                            mock_pickle.side_effect = [
                                mock_meta,  # META_LR
                                mock_platt,  # PLATT
                                {'weights': [0.5], 'threshold': 0.52}  # WEIGHTS
                            ]
                            ensemble = loader.load_ensemble(verbose=False)
        
        assert ensemble.n_models == 1
        assert ensemble.meta_clf is mock_meta
        assert ensemble.platt is mock_platt
        assert ensemble.ensemble_weights == [0.5]
        assert ensemble.ensemble_threshold == 0.52
    
    def test_load_ensemble_without_optional_artifacts(self, loader):
        """Test loading when optional artifacts are missing"""
        mock_model = Mock()
        mock_scaler = StandardScaler()
        
        def pickle_side_effect(path):
            raise FileNotFoundError()
        
        with patch.object(loader, 'load_model_types', return_value=['xgboost']):
            with patch.object(loader, '_load_xgboost_model', return_value=mock_model):
                with patch.object(loader, 'load_scalers', return_value=[mock_scaler]):
                    with patch.object(loader, 'load_feature_cols', return_value=['feat']):
                        with patch.object(loader, '_load_pickle', side_effect=pickle_side_effect):
                            ensemble = loader.load_ensemble(verbose=False)
        
        # Should still work without optional artifacts
        assert ensemble.n_models == 1
        assert ensemble.meta_clf is None
        assert ensemble.platt is None


class TestModelLoaderVerbose:
    """Tests for verbose output in loader"""
    
    @pytest.fixture
    def loader(self):
        """Create a loader with mocked path resolver"""
        mock_resolver = Mock(side_effect=lambda name: f"/models/{name}")
        return ModelLoader(path_resolver=mock_resolver)
    
    def test_load_ensemble_verbose_prints(self, loader, capsys):
        """Test that verbose=True prints loading messages"""
        mock_model = Mock()
        mock_scaler = Mock()
        
        with patch.object(loader, 'load_model_types', return_value=['random_forest']):
            with patch.object(loader, '_load_pickle', return_value=mock_model):
                with patch.object(loader, 'load_scalers', return_value=[mock_scaler]):
                    with patch.object(loader, 'load_feature_cols', return_value=['feat1']):
                        ensemble = loader.load_ensemble(verbose=True)
        
        captured = capsys.readouterr()
        assert "Loading ensemble models" in captured.out
    
    def test_load_ensemble_verbose_false_no_prints(self, loader, capsys):
        """Test that verbose=False suppresses output"""
        mock_model = Mock()
        mock_scaler = Mock()
        
        with patch.object(loader, 'load_model_types', return_value=['logistic']):
            with patch.object(loader, '_load_pickle', return_value=mock_model):
                with patch.object(loader, 'load_scalers', return_value=[mock_scaler]):
                    with patch.object(loader, 'load_feature_cols', return_value=['feat1']):
                        ensemble = loader.load_ensemble(verbose=False)
        
        captured = capsys.readouterr()
        assert captured.out == ""
    
    def test_load_ensemble_rf_model(self, loader):
        """Test loading Random Forest model"""
        mock_model = Mock()
        mock_scaler = Mock()
        
        with patch.object(loader, 'load_model_types', return_value=['random_forest']):
            with patch.object(loader, '_load_pickle', return_value=mock_model):
                with patch.object(loader, 'load_scalers', return_value=[mock_scaler]):
                    with patch.object(loader, 'load_feature_cols', return_value=['f1']):
                        ensemble = loader.load_ensemble(verbose=False)
        
        assert ensemble.n_models == 1
        assert ensemble.model_types == ['random_forest']
    
    def test_load_ensemble_logistic_model(self, loader):
        """Test loading Logistic model"""
        mock_model = Mock()
        mock_scaler = Mock()
        
        with patch.object(loader, 'load_model_types', return_value=['logistic']):
            with patch.object(loader, '_load_pickle', return_value=mock_model):
                with patch.object(loader, 'load_scalers', return_value=[mock_scaler]):
                    with patch.object(loader, 'load_feature_cols', return_value=['f1']):
                        ensemble = loader.load_ensemble(verbose=False)
        
        assert ensemble.n_models == 1
        assert ensemble.model_types == ['logistic']
    
    def test_path_resolver_lazy_load(self):
        """Test that path resolver is lazily loaded"""
        loader = ModelLoader(path_resolver=None)
        # Accessing path_resolver should trigger lazy load
        resolver = loader.path_resolver
        assert resolver is not None