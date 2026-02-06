"""
Tests for Model Trainers Module
===============================
Tests for src/training/trainers.py

Note: These tests mock actual model training to keep tests fast.
Integration tests with real training are in test_training_integration.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from sklearn.preprocessing import StandardScaler

from src.training.trainers import (
    TrainerConfig,
    XGBoostConfig,
    RandomForestConfig,
    LogisticConfig,
    LSTMConfig,
    TrainingResult,
    XGBoostTrainer,
    RandomForestTrainer,
    LogisticTrainer,
    LSTMTrainer,
    TrainerFactory
)
from src.training.data_prep import ScaledData


class TestTrainerConfigs:
    """Tests for configuration dataclasses"""
    
    def test_base_config_defaults(self):
        """Test TrainerConfig default values"""
        config = TrainerConfig()
        assert config.random_state == 42
        assert config.verbose == True
    
    def test_xgboost_config_defaults(self):
        """Test XGBoostConfig optimized defaults"""
        config = XGBoostConfig()
        assert config.n_estimators == 1253
        assert config.max_depth == 2
        assert config.learning_rate == 0.015
        assert config.early_stopping_rounds == 50
    
    def test_rf_config_defaults(self):
        """Test RandomForestConfig defaults"""
        config = RandomForestConfig()
        assert config.n_estimators == 500
        assert config.max_depth is None
        assert config.n_top_features == 40
    
    def test_logistic_config_defaults(self):
        """Test LogisticConfig defaults"""
        config = LogisticConfig()
        assert config.C == 0.5
        assert config.penalty == 'l2'
        assert config.class_weight == 'balanced'
    
    def test_lstm_config_defaults(self):
        """Test LSTMConfig defaults"""
        config = LSTMConfig()
        assert config.architecture == 'lstm'
        assert config.epochs == 200
        assert config.batch_size == 64
    
    def test_config_custom_values(self):
        """Test config with custom values"""
        config = XGBoostConfig(
            n_estimators=100,
            max_depth=5,
            random_state=123
        )
        assert config.n_estimators == 100
        assert config.max_depth == 5
        assert config.random_state == 123


class TestTrainingResult:
    """Tests for TrainingResult dataclass"""
    
    def test_training_result_creation(self):
        """Test TrainingResult creation"""
        model = Mock()
        scaler = StandardScaler()
        
        result = TrainingResult(
            model=model,
            scaler=scaler,
            accuracy=0.65,
            y_pred=np.array([0, 1, 1]),
            y_pred_prob=np.array([0.3, 0.7, 0.8]),
            y_test=np.array([0, 1, 0]),
            model_type='xgboost'
        )
        
        assert result.accuracy == 0.65
        assert result.model_type == 'xgboost'
        assert result.model is model
    
    def test_training_result_with_feature_indices(self):
        """Test TrainingResult with feature selection indices"""
        result = TrainingResult(
            model=Mock(),
            scaler=StandardScaler(),
            accuracy=0.60,
            y_pred=np.array([0, 1]),
            y_pred_prob=np.array([0.4, 0.6]),
            y_test=np.array([0, 1]),
            model_type='random_forest',
            feature_indices=np.array([0, 5, 10, 15])
        )
        
        assert result.feature_indices is not None
        assert len(result.feature_indices) == 4


@pytest.fixture
def mock_scaled_data():
    """Create mock scaled data for testing"""
    np.random.seed(42)
    return ScaledData(
        X_train_scaled=np.random.randn(100, 50),
        X_test_scaled=np.random.randn(20, 50),
        y_train=np.random.randint(0, 2, 100),
        y_test=np.random.randint(0, 2, 20),
        scaler=StandardScaler()
    )


class TestXGBoostTrainer:
    """Tests for XGBoostTrainer"""
    
    def test_model_type_property(self):
        """Test model type is correct"""
        trainer = XGBoostTrainer()
        assert trainer.model_type == 'xgboost'
    
    def test_create_model(self):
        """Test model creation with config"""
        config = XGBoostConfig(n_estimators=10, max_depth=3)
        trainer = XGBoostTrainer(config)
        
        model = trainer.create_model()
        
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict_proba')
    
    def test_train_creates_result(self, mock_scaled_data):
        """Test that train returns a proper TrainingResult"""
        # Use minimal config for speed
        config = XGBoostConfig(n_estimators=5, max_depth=2, early_stopping_rounds=2, verbose=False)
        trainer = XGBoostTrainer(config)
        
        result = trainer.train(mock_scaled_data)
        
        assert isinstance(result, TrainingResult)
        assert result.model_type == 'xgboost'
        assert 0 <= result.accuracy <= 1
    
    def test_predict_proba(self):
        """Test predict_proba interface"""
        trainer = XGBoostTrainer()
        
        model = Mock()
        model.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
        
        probs = trainer.predict_proba(model, np.random.randn(2, 10))
        
        assert len(probs) == 2
        assert probs[0] == 0.7  # Second column (positive class)


class TestRandomForestTrainer:
    """Tests for RandomForestTrainer"""
    
    def test_model_type_property(self):
        """Test model type is correct"""
        trainer = RandomForestTrainer()
        assert trainer.model_type == 'random_forest'
    
    def test_select_features_returns_indices(self, mock_scaled_data):
        """Test feature selection returns correct number of indices"""
        trainer = RandomForestTrainer(RandomForestConfig(n_top_features=10))
        
        _, top_indices = trainer.select_features(
            mock_scaled_data.X_train_scaled,
            mock_scaled_data.y_train
        )
        
        assert len(top_indices) == 10
    
    def test_predict_proba_with_feature_selection(self):
        """Test predict_proba applies feature selection"""
        trainer = RandomForestTrainer()
        
        model = Mock()
        model._top_feature_indices = np.array([0, 1, 2])
        model.predict_proba.return_value = np.array([[0.4, 0.6]])
        
        X = np.random.randn(1, 10)
        probs = trainer.predict_proba(model, X)
        
        # Should have selected only 3 features
        model.predict_proba.assert_called_once()
        called_X = model.predict_proba.call_args[0][0]
        assert called_X.shape[1] == 3


class TestLogisticTrainer:
    """Tests for LogisticTrainer"""
    
    def test_model_type_property(self):
        """Test model type is correct"""
        trainer = LogisticTrainer()
        assert trainer.model_type == 'logistic'
    
    def test_create_model(self):
        """Test model creation"""
        config = LogisticConfig(C=1.0, max_iter=500)
        trainer = LogisticTrainer(config)
        
        model = trainer.create_model()
        
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict_proba')
    
    @patch('sklearn.linear_model.LogisticRegression')
    def test_train(self, mock_lr_class, mock_scaled_data):
        """Test training logistic regression"""
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.random.rand(20, 2)
        mock_lr_class.return_value = mock_model
        
        trainer = LogisticTrainer()
        result = trainer.train(mock_scaled_data)
        
        mock_model.fit.assert_called_once()
        assert result.model_type == 'logistic'


class TestLSTMTrainer:
    """Tests for LSTMTrainer"""
    
    def test_model_type_property(self):
        """Test model type is correct"""
        trainer = LSTMTrainer()
        assert trainer.model_type == 'keras'
    
    def test_init_with_model_builder(self):
        """Test initialization with custom model builder"""
        mock_builder = Mock()
        trainer = LSTMTrainer(model_builder=mock_builder)
        
        assert trainer._model_builder is mock_builder
    
    def test_create_callbacks(self):
        """Test callback creation"""
        trainer = LSTMTrainer(LSTMConfig(
            early_stopping_patience=10,
            reduce_lr_patience=5
        ))
        
        callbacks = trainer.create_callbacks()
        
        assert len(callbacks) == 2
        # Check callback types by name
        callback_names = [type(cb).__name__ for cb in callbacks]
        assert 'EarlyStopping' in callback_names
        assert 'ReduceLROnPlateau' in callback_names
    
    def test_predict_proba(self):
        """Test predict_proba for keras models"""
        trainer = LSTMTrainer()
        
        model = Mock()
        model.predict.return_value = np.array([[0.7], [0.3]])
        
        probs = trainer.predict_proba(model, np.random.randn(2, 10))
        
        assert len(probs) == 2
        assert probs[0] == 0.7
        model.predict.assert_called_once()


class TestTrainerFactory:
    """Tests for TrainerFactory"""
    
    def test_create_xgboost(self):
        """Test creating XGBoost trainer"""
        trainer = TrainerFactory.create('xgboost')
        assert isinstance(trainer, XGBoostTrainer)
    
    def test_create_random_forest(self):
        """Test creating Random Forest trainer"""
        trainer = TrainerFactory.create('random_forest')
        assert isinstance(trainer, RandomForestTrainer)
    
    def test_create_logistic(self):
        """Test creating Logistic trainer"""
        trainer = TrainerFactory.create('logistic')
        assert isinstance(trainer, LogisticTrainer)
    
    def test_create_lstm(self):
        """Test creating LSTM trainer"""
        trainer = TrainerFactory.create('lstm')
        assert isinstance(trainer, LSTMTrainer)
    
    def test_create_deep(self):
        """Test creating deep MLP trainer"""
        trainer = TrainerFactory.create('deep')
        assert isinstance(trainer, LSTMTrainer)
        assert trainer.config.architecture == 'deep'
    
    def test_create_with_config(self):
        """Test creating trainer with custom config"""
        config = XGBoostConfig(n_estimators=50)
        trainer = TrainerFactory.create('xgboost', config)
        
        assert trainer.config.n_estimators == 50
    
    def test_create_unknown_raises(self):
        """Test that unknown architecture raises error"""
        with pytest.raises(ValueError, match="Unknown architecture"):
            TrainerFactory.create('unknown_model')
    
    def test_available_architectures(self):
        """Test listing available architectures"""
        archs = TrainerFactory.available_architectures()
        
        assert 'xgboost' in archs
        assert 'random_forest' in archs
        assert 'logistic' in archs
        assert 'lstm' in archs
        assert 'deep' in archs
    
    def test_register_new_trainer(self):
        """Test registering a custom trainer"""
        class CustomTrainer:
            pass
        
        TrainerFactory.register('custom', CustomTrainer)
        
        assert 'custom' in TrainerFactory.available_architectures()
        
        # Cleanup
        del TrainerFactory._trainers['custom']
