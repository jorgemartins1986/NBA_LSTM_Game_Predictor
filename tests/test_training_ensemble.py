"""
Tests for src/training/ensemble.py - Ensemble Training Coordinator
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, Mock, patch
from dataclasses import dataclass, field

from src.training.ensemble import (
    EnsembleConfig, EnsembleResult, EnsembleTrainer, train_ensemble_models
)
from src.training.trainers import TrainingResult, TrainerConfig
from src.training.data_prep import TrainTestData


@pytest.fixture
def sample_matchup_df():
    """Create sample matchup dataframe for testing"""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'HOME_WIN': np.random.randint(0, 2, n),
        'HOME_ELO': np.random.uniform(1400, 1600, n),
        'AWAY_ELO': np.random.uniform(1400, 1600, n),
        'HOME_WIN_PCT': np.random.uniform(0.3, 0.7, n),
        'AWAY_WIN_PCT': np.random.uniform(0.3, 0.7, n),
        'HOME_OFF_RTG': np.random.uniform(105, 115, n),
        'AWAY_OFF_RTG': np.random.uniform(105, 115, n),
        'HOME_DEF_RTG': np.random.uniform(105, 115, n),
        'AWAY_DEF_RTG': np.random.uniform(105, 115, n),
        'HOME_NET_RTG': np.random.uniform(-5, 5, n),
        'AWAY_NET_RTG': np.random.uniform(-5, 5, n),
    })


@pytest.fixture
def sample_train_test_data():
    """Create sample train/test data"""
    np.random.seed(42)
    return TrainTestData(
        X_train=np.random.randn(80, 10),
        X_test=np.random.randn(20, 10),
        y_train=np.random.randint(0, 2, 80),
        y_test=np.random.randint(0, 2, 20),
        feature_cols=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']
    )


@pytest.fixture
def mock_training_result():
    """Create a mock training result"""
    return TrainingResult(
        model=MagicMock(),
        scaler=MagicMock(),
        accuracy=0.65,
        model_type='xgboost',
        y_pred=np.array([0, 1, 0, 1]),
        y_pred_prob=np.array([0.4, 0.6, 0.3, 0.7]),
        y_test=np.array([0, 1, 0, 1])
    )


class TestEnsembleConfig:
    """Tests for EnsembleConfig dataclass"""
    
    def test_default_architectures(self):
        config = EnsembleConfig()
        assert config.architectures == ['xgboost', 'random_forest', 'logistic', 'lstm']
    
    def test_default_test_size(self):
        config = EnsembleConfig()
        assert config.test_size == 0.2
    
    def test_custom_architectures(self):
        config = EnsembleConfig(architectures=['xgboost', 'logistic'])
        assert len(config.architectures) == 2
    
    def test_custom_settings(self):
        config = EnsembleConfig(
            test_size=0.3,
            random_state=123,
            use_sample_weights=False,
            verbose=False
        )
        assert config.test_size == 0.3
        assert config.random_state == 123
        assert config.use_sample_weights is False
        assert config.verbose is False


class TestEnsembleResult:
    """Tests for EnsembleResult dataclass"""
    
    def test_n_models_property(self):
        result = EnsembleResult(
            models=[MagicMock(), MagicMock(), MagicMock()],
            scalers=[MagicMock(), MagicMock(), MagicMock()],
            feature_cols=['f1', 'f2'],
            model_types=['xgboost', 'rf', 'logistic'],
            training_results=[]
        )
        assert result.n_models == 3
    
    def test_best_model_index_with_accuracies(self):
        result = EnsembleResult(
            models=[MagicMock(), MagicMock(), MagicMock()],
            scalers=[MagicMock(), MagicMock(), MagicMock()],
            feature_cols=['f1', 'f2'],
            model_types=['xgboost', 'rf', 'logistic'],
            training_results=[],
            individual_accuracies=[0.6, 0.75, 0.65]
        )
        assert result.best_model_index == 1  # RF has highest accuracy
    
    def test_best_model_index_without_accuracies(self):
        result = EnsembleResult(
            models=[MagicMock(), MagicMock()],
            scalers=[MagicMock(), MagicMock()],
            feature_cols=['f1'],
            model_types=['xgboost', 'rf'],
            training_results=[]
        )
        assert result.best_model_index == 0  # Default when no accuracies
    
    def test_average_accuracy_property(self):
        mock_results = [
            TrainingResult(
                model=MagicMock(), scaler=MagicMock(), accuracy=0.6, model_type='a',
                y_pred=np.array([0, 1]), y_pred_prob=np.array([0.4, 0.6]), y_test=np.array([0, 1])
            ),
            TrainingResult(
                model=MagicMock(), scaler=MagicMock(), accuracy=0.7, model_type='b',
                y_pred=np.array([0, 1]), y_pred_prob=np.array([0.4, 0.6]), y_test=np.array([0, 1])
            ),
            TrainingResult(
                model=MagicMock(), scaler=MagicMock(), accuracy=0.8, model_type='c',
                y_pred=np.array([0, 1]), y_pred_prob=np.array([0.4, 0.6]), y_test=np.array([0, 1])
            ),
        ]
        result = EnsembleResult(
            models=[MagicMock()] * 3,
            scalers=[MagicMock()] * 3,
            feature_cols=['f1'],
            model_types=['a', 'b', 'c'],
            training_results=mock_results
        )
        # Average of 0.6, 0.7, 0.8 is 0.7
        assert abs(result.average_accuracy - 0.7) < 0.001


class TestEnsembleTrainer:
    """Tests for EnsembleTrainer class"""
    
    def test_init_default(self):
        trainer = EnsembleTrainer()
        assert trainer.config is not None
        assert trainer.data_prep is not None
        assert trainer.evaluator is not None
    
    def test_init_with_config(self):
        config = EnsembleConfig(architectures=['xgboost'], verbose=False)
        trainer = EnsembleTrainer(config=config)
        assert trainer.config.architectures == ['xgboost']
        assert trainer.config.verbose is False
    
    def test_init_with_custom_print_fn(self):
        captured = []
        trainer = EnsembleTrainer(
            config=EnsembleConfig(verbose=True),
            print_fn=lambda x: captured.append(x)
        )
        trainer._log("Test message")
        assert "Test message" in captured
    
    def test_log_when_verbose(self):
        captured = []
        trainer = EnsembleTrainer(
            config=EnsembleConfig(verbose=True),
            print_fn=lambda x: captured.append(x)
        )
        trainer._log("Hello")
        assert len(captured) == 1
    
    def test_log_when_not_verbose(self):
        captured = []
        trainer = EnsembleTrainer(
            config=EnsembleConfig(verbose=False),
            print_fn=lambda x: captured.append(x)
        )
        trainer._log("Hello")
        assert len(captured) == 0
    
    def test_compute_sample_weights_enabled(self):
        trainer = EnsembleTrainer(
            config=EnsembleConfig(use_sample_weights=True, weight_decay=1.2)
        )
        weights = trainer.compute_sample_weights(100)
        assert weights is not None
        assert len(weights) == 100
        assert weights[-1] > weights[0]  # Recent samples should have higher weight
    
    def test_compute_sample_weights_disabled(self):
        trainer = EnsembleTrainer(
            config=EnsembleConfig(use_sample_weights=False)
        )
        weights = trainer.compute_sample_weights(100)
        assert weights is None
    
    def test_get_trainer_config_xgboost(self):
        trainer = EnsembleTrainer()
        config = trainer._get_trainer_config('xgboost', 0)
        assert config.random_state == 42  # Default random_state
    
    def test_get_trainer_config_random_forest(self):
        trainer = EnsembleTrainer()
        config = trainer._get_trainer_config('random_forest', 1)
        assert config.random_state == 43  # 42 + 1
    
    def test_get_trainer_config_logistic(self):
        trainer = EnsembleTrainer()
        config = trainer._get_trainer_config('logistic', 2)
        assert config.random_state == 44
    
    def test_get_trainer_config_lstm(self):
        trainer = EnsembleTrainer()
        config = trainer._get_trainer_config('lstm', 0)
        assert hasattr(config, 'architecture')
    
    def test_get_trainer_config_unknown(self):
        trainer = EnsembleTrainer()
        config = trainer._get_trainer_config('unknown_type', 0)
        assert config.random_state == 42
    
    def test_prepare_data(self, sample_matchup_df):
        captured = []
        trainer = EnsembleTrainer(
            config=EnsembleConfig(verbose=True),
            print_fn=lambda x: captured.append(x)
        )
        data, feature_cols = trainer.prepare_data(sample_matchup_df)
        
        assert data is not None
        assert len(feature_cols) > 0
        assert data.n_train_samples > 0
        assert data.n_test_samples > 0
        assert any("Preparing data" in msg for msg in captured)


class TestEnsembleTrainerIntegration:
    """Integration tests for ensemble training"""
    
    def test_train_single_model_with_mock_factory(self, sample_train_test_data, mock_training_result):
        """Test training a single model with mocked factory"""
        # Create mock factory
        mock_factory = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_training_result
        mock_factory.create.return_value = mock_trainer
        
        # Create mock data prep
        mock_data_prep = MagicMock()
        mock_scaled = MagicMock()
        mock_scaled.X_train_scaled = np.zeros((80, 10))
        mock_scaled.X_test_scaled = np.zeros((20, 10))
        mock_data_prep.scale_features.return_value = mock_scaled
        
        trainer = EnsembleTrainer(
            config=EnsembleConfig(verbose=False),
            trainer_factory=mock_factory,
            data_prep=mock_data_prep
        )
        
        result = trainer.train_single_model('xgboost', sample_train_test_data, 0)
        
        assert result.model_type == 'xgboost'
        assert result.accuracy == 0.65
        mock_factory.create.assert_called_once()
    
    def test_train_full_ensemble_with_mock_factory(self, sample_matchup_df):
        """Test full ensemble training with mocked components"""
        # Mock data prep
        mock_data_prep = MagicMock()
        mock_train_test = MagicMock()
        mock_train_test.n_train_samples = 80
        mock_train_test.n_test_samples = 20
        mock_train_test.y_train = np.zeros(80)
        mock_train_test.X_test = np.zeros((20, 10))
        mock_data_prep.prepare_for_training.return_value = (mock_train_test, ['f1', 'f2'])
        
        mock_scaled = MagicMock()
        mock_scaled.X_train_scaled = np.zeros((80, 10))
        mock_scaled.X_test_scaled = np.zeros((20, 10))
        mock_data_prep.scale_features.return_value = mock_scaled
        
        # Mock trainer factory
        mock_factory = MagicMock()
        mock_trainer = MagicMock()
        mock_training_result = TrainingResult(
            model=MagicMock(),
            scaler=MagicMock(),
            accuracy=0.65,
            model_type='test',
            y_pred=np.array([0, 1]),
            y_pred_prob=np.array([0.4, 0.6]),
            y_test=np.array([0, 1])
        )
        mock_trainer.train.return_value = mock_training_result
        mock_factory.create.return_value = mock_trainer
        
        trainer = EnsembleTrainer(
            config=EnsembleConfig(architectures=['xgboost', 'logistic'], verbose=False),
            trainer_factory=mock_factory,
            data_prep=mock_data_prep
        )
        
        result = trainer.train(sample_matchup_df)
        
        assert result.n_models == 2
        assert len(result.training_results) == 2
        assert result.individual_accuracies is not None


class TestBackwardCompatibleWrapper:
    """Tests for train_ensemble_models() backward-compatible wrapper"""
    
    def test_wrapper_returns_tuple(self, sample_matchup_df):
        """Test that wrapper returns the expected tuple format"""
        with patch('src.training.ensemble.EnsembleTrainer') as MockTrainer:
            # Mock the result
            mock_result = MagicMock()
            mock_result.models = [MagicMock(), MagicMock()]
            mock_result.scalers = [MagicMock(), MagicMock()]
            mock_result.feature_cols = ['f1', 'f2']
            mock_result.model_types = ['xgboost', 'rf']
            
            MockTrainer.return_value.train.return_value = mock_result
            
            models, scalers, feature_cols, model_types = train_ensemble_models(
                sample_matchup_df, n_models=2
            )
            
            assert len(models) == 2
            assert len(scalers) == 2
            assert feature_cols == ['f1', 'f2']
            assert model_types == ['xgboost', 'rf']
    
    def test_wrapper_with_custom_architectures(self, sample_matchup_df):
        """Test wrapper with explicit architectures"""
        with patch('src.training.ensemble.EnsembleTrainer') as MockTrainer:
            mock_result = MagicMock()
            mock_result.models = [MagicMock()]
            mock_result.scalers = [MagicMock()]
            mock_result.feature_cols = ['f1']
            mock_result.model_types = ['xgboost']
            
            MockTrainer.return_value.train.return_value = mock_result
            
            models, scalers, feature_cols, model_types = train_ensemble_models(
                sample_matchup_df, architectures=['xgboost']
            )
            
            # Verify config was created with specified architectures
            call_args = MockTrainer.call_args
            config = call_args[0][0] if call_args[0] else call_args[1].get('config')
            assert config.architectures == ['xgboost']
    
    def test_wrapper_default_architectures(self, sample_matchup_df):
        """Test wrapper uses default architectures based on n_models"""
        with patch('src.training.ensemble.EnsembleTrainer') as MockTrainer:
            mock_result = MagicMock()
            mock_result.models = [MagicMock()] * 3
            mock_result.scalers = [MagicMock()] * 3
            mock_result.feature_cols = ['f1']
            mock_result.model_types = ['xgboost', 'rf', 'logistic']
            
            MockTrainer.return_value.train.return_value = mock_result
            
            train_ensemble_models(sample_matchup_df, n_models=3)
            
            # Verify config was created with 3 architectures
            call_args = MockTrainer.call_args
            config = call_args[0][0] if call_args[0] else call_args[1].get('config')
            assert len(config.architectures) == 3

class TestEnsembleTrainerSave:
    """Tests for EnsembleTrainer.save() method - behavior verification"""
    
    def test_save_method_exists(self):
        """Test that save method exists and is callable"""
        trainer = EnsembleTrainer(config=EnsembleConfig(verbose=False))
        assert hasattr(trainer, 'save')
        assert callable(trainer.save)


class TestEnsembleTrainerEvaluate:
    """Tests for EnsembleTrainer.evaluate() method"""
    
    def test_evaluate_ensemble(self, sample_matchup_df):
        """Test ensemble evaluation"""
        # Mock data prep
        mock_data_prep = MagicMock()
        mock_train_test = MagicMock()
        mock_train_test.X_test = np.random.randn(20, 10)
        mock_train_test.y_test = np.random.randint(0, 2, 20)
        mock_data_prep.prepare_for_training.return_value = (mock_train_test, ['f1'])
        
        # Mock trainer factory
        mock_factory = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.predict_proba.return_value = np.random.uniform(0.3, 0.7, 20)
        mock_factory.create.return_value = mock_trainer
        
        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_eval_result = MagicMock()
        mock_eval_result.accuracy = 0.65
        mock_eval_result.classification_report = "Classification Report"
        mock_evaluator.evaluate_ensemble.return_value = mock_eval_result
        
        trainer = EnsembleTrainer(
            config=EnsembleConfig(verbose=False),
            trainer_factory=mock_factory,
            data_prep=mock_data_prep,
            evaluator=mock_evaluator
        )
        
        # Create result to evaluate
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.random.randn(20, 10)
        
        ensemble_result = EnsembleResult(
            models=[MagicMock(), MagicMock()],
            scalers=[mock_scaler, mock_scaler],
            feature_cols=['f1'],
            model_types=['xgboost', 'logistic'],
            training_results=[]
        )
        
        eval_result = trainer.evaluate(ensemble_result, sample_matchup_df)
        
        assert eval_result.accuracy == 0.65
        mock_evaluator.evaluate_ensemble.assert_called_once()
    
    def test_evaluate_logs_accuracy(self, sample_matchup_df):
        """Test that evaluation logs accuracy"""
        captured = []
        
        # Mock data prep
        mock_data_prep = MagicMock()
        mock_train_test = MagicMock()
        mock_train_test.X_test = np.random.randn(20, 10)
        mock_train_test.y_test = np.random.randint(0, 2, 20)
        mock_data_prep.prepare_for_training.return_value = (mock_train_test, ['f1'])
        
        # Mock trainer factory
        mock_factory = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.predict_proba.return_value = np.random.uniform(0.3, 0.7, 20)
        mock_factory.create.return_value = mock_trainer
        
        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_eval_result = MagicMock()
        mock_eval_result.accuracy = 0.65
        mock_eval_result.classification_report = "Report"
        mock_evaluator.evaluate_ensemble.return_value = mock_eval_result
        
        trainer = EnsembleTrainer(
            config=EnsembleConfig(verbose=True),
            trainer_factory=mock_factory,
            data_prep=mock_data_prep,
            evaluator=mock_evaluator,
            print_fn=lambda x: captured.append(x)
        )
        
        # Create result to evaluate
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.random.randn(20, 10)
        
        ensemble_result = EnsembleResult(
            models=[MagicMock()],
            scalers=[mock_scaler],
            feature_cols=['f1'],
            model_types=['xgboost'],
            training_results=[]
        )
        
        trainer.evaluate(ensemble_result, sample_matchup_df)
        
        # Check that accuracy was logged
        assert any('ENSEMBLE ACCURACY' in msg or 'EVALUATING' in msg for msg in captured)