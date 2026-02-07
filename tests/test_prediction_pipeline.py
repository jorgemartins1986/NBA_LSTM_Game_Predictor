"""
Tests for Prediction Pipeline Module
====================================
Tests for src/prediction/pipeline.py
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from sklearn.preprocessing import StandardScaler

from src.prediction.pipeline import (
    PredictionResult,
    PredictionPipeline,
    predict_with_ensemble_compat
)
from src.prediction.loader import LoadedEnsemble
from src.prediction.features import FeatureComputer


class TestPredictionResult:
    """Tests for PredictionResult dataclass"""
    
    def test_basic_creation(self):
        """Test basic creation"""
        result = PredictionResult(
            home_win_probability=0.65,
            away_win_probability=0.35,
            predicted_winner='HOME',
            confidence=0.3,
            individual_predictions=[0.6, 0.7, 0.65],
            prediction_std=0.05
        )
        
        assert result.home_win_probability == 0.65
        assert result.is_home_favorite == True
    
    def test_away_favorite(self):
        """Test away favorite case"""
        result = PredictionResult(
            home_win_probability=0.35,
            away_win_probability=0.65,
            predicted_winner='AWAY',
            confidence=0.3,
            individual_predictions=[0.3, 0.4, 0.35],
            prediction_std=0.05
        )
        
        assert result.is_home_favorite == False
    
    def test_is_high_confidence(self):
        """Test high confidence detection"""
        high = PredictionResult(
            home_win_probability=0.75,
            away_win_probability=0.25,
            predicted_winner='HOME',
            confidence=0.5,
            individual_predictions=[0.75],
            prediction_std=0.0
        )
        
        low = PredictionResult(
            home_win_probability=0.55,
            away_win_probability=0.45,
            predicted_winner='HOME',
            confidence=0.1,
            individual_predictions=[0.55],
            prediction_std=0.0
        )
        
        assert high.is_high_confidence == True
        assert low.is_high_confidence == False
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        result = PredictionResult(
            home_win_probability=0.65,
            away_win_probability=0.35,
            predicted_winner='HOME',
            confidence=0.3,
            individual_predictions=[0.6, 0.7],
            prediction_std=0.05,
            model_agreement=0.8
        )
        
        d = result.to_dict()
        
        assert d['home_win_probability'] == 0.65
        assert d['predicted_winner'] == 'HOME'
        assert d['model_agreement'] == 0.8


@pytest.fixture
def mock_ensemble():
    """Create a mock ensemble for testing"""
    # Create mock models
    model1 = Mock()
    model1.predict_proba = Mock(return_value=np.array([[0.4, 0.6]]))
    
    model2 = Mock()
    model2.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
    
    model3 = Mock()
    model3.predict = Mock(return_value=np.array([[0.65]]))
    
    # Create mock scalers
    scaler1 = Mock()
    scaler1.transform = Mock(side_effect=lambda x: x)
    scaler2 = Mock()
    scaler2.transform = Mock(side_effect=lambda x: x)
    scaler3 = Mock()
    scaler3.transform = Mock(side_effect=lambda x: x)
    
    return LoadedEnsemble(
        models=[model1, model2, model3],
        scalers=[scaler1, scaler2, scaler3],
        feature_cols=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'],
        model_types=['xgboost', 'logistic', 'lstm']
    )


class TestPredictionPipeline:
    """Tests for PredictionPipeline class"""
    
    def test_init(self, mock_ensemble):
        """Test initialization"""
        pipeline = PredictionPipeline(mock_ensemble)
        
        assert pipeline.ensemble is mock_ensemble
        assert isinstance(pipeline.feature_computer, FeatureComputer)
    
    def test_init_with_custom_feature_computer(self, mock_ensemble):
        """Test initialization with custom feature computer"""
        custom_computer = FeatureComputer(window_size=30)
        pipeline = PredictionPipeline(mock_ensemble, feature_computer=custom_computer)
        
        assert pipeline.feature_computer is custom_computer
    
    def test_get_model_prediction_xgboost(self, mock_ensemble):
        """Test prediction from XGBoost model"""
        pipeline = PredictionPipeline(mock_ensemble)
        
        model = mock_ensemble.models[0]
        scaler = mock_ensemble.scalers[0]
        features = np.random.randn(1, 5)
        
        pred = pipeline.get_model_prediction(model, scaler, 'xgboost', features)
        
        assert pred == 0.6
        model.predict_proba.assert_called_once()
    
    def test_get_model_prediction_keras(self, mock_ensemble):
        """Test prediction from Keras model"""
        pipeline = PredictionPipeline(mock_ensemble)
        
        model = mock_ensemble.models[2]
        scaler = mock_ensemble.scalers[2]
        features = np.random.randn(1, 5)
        
        pred = pipeline.get_model_prediction(model, scaler, 'keras', features)
        
        assert pred == 0.65
        model.predict.assert_called_once()
    
    def test_get_model_prediction_rf_with_feature_selection(self, mock_ensemble):
        """Test prediction from RF with feature selection"""
        pipeline = PredictionPipeline(mock_ensemble)
        
        # Create RF model with feature selection
        rf_model = Mock()
        rf_model._top_feature_indices = np.array([0, 2, 4])  # Select 3 features
        rf_model.predict_proba = Mock(return_value=np.array([[0.35, 0.65]]))
        
        scaler = Mock()
        scaler.transform = Mock(side_effect=lambda x: x)
        
        features = np.random.randn(1, 5)
        
        pred = pipeline.get_model_prediction(rf_model, scaler, 'random_forest', features)
        
        assert pred == 0.65
        # Check that only 3 features were passed
        call_args = rf_model.predict_proba.call_args[0][0]
        assert call_args.shape[1] == 3
    
    def test_aggregate_predictions_simple_avg(self, mock_ensemble):
        """Test simple average aggregation"""
        pipeline = PredictionPipeline(mock_ensemble)
        
        preds = [0.6, 0.7, 0.65]
        result = pipeline.aggregate_predictions(preds)
        
        assert abs(result - 0.65) < 0.001
    
    def test_aggregate_predictions_weighted(self, mock_ensemble):
        """Test weighted average aggregation"""
        pipeline = PredictionPipeline(mock_ensemble)
        
        preds = [0.5, 0.9]
        weights = [0.2, 0.8]
        
        result = pipeline.aggregate_predictions(preds, weights)
        
        # weighted: 0.5 * 0.2 + 0.9 * 0.8 = 0.1 + 0.72 = 0.82
        assert abs(result - 0.82) < 0.001
    
    def test_compute_model_agreement_unanimous(self, mock_ensemble):
        """Test agreement when all models predict similar probabilities"""
        pipeline = PredictionPipeline(mock_ensemble)
        
        preds = [0.7, 0.8, 0.65]  # Similar predictions
        agreement = pipeline.compute_model_agreement(preds)
        
        # std ≈ 0.0624, so agreement ≈ 0.9376
        assert abs(agreement - 0.9376) < 0.01
    
    def test_compute_model_agreement_split(self, mock_ensemble):
        """Test agreement when models predict different probabilities"""
        pipeline = PredictionPipeline(mock_ensemble)
        
        preds = [0.7, 0.3, 0.6]  # More varied predictions
        agreement = pipeline.compute_model_agreement(preds)
        
        # std ≈ 0.17, so agreement ≈ 0.83
        assert abs(agreement - 0.83) < 0.01
    
    def test_predict_from_features(self, mock_ensemble):
        """Test prediction from feature vector"""
        pipeline = PredictionPipeline(mock_ensemble)
        
        features = np.random.randn(1, 5)
        result = pipeline.predict_from_features(features)
        
        assert isinstance(result, PredictionResult)
        assert 0 <= result.home_win_probability <= 1
        assert len(result.individual_predictions) == 3
    
    def test_predict_from_features_with_elo(self, mock_ensemble):
        """Test prediction with ELO difference"""
        # Add meta model
        mock_ensemble.meta_clf = Mock()
        mock_ensemble.meta_clf.predict_proba = Mock(
            return_value=np.array([[0.3, 0.7]])
        )
        
        pipeline = PredictionPipeline(mock_ensemble)
        
        features = np.random.randn(1, 5)
        result = pipeline.predict_from_features(features, elo_diff=50.0)
        
        # Should have stacked probability
        assert result.stacked_probability is not None
    
    def test_predict_from_dict_features(self, mock_ensemble):
        """Test prediction from feature dictionaries"""
        pipeline = PredictionPipeline(mock_ensemble)
        
        # Mock feature computer to return expected shape
        mock_computer = Mock()
        mock_computer.build_feature_vector = Mock(
            return_value=np.random.randn(1, 5)
        )
        pipeline.feature_computer = mock_computer
        
        result = pipeline.predict(
            home_features={'PTS': 110},
            away_features={'PTS': 105}
        )
        
        assert isinstance(result, PredictionResult)
        mock_computer.build_feature_vector.assert_called_once()
    
    def test_batch_predict(self, mock_ensemble):
        """Test batch prediction"""
        pipeline = PredictionPipeline(mock_ensemble)
        
        # Mock feature computer
        mock_computer = Mock()
        mock_computer.build_feature_vector = Mock(
            return_value=np.random.randn(1, 5)
        )
        pipeline.feature_computer = mock_computer
        
        games = [
            {'home_features': {'PTS': 110}, 'away_features': {'PTS': 105}},
            {'home_features': {'PTS': 115}, 'away_features': {'PTS': 108}},
        ]
        
        results = pipeline.batch_predict(games)
        
        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)
    
    def test_apply_stacking_with_platt(self, mock_ensemble):
        """Test stacking with Platt calibration"""
        mock_ensemble.meta_clf = Mock()
        mock_ensemble.meta_clf.predict_proba = Mock(
            return_value=np.array([[0.35, 0.65]])
        )
        mock_ensemble.platt = Mock()
        mock_ensemble.platt.predict_proba = Mock(
            return_value=np.array([[0.3, 0.7]])
        )
        
        pipeline = PredictionPipeline(mock_ensemble)
        
        stacked = pipeline.apply_stacking([0.6, 0.7, 0.65])
        
        # Should return calibrated value
        assert stacked == 0.7
    
    def test_apply_stacking_no_meta_model(self, mock_ensemble):
        """Test stacking returns None when no meta model"""
        mock_ensemble.meta_clf = None
        
        pipeline = PredictionPipeline(mock_ensemble)
        
        stacked = pipeline.apply_stacking([0.6, 0.7, 0.65])
        
        assert stacked is None


class TestBackwardCompatibility:
    """Tests for backward compatibility wrapper"""
    
    def test_predict_with_ensemble_compat(self, mock_ensemble):
        """Test backward compatible prediction function"""
        import pandas as pd
        
        # Create minimal matchup_df
        matchup_df = pd.DataFrame({
            'feat1': [1.0],
            'feat2': [2.0],
            'feat3': [3.0],
            'feat4': [4.0],
            'feat5': [5.0],
        })
        
        result = predict_with_ensemble_compat(
            models=mock_ensemble.models,
            scalers=mock_ensemble.scalers,
            feature_cols=mock_ensemble.feature_cols,
            matchup_df=matchup_df,
            game_idx=0,
            model_types=mock_ensemble.model_types
        )
        
        # Should return dict in original format
        assert isinstance(result, dict)
        assert 'home_win_probability' in result
        assert 'predicted_winner' in result
        assert 'confidence' in result
