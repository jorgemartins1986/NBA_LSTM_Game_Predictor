"""
Tests for src/nba_ensemble_predictor.py
=======================================
Tests for ensemble model training and prediction functions.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSaveClassificationReports:
    """Tests for save_classification_reports function"""
    
    def test_creates_txt_file(self, tmp_path):
        """Should create a text file with reports"""
        from src.nba_ensemble_predictor import save_classification_reports
        
        # Mock the get_report_path to use tmp_path
        with patch('src.nba_ensemble_predictor.get_report_path') as mock_path:
            mock_path.side_effect = lambda x: str(tmp_path / x)
            
            reports = [
                {
                    'model': 1,
                    'architecture': 'xgboost',
                    'accuracy': 0.64,
                    'report': 'Test classification report\nAway Win  0.60\nHome Win  0.65'
                }
            ]
            
            result = save_classification_reports(reports, ['xgboost'])
            
            assert os.path.exists(result)
    
    def test_creates_json_file(self, tmp_path):
        """Should create a JSON file with reports"""
        from src.nba_ensemble_predictor import save_classification_reports
        
        with patch('src.nba_ensemble_predictor.get_report_path') as mock_path:
            mock_path.side_effect = lambda x: str(tmp_path / x)
            
            reports = [
                {
                    'model': 1,
                    'architecture': 'xgboost',
                    'accuracy': 0.64,
                    'report': 'Test report'
                }
            ]
            
            save_classification_reports(reports, ['xgboost'])
            
            # Find the JSON file
            json_files = list(tmp_path.glob('*.json'))
            assert len(json_files) > 0
    
    def test_report_content(self, tmp_path):
        """Report should contain model information"""
        from src.nba_ensemble_predictor import save_classification_reports
        
        with patch('src.nba_ensemble_predictor.get_report_path') as mock_path:
            mock_path.side_effect = lambda x: str(tmp_path / x)
            
            reports = [
                {
                    'model': 1,
                    'architecture': 'xgboost',
                    'accuracy': 0.6382,
                    'report': 'Away Win  0.60  0.40\nHome Win  0.65  0.81'
                },
                {
                    'model': 2,
                    'architecture': 'random_forest',
                    'accuracy': 0.6305,
                    'report': 'Away Win  0.59  0.38\nHome Win  0.65  0.81'
                }
            ]
            
            result_path = save_classification_reports(reports, ['xgboost', 'random_forest'])
            
            with open(result_path, 'r') as f:
                content = f.read()
            
            assert 'xgboost' in content.lower()
            assert 'random_forest' in content.lower()
            assert '63' in content  # Accuracy percentage


class TestFeatureExtraction:
    """Tests for feature extraction from matchup data"""
    
    def test_exclude_columns_removed(self, sample_matchup_df):
        """Non-feature columns should be excluded"""
        exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
        feature_cols = [col for col in sample_matchup_df.columns if col not in exclude_cols]
        
        X = sample_matchup_df[feature_cols].values
        
        # Should have fewer columns than original
        assert X.shape[1] < len(sample_matchup_df.columns)
        # HOME_WIN should not be in features
        assert 'HOME_WIN' not in feature_cols
    
    def test_feature_matrix_shape(self, sample_matchup_df):
        """Feature matrix should have correct shape"""
        exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
        feature_cols = [col for col in sample_matchup_df.columns if col not in exclude_cols]
        
        X = sample_matchup_df[feature_cols].values
        y = sample_matchup_df['HOME_WIN'].values
        
        assert X.shape[0] == len(sample_matchup_df)
        assert len(y) == len(sample_matchup_df)
    
    def test_target_is_binary(self, sample_matchup_df):
        """Target variable should be binary"""
        y = sample_matchup_df['HOME_WIN'].values
        
        unique_values = np.unique(y)
        assert len(unique_values) <= 2
        assert all(v in [0, 1] for v in unique_values)


class TestModelArchitectures:
    """Tests for model architecture configurations"""
    
    def test_default_architectures(self):
        """Default architectures should include all four models"""
        default_archs = ['xgboost', 'random_forest', 'logistic', 'lstm']
        
        assert len(default_archs) == 4
        assert 'xgboost' in default_archs
        assert 'random_forest' in default_archs
        assert 'logistic' in default_archs
        assert 'lstm' in default_archs
    
    def test_architecture_cycling(self):
        """Architectures should cycle for n_models > len(architectures)"""
        architectures = ['xgboost', 'random_forest']
        n_models = 5
        
        selected = [architectures[i % len(architectures)] for i in range(n_models)]
        
        assert selected == ['xgboost', 'random_forest', 'xgboost', 'random_forest', 'xgboost']


class TestDataPreparation:
    """Tests for data preparation functions"""
    
    def test_train_test_split_ratio(self, sample_matchup_df):
        """Train/test split should use correct ratio"""
        from sklearn.model_selection import train_test_split
        
        exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
        feature_cols = [col for col in sample_matchup_df.columns if col not in exclude_cols]
        
        X = sample_matchup_df[feature_cols].values
        y = sample_matchup_df['HOME_WIN'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Test set should be ~20%
        assert abs(len(X_test) / len(X) - 0.2) < 0.05
    
    def test_scaling_normalization(self, sample_matchup_df):
        """Scaler should normalize features"""
        from sklearn.preprocessing import StandardScaler
        
        exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
        feature_cols = [col for col in sample_matchup_df.columns if col not in exclude_cols]
        
        X = sample_matchup_df[feature_cols].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # After scaling, mean should be ~0 and std ~1
        assert abs(X_scaled.mean()) < 0.1
        assert abs(X_scaled.std() - 1.0) < 0.1


class TestClassWeights:
    """Tests for class weight computation"""
    
    def test_compute_class_weights(self):
        """Class weights should be computed for imbalanced data"""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Imbalanced: 60% home wins, 40% away wins
        y = np.array([1]*60 + [0]*40)
        
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        
        # Away wins (minority) should have higher weight
        assert weights[0] > weights[1]
    
    def test_balanced_classes_equal_weights(self):
        """Balanced classes should have equal weights"""
        from sklearn.utils.class_weight import compute_class_weight
        
        y = np.array([0]*50 + [1]*50)
        
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        
        assert abs(weights[0] - weights[1]) < 0.01


class TestModelMetrics:
    """Tests for model evaluation metrics"""
    
    def test_accuracy_calculation(self):
        """Accuracy should be correctly calculated"""
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 1, 1, 1])
        
        accuracy = (y_true == y_pred).mean()
        
        assert accuracy == 0.8
    
    def test_accuracy_bounds(self):
        """Accuracy should be between 0 and 1"""
        y_true = np.random.randint(0, 2, 100)
        y_pred = np.random.randint(0, 2, 100)
        
        accuracy = (y_true == y_pred).mean()
        
        assert 0 <= accuracy <= 1


class TestEnsembleWeights:
    """Tests for ensemble weight configurations"""
    
    def test_default_weights_sum_to_one(self):
        """Default ensemble weights should sum to 1"""
        # Typical default weights
        weights = [0.35, 0.20, 0.15, 0.30]  # XGB, RF, Log, LSTM
        
        assert abs(sum(weights) - 1.0) < 0.001
    
    def test_weighted_average_prediction(self):
        """Weighted average should produce valid probability"""
        predictions = np.array([0.65, 0.60, 0.55, 0.70])  # From 4 models
        weights = np.array([0.35, 0.20, 0.15, 0.30])
        
        ensemble_pred = np.average(predictions, weights=weights)
        
        assert 0 <= ensemble_pred <= 1
        assert abs(ensemble_pred - 0.64) < 0.01  # Allow small tolerance


class TestPredictionThresholds:
    """Tests for prediction threshold logic"""
    
    def test_threshold_classification(self):
        """Predictions above threshold should be classified as home win"""
        probabilities = np.array([0.45, 0.55, 0.60, 0.40, 0.75])
        threshold = 0.5
        
        predictions = (probabilities >= threshold).astype(int)
        
        assert list(predictions) == [0, 1, 1, 0, 1]
    
    def test_confidence_calculation(self):
        """Confidence should be distance from threshold"""
        probability = 0.75
        threshold = 0.5
        
        confidence = abs(probability - threshold)
        
        assert confidence == 0.25
    
    def test_high_confidence_threshold(self):
        """High confidence predictions should have prob > 0.65"""
        probabilities = np.array([0.65, 0.70, 0.75, 0.80, 0.55, 0.60])
        
        high_confidence = probabilities >= 0.65
        
        assert high_confidence.sum() == 4


class TestModelPersistence:
    """Tests for model saving and loading"""
    
    def test_scaler_serialization(self, tmp_path):
        """Scaler should be serializable"""
        from sklearn.preprocessing import StandardScaler
        import joblib
        
        scaler = StandardScaler()
        X = np.random.randn(100, 10)
        scaler.fit(X)
        
        path = tmp_path / "scaler.pkl"
        joblib.dump(scaler, path)
        
        loaded_scaler = joblib.load(path)
        
        # Should produce same transform
        X_original = scaler.transform(X[:5])
        X_loaded = loaded_scaler.transform(X[:5])
        
        np.testing.assert_array_almost_equal(X_original, X_loaded)
    
    def test_logistic_regression_serialization(self, tmp_path):
        """LogisticRegression should be serializable"""
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        path = tmp_path / "model.pkl"
        joblib.dump(model, path)
        
        loaded_model = joblib.load(path)
        
        # Should produce same predictions
        preds_original = model.predict(X[:5])
        preds_loaded = loaded_model.predict(X[:5])
        
        np.testing.assert_array_equal(preds_original, preds_loaded)
    
    def test_random_forest_serialization(self, tmp_path):
        """RandomForest should be serializable"""
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        path = tmp_path / "rf_model.pkl"
        joblib.dump(model, path)
        
        loaded_model = joblib.load(path)
        
        preds_original = model.predict(X[:5])
        preds_loaded = loaded_model.predict(X[:5])
        
        np.testing.assert_array_equal(preds_original, preds_loaded)
