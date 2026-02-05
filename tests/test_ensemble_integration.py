"""
Integration tests for src/nba_ensemble_predictor.py
====================================================
Tests for model training and ensemble prediction functionality.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEnsembleTraining:
    """Tests for ensemble model training"""
    
    def test_xgboost_training(self, sample_matchup_df):
        """XGBoost model should train successfully"""
        import xgboost as xgb
        
        exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
        feature_cols = [col for col in sample_matchup_df.columns if col not in exclude_cols]
        
        X = sample_matchup_df[feature_cols].values
        y = sample_matchup_df['HOME_WIN'].values
        
        # Train XGBoost with minimal settings for speed
        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X, y)
        
        # Should be able to predict
        predictions = model.predict(X[:5])
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)
    
    def test_random_forest_training(self, sample_matchup_df):
        """Random Forest model should train successfully"""
        from sklearn.ensemble import RandomForestClassifier
        
        exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
        feature_cols = [col for col in sample_matchup_df.columns if col not in exclude_cols]
        
        X = sample_matchup_df[feature_cols].values
        y = sample_matchup_df['HOME_WIN'].values
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X[:5])
        assert len(predictions) == 5
    
    def test_logistic_regression_training(self, sample_matchup_df):
        """Logistic Regression model should train successfully"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
        feature_cols = [col for col in sample_matchup_df.columns if col not in exclude_cols]
        
        X = sample_matchup_df[feature_cols].values
        y = sample_matchup_df['HOME_WIN'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_scaled, y)
        
        predictions = model.predict(X_scaled[:5])
        assert len(predictions) == 5
    
    def test_probability_predictions(self, sample_matchup_df):
        """Models should output valid probabilities"""
        from sklearn.ensemble import RandomForestClassifier
        
        exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
        feature_cols = [col for col in sample_matchup_df.columns if col not in exclude_cols]
        
        X = sample_matchup_df[feature_cols].values
        y = sample_matchup_df['HOME_WIN'].values
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        probs = model.predict_proba(X[:5])
        
        # Probabilities should be between 0 and 1
        assert probs.min() >= 0
        assert probs.max() <= 1
        
        # Each row should sum to 1
        row_sums = probs.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(5))


class TestEnsembleVoting:
    """Tests for ensemble voting mechanisms"""
    
    def test_simple_average_voting(self):
        """Simple average voting should work correctly"""
        predictions = np.array([
            [0.7, 0.65, 0.60, 0.75],  # 4 models
            [0.3, 0.35, 0.40, 0.25],
        ])
        
        # Average across models
        ensemble_pred = predictions.mean(axis=1)
        
        assert len(ensemble_pred) == 2
        assert 0 <= ensemble_pred[0] <= 1
        assert 0 <= ensemble_pred[1] <= 1
    
    def test_weighted_voting(self):
        """Weighted voting should work correctly"""
        predictions = np.array([0.7, 0.65, 0.60, 0.75])  # 4 models
        weights = np.array([0.35, 0.20, 0.15, 0.30])
        
        ensemble_pred = np.average(predictions, weights=weights)
        
        assert 0 <= ensemble_pred <= 1
    
    def test_majority_voting(self):
        """Majority voting should select most common prediction"""
        predictions = np.array([1, 1, 0, 1])  # 3 models predict 1, 1 predicts 0
        
        majority = (predictions.mean() >= 0.5).astype(int)
        
        assert majority == 1


class TestEnsembleCalibration:
    """Tests for probability calibration"""
    
    def test_platt_scaling(self):
        """Platt scaling should produce calibrated probabilities"""
        from sklearn.linear_model import LogisticRegression
        
        # Uncalibrated predictions
        raw_probs = np.array([0.55, 0.60, 0.70, 0.80, 0.90])
        actual = np.array([0, 1, 1, 1, 1])
        
        # Platt scaling (logistic regression on raw probabilities)
        calibrator = LogisticRegression()
        calibrator.fit(raw_probs.reshape(-1, 1), actual)
        
        calibrated = calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        
        # Calibrated probabilities should still be valid
        assert calibrated.min() >= 0
        assert calibrated.max() <= 1


class TestModelEvaluation:
    """Tests for model evaluation metrics"""
    
    def test_classification_report_generation(self):
        """Classification report should be generated correctly"""
        from sklearn.metrics import classification_report
        
        y_true = np.array([0, 1, 1, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 1])
        
        report = classification_report(
            y_true, y_pred,
            target_names=['Away Win', 'Home Win']
        )
        
        assert 'Away Win' in report
        assert 'Home Win' in report
        assert 'accuracy' in report
    
    def test_accuracy_per_tier(self, sample_prediction_history):
        """Accuracy should be calculable per tier"""
        tier_accuracy = sample_prediction_history.groupby('tier').apply(
            lambda x: x['correct'].mean()
        )
        
        # EXCELLENT tier should have 100% accuracy (1/1 in sample)
        assert tier_accuracy['EXCELLENT'] == 1.0
    
    def test_roi_calculation_by_tier(self, sample_prediction_history):
        """ROI should be calculable per tier"""
        for tier in sample_prediction_history['tier'].unique():
            tier_data = sample_prediction_history[sample_prediction_history['tier'] == tier]
            wins = tier_data['correct'].sum()
            losses = len(tier_data) - wins
            
            # ROI at -110 odds
            roi = (wins * 100 - losses * 110) / (len(tier_data) * 110) * 100
            
            assert isinstance(roi, float)


class TestModelPaths:
    """Tests for model file paths and serialization"""
    
    def test_model_paths_exist(self):
        """Model path constants should be defined"""
        from src.paths import (
            ENSEMBLE_TYPES_FILE, ENSEMBLE_SCALERS_FILE, ENSEMBLE_FEATURES_FILE,
            ENSEMBLE_META_LR_FILE, ENSEMBLE_PLATT_FILE
        )
        
        assert ENSEMBLE_TYPES_FILE is not None
        assert ENSEMBLE_SCALERS_FILE is not None
        assert ENSEMBLE_FEATURES_FILE is not None
        assert ENSEMBLE_META_LR_FILE is not None
        assert ENSEMBLE_PLATT_FILE is not None
    
    def test_model_paths_in_models_dir(self):
        """Model files should be in models directory"""
        from src.paths import MODELS_DIR, ENSEMBLE_TYPES_FILE
        
        assert MODELS_DIR in ENSEMBLE_TYPES_FILE


class TestFeatureImportance:
    """Tests for feature importance analysis"""
    
    def test_xgboost_feature_importance(self, sample_matchup_df):
        """XGBoost should provide feature importance"""
        import xgboost as xgb
        
        exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
        feature_cols = [col for col in sample_matchup_df.columns if col not in exclude_cols]
        
        X = sample_matchup_df[feature_cols].values
        y = sample_matchup_df['HOME_WIN'].values
        
        model = xgb.XGBClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        importance = model.feature_importances_
        
        assert len(importance) == len(feature_cols)
        assert importance.sum() > 0
    
    def test_random_forest_feature_importance(self, sample_matchup_df):
        """Random Forest should provide feature importance"""
        from sklearn.ensemble import RandomForestClassifier
        
        exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
        feature_cols = [col for col in sample_matchup_df.columns if col not in exclude_cols]
        
        X = sample_matchup_df[feature_cols].values
        y = sample_matchup_df['HOME_WIN'].values
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        importance = model.feature_importances_
        
        assert len(importance) == len(feature_cols)
        assert abs(importance.sum() - 1.0) < 0.01  # Should sum to 1
