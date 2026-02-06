"""
Tests for Data Preparation Module
=================================
Tests for src/training/data_prep.py
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.training.data_prep import (
    DataPreparation,
    TrainTestData,
    ScaledData,
    validate_dataframe
)


class TestTrainTestData:
    """Tests for TrainTestData dataclass"""
    
    def test_basic_properties(self):
        """Test basic TrainTestData creation and properties"""
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_test = np.array([[7, 8]])
        y_train = np.array([0, 0, 1])
        y_test = np.array([1])
        
        data = TrainTestData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_cols=['feat1', 'feat2']
        )
        
        assert data.n_features == 2
        assert data.n_train_samples == 3
        assert data.n_test_samples == 1
    
    def test_empty_data(self):
        """Test with empty arrays"""
        data = TrainTestData(
            X_train=np.array([]).reshape(0, 5),
            X_test=np.array([]).reshape(0, 5),
            y_train=np.array([]),
            y_test=np.array([]),
            feature_cols=[]
        )
        
        assert data.n_train_samples == 0
        assert data.n_test_samples == 0


class TestDataPreparation:
    """Tests for DataPreparation class"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample matchup dataframe"""
        np.random.seed(42)
        return pd.DataFrame({
            'HOME_WIN': [1, 0, 1, 0, 1] * 20,  # 100 samples
            'GAME_ID': range(100),
            'GAME_DATE': pd.date_range('2024-01-01', periods=100),
            'HOME_TEAM_ID': [1] * 100,
            'AWAY_TEAM_ID': [2] * 100,
            'HOME_PTS': np.random.randint(90, 130, 100),
            'AWAY_PTS': np.random.randint(90, 130, 100),
            'HOME_REB': np.random.randint(30, 60, 100),
            'AWAY_REB': np.random.randint(30, 60, 100),
            'HOME_AST': np.random.randint(15, 35, 100),
            'AWAY_AST': np.random.randint(15, 35, 100),
        })
    
    @pytest.fixture
    def data_prep(self):
        """Create DataPreparation instance"""
        return DataPreparation(test_size=0.2, random_state=42)
    
    def test_init(self):
        """Test initialization with parameters"""
        dp = DataPreparation(test_size=0.3, random_state=123)
        assert dp.test_size == 0.3
        assert dp.random_state == 123
    
    def test_get_feature_columns(self, data_prep, sample_df):
        """Test feature column extraction"""
        feature_cols = data_prep.get_feature_columns(sample_df)
        
        # Should not include excluded columns
        assert 'HOME_WIN' not in feature_cols
        assert 'GAME_ID' not in feature_cols
        assert 'GAME_DATE' not in feature_cols
        
        # Should include feature columns
        assert 'HOME_PTS' in feature_cols
        assert 'AWAY_PTS' in feature_cols
    
    def test_get_feature_columns_with_custom_exclude(self, data_prep, sample_df):
        """Test feature extraction with additional exclusions"""
        feature_cols = data_prep.get_feature_columns(
            sample_df, 
            exclude_cols=['HOME_PTS']
        )
        
        assert 'HOME_PTS' not in feature_cols
        assert 'AWAY_PTS' in feature_cols
    
    def test_prepare_features(self, data_prep, sample_df):
        """Test feature and label extraction"""
        X, y = data_prep.prepare_features(sample_df)
        
        assert len(X) == 100
        assert len(y) == 100
        assert X.shape[1] == 6  # 6 numeric feature columns
        assert set(y) == {0, 1}
    
    def test_chronological_split(self, data_prep, sample_df):
        """Test chronological train/test split"""
        feature_cols = data_prep.get_feature_columns(sample_df)
        X, y = data_prep.prepare_features(sample_df, feature_cols)
        
        data = data_prep.chronological_split(X, y)
        
        # Check sizes (80/20 split)
        assert data.n_train_samples == 80
        assert data.n_test_samples == 20
        
        # Check that train comes before test (chronological)
        # First test sample should have higher index than last train sample
        # In a chronological split with shuffle=False, this is guaranteed
        assert len(data.X_train) + len(data.X_test) == len(X)
    
    def test_scale_features(self, data_prep, sample_df):
        """Test feature scaling"""
        feature_cols = data_prep.get_feature_columns(sample_df)
        X, y = data_prep.prepare_features(sample_df, feature_cols)
        data = data_prep.chronological_split(X, y)
        
        scaled = data_prep.scale_features(data)
        
        assert isinstance(scaled.scaler, StandardScaler)
        assert scaled.X_train_scaled.shape == data.X_train.shape
        assert scaled.X_test_scaled.shape == data.X_test.shape
        
        # Check scaling (mean ~ 0, std ~ 1 for train)
        assert abs(scaled.X_train_scaled.mean()) < 0.1
        assert abs(scaled.X_train_scaled.std() - 1.0) < 0.2
    
    def test_compute_sample_weights(self):
        """Test sample weight computation"""
        weights = DataPreparation.compute_sample_weights(100, decay_factor=1.2)
        
        assert len(weights) == 100
        # Oldest sample should have lower weight
        assert weights[0] < weights[-1]
        # Newest sample should be ~1.0
        assert abs(weights[-1] - 1.0) < 0.01
        # Oldest should be ~0.3
        assert weights[0] < 0.35
        assert weights[0] > 0.25
    
    def test_compute_sample_weights_different_decay(self):
        """Test sample weights with different decay factors"""
        weights_low = DataPreparation.compute_sample_weights(100, decay_factor=0.5)
        weights_high = DataPreparation.compute_sample_weights(100, decay_factor=2.0)
        
        # Higher decay = smaller oldest weight
        assert weights_high[0] < weights_low[0]
    
    def test_select_top_features(self):
        """Test top feature selection"""
        importances = np.array([0.1, 0.5, 0.2, 0.8, 0.05])
        
        top_3 = DataPreparation.select_top_features(importances, n_features=3)
        
        # Should return indices of top 3 most important
        assert 3 in top_3  # 0.8 - highest
        assert 1 in top_3  # 0.5 - second
        assert 2 in top_3  # 0.2 - third
        assert len(top_3) == 3
    
    def test_select_top_features_order(self):
        """Test that top features are in descending importance order"""
        importances = np.array([0.1, 0.5, 0.2, 0.8, 0.05])
        
        top = DataPreparation.select_top_features(importances, n_features=5)
        
        # First index should be the most important
        assert top[0] == 3  # 0.8
        assert top[1] == 1  # 0.5
    
    def test_prepare_for_training(self, data_prep, sample_df):
        """Test complete preparation pipeline"""
        data, feature_cols = data_prep.prepare_for_training(sample_df)
        
        assert isinstance(data, TrainTestData)
        assert len(feature_cols) == 6
        assert data.feature_cols == feature_cols
        assert data.n_train_samples == 80
        assert data.n_test_samples == 20


class TestValidateDataframe:
    """Tests for validate_dataframe function"""
    
    def test_valid_dataframe(self):
        """Test validation of valid dataframe"""
        df = pd.DataFrame({'HOME_WIN': [1, 0, 1]})
        assert validate_dataframe(df) == True
    
    def test_empty_dataframe(self):
        """Test validation rejects empty dataframe"""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df)
    
    def test_none_dataframe(self):
        """Test validation rejects None"""
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(None)
    
    def test_missing_home_win(self):
        """Test validation requires HOME_WIN column"""
        df = pd.DataFrame({'OTHER': [1, 2, 3]})
        with pytest.raises(ValueError, match="HOME_WIN"):
            validate_dataframe(df)
    
    def test_missing_required_cols(self):
        """Test validation checks required columns"""
        df = pd.DataFrame({'HOME_WIN': [1, 0], 'COL_A': [1, 2]})
        
        # Should pass with satisfied requirements
        assert validate_dataframe(df, required_cols=['COL_A']) == True
        
        # Should fail with unsatisfied requirements
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe(df, required_cols=['COL_B'])
