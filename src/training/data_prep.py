"""
Data Preparation for Training
=============================
Pure functions and classes for preparing data for model training.
All functions are deterministic and have no side effects.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainTestData:
    """Container for train/test split data"""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_cols: List[str]
    
    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]
    
    @property
    def n_train_samples(self) -> int:
        return len(self.y_train)
    
    @property
    def n_test_samples(self) -> int:
        return len(self.y_test)


@dataclass
class ScaledData:
    """Container for scaled train/test data with the scaler"""
    X_train_scaled: np.ndarray
    X_test_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


class DataPreparation:
    """
    Handles all data preparation for model training.
    
    Design decisions:
    - Chronological split (shuffle=False) to prevent temporal leakage
    - Sample weights decay for recency weighting
    - Pure functions where possible for testability
    """
    
    # Columns to exclude from features (always)
    EXCLUDE_COLS = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize data preparation.
        
        Args:
            test_size: Fraction of data to use for testing (default 0.2)
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
    
    def get_feature_columns(self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
        """
        Get list of feature column names from dataframe.
        
        Args:
            df: Matchup dataframe
            exclude_cols: Additional columns to exclude (beyond defaults)
            
        Returns:
            List of feature column names
        """
        exclude = set(self.EXCLUDE_COLS)
        if exclude_cols:
            exclude.update(exclude_cols)
        
        return [col for col in df.columns if col not in exclude]
    
    def prepare_features(self, df: pd.DataFrame, 
                         feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features X and labels y from dataframe.
        
        Args:
            df: Matchup dataframe with features and HOME_WIN label
            feature_cols: Optional specific columns to use (auto-detected if None)
            
        Returns:
            Tuple of (X features array, y labels array)
        """
        if feature_cols is None:
            feature_cols = self.get_feature_columns(df)
        
        X = df[feature_cols].values
        y = df['HOME_WIN'].values
        
        return X, y
    
    def chronological_split(self, X: np.ndarray, y: np.ndarray) -> TrainTestData:
        """
        Split data chronologically (older games train, newer test).
        
        Uses shuffle=False to prevent temporal leakage - this is critical
        for time-series data like NBA games.
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            TrainTestData with train/test splits
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            shuffle=False  # CRITICAL: chronological split
        )
        
        return TrainTestData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_cols=[]  # Set by caller if needed
        )
    
    def scale_features(self, data: TrainTestData) -> ScaledData:
        """
        Scale features using StandardScaler.
        
        Scaler is fit ONLY on training data to prevent data leakage.
        
        Args:
            data: TrainTestData with unscaled features
            
        Returns:
            ScaledData with scaled features and fitted scaler
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(data.X_train)
        X_test_scaled = scaler.transform(data.X_test)
        
        return ScaledData(
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=data.y_train,
            y_test=data.y_test,
            scaler=scaler
        )
    
    @staticmethod
    def compute_sample_weights(n_samples: int, 
                               decay_factor: float = 1.2) -> np.ndarray:
        """
        Compute exponentially decaying sample weights.
        
        Gives more weight to recent games (end of dataset).
        Oldest games get ~0.3 weight, newest get 1.0.
        
        Args:
            n_samples: Number of samples
            decay_factor: Decay strength (higher = more decay)
            
        Returns:
            Array of sample weights
        """
        return np.exp(np.linspace(-decay_factor, 0, n_samples))
    
    @staticmethod
    def select_top_features(importances: np.ndarray, 
                           n_features: int = 40) -> np.ndarray:
        """
        Select indices of top N most important features.
        
        Args:
            importances: Feature importance scores
            n_features: Number of top features to select
            
        Returns:
            Array of feature indices (sorted by importance, descending)
        """
        return np.argsort(importances)[::-1][:n_features]
    
    def prepare_for_training(self, df: pd.DataFrame) -> Tuple[TrainTestData, List[str]]:
        """
        Complete data preparation pipeline.
        
        Combines feature extraction, chronological split.
        Scaling is NOT included here as different models may need different scalers.
        
        Args:
            df: Matchup dataframe
            
        Returns:
            Tuple of (TrainTestData, feature_cols list)
        """
        feature_cols = self.get_feature_columns(df)
        X, y = self.prepare_features(df, feature_cols)
        
        data = self.chronological_split(X, y)
        data.feature_cols = feature_cols
        
        return data, feature_cols


def validate_dataframe(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> bool:
    """
    Validate that dataframe has required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        
    Returns:
        True if valid, raises ValueError if not
    """
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is empty or None")
    
    if 'HOME_WIN' not in df.columns:
        raise ValueError("DataFrame missing required 'HOME_WIN' column")
    
    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
    
    return True
