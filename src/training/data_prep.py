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
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainTestData:
    """Container for train/test split data"""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_cols: List[str]
    train_dates: Optional[np.ndarray] = None
    test_dates: Optional[np.ndarray] = None
    
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
    - Explicit date ordering and disjoint calendar dates at the split
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
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
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
    
    def chronological_split(self, X: np.ndarray, y: np.ndarray,
                            game_dates, game_ids=None) -> TrainTestData:
        """
        Split data chronologically (older games train, newer test).
        
        Sort features and labels together by GAME_DATE (then GAME_ID when
        available). Move the entire boundary date into the later partition.
        The requested test fraction is approximate when dates contain many games.
        Dates are required: row order alone cannot establish chronology.
        
        Args:
            X: Feature array
            y: Label array
            game_dates: One valid date per row; timestamps are grouped by day.
            game_ids: Optional IDs for deterministic ordering within a date.
            
        Returns:
            TrainTestData with train/test splits
        """
        X, y = np.asarray(X), np.asarray(y)
        if len(X) != len(y) or len(X) != len(game_dates):
            raise ValueError("Features, labels, and GAME_DATE must have matching lengths")
        if len(X) < 2:
            raise ValueError("At least two games on distinct dates are required")
        raw_dates = pd.Series(np.asarray(game_dates))
        if any(isinstance(value, (int, float, np.number)) for value in raw_dates):
            raise ValueError("GAME_DATE must contain dates, not numeric timestamps")
        dates = pd.to_datetime(raw_dates, format='mixed', errors='coerce').dt.normalize()
        if dates.isna().any():
            raise ValueError("GAME_DATE contains missing or invalid dates")
        order_frame = pd.DataFrame({'date': dates, 'position': np.arange(len(X))})
        sort_keys = ['date']
        if game_ids is not None:
            if len(game_ids) != len(X):
                raise ValueError("GAME_ID must have one value per game")
            ids = pd.Series(np.asarray(game_ids))
            if ids.isna().any() or ids.astype(str).duplicated().any():
                raise ValueError("GAME_ID must be nonmissing and unique")
            order_frame['game_id'] = ids.astype(str)
            sort_keys.append('game_id')
        order = order_frame.sort_values(sort_keys, kind='stable')['position'].to_numpy()
        ordered_dates = dates.iloc[order].to_numpy()
        requested_train = len(X) - int(np.ceil(len(X) * self.test_size))
        if requested_train < 1:
            raise ValueError("test_size leaves no training games")
        cutoff = ordered_dates[requested_train]
        split_at = int((ordered_dates < cutoff).sum())
        if split_at == 0:
            raise ValueError("Cannot split into nonempty periods without sharing a GAME_DATE")
        train_indices, test_indices = order[:split_at], order[split_at:]
        
        return TrainTestData(
            X_train=X[train_indices],
            X_test=X[test_indices],
            y_train=y[train_indices],
            y_test=y[test_indices],
            feature_cols=[],  # Set by caller if needed
            train_dates=ordered_dates[:split_at],
            test_dates=ordered_dates[split_at:]
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
        
        Requires valid GAME_DATE values and sorts even unordered cache inputs.
        No calendar date is shared between the two resulting periods.
        Scaling is NOT included here as different models may need different scalers.
        
        Args:
            df: Matchup dataframe
            
        Returns:
            Tuple of (TrainTestData, feature_cols list)
        """
        validate_dataframe(df, required_cols=['GAME_DATE'])
        feature_cols = self.get_feature_columns(df)
        X, y = self.prepare_features(df, feature_cols)
        
        data = self.chronological_split(X, y, df['GAME_DATE'],
                                        df['GAME_ID'] if 'GAME_ID' in df else None)
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
