"""
Prediction Pipeline
===================
Orchestrates the prediction process with injectable dependencies.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from .features import FeatureComputer, GameFeatures
from .loader import LoadedEnsemble


@dataclass
class PredictionResult:
    """Result of a single game prediction"""
    home_win_probability: float
    away_win_probability: float
    predicted_winner: str  # 'HOME' or 'AWAY'
    confidence: float  # 0 to 1
    individual_predictions: List[float]
    prediction_std: float  # Standard deviation between models
    model_agreement: float = 0.0  # 0 if models disagree, 1 if all agree
    
    # Optional calibrated values
    calibrated_probability: Optional[float] = None
    stacked_probability: Optional[float] = None
    
    @property
    def is_home_favorite(self) -> bool:
        return self.predicted_winner == 'HOME'
    
    @property
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.3
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'home_win_probability': self.home_win_probability,
            'away_win_probability': self.away_win_probability,
            'predicted_winner': self.predicted_winner,
            'confidence': self.confidence,
            'individual_predictions': self.individual_predictions,
            'prediction_std': self.prediction_std,
            'model_agreement': self.model_agreement,
            'calibrated_probability': self.calibrated_probability,
            'stacked_probability': self.stacked_probability,
        }


class PredictionPipeline:
    """
    Orchestrates game predictions using the ensemble.
    
    Designed for testability:
    - All dependencies are injectable
    - Pure functions for core logic
    - No direct I/O operations
    
    Example usage:
        ensemble = ModelLoader().load_ensemble()
        pipeline = PredictionPipeline(ensemble)
        result = pipeline.predict(home_features, away_features)
    
    For testing:
        mock_ensemble = create_mock_ensemble()
        pipeline = PredictionPipeline(mock_ensemble)
    """
    
    def __init__(self, 
                 ensemble: LoadedEnsemble,
                 feature_computer: Optional[FeatureComputer] = None):
        """
        Initialize prediction pipeline.
        
        Args:
            ensemble: Loaded ensemble models
            feature_computer: Optional feature computer (for computing feature vectors)
        """
        self.ensemble = ensemble
        self.feature_computer = feature_computer or FeatureComputer()
    
    def get_model_prediction(self, 
                            model: Any, 
                            scaler: Any,
                            model_type: str,
                            features: np.ndarray) -> float:
        """
        Get prediction from a single model.
        
        Args:
            model: Trained model
            scaler: Scaler for this model
            model_type: Type of model ('xgboost', 'random_forest', etc.)
            features: Feature vector (unscaled)
            
        Returns:
            Probability of home win
        """
        # Scale features
        features_scaled = scaler.transform(features)
        
        if model_type == 'xgboost':
            return float(model.predict_proba(features_scaled)[0][1])
        
        elif model_type == 'random_forest':
            # Apply feature selection if present
            if hasattr(model, '_top_feature_indices'):
                features_scaled = features_scaled[:, model._top_feature_indices]
            return float(model.predict_proba(features_scaled)[0][1])
        
        elif model_type == 'logistic':
            return float(model.predict_proba(features_scaled)[0][1])
        
        else:  # keras
            return float(model.predict(features_scaled, verbose=0)[0][0])
    
    def aggregate_predictions(self,
                             predictions: List[float],
                             weights: Optional[List[float]] = None) -> float:
        """
        Aggregate predictions from multiple models.
        
        Args:
            predictions: List of probability predictions
            weights: Optional weights for weighted average
            
        Returns:
            Aggregated probability
        """
        if weights is None or len(weights) != len(predictions):
            # Fall back to simple average if weights are missing or mismatched
            return float(np.mean(predictions))
        else:
            weights = np.array(weights)
            predictions = np.array(predictions)
            weights = weights / weights.sum()  # Normalize
            return float(np.average(predictions, weights=weights))
    
    def compute_model_agreement(self, predictions: List[float]) -> float:
        """
        Compute how much the models agree based on standard deviation.
        
        Lower std = models predicting similar probabilities = higher agreement.
        
        Returns:
            Agreement score (0 to 1, where 1 = all models predict exact same probability)
        """
        # Agreement = 1 - std (higher when models predict similar probabilities)
        return float(1 - np.std(predictions))
    
    def apply_stacking(self, 
                      predictions: List[float],
                      elo_diff: Optional[float] = None) -> Optional[float]:
        """
        Apply stacking meta-model if available.
        
        Args:
            predictions: Individual model predictions
            elo_diff: Optional ELO difference feature
            
        Returns:
            Stacked probability or None if stacking not available
        """
        if self.ensemble.meta_clf is None:
            return None
        
        # Build meta-features
        raw_probs = np.array(predictions).reshape(1, -1)
        confidences = np.abs(raw_probs - 0.5)
        
        meta_features = [raw_probs, confidences]
        
        if elo_diff is not None:
            meta_features.append(np.array([[elo_diff]]))
        
        meta_X = np.hstack(meta_features)
        
        # Get stacked prediction
        stacked_prob = self.ensemble.meta_clf.predict_proba(meta_X)[0, 1]
        
        # Apply Platt calibration if available
        if self.ensemble.platt is not None:
            stacked_prob = self.ensemble.platt.predict_proba(
                np.array([[stacked_prob]])
            )[0, 1]
        
        return float(stacked_prob)
    
    def predict_from_features(self, 
                             features: np.ndarray,
                             elo_diff: Optional[float] = None) -> PredictionResult:
        """
        Make prediction from pre-computed feature vector.
        
        Args:
            features: Feature vector (shape: [1, n_features])
            elo_diff: Optional ELO difference for stacking
            
        Returns:
            PredictionResult
        """
        # Get predictions from each model
        predictions = []
        for model, scaler, model_type in zip(
            self.ensemble.models, 
            self.ensemble.scalers, 
            self.ensemble.model_types
        ):
            pred = self.get_model_prediction(model, scaler, model_type, features)
            predictions.append(pred)
        
        # Aggregate
        ensemble_prob = self.aggregate_predictions(
            predictions, 
            self.ensemble.ensemble_weights
        )
        
        # Compute agreement
        agreement = self.compute_model_agreement(predictions)
        
        # Apply stacking if available
        stacked_prob = self.apply_stacking(predictions, elo_diff)
        
        # Determine winner
        threshold = self.ensemble.ensemble_threshold or 0.5
        predicted_winner = 'HOME' if ensemble_prob > threshold else 'AWAY'
        
        return PredictionResult(
            home_win_probability=ensemble_prob,
            away_win_probability=1 - ensemble_prob,
            predicted_winner=predicted_winner,
            confidence=abs(ensemble_prob - 0.5) * 2,
            individual_predictions=predictions,
            prediction_std=float(np.std(predictions)),
            model_agreement=agreement,
            stacked_probability=stacked_prob
        )
    
    def predict(self,
               home_features: Dict[str, float],
               away_features: Dict[str, float],
               h2h_home: Optional[Dict] = None,
               h2h_away: Optional[Dict] = None,
               home_standings: Optional[Dict] = None,
               away_standings: Optional[Dict] = None,
               home_odds: Optional[Dict] = None,
               away_odds: Optional[Dict] = None,
               elo_diff: Optional[float] = None) -> PredictionResult:
        """
        Make prediction from feature dictionaries.
        
        Args:
            home_features: Home team rolling features
            away_features: Away team rolling features
            h2h_*: Head-to-head features
            *_standings: Standings features
            *_odds: Odds features
            elo_diff: ELO difference (home - away)
            
        Returns:
            PredictionResult
        """
        # Build feature vector
        features = self.feature_computer.build_feature_vector(
            home_features=home_features,
            away_features=away_features,
            feature_cols=self.ensemble.feature_cols,
            h2h_home=h2h_home,
            h2h_away=h2h_away,
            home_standings=home_standings,
            away_standings=away_standings,
            home_odds=home_odds,
            away_odds=away_odds
        )
        
        return self.predict_from_features(features, elo_diff)
    
    def batch_predict(self, 
                     games: List[Dict]) -> List[PredictionResult]:
        """
        Predict multiple games.
        
        Args:
            games: List of game feature dicts
            
        Returns:
            List of PredictionResult
        """
        results = []
        for game in games:
            result = self.predict(
                home_features=game.get('home_features', {}),
                away_features=game.get('away_features', {}),
                h2h_home=game.get('h2h_home'),
                h2h_away=game.get('h2h_away'),
                home_standings=game.get('home_standings'),
                away_standings=game.get('away_standings'),
                home_odds=game.get('home_odds'),
                away_odds=game.get('away_odds'),
                elo_diff=game.get('elo_diff')
            )
            results.append(result)
        return results


def predict_with_ensemble_compat(models, scalers, feature_cols, matchup_df, game_idx, model_types):
    """
    Backward-compatible wrapper for predict_with_ensemble.
    
    Maintains the original interface while using the refactored implementation.
    """
    # Create ensemble container
    ensemble = LoadedEnsemble(
        models=models,
        scalers=scalers,
        feature_cols=feature_cols,
        model_types=model_types
    )
    
    # Get game features from dataframe
    game_features = matchup_df[feature_cols].iloc[game_idx:game_idx+1].values
    
    # Create pipeline and predict
    pipeline = PredictionPipeline(ensemble)
    result = pipeline.predict_from_features(game_features)
    
    # Return in original format
    return {
        'home_win_probability': result.home_win_probability,
        'away_win_probability': result.away_win_probability,
        'predicted_winner': result.predicted_winner,
        'confidence': result.confidence,
        'individual_predictions': result.individual_predictions,
        'prediction_std': result.prediction_std
    }
