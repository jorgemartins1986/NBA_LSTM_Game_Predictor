"""
Model Evaluation
================
Functions and classes for evaluating trained models.
Pure functions with no side effects for testability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


@dataclass
class EvaluationResult:
    """Result of evaluating a model or ensemble"""
    accuracy: float
    y_true: np.ndarray
    y_pred: np.ndarray
    y_pred_prob: np.ndarray
    classification_report: str = ""
    individual_accuracies: Optional[List[float]] = None  # For ensemble
    model_names: Optional[List[str]] = None
    
    @property
    def precision_away(self) -> Optional[float]:
        """Extract away win precision from report"""
        return self._extract_metric('Away Win', 'precision')
    
    @property
    def precision_home(self) -> Optional[float]:
        """Extract home win precision from report"""
        return self._extract_metric('Home Win', 'precision')
    
    def _extract_metric(self, class_name: str, metric: str) -> Optional[float]:
        """Extract a metric from the classification report string"""
        for line in self.classification_report.split('\n'):
            if class_name in line:
                parts = line.split()
                try:
                    if metric == 'precision' and len(parts) > 2:
                        return float(parts[1])
                    elif metric == 'recall' and len(parts) > 3:
                        return float(parts[2])
                    elif metric == 'f1-score' and len(parts) > 4:
                        return float(parts[3])
                except (ValueError, IndexError):
                    pass
        return None


class ModelEvaluator:
    """
    Evaluates individual models and ensembles.
    
    All methods are stateless and can be tested independently.
    """
    
    @staticmethod
    def evaluate_predictions(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            y_pred_prob: Optional[np.ndarray] = None) -> EvaluationResult:
        """
        Evaluate predictions against ground truth.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (0 or 1)
            y_pred_prob: Optional probability predictions
            
        Returns:
            EvaluationResult with metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred,
                                       target_names=['Away Win', 'Home Win'])
        
        if y_pred_prob is None:
            y_pred_prob = y_pred.astype(float)
        
        return EvaluationResult(
            accuracy=accuracy,
            y_true=y_true,
            y_pred=y_pred,
            y_pred_prob=y_pred_prob,
            classification_report=report
        )
    
    @staticmethod
    def compute_ensemble_predictions(model_predictions: List[np.ndarray],
                                    weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Combine predictions from multiple models.
        
        Args:
            model_predictions: List of probability arrays from each model
            weights: Optional weights for weighted average
            
        Returns:
            Ensemble probability predictions
        """
        if weights is None:
            # Simple average
            return np.mean(model_predictions, axis=0)
        else:
            # Weighted average
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            return np.average(model_predictions, axis=0, weights=weights)
    
    @staticmethod
    def probabilities_to_predictions(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Convert probabilities to binary predictions.
        
        Args:
            probs: Probability of positive class (home win)
            threshold: Decision threshold (default 0.5)
            
        Returns:
            Binary predictions (0 or 1)
        """
        return (probs > threshold).astype(int)
    
    @staticmethod
    def compute_calibration_error(y_true: np.ndarray, 
                                  y_prob: np.ndarray,
                                  n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Lower is better. A well-calibrated model has prob predictions
        that match actual frequencies.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Expected Calibration Error (0 to 1)
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            bin_size = mask.sum() / len(y_true)
            
            ece += bin_size * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    @staticmethod
    def compute_model_agreement(predictions: List[np.ndarray]) -> np.ndarray:
        """
        Compute agreement between model predictions.
        
        High agreement = all models predict same direction.
        Low agreement = models disagree.
        
        Args:
            predictions: List of probability arrays
            
        Returns:
            Agreement score per sample (0 to 1)
        """
        # Convert to binary
        binary = [(p > 0.5).astype(int) for p in predictions]
        stacked = np.vstack(binary)
        
        # Agreement is how close to unanimous
        # 1.0 if all models agree, lower if they disagree
        votes = stacked.mean(axis=0)  # Fraction voting home
        agreement = 2 * np.abs(votes - 0.5)  # 0 if 50/50, 1 if unanimous
        
        return agreement
    
    def evaluate_ensemble(self, 
                         model_predictions: List[np.ndarray],
                         y_true: np.ndarray,
                         model_names: Optional[List[str]] = None,
                         weights: Optional[List[float]] = None) -> EvaluationResult:
        """
        Evaluate an ensemble of models.
        
        Args:
            model_predictions: List of probability arrays from each model
            y_true: True labels
            model_names: Optional list of model names
            weights: Optional ensemble weights
            
        Returns:
            EvaluationResult with ensemble metrics
        """
        # Compute individual accuracies
        individual_accs = []
        for preds in model_predictions:
            y_pred = self.probabilities_to_predictions(preds)
            acc = accuracy_score(y_true, y_pred)
            individual_accs.append(acc)
        
        # Compute ensemble predictions
        ensemble_probs = self.compute_ensemble_predictions(model_predictions, weights)
        ensemble_pred = self.probabilities_to_predictions(ensemble_probs)
        
        # Evaluate ensemble
        result = self.evaluate_predictions(y_true, ensemble_pred, ensemble_probs)
        result.individual_accuracies = individual_accs
        result.model_names = model_names or [f"Model_{i+1}" for i in range(len(model_predictions))]
        
        return result
    
    @staticmethod
    def compute_prediction_confidence(prob: float) -> float:
        """
        Compute confidence score from probability.
        
        Confidence is how far from 0.5 the prediction is.
        
        Args:
            prob: Probability of home win
            
        Returns:
            Confidence (0 to 1)
        """
        return abs(prob - 0.5) * 2
    
    @staticmethod
    def compute_prediction_stats(probs: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics for a set of predictions.
        
        Args:
            probs: Array of probabilities
            
        Returns:
            Dict with mean, std, min, max, etc.
        """
        return {
            'mean': float(np.mean(probs)),
            'std': float(np.std(probs)),
            'min': float(np.min(probs)),
            'max': float(np.max(probs)),
            'median': float(np.median(probs)),
        }
