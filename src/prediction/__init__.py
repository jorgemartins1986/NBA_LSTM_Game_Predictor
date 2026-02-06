"""
Prediction Module
=================
Modular prediction components for the NBA ensemble predictor.
Designed for testability with dependency injection.
"""

from .features import FeatureComputer, GameFeatures
from .pipeline import PredictionPipeline, PredictionResult
from .loader import ModelLoader, LoadedEnsemble

__all__ = [
    'FeatureComputer',
    'GameFeatures',
    'PredictionPipeline',
    'PredictionResult',
    'ModelLoader',
    'LoadedEnsemble',
]
