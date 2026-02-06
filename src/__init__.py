"""
NBA Game Predictor Package
==========================
Ensemble machine learning models for predicting NBA game outcomes.

Modules:
- training: Model training components (trainers, data prep, evaluation)
- prediction: Prediction pipeline and feature computation
- nba_data_manager: Data fetching and caching
- nba_predictor: Feature engineering and base models
"""

from .paths import (
    PROJECT_ROOT, MODELS_DIR, CACHE_DIR, CONFIG_DIR, REPORTS_DIR,
    get_model_path, get_cache_path, get_config_path, get_report_path
)

# Re-export training module components
from .training import (
    DataPreparation, TrainTestData,
    ModelTrainer, XGBoostTrainer, RandomForestTrainer, LogisticTrainer, LSTMTrainer,
    TrainerFactory,
    EnsembleTrainer, EnsembleResult,
    ModelEvaluator, EvaluationResult
)

# Re-export prediction module components  
from .prediction import (
    FeatureComputer, GameFeatures,
    PredictionPipeline, PredictionResult,
    ModelLoader, LoadedEnsemble
)

__version__ = "2.0.0"  # Major version bump for refactoring
__author__ = "Jorge"
