"""
Training Module
===============
Modular training components for the NBA ensemble predictor.
Designed for testability with dependency injection and single-responsibility classes.
"""

from .data_prep import DataPreparation, TrainTestData
from .trainers import (
    ModelTrainer,
    XGBoostTrainer,
    RandomForestTrainer,
    LogisticTrainer,
    LSTMTrainer,
    TrainerFactory
)
from .ensemble import EnsembleTrainer, EnsembleResult, EnsembleConfig
from .evaluation import ModelEvaluator, EvaluationResult, save_classification_reports

__all__ = [
    'DataPreparation',
    'TrainTestData',
    'ModelTrainer',
    'XGBoostTrainer',
    'RandomForestTrainer',
    'LogisticTrainer',
    'LSTMTrainer',
    'TrainerFactory',
    'EnsembleTrainer',
    'EnsembleResult',
    'EnsembleConfig',
    'ModelEvaluator',
    'EvaluationResult',
    'save_classification_reports',
]
