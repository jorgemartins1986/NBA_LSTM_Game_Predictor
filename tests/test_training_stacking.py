"""
Tests for EnsembleTrainer.train_stacking
=========================================
Covers the stacking meta-model and Platt calibrator training logic.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from src.training.ensemble import (
    EnsembleConfig, EnsembleResult, EnsembleTrainer,
)
from src.training.trainers import TrainingResult
from src.training.data_prep import TrainTestData


@pytest.fixture
def stacking_setup():
    """
    Set up an EnsembleTrainer with mocked deps and a fake EnsembleResult
    containing two base models that return deterministic probabilities.
    """
    np.random.seed(42)
    n_test = 40

    # Deterministic test data
    X_test = np.random.randn(n_test, 5)
    y_test = np.concatenate([np.ones(20), np.zeros(20)])  # balanced

    mock_train_test = TrainTestData(
        X_train=np.random.randn(80, 5),
        X_test=X_test,
        y_train=np.random.randint(0, 2, 80),
        y_test=y_test,
        feature_cols=['f1', 'f2', 'f3', 'f4', 'f5'],
    )

    # Mock data_prep to return our data
    mock_data_prep = MagicMock()
    mock_data_prep.prepare_for_training.return_value = (mock_train_test, ['f1', 'f2', 'f3', 'f4', 'f5'])

    # Two mock scalers that pass through
    mock_scaler1 = MagicMock()
    mock_scaler1.transform.return_value = X_test
    mock_scaler2 = MagicMock()
    mock_scaler2.transform.return_value = X_test

    # Two mock base models
    model1 = MagicMock()
    model2 = MagicMock()

    # Mock trainer factory: model1 outputs mostly right, model2 is weaker
    preds1 = np.where(y_test == 1, np.random.uniform(0.6, 0.9, n_test),
                       np.random.uniform(0.1, 0.4, n_test))
    preds2 = np.where(y_test == 1, np.random.uniform(0.5, 0.7, n_test),
                       np.random.uniform(0.3, 0.5, n_test))

    mock_factory = MagicMock()
    mock_trainer1 = MagicMock()
    mock_trainer1.predict_proba.return_value = preds1
    mock_trainer2 = MagicMock()
    mock_trainer2.predict_proba.return_value = preds2
    trainers_iter = iter([mock_trainer1, mock_trainer2])
    mock_factory.create.side_effect = lambda mt: next(trainers_iter)

    ensemble_result = EnsembleResult(
        models=[model1, model2],
        scalers=[mock_scaler1, mock_scaler2],
        feature_cols=['f1', 'f2', 'f3', 'f4', 'f5'],
        model_types=['xgboost', 'logistic'],
        training_results=[],
    )

    trainer = EnsembleTrainer(
        config=EnsembleConfig(verbose=False),
        data_prep=mock_data_prep,
        trainer_factory=mock_factory,
    )

    matchup_df = pd.DataFrame({'dummy': [1]})
    return trainer, ensemble_result, matchup_df


class TestTrainStacking:
    """Tests for EnsembleTrainer.train_stacking"""

    def test_sets_meta_clf_and_platt(self, stacking_setup):
        trainer, result, matchup_df = stacking_setup

        updated = trainer.train_stacking(result, matchup_df)

        assert updated.meta_clf is not None
        assert updated.platt is not None

    def test_meta_clf_is_fitted_logistic(self, stacking_setup):
        trainer, result, matchup_df = stacking_setup

        updated = trainer.train_stacking(result, matchup_df)

        # A fitted LogisticRegression has coef_ attribute
        assert hasattr(updated.meta_clf, 'coef_')
        # Meta features should be 2 * n_models = 4 columns
        assert updated.meta_clf.coef_.shape[1] == 4

    def test_platt_is_fitted_logistic(self, stacking_setup):
        trainer, result, matchup_df = stacking_setup

        updated = trainer.train_stacking(result, matchup_df)

        assert hasattr(updated.platt, 'coef_')
        # Platt takes a single column (meta-classifier probability)
        assert updated.platt.coef_.shape[1] == 1

    def test_meta_clf_produces_valid_probabilities(self, stacking_setup):
        trainer, result, matchup_df = stacking_setup

        updated = trainer.train_stacking(result, matchup_df)

        # Build test meta-features the same way the code does
        dummy_probs = np.array([[0.7, 0.6]])
        confidences = np.abs(dummy_probs - 0.5)
        meta_X = np.hstack([dummy_probs, confidences])

        proba = updated.meta_clf.predict_proba(meta_X)
        assert proba.shape == (1, 2)
        assert 0 <= proba[0, 1] <= 1

    def test_platt_calibrates_probabilities(self, stacking_setup):
        trainer, result, matchup_df = stacking_setup

        updated = trainer.train_stacking(result, matchup_df)

        # Feed a raw probability through Platt
        raw = np.array([[0.75]])
        calibrated = updated.platt.predict_proba(raw)[:, 1]
        assert 0 <= calibrated[0] <= 1

    def test_logs_when_verbose(self):
        """Verbose mode should log stacking progress."""
        np.random.seed(42)
        n_test = 30
        y_test = np.concatenate([np.ones(15), np.zeros(15)])

        mock_data_prep = MagicMock()
        mock_data_prep.prepare_for_training.return_value = (
            TrainTestData(
                X_train=np.random.randn(60, 3),
                X_test=np.random.randn(n_test, 3),
                y_train=np.random.randint(0, 2, 60),
                y_test=y_test,
                feature_cols=['a', 'b', 'c'],
            ),
            ['a', 'b', 'c']
        )

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.random.randn(n_test, 3)

        mock_factory = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.predict_proba.return_value = np.random.uniform(0.3, 0.7, n_test)
        mock_factory.create.return_value = mock_trainer

        captured = []
        trainer = EnsembleTrainer(
            config=EnsembleConfig(verbose=True),
            data_prep=mock_data_prep,
            trainer_factory=mock_factory,
            print_fn=lambda x: captured.append(x),
        )

        result = EnsembleResult(
            models=[MagicMock()],
            scalers=[mock_scaler],
            feature_cols=['a', 'b', 'c'],
            model_types=['xgboost'],
            training_results=[],
        )

        trainer.train_stacking(result, pd.DataFrame({'x': [1]}))

        log_text = '\n'.join(captured)
        assert 'STACKING' in log_text
        assert 'meta-classifier' in log_text.lower() or 'meta' in log_text.lower()
        assert 'STACKED ACCURACY' in log_text
