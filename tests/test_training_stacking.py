"""Stacking and calibration are fitted on separate frozen periods."""

import numpy as np
import pandas as pd
import pytest
from src.training.ensemble import EnsembleConfig, EnsembleTrainer


@pytest.fixture
def stacking_setup():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(1200, 5)), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
    frame['HOME_WIN'] = np.arange(len(frame)) % 2
    frame['GAME_DATE'] = pd.date_range('2020-01-01', periods=len(frame))
    trainer = EnsembleTrainer(EnsembleConfig(architectures=['logistic', 'logistic'], verbose=False))
    result = trainer.train(frame)
    return trainer, result, frame


class TestTrainStacking:
    def test_sets_meta_clf_and_platt(self, stacking_setup):
        trainer, result, frame = stacking_setup
        updated = trainer.train_stacking(result, frame)
        assert updated.meta_clf is not None
        assert updated.platt is not None
        assert updated.final_evaluations == {}

    def test_meta_clf_is_fitted_logistic(self, stacking_setup):
        trainer, result, _ = stacking_setup
        trainer.train_stacking(result)
        assert result.meta_clf.coef_.shape[1] == 4

    def test_platt_is_fitted_logistic(self, stacking_setup):
        trainer, result, _ = stacking_setup
        trainer.train_stacking(result)
        assert result.platt.coef_.shape[1] == 1

    def test_meta_clf_produces_valid_probabilities(self, stacking_setup):
        trainer, result, _ = stacking_setup
        trainer.train_stacking(result)
        p = result.meta_clf.predict_proba([[.7, .6, .2, .1]])[0, 1]
        assert 0 <= p <= 1

    def test_platt_calibrates_probabilities(self, stacking_setup):
        trainer, result, _ = stacking_setup
        trainer.train_stacking(result)
        assert 0 <= result.platt.predict_proba([[.75]])[0, 1] <= 1

    def test_logs_when_verbose(self, stacking_setup):
        trainer, result, _ = stacking_setup
        captured = []
        trainer.config.verbose = True
        trainer.print_fn = captured.append
        trainer.train_stacking(result)
        assert any('STACKING' in item for item in captured)
        assert not any('STACKED ACCURACY' in item for item in captured)
        assert not result.final_evaluations
