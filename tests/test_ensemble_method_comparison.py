"""Checks for separation of evaluation periods and paired comparison arithmetic."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.compare_ensemble_methods import (
    chronological_blocks, meta_features, method_probabilities, metrics, paired_comparison,
)


def test_periods_are_chronological_date_disjoint_and_cover_every_game():
    frame = pd.DataFrame({'GAME_ID': [str(i) for i in range(200)],
                          'GAME_DATE': pd.date_range('2020-01-01', periods=100).repeat(2),
                          'HOME_WIN': [0, 1] * 100}).sample(frac=1, random_state=3)
    blocks = chronological_blocks(frame)
    assert sum(len(block) for block in blocks.values()) == len(frame)
    previous = None
    for block in blocks.values():
        assert block.GAME_DATE.is_monotonic_increasing
        if previous is not None:
            assert previous < block.GAME_DATE.min()
        previous = block.GAME_DATE.max()


def test_stacking_and_calibration_are_compared_independently_of_average():
    class Stacker:
        def predict_proba(self, matrix):
            np.testing.assert_allclose(matrix, meta_features(np.array([[.8, .6], [.3, .5]])))
            return np.array([[.8, .2], [.1, .9]])

    class Calibrator:
        def predict_proba(self, matrix):
            np.testing.assert_allclose(matrix[:, 0], [.2, .9])
            return np.array([[.7, .3], [.2, .8]])

    ensemble = SimpleNamespace(ensemble_weights=[3, 1], meta_clf=Stacker(), platt=Calibrator())
    output = method_probabilities(ensemble, np.array([[.8, .6], [.3, .5]]))
    np.testing.assert_allclose(output['average'], [.75, .35])
    np.testing.assert_allclose(output['stacking'], [.2, .9])
    np.testing.assert_allclose(output['calibrated_stacking'], [.3, .8])


def test_metrics_and_paired_disagreements_use_same_games_and_thresholds():
    y = np.array([1, 0, 1, 0])
    average = np.array([.6, .6, .4, .4])
    stack = np.array([.7, .3, .8, .2])
    dates = pd.date_range('2024-01-01', periods=4)
    assert metrics(y, average)['correct'] == 2
    assert metrics(y, stack)['correct'] == 4
    result = paired_comparison(y, average, stack, dates)
    assert result['stacking_only_correct'] == 2
    assert result['average_only_correct'] == 0
    assert result['both_correct'] == 2
    assert result['accuracy_difference'] == .5
    assert result['brier_difference'] < 0


def test_nba_only_summary_removes_same_ineligible_game_from_both_methods(monkeypatch):
    from scripts import compare_ensemble_methods as comparison

    source = pd.DataFrame({'GAME_ID': ['1', '2', '3', '4'],
                           'HOME_TEAM_ID': [1610612738] * 4,
                           'AWAY_TEAM_ID': [1610612747] * 3 + [999]})
    predictions = pd.DataFrame({'GAME_ID': ['1', '2', '3', '4'], 'HOME_WIN': [1, 0, 1, 0],
                                'GAME_DATE': pd.date_range('2025-01-01', periods=4),
                                'average': [.6, .7, .4, .1], 'stacking': [.7, .2, .8, .9],
                                'calibrated_stacking': [.7, .2, .8, .9]})
    responses = iter([source, predictions])
    monkeypatch.setattr(comparison.pd, 'read_csv', lambda *args, **kwargs: next(responses))
    report = {'clean_holdout': {'average_threshold': .5}}
    comparison.add_nba_only_results(report, Path('report.json'))
    result = report['clean_holdout']['nba_franchises_only']
    assert result['excluded_non_nba_games'] == 1
    assert result['methods']['average']['games'] == 3
    assert result['methods']['average']['correct'] == 1
    assert result['methods']['calibrated_stacking']['correct'] == 3
