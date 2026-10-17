"""Verify that no holdout inputs or labels can affect any fitted component."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.training.data_prep import DataPreparation
from src.training.ensemble import EnsembleConfig, EnsembleResult, EnsembleTrainer
from src.training.partitions import prepare_partitions


def games(n=1000):
    rng = np.random.default_rng(42)
    return pd.DataFrame({'GAME_ID': [f'{i:05}' for i in range(n)],
        'GAME_DATE': pd.date_range('2020-01-01', periods=n // 2).repeat(2),
        'HOME_WIN': np.arange(n) % 2, 'ROW_MARKER': np.arange(n, dtype=float),
        'SIGNAL': rng.normal(size=n)})


def trainer():
    return EnsembleTrainer(EnsembleConfig(architectures=['logistic'], verbose=False))


def fitted(frame=None):
    coordinator = trainer()
    result = coordinator.train(games() if frame is None else frame)
    coordinator.train_stacking(result)
    return coordinator, result


def test_five_periods_are_date_disjoint_complete_and_shuffle_invariant():
    frame = games()
    a = prepare_partitions(frame, DataPreparation(test_size=.1))
    b = prepare_partitions(frame.sample(frac=1, random_state=2), DataPreparation(test_size=.1))
    assert a.input_sha256 == b.input_sha256
    names = list(a.summary())
    assert [len(getattr(a, key).y) for key in names] == [700, 100, 50, 50, 100]
    for previous, current in zip(names, names[1:]):
        assert getattr(a, previous).dates.max() < getattr(a, current).dates.min()
    combined = np.concatenate([getattr(a, key).X[:, 0] for key in names])
    np.testing.assert_array_equal(combined, np.arange(len(frame)))


def test_each_fitting_stage_receives_only_its_own_period(monkeypatch):
    coordinator = trainer()
    result = coordinator.train(games())
    p = result.partitions
    assert result.scalers[0].mean_[0] == p.base_train.X[:, 0].mean()
    assert result.training_results[0].y_test.tolist() == p.early_stopping.y.tolist()
    seen = []
    original = coordinator._period_predictions

    def record(result, X):
        seen.append(X[:, 0].copy())
        return original(result, X)

    monkeypatch.setattr(coordinator, '_period_predictions', record)
    coordinator.train_stacking(result)
    np.testing.assert_array_equal(seen[0], p.stacker_fit.X[:, 0])
    np.testing.assert_array_equal(seen[1], p.calibrator_fit.X[:, 0])
    assert len(seen) == 2
    assert result.final_evaluations == {}
    coordinator.evaluate(result)
    np.testing.assert_array_equal(seen[2], p.final_test.X[:, 0])
    for name, evaluated in result.final_evaluations.items():
        np.testing.assert_array_equal(evaluated.y_true, p.final_test.y)
        assert evaluated.metrics()['games'] == len(p.final_test.y)


def test_changing_final_holdout_features_and_labels_cannot_change_fitted_models():
    frame = games()
    changed = frame.copy()
    final = changed.index >= 900
    changed.loc[final, 'HOME_WIN'] = 1 - changed.loc[final, 'HOME_WIN']
    changed.loc[final, ['ROW_MARKER', 'SIGNAL']] = 1000000.
    _, original = fitted(frame)
    _, modified = fitted(changed)
    np.testing.assert_array_equal(original.models[0].coef_, modified.models[0].coef_)
    np.testing.assert_array_equal(original.scalers[0].mean_, modified.scalers[0].mean_)
    np.testing.assert_array_equal(original.meta_clf.coef_, modified.meta_clf.coef_)
    np.testing.assert_array_equal(original.platt.coef_, modified.platt.coef_)


def test_final_model_metrics_use_same_games_and_distinguish_mean_from_ensemble(monkeypatch):
    coordinator = EnsembleTrainer(EnsembleConfig(architectures=['logistic'] * 4, verbose=False))
    result = coordinator.train(games())
    coordinator.train_stacking(result)
    captured = []
    coordinator.config.verbose = True
    coordinator.print_fn = captured.append
    raw = np.tile([[.1, .6, .2, .8], [.9, .8, .4, .9],
                   [.9, .2, .3, .2], [.1, .7, .3, .7]], (25, 1))
    calls = []

    def predictions(received_result, X):
        assert received_result is result
        np.testing.assert_array_equal(X, result.partitions.final_test.X)
        calls.append(X)
        return raw

    monkeypatch.setattr(coordinator, '_period_predictions', predictions)
    coordinator.evaluate(result)
    assert len(calls) == 1  # Reporting reuses predictions; it does not refit/infer again.
    assert len(result.final_model_evaluations) == 4
    y = result.partitions.final_test.y
    expected_accuracies = [.5, .75, .5, .75]
    for i, evaluated in enumerate(result.final_model_evaluations):
        np.testing.assert_array_equal(evaluated.y_true, y)
        np.testing.assert_array_equal(evaluated.y_pred_prob, raw[:, i])
        scores = evaluated.metrics()
        assert scores['accuracy'] == expected_accuracies[i]
        assert scores['brier'] == pytest.approx(np.mean((raw[:, i] - y) ** 2))
        assert scores['log_loss'] == pytest.approx(-np.mean(
            y * np.log(raw[:, i]) + (1 - y) * np.log(1 - raw[:, i])))
        assert any(f'Model {i + 1} (logistic)' in line and
                   f'{int(100 * expected_accuracies[i])} / 100' in line for line in captured)
    assert result.ensemble_accuracy == .75
    assert any('mean of individual accuracies: 62.50%' in line for line in captured)
    assert any('+12.50 percentage points' in line for line in captured)
    assert any('100 games, 2021-03-26 to 2021-05-14' in line for line in captured)


def test_frozen_training_snapshot_is_not_replaced_by_later_dataframe_arguments():
    frame = games()
    coordinator = trainer()
    result = coordinator.train(frame)
    expected = result.partitions.final_test.y.copy()
    frame.loc[:, 'HOME_WIN'] = 1 - frame.HOME_WIN
    coordinator.train_stacking(result, frame)
    evaluated = coordinator.evaluate(result, frame)
    np.testing.assert_array_equal(evaluated.y_true, expected)


def test_evaluation_before_stacking_and_refitting_after_evaluation_are_rejected():
    coordinator = trainer()
    result = coordinator.train(games())
    with pytest.raises(ValueError, match='before inspecting'):
        coordinator.evaluate(result)
    coordinator.train_stacking(result)
    coordinator.evaluate(result)
    with pytest.raises(ValueError, match='Cannot refit'):
        coordinator.train_stacking(result)


def test_unpartitioned_results_and_changed_feature_schema_are_rejected():
    coordinator = trainer()
    result = EnsembleResult([], [], [], [], [])
    with pytest.raises(ValueError, match='five chronological periods'):
        coordinator.train_stacking(result)
    result = coordinator.train(games())
    result.feature_cols = ['changed']
    with pytest.raises(ValueError, match='Feature schema'):
        coordinator.train_stacking(result)


def test_incomplete_model_set_is_rejected():
    coordinator, result = fitted()
    result.scalers = []
    with pytest.raises(ValueError, match='matching nonzero lengths'):
        coordinator.evaluate(result)


@pytest.mark.parametrize('invalid', [float('nan'), 1.1])
def test_invalid_model_probabilities_stop_stacking(monkeypatch, invalid):
    from types import SimpleNamespace
    coordinator = trainer()
    result = coordinator.train(games())
    coordinator.trainer_factory = SimpleNamespace(create=lambda kind: SimpleNamespace(
        predict_proba=lambda model, X: np.full(len(X), invalid)))
    with pytest.raises(ValueError, match='Invalid model probabilities'):
        coordinator.train_stacking(result)


def test_single_class_final_holdout_is_scored_without_moving_boundaries():
    frame = games()
    frame.loc[900:, 'HOME_WIN'] = 1
    coordinator, result = fitted(frame)
    evaluated = coordinator.evaluate(result)
    assert len(evaluated.y_true) == 100
    assert evaluated.y_true.tolist() == [1] * 100


@pytest.mark.parametrize('bad', [0, -.1, float('nan'), float('inf'), .9])
def test_invalid_period_fractions_fail_closed(bad):
    with pytest.raises(ValueError, match='fractions'):
        prepare_partitions(games(), DataPreparation(.1), validation_size=bad)


def test_insufficient_dates_and_single_class_fitting_periods_are_rejected():
    with pytest.raises(ValueError, match='five nonempty'):
        prepare_partitions(games(8), DataPreparation(.1))
    frame = games()
    frame.loc[800:849, 'HOME_WIN'] = 1
    with pytest.raises(ValueError, match='stacker fitting'):
        prepare_partitions(frame, DataPreparation(.1))


@pytest.mark.parametrize('column,value', [('SIGNAL', float('inf')), ('HOME_WIN', 2)])
def test_nonfinite_features_and_nonbinary_labels_are_rejected(column, value):
    frame = games()
    frame.loc[2, column] = value
    with pytest.raises(ValueError, match='finite features and binary'):
        prepare_partitions(frame, DataPreparation(.1))


def test_save_records_periods_cutoff_metrics_and_actual_artifact_hashes(tmp_path):
    coordinator, result = fitted()
    coordinator.evaluate(result)
    coordinator.save(result, output_dir=tmp_path)
    manifest = json.loads((tmp_path / 'ensemble_training_manifest.json').read_text())
    assert manifest['protocol'] == 'chronological-five-period-v1'
    assert manifest['periods'] == result.partitions.summary()
    assert manifest['input_sha256'] == result.partitions.input_sha256
    assert manifest['prediction_method'] == 'average'
    assert manifest['method_selected_using_final_test'] is False
    assert manifest['final_holdout_metrics']['average']['accuracy'] == result.ensemble_accuracy
    assert manifest['final_holdout_mean_individual_accuracy'] == result.final_model_evaluations[0].accuracy
    individual = manifest['final_holdout_individual_metrics']
    assert len(individual) == 1
    assert individual[0]['model_index'] == 1
    assert individual[0]['model_type'] == 'logistic'
    for key, value in result.final_model_evaluations[0].metrics().items():
        assert individual[0][key] == value
    assert individual[0]['correct'] == int(np.sum(
        result.final_model_evaluations[0].y_pred == result.partitions.final_test.y))
    for name, digest in manifest['artifact_sha256'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest


def test_save_without_final_evaluation_or_with_stale_weights_fails_before_writing(tmp_path):
    coordinator, result = fitted()
    with pytest.raises(ValueError, match='Evaluate all'):
        coordinator.save(result, output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []
    coordinator.evaluate(result)
    weights = tmp_path / 'ensemble_weights.pkl'
    weights.write_bytes(b'old custom weights')
    with pytest.raises(ValueError, match='revalidated'):
        coordinator.save(result, output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == [weights]
