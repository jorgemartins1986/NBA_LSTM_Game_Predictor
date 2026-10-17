"""Regression tests for unordered cache rows leaking future dates into training."""

import numpy as np
import pandas as pd
import pytest

from src.training.data_prep import DataPreparation
from src.training.ensemble import EnsembleConfig, EnsembleTrainer


def games():
    return pd.DataFrame({
        'GAME_ID': [f'{i:03}' for i in range(10)],
        'GAME_DATE': ['2024-01-01'] + ['2024-01-02'] * 2 + ['2024-01-03'] * 2
                     + ['2024-01-04'] * 2 + ['2024-01-05'] * 3,
        'HOME_WIN': np.arange(10) % 2,
        'ROW_MARKER': np.arange(10),
    })


@pytest.mark.parametrize('order', ['shuffled', 'reversed'])
def test_cache_order_cannot_change_date_partition_or_label_alignment(order):
    original = games()
    frame = original.sample(frac=1, random_state=3) if order == 'shuffled' else original.iloc[::-1]
    before = frame.copy(deep=True)
    data, features = DataPreparation().prepare_for_training(frame)
    assert features == ['ROW_MARKER']
    np.testing.assert_array_equal(data.X_train[:, 0], np.arange(7))
    np.testing.assert_array_equal(data.X_test[:, 0], np.arange(7, 10))
    np.testing.assert_array_equal(data.y_train, np.arange(7) % 2)
    np.testing.assert_array_equal(data.y_test, np.arange(7, 10) % 2)
    assert data.train_dates.max() < data.test_dates.min()
    assert set(data.train_dates).isdisjoint(data.test_dates)
    pd.testing.assert_frame_equal(frame, before)


def test_boundary_timestamps_on_same_calendar_date_stay_together():
    frame = games()
    frame.loc[7:, 'GAME_DATE'] = ['2024-01-05 01:00:00', '2024-01-05 12:00:00', '2024-01-05 23:59:00']
    frame['GAME_DATE'] = pd.to_datetime(frame.GAME_DATE, format='mixed')
    data, _ = DataPreparation().prepare_for_training(frame.sample(frac=1, random_state=4))
    assert len(data.X_test) == 3  # The requested 20% must not divide January 5.
    assert data.train_dates.max() == np.datetime64('2024-01-04')
    assert data.test_dates.min() == np.datetime64('2024-01-05')


def test_missing_date_column_fails_closed():
    with pytest.raises(ValueError, match='GAME_DATE'):
        DataPreparation().prepare_for_training(games().drop(columns='GAME_DATE'))


@pytest.mark.parametrize('bad_date', [None, pd.NaT, 'not-a-date', 12345])
def test_invalid_date_values_fail_closed(bad_date):
    frame = games()
    frame.loc[3, 'GAME_DATE'] = bad_date
    with pytest.raises(ValueError, match='GAME_DATE'):
        DataPreparation().prepare_for_training(frame)


@pytest.mark.parametrize('invalid_id', [None, '000'])
def test_duplicate_or_missing_game_id_fails_closed(invalid_id):
    frame = games()
    frame.loc[1, 'GAME_ID'] = invalid_id
    with pytest.raises(ValueError, match='GAME_ID'):
        DataPreparation().prepare_for_training(frame)


def test_one_calendar_date_cannot_be_split_between_training_and_validation():
    frame = games()
    frame['GAME_DATE'] = '2024-01-01'
    with pytest.raises(ValueError, match='sharing a GAME_DATE'):
        DataPreparation().prepare_for_training(frame)


@pytest.mark.parametrize('size', [0, 1, -.1, 1.1])
def test_invalid_test_fraction_rejected(size):
    with pytest.raises(ValueError, match='test_size'):
        DataPreparation(test_size=size)


def test_scaler_is_fit_only_on_older_dates():
    frame = games().sample(frac=1, random_state=8)
    prep = DataPreparation()
    data, _ = prep.prepare_for_training(frame)
    scaled = prep.scale_features(data)
    assert scaled.scaler.mean_[0] == 3  # Mean of markers 0..6, not all rows.
    assert scaled.scaler.mean_[0] != frame.ROW_MARKER.mean()


def test_training_evaluation_and_stacking_reconstruct_identical_partitions():
    class RecordingPreparation(DataPreparation):
        def __init__(self):
            super().__init__()
            self.seen = []

        def prepare_for_training(self, frame):
            result = super().prepare_for_training(frame)
            self.seen.append(result[0])
            return result

    frame = pd.DataFrame({'GAME_ID': [f'{i:03}' for i in range(120)],
                          'GAME_DATE': pd.date_range('2024-01-01', periods=40).repeat(3),
                          'HOME_WIN': np.arange(120) % 2, 'ROW_MARKER': np.arange(120)})
    prep = RecordingPreparation()
    trainer = EnsembleTrainer(EnsembleConfig(architectures=['logistic'], verbose=False), data_prep=prep)
    result = trainer.train(frame.sample(frac=1, random_state=2))
    trainer.train_stacking(result, frame.sample(frac=1, random_state=6))
    trainer.evaluate(result, frame.sample(frac=1, random_state=4))
    assert len(prep.seen) == 1  # One snapshot, never rebuilt from later arguments.
    periods = result.partitions
    assert periods.base_train.dates.max() < periods.early_stopping.dates.min()
    assert periods.early_stopping.dates.max() < periods.stacker_fit.dates.min()
    assert periods.stacker_fit.dates.max() < periods.calibrator_fit.dates.min()
    assert periods.calibrator_fit.dates.max() < periods.final_test.dates.min()
    assert result.scalers[0].mean_[0] == periods.base_train.X.mean()


def test_split_requires_aligned_dates_and_features():
    with pytest.raises(ValueError, match='matching lengths'):
        DataPreparation().chronological_split(np.ones((3, 1)), np.array([0, 1, 0]), ['2024-01-01'])


def test_legacy_training_uses_same_partition_before_fitting_or_writing(monkeypatch):
    from src.nba_predictor import NBAPredictor

    class SplitObserved(Exception):
        pass

    predictor = NBAPredictor()

    def inspect_training(matrix):
        np.testing.assert_array_equal(matrix[:, 0], np.arange(5))
        raise SplitObserved()

    monkeypatch.setattr(predictor.scaler, 'fit_transform', inspect_training)
    with pytest.raises(SplitObserved):
        predictor.train(games().sample(frac=1, random_state=7))


def test_split_rejects_single_row_or_fraction_leaving_no_training_rows():
    with pytest.raises(ValueError, match='At least two'):
        DataPreparation().prepare_for_training(games().iloc[:1])
    with pytest.raises(ValueError, match='no training games'):
        DataPreparation(test_size=.99).prepare_for_training(games())


def test_supplied_game_ids_must_match_feature_count():
    with pytest.raises(ValueError, match='one value per game'):
        DataPreparation().chronological_split(np.ones((3, 1)), np.array([0, 1, 0]),
            pd.date_range('2024-01-01', periods=3), ['1'])
