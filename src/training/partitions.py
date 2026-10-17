"""Five chronological periods with a final holdout excluded from all fitting."""

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from .data_prep import DataPreparation, TrainTestData


@dataclass
class TrainingPeriod:
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray

    def summary(self):
        return {'games': len(self.y), 'start': str(pd.Timestamp(self.dates[0]).date()),
                'end': str(pd.Timestamp(self.dates[-1]).date())}


@dataclass
class TrainingPartitions:
    base_train: TrainingPeriod
    early_stopping: TrainingPeriod
    stacker_fit: TrainingPeriod
    calibrator_fit: TrainingPeriod
    final_test: TrainingPeriod
    feature_cols: list
    input_sha256: str

    def base_model_data(self):
        return TrainTestData(self.base_train.X, self.early_stopping.X,
                             self.base_train.y, self.early_stopping.y, self.feature_cols,
                             self.base_train.dates, self.early_stopping.dates)

    def summary(self):
        return {name: getattr(self, name).summary() for name in
                ('base_train', 'early_stopping', 'stacker_fit', 'calibrator_fit', 'final_test')}


def prepare_partitions(frame: pd.DataFrame, prep: DataPreparation,
                       validation_size=.1, stacking_size=.05, calibration_size=.05,
                       test_size=None) -> TrainingPartitions:
    """Use approximately 70/10/5/5/10 by default; whole dates stay together.

    prep.test_size defines the final holdout fraction. Boundaries depend only on
    dates/counts, never outcomes. All stages fail closed if boundaries collapse.
    """
    fractions = np.array([validation_size, stacking_size, calibration_size,
                          prep.test_size if test_size is None else test_size], dtype=float)
    if not np.isfinite(fractions).all() or (fractions <= 0).any() or fractions.sum() >= 1:
        raise ValueError('Period fractions must be positive and leave a nonempty base-training fraction')
    # Reuse the strict date/ID validation and deterministic ordering boundary.
    ordered, features = prep.prepare_for_training(frame)
    X = np.concatenate([ordered.X_train, ordered.X_test])
    y = np.concatenate([ordered.y_train, ordered.y_test])
    dates = np.concatenate([ordered.train_dates, ordered.test_dates])
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Training requires finite features and binary HOME_WIN labels')
    cumulative = np.round(np.cumsum([1 - fractions.sum(), *fractions[:3]]), 12)
    indices = [int(np.searchsorted(dates, dates[int(len(dates) * fraction)], side='left'))
               for fraction in cumulative]
    bounds = [0, *indices, len(dates)]
    if any(a >= b for a, b in zip(bounds, bounds[1:])):
        raise ValueError('Not enough distinct dates to create five nonempty date-disjoint periods')
    parts = [TrainingPeriod(X[a:b], y[a:b], dates[a:b]) for a, b in zip(bounds, bounds[1:])]
    for name, index in [('base training', 0), ('stacker fitting', 2), ('calibration', 3)]:
        if len(np.unique(parts[index].y)) != 2:
            raise ValueError(f'The {name} period must contain both HOME_WIN classes')
    digest = hashlib.sha256()
    for values in (X, y, dates.astype('datetime64[ns]').astype('int64')):
        digest.update(np.ascontiguousarray(values).tobytes())
    digest.update('\n'.join(features).encode('utf-8'))
    return TrainingPartitions(*parts, features, digest.hexdigest())
