"""Read-only saved-model replay plus a separately trained chronological comparison.

Production models and history are never written. New experimental models live
only in memory; outputs go to reports/ by default.
"""

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from scripts.evaluate_saved_ensemble_history import model_manifest
from src.paths import MATCHUPS_CSV, REPORTS_DIR
from src.prediction import ModelLoader, PredictionPipeline
from src.training.data_prep import DataPreparation, TrainTestData
from src.training.ensemble import EnsembleConfig, EnsembleTrainer


def chronological_blocks(frame):
    """70/10/5/5/10 split; never put one calendar date in two blocks."""
    frame = frame.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    cutoffs = [frame.iloc[int(len(frame) * q)]['GAME_DATE'] for q in (.7, .8, .85, .9)]
    if len(set(cutoffs)) != 4:
        raise ValueError('Insufficient distinct dates for five disjoint periods')
    date = frame['GAME_DATE']
    groups = [frame[date < cutoffs[0]]]
    groups.extend(frame[(date >= lower) & (date < upper)]
                  for lower, upper in zip(cutoffs, cutoffs[1:]))
    groups.append(frame[date >= cutoffs[-1]])
    if any(group.empty or group['HOME_WIN'].nunique() != 2 for group in groups):
        raise ValueError('Every period must contain both home and away wins')
    return dict(zip(('base_train', 'early_stopping', 'stacker_fit', 'calibrator_fit', 'final_test'), groups))


def base_probabilities(ensemble, matrix):
    columns = []
    for model, scaler, kind in zip(ensemble.models, ensemble.scalers, ensemble.model_types):
        scaled = scaler.transform(matrix)
        if kind == 'random_forest' and hasattr(model, '_top_feature_indices'):
            scaled = scaled[:, model._top_feature_indices]
        if kind in ('xgboost', 'random_forest', 'logistic'):
            probability = model.predict_proba(scaled)[:, 1]
        else:
            probability = model.predict(scaled, batch_size=128, verbose=0).reshape(-1)
        columns.append(probability)
    raw = np.column_stack(columns)
    if not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise ValueError('Invalid base-model probabilities')
    return raw


def meta_features(raw):
    return np.hstack([raw, np.abs(raw - .5)])


def method_probabilities(ensemble, raw):
    weights = ensemble.ensemble_weights
    average = (raw.mean(axis=1) if weights is None or len(weights) != raw.shape[1]
               else np.average(raw, axis=1, weights=weights))
    stacked = ensemble.meta_clf.predict_proba(meta_features(raw))[:, 1]
    calibrated = ensemble.platt.predict_proba(stacked.reshape(-1, 1))[:, 1]
    return {'average': average, 'stacking': stacked, 'calibrated_stacking': calibrated}


def metrics(y, probability, threshold=.5):
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    correct = (probability > threshold) == y
    return {'games': len(y), 'correct': int(correct.sum()), 'accuracy': float(correct.mean()),
            'brier': float(np.mean((p - y) ** 2)),
            'log_loss': float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}


def paired_comparison(y, average, stacked, dates, average_threshold=.5):
    """Paired uncertainty, resampling calendar dates rather than individual games."""
    a = (average > average_threshold) == y
    s = (stacked > .5) == y
    frame = pd.DataFrame({'date': dates, 'n': 1, 'accuracy': s.astype(int) - a.astype(int),
                          'brier': (stacked - y) ** 2 - (average - y) ** 2})
    days = frame.groupby('date')[['n', 'accuracy', 'brier']].sum().to_numpy()
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(days), size=(3000, len(days)))
    totals = days[draws].sum(axis=1)
    from scipy.stats import binomtest
    wins, losses = int((s & ~a).sum()), int((a & ~s).sum())
    return {'stacking_only_correct': wins, 'average_only_correct': losses,
            'both_correct': int((s & a).sum()), 'both_wrong': int((~s & ~a).sum()),
            'accuracy_difference': float((s.astype(int) - a.astype(int)).mean()),
            'accuracy_difference_95pct_date_bootstrap': np.quantile(totals[:, 1] / totals[:, 0], [.025, .975]).tolist(),
            'brier_difference': float(frame.brier.mean()),
            'brier_difference_95pct_date_bootstrap': np.quantile(totals[:, 2] / totals[:, 0], [.025, .975]).tolist(),
            'mcnemar_exact_p': float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0}


def evaluate(ensemble, frame, label, output):
    matrix = frame[ensemble.feature_cols].to_numpy(dtype=float)
    y = frame.HOME_WIN.to_numpy(dtype=int)
    raw = base_probabilities(ensemble, matrix)
    probabilities = method_probabilities(ensemble, raw)
    pipeline = PredictionPipeline(ensemble)
    for i in sorted({0, len(frame) // 2, len(frame) - 1}):
        actual = pipeline.predict_from_features(matrix[i:i + 1])
        np.testing.assert_allclose(actual.home_win_probability, probabilities['average'][i], atol=1e-6)
        np.testing.assert_allclose(actual.stacked_probability, probabilities['calibrated_stacking'][i], atol=1e-6)
    threshold = ensemble.ensemble_threshold or .5
    report = {'label': label, 'start': str(frame.GAME_DATE.min().date()),
              'end': str(frame.GAME_DATE.max().date()), 'average_threshold': threshold,
              'stacking_threshold': .5,
              'methods': {name: metrics(y, p, threshold if name == 'average' else .5)
                          for name, p in probabilities.items()},
              'individual_models': {kind: metrics(y, raw[:, i]) for i, kind in enumerate(ensemble.model_types)},
              'always_home_accuracy': float(y.mean()),
              'paired_calibrated_vs_average': paired_comparison(
                  y, probabilities['average'], probabilities['calibrated_stacking'],
                  frame.GAME_DATE.to_numpy(), threshold)}
    per_game = frame[['GAME_ID', 'GAME_DATE', 'HOME_WIN']].copy()
    for name, probability in probabilities.items():
        per_game[name] = probability
    per_game.to_csv(output, index=False)
    print(json.dumps(report, indent=2), flush=True)
    return report


def fit_experiment(blocks, features):
    trainer = EnsembleTrainer(EnsembleConfig(verbose=False))
    train, stopping = blocks['base_train'], blocks['early_stopping']
    data = TrainTestData(train[features].to_numpy(), stopping[features].to_numpy(),
                         train.HOME_WIN.to_numpy(), stopping.HOME_WIN.to_numpy(), features)
    results = []
    for i, kind in enumerate(trainer.config.architectures):
        print(f'Fitting experimental {kind}; validation uses only the early-stopping period.', flush=True)
        results.append(trainer.train_single_model(kind, data, i))
    ensemble = SimpleNamespace(models=[r.model for r in results], scalers=[r.scaler for r in results],
        model_types=[r.model_type for r in results], feature_cols=features,
        ensemble_weights=None, ensemble_threshold=None)
    stack = blocks['stacker_fit']
    raw = base_probabilities(ensemble, stack[features].to_numpy())
    ensemble.meta_clf = LogisticRegression(C=1., class_weight='balanced', max_iter=1000, random_state=42)
    ensemble.meta_clf.fit(meta_features(raw), stack.HOME_WIN.to_numpy())
    calibrate = blocks['calibrator_fit']
    raw = base_probabilities(ensemble, calibrate[features].to_numpy())
    probabilities = ensemble.meta_clf.predict_proba(meta_features(raw))[:, 1]
    ensemble.platt = LogisticRegression(C=1., max_iter=1000, random_state=42)
    ensemble.platt.fit(probabilities.reshape(-1, 1), calibrate.HOME_WIN.to_numpy())
    return ensemble


def add_nba_only_results(report, output):
    """Report franchise-only samples, excluding cached international opponents.

    This only filters already generated paired predictions; it never refits or
    selects models. Training still uses the same cache as standard training.
    """
    from nba_api.stats.static import teams
    valid = {team['id'] for team in teams.get_teams()}
    source = pd.read_csv(MATCHUPS_CSV, dtype={'GAME_ID': str})
    eligible = set(source.loc[source.HOME_TEAM_ID.isin(valid) &
                              source.AWAY_TEAM_ID.isin(valid), 'GAME_ID'])
    for key, suffix in [('saved_model_diagnostic', '_saved_games.csv'), ('clean_holdout', '_holdout_games.csv')]:
        if key not in report:
            continue
        rows = pd.read_csv(output.with_name(output.stem + suffix), dtype={'GAME_ID': str},
                           parse_dates=['GAME_DATE'])
        included = rows[rows.GAME_ID.isin(eligible)]
        y = included.HOME_WIN.to_numpy()
        threshold = report[key]['average_threshold']
        report[key]['nba_franchises_only'] = {
            'excluded_non_nba_games': len(rows) - len(included),
            'methods': {name: metrics(y, included[name].to_numpy(), threshold if name == 'average' else .5)
                        for name in ('average', 'stacking', 'calibrated_stacking')},
            'paired_calibrated_vs_average': paired_comparison(y, included.average.to_numpy(),
                included.calibrated_stacking.to_numpy(), included.GAME_DATE.to_numpy(), threshold)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(Path(REPORTS_DIR) / 'ensemble_method_comparison.json'))
    parser.add_argument('--saved-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    before = model_manifest()
    source_digest = hashlib.sha256(Path(MATCHUPS_CSV).read_bytes()).hexdigest()
    frame = pd.read_csv(MATCHUPS_CSV, dtype={'GAME_ID': str}, parse_dates=['GAME_DATE'])
    if frame.GAME_ID.duplicated().any():
        raise ValueError('Duplicate games in matchup input')
    saved = ModelLoader().load_ensemble()
    if saved.n_models != 4 or saved.meta_clf is None or saved.platt is None:
        raise ValueError('Four saved models, a stacker, and a calibrator are required')
    if not np.isfinite(frame[saved.feature_cols].to_numpy()).all():
        raise ValueError('Nonfinite input features')
    report = {'input': {'path': MATCHUPS_CSV, 'sha256': source_digest, 'rows': len(frame),
                        'original_rows_date_ordered': bool(frame.GAME_DATE.is_monotonic_increasing)},
              'production_models': before, 'feature_count': len(saved.feature_cols),
              'limitations': ['Saved-model replay reuses possible training/validation/calibration data; it is not an independent holdout.',
                  'Clean experiment fits fresh models; its results do not directly certify the current saved artifacts.',
                  'All methods use the same cached training features; this does not test all live feature/season-policy differences.',
                  'Single chronological split and one random seed; no historical odds or betting ROI evaluated.']}
    config_trainer = EnsembleTrainer(EnsembleConfig(verbose=False))
    report['experiment_configuration'] = {
        'seed': 42, 'period_fractions': [0.7, 0.1, 0.05, 0.05, 0.1],
        'packages': {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'tensorflow', 'xgboost')},
        'base_trainers': {kind: asdict(config_trainer._get_trainer_config(kind, i))
                          for i, kind in enumerate(config_trainer.config.architectures)}}
    tail = frame.iloc[len(frame) - math.ceil(.2 * len(frame)):]
    report['saved_model_diagnostic'] = evaluate(saved, tail,
        'Saved artifacts on reconstructed last 20% of original row order: development diagnostic only',
        output.with_name(output.stem + '_saved_games.csv'))
    if not args.saved_only:
        blocks = chronological_blocks(frame)
        report['chronological_periods'] = {key: {'games': len(value),
            'start': str(value.GAME_DATE.min().date()), 'end': str(value.GAME_DATE.max().date())}
            for key, value in blocks.items()}
        print(json.dumps(report['chronological_periods'], indent=2), flush=True)
        # Persist the quick diagnostic before the longer isolated training run.
        output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        experimental = fit_experiment(blocks, saved.feature_cols)
        report['clean_holdout'] = evaluate(experimental, blocks['final_test'],
            'Fresh models; test dates unused for fitting, early stopping, stacking, or calibration',
            output.with_name(output.stem + '_holdout_games.csv'))
    if model_manifest() != before or hashlib.sha256(Path(MATCHUPS_CSV).read_bytes()).hexdigest() != source_digest:
        raise RuntimeError('Production model or input files changed during comparison')
    report['production_models_and_matchup_input_unchanged'] = True
    add_nba_only_results(report, output)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Comparison saved to {output}', flush=True)


if __name__ == '__main__':
    main()
