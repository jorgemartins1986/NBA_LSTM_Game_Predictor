"""Compare rolling-history policies with the actual saved production ensemble.

Reads local data/models only. No retraining, API requests, or prediction-history
writes. Old-season replays are diagnostics because model training provenance is
incomplete; the latest sample is screened by saved-file modification dates.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nba_api.stats.static import teams

from src.paths import GAMES_CACHE_FILE, MODELS_DIR
from src.prediction import ModelLoader, PredictionPipeline
from src.prediction.nba_data import compute_head_to_head, get_recent_team_stats
from src.prediction.rolling_history import upcoming_team_stats
from src.season_config import filter_prediction_games


def predict_batch(ensemble, matrix):
    """Same scaling/selection/aggregation as PredictionPipeline, batched."""
    columns = []
    for model, scaler, kind in zip(ensemble.models, ensemble.scalers, ensemble.model_types):
        scaled = scaler.transform(matrix)
        if kind == 'random_forest' and hasattr(model, '_top_feature_indices'):
            scaled = scaled[:, model._top_feature_indices]
        if kind in ['xgboost', 'random_forest', 'logistic']:
            probability = model.predict_proba(scaled)[:, 1]
        else:
            probability = model.predict(scaled, batch_size=128, verbose=0).reshape(-1)
        columns.append(probability)
    raw = np.column_stack(columns)
    weights = ensemble.ensemble_weights
    if weights is None or len(weights) != raw.shape[1]:
        return raw.mean(axis=1)
    return np.average(raw, axis=1, weights=np.asarray(weights) / np.sum(weights))


def model_manifest():
    records = []
    for path in sorted(Path(MODELS_DIR).glob('*')):
        if not path.is_file() or path.suffix not in ['.json', '.pkl', '.keras']:
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        records.append({'name': path.name, 'sha256': digest.hexdigest(),
                        'modified_at': datetime.fromtimestamp(path.stat().st_mtime,
                            ZoneInfo('Europe/Lisbon')).isoformat()})
    return records


def target_games(raw, season, start=None, end=None, opening_limit=None):
    frame = raw[(raw['SEASON'] == season) & raw['WL'].isin(['W', 'L'])].copy()
    if opening_limit is not None:
        frame = frame[frame['SEASON_ID'].astype(str).str.startswith('2')]
        frame = frame.sort_values('GAME_DATE')
        frame['season_game_number'] = frame.groupby('TEAM_ID').cumcount() + 1
    if start is not None:
        frame = frame[frame['GAME_DATE'] > pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame['GAME_DATE'] < pd.Timestamp(end)]
    valid_ids = {team['id'] for team in teams.get_teams()}
    targets = []
    for game_id, game in frame.groupby('GAME_ID'):
        if len(game) != 2 or not set(game['TEAM_ID']).issubset(valid_ids):
            continue
        home = game[game['MATCHUP'].str.contains(' vs. ', regex=False)]
        away = game[game['MATCHUP'].str.contains(' @ ', regex=False)]
        if len(home) != 1 or len(away) != 1:
            continue
        home, away = home.iloc[0], away.iloc[0]
        if opening_limit and max(home['season_game_number'], away['season_game_number']) > opening_limit:
            continue
        targets.append({'game_id': int(game_id), 'date': home['GAME_DATE'],
                        'home': int(home['TEAM_ID']), 'away': int(away['TEAM_ID']),
                        'home_win': int(home['WL'] == 'W')})
    return sorted(targets, key=lambda game: (game['date'], game['game_id']))


def summary(rows, policy, keys=None):
    rows = [row for row in rows if row['policy'] == policy and
            (keys is None or row['game_id'] in keys)]
    if not rows:
        return {'games': 0}
    y = np.array([row['home_win'] for row in rows])
    p = np.clip([row['probability'] for row in rows], 1e-6, 1 - 1e-6)
    correct = np.array([row['predicted_home'] == row['home_win'] for row in rows])
    return {'games': len(rows), 'correct': int(correct.sum()),
            'accuracy': float(correct.mean()), 'brier': float(((p - y) ** 2).mean()),
            'log_loss': float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())}


def evaluate(raw, targets, season, pipeline, preseason, label):
    from src.nba_predictor import compute_standings_at_date

    policies = ['legacy_current_only', 'current_only', 'carry_previous']
    matrix, rows = [], []
    current = filter_prediction_games(raw[raw['SEASON'] == season], preseason)
    regular = current[current['SEASON_ID'].astype(str).str.startswith('2')]
    history = raw[raw['SEASON'].isin([season,
                 f'{int(season[:4])-1}-{int(season[:4]) % 100:02d}'])]
    for game in targets:
        prior_current = current[current['GAME_DATE'] < game['date']]
        hs = compute_standings_at_date(regular, game['date'], game['home'])
        aws = compute_standings_at_date(regular, game['date'], game['away'])
        h2h_h = compute_head_to_head(prior_current, game['home'], game['away'])
        h2h_a = compute_head_to_head(prior_current, game['away'], game['home'])
        for policy in policies:
            if policy == 'legacy_current_only':
                hf = get_recent_team_stats(game['home'], prior_current)
                af = get_recent_team_stats(game['away'], prior_current)
            else:
                kwargs = dict(carry_previous=policy == 'carry_previous', preseason_enabled=preseason)
                hf = upcoming_team_stats(game['home'], history, season, game['date'], True, **kwargs)
                af = upcoming_team_stats(game['away'], history, season, game['date'], False, **kwargs)
            if hf is None or af is None:
                continue
            vector = pipeline.feature_computer.build_feature_vector(
                hf, af, pipeline.ensemble.feature_cols, h2h_home=h2h_h, h2h_away=h2h_a,
                home_standings=hs, away_standings=aws)
            assert np.isfinite(vector).all()
            matrix.append(vector[0])
            rows.append({**game, 'date': game['date'].strftime('%Y-%m-%d'), 'policy': policy})
    if not matrix:
        return {'label': label, 'season': season, 'preseason_inputs': preseason,
                'target_games': len(targets), 'all_available': {
                    policy: {'games': 0} for policy in policies},
                'common_games': {policy: {'games': 0} for policy in policies},
                'extra_predictions_with_carryover': 0, 'games': []}
    probabilities = predict_batch(pipeline.ensemble, np.asarray(matrix))
    # Confirm batch inference agrees with the real production pipeline.
    for index in sorted({0, len(matrix) // 2, len(matrix) - 1}):
        actual = pipeline.predict_from_features(np.asarray(matrix[index]).reshape(1, -1))
        assert abs(actual.home_win_probability - probabilities[index]) < 1e-6
    threshold = pipeline.ensemble.ensemble_threshold or .5
    for row, probability in zip(rows, probabilities):
        row['probability'] = float(probability)
        row['predicted_home'] = int(probability > threshold)
    keys = {policy: {row['game_id'] for row in rows if row['policy'] == policy} for policy in policies}
    common = keys['current_only'] & keys['carry_previous']
    report = {'label': label, 'season': season, 'preseason_inputs': preseason,
              'target_games': len(targets), 'all_available': {
                  policy: summary(rows, policy) for policy in policies},
              'common_games': {policy: summary(rows, policy, common) for policy in policies},
              'extra_predictions_with_carryover': len(keys['carry_previous'] - keys['current_only']),
              'games': rows}
    print(json.dumps({key: value for key, value in report.items() if key != 'games'}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/saved_ensemble_history_evaluation.json')
    parser.add_argument('--recent-only', action='store_true')
    args = parser.parse_args()
    raw = pd.read_csv(GAMES_CACHE_FILE)
    raw['GAME_DATE'] = pd.to_datetime(raw['GAME_DATE'])
    raw = raw.drop_duplicates(['TEAM_ID', 'GAME_ID'])
    manifest = model_manifest()
    latest_date = max(record['modified_at'][:10] for record in manifest)
    today = datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')
    recent_season = raw.sort_values('GAME_DATE').iloc[-1]['SEASON']
    recent = target_games(raw, recent_season, start=latest_date, end=today)
    ensemble = ModelLoader().load_ensemble(verbose=True)
    assert ensemble.n_models == 4, 'Evaluation requires all four production models'
    pipeline = PredictionPipeline(ensemble)
    reports = []
    for preseason in [True, False]:
        if recent:
            reports.append(evaluate(raw, recent, recent_season, pipeline, preseason,
                                    'After saved-model file date; timestamp-screened, not audited provenance'))
        if not args.recent_only:
            historical = target_games(raw, '2025-26', opening_limit=20)
            reports.append(evaluate(raw, historical, '2025-26', pipeline, preseason,
                                    'Retrospective diagnostic; training/validation overlap possible'))
    assert model_manifest() == manifest, 'Production models changed during evaluation'
    result = {'models': manifest, 'model_types': ensemble.model_types,
              'feature_count': len(ensemble.feature_cols), 'results': reports,
              'limitations': [
                  'Historical replays are not independent holdouts: validation influenced model selection and calibration.',
                  'Recent games are screened against artifact modification dates, not an audited training-data manifest.',
                  'Recent games are preseason only; small samples cannot establish regular-season accuracy improvement.',
                  'Historical standings are reconstructed from regular-season results; tied ranks may differ from the NBA API.',
                  'No historical bookmaker odds or ROI evaluation. Stacking output is not the CLI winner decision; production aggregation is preserved.',
              ]}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
