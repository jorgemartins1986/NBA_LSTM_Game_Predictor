"""Offline chronological comparison; never writes production models or history.

Run: python -m scripts.evaluate_rolling_history
Uses an independently fitted logistic baseline because saved ensemble artifacts
have no auditable training cutoff. This is evidence about inputs, not an
out-of-sample validation of the deployed ensemble or betting returns.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.nba_predictor import FeatureEngineering
from src.paths import GAMES_CACHE_FILE
from src.prediction.rolling_history import previous_season
from src.season_config import filter_prediction_games


FORM = ['WIN_STREAK', 'RECENT_FORM', 'WIN_PATTERN_3GAME', 'MOMENTUM_TREND',
        'WIN_CONSISTENCY', 'DAYS_REST', 'IS_BACK_TO_BACK', 'IS_3_IN_4']


def build_dataset(raw, preseason):
    """Compute features on game rows, with shift(1) excluding each target result."""
    raw = filter_prediction_games(raw, preseason)
    raw = raw[raw['WL'].isin(['W', 'L'])].copy()
    raw['GAME_DATE'] = pd.to_datetime(raw['GAME_DATE'])
    raw = raw.sort_values('GAME_DATE').drop_duplicates(['TEAM_ID', 'GAME_ID'])
    fe = FeatureEngineering(window_size=20)
    raw = fe.calculate_four_factors(raw)
    pieces = []
    feature_cols = None
    for (season, team), current in raw.groupby(['SEASON', 'TEAM_ID'], sort=True):
        if season < '2015-16' or season > '2025-26':
            continue
        prior = raw[(raw['SEASON'] == previous_season(season)) & (raw['TEAM_ID'] == team)]
        # Only the latest twenty previous-season games can enter the window.
        baseline = fe.create_rolling_features(current, team).copy()
        # Match live season-only form defaults and upcoming-game fatigue.
        defaults = {'WIN_STREAK': 0, 'RECENT_FORM': .5, 'WIN_PATTERN_3GAME': 3.5,
                    'MOMENTUM_TREND': 0, 'WIN_CONSISTENCY': .5}
        for col, value in defaults.items():
            baseline[col] = baseline[col].fillna(value)
        baseline['IS_3_IN_4'] = (
            (current['GAME_DATE'] - current['GAME_DATE'].shift(2)).dt.days <= 4
        ).astype(int).values
        carried = fe.create_rolling_features(pd.concat([prior.tail(20), current]), team)
        carried = carried[carried['SEASON'] == season].copy()
        if feature_cols is None:
            feature_cols = [col for col in baseline if 'ROLL' in col] + FORM
        # Both policies reset season form/fatigue, and both include the latest
        # completed result when predicting the next game.
        for col in FORM:
            carried[col] = baseline[col].values
        count = np.arange(len(current))
        baseline['current_count'] = count
        baseline['prior_count'] = 0
        carried['current_count'] = np.minimum(count, 20)
        carried['prior_count'] = np.minimum(np.maximum(20 - count, 0), len(prior))
        for mode, frame in [('current_only', baseline), ('carry_previous', carried)]:
            frame['policy'] = mode
            frame['regular_number'] = (frame['SEASON_ID'].astype(str).str.startswith('2')).cumsum()
            pieces.append(frame[['SEASON', 'TEAM_ID', 'GAME_ID', 'GAME_DATE', 'MATCHUP',
                                 'WL', 'SEASON_ID', 'policy', 'current_count', 'prior_count',
                                 'regular_number'] + feature_cols])
    features = pd.concat(pieces, ignore_index=True)
    # A game must have both NBA team records. Exhibition opponents are excluded.
    home = features[features['MATCHUP'].str.contains(' vs. ', regex=False)].copy()
    away = features[features['MATCHUP'].str.contains(' @ ', regex=False)].copy()
    matched = home.merge(away, on=['SEASON', 'GAME_ID', 'policy'], suffixes=('_home', '_away'))
    matched = matched[matched['SEASON_ID_home'].astype(str).str.startswith('2')].copy()
    for col in feature_cols:
        matched[f'diff_{col}'] = matched[f'{col}_home'] - matched[f'{col}_away']
    model_cols = [f'{col}_{side}' for side in ['home', 'away'] for col in feature_cols]
    model_cols += [f'diff_{col}' for col in feature_cols]
    matched[model_cols] = matched[model_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    matched = matched.copy()
    matched['target'] = (matched['WL_home'] == 'W').astype(int)
    matched['eligible'] = ((matched['current_count_home'] + matched['prior_count_home'] > 0)
                           & (matched['current_count_away'] + matched['prior_count_away'] > 0))
    return matched, model_cols


def metrics(frame):
    y, p = frame['target'].to_numpy(), frame['probability'].to_numpy()
    return {'games': len(frame), 'accuracy': float(((p >= .5) == y).mean()),
            'brier': float(np.mean((p - y) ** 2)),
            'log_loss': float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}


def evaluate(raw, preseason):
    dataset, cols = build_dataset(raw, preseason)
    outputs = []
    seasons = ['2022-23', '2023-24', '2024-25', '2025-26']
    for season in seasons:
        # Identical model for both policies; train on earlier-season games with
        # full current-season windows, where the two feature policies coincide.
        train = dataset[(dataset['SEASON'] < season) & (dataset['policy'] == 'current_only')
                        & (dataset['current_count_home'] >= 20)
                        & (dataset['current_count_away'] >= 20)]
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000))
        model.fit(train[cols], train['target'])
        early = dataset[(dataset['SEASON'] == season)
                        & (dataset['regular_number_home'] <= 20)
                        & (dataset['regular_number_away'] <= 20)].copy()
        eligible = early[early['eligible']].copy()
        eligible['probability'] = np.clip(model.predict_proba(eligible[cols])[:, 1], 1e-6, 1 - 1e-6)
        outputs.append(eligible)
        print(f'{season}: trained on {len(train)} earlier games; evaluated {len(early)//2} opening games', flush=True)
    results = pd.concat(outputs)
    baseline = results[results['policy'] == 'current_only']
    carried = results[results['policy'] == 'carry_previous']
    paired = baseline.merge(carried, on=['SEASON', 'GAME_ID'], suffixes=('_base', '_carry'))
    y = paired['target_base'].to_numpy()
    delta = (paired['probability_carry'].to_numpy() - y) ** 2 - (paired['probability_base'].to_numpy() - y) ** 2
    # Resample calendar dates, preserving correlation among same-day games.
    paired['delta'] = delta
    dates = paired.groupby('GAME_DATE_home_base')['delta'].agg(['sum', 'count']).to_numpy()
    rng = np.random.default_rng(42)
    samples = dates[rng.integers(0, len(dates), size=(5000, len(dates)))]
    bootstrap = samples[:, :, 0].sum(axis=1) / samples[:, :, 1].sum(axis=1)
    common_base = paired.rename(columns={'target_base': 'target', 'probability_base': 'probability'})
    common_carry = paired.rename(columns={'target_carry': 'target', 'probability_carry': 'probability'})
    return {
        'preseason_included': preseason,
        'evaluation_seasons': seasons,
        'method': 'Expanding chronological training, identical logistic baseline; first 20 regular-season games per team; date-cluster bootstrap.',
        'current_only': metrics(baseline), 'carry_previous': metrics(carried),
        'common_games_current_only': metrics(common_base),
        'common_games_carry_previous': metrics(common_carry),
        'paired_brier_delta_carry_minus_current': float(delta.mean()),
        'paired_brier_delta_95pct_ci': np.quantile(bootstrap, [.025, .975]).tolist(),
        'per_season_common_games': {
            season: {
                'current_only': metrics(common_base[common_base['SEASON'] == season]),
                'carry_previous': metrics(common_carry[common_carry['SEASON'] == season]),
            } for season in seasons
        },
        'limitation': 'Independent baseline, not the production ensemble; no historical odds or ROI evaluation. Feature/result cutoff is chronological; team roster changes are not modeled.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='docs/rolling_history_evaluation.json')
    args = parser.parse_args()
    raw = pd.read_csv(GAMES_CACHE_FILE)
    report = [evaluate(raw, preseason) for preseason in [True, False]]
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
