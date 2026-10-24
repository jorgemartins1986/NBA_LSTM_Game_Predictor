"""Shared pre-date prediction inputs and model checks for historical CLI runs."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from nba_api.stats.static import teams

from ..paths import MODELS_DIR


def season_for_date(date_str):
    target = datetime.strptime(date_str, '%Y-%m-%d')
    year = target.year if target.month >= 10 else target.year - 1
    return f'{year}-{(year + 1) % 100:02d}'


def validate_historical_model(ensemble, date_str, model_dir=None):
    """Require artifacts and all fitting/evaluation to precede the target day.

    A model trained after a past game cannot recreate a pick available before
    that game, even if a particular estimator's fitting data ended earlier.
    """
    root = Path(model_dir or MODELS_DIR)
    try:
        content = (root / 'ensemble_training_manifest.json').read_bytes()
        manifest = json.loads(content)
        if manifest['protocol'] != 'chronological-five-period-v1':
            raise ValueError('Unrecognized training protocol')
        if manifest['method_selected_using_final_test'] is not False:
            raise ValueError('Final-test model selection cannot be certified')
        if ensemble.model_types != manifest['model_types'] or ensemble.feature_cols != manifest['feature_cols']:
            raise ValueError('Loaded model set or feature schema differs from the manifest')
        if not ensemble.models or len(ensemble.models) != len(manifest['model_types']):
            raise ValueError('Incomplete model set')
        target = pd.Timestamp(date_str).normalize()
        created = pd.Timestamp(manifest['created_at_utc'])
        if created.tzinfo is None:
            raise ValueError('Manifest creation time has no timezone')
        if created.tz_convert(ZoneInfo('America/New_York')).date() >= target.date():
            raise ValueError('This model version was created on or after the requested date')
        stages = ('base_train', 'early_stopping', 'stacker_fit', 'calibrator_fit', 'final_test')
        ends = [pd.Timestamp(manifest['periods'][stage]['end']).normalize() for stage in stages]
        if any(pd.isna(end) for end in ends):
            raise ValueError('Manifest period dates are invalid')
        if max(ends) >= target:
            raise ValueError('Model fitting, calibration, or evaluation includes the requested date or later')
        hashes = manifest['artifact_sha256']
        required = {'ensemble_scalers.pkl', 'ensemble_features.pkl', 'ensemble_model_types.pkl',
                    'ensemble_stack_meta_lr.pkl', 'ensemble_stack_platt.pkl'}
        for i, kind in enumerate(manifest['model_types'], 1):
            name = {'xgboost': f'nba_ensemble_xgboost_{i}.json',
                    'random_forest': f'nba_ensemble_rf_{i}.pkl',
                    'logistic': f'nba_ensemble_logistic_{i}.pkl'}.get(kind, f'nba_ensemble_model_{i}.keras')
            required.add(name)
        if not required.issubset(hashes):
            raise ValueError('Manifest lacks required artifact fingerprints')
        for name in required:
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != hashes[name]:
                raise ValueError(f'Model artifact fingerprint mismatch: {name}')
        if (root / 'ensemble_weights.pkl').exists():
            raise ValueError('Custom weights/threshold lack verified historical provenance')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(f'Historical prediction blocked: {error}. Restore a complete model set '
                         'with a matching training manifest from before this date.') from error


def games_before_date(games, date_str):
    """Use the same completed, deduplicated prior games for every entry point."""
    prior = games.copy()
    prior['GAME_DATE'] = pd.to_datetime(prior['GAME_DATE'])
    prior = prior[(prior.GAME_DATE < pd.Timestamp(date_str).normalize()) & prior.WL.isin(['W', 'L'])]
    return prior.sort_values(['GAME_DATE', 'GAME_ID']).drop_duplicates(['TEAM_ID', 'GAME_ID'], keep='last')


def historical_standings(games, date_str, season):
    """Reconstruct regular-season standings using completed prior-day games.

    Rank ties use equal win percentages; official multi-stage NBA tiebreakers
    are not reconstructed. No process-global standings cache is consulted.
    """
    from ..nba_predictor import EASTERN_IDS

    prior = games_before_date(games, date_str)
    if 'SEASON' in prior:
        prior = prior[prior.SEASON == season]
    if 'SEASON_ID' in prior:
        prior = prior[prior.SEASON_ID.astype(str).str.startswith('2')]
    else:
        ids = prior.GAME_ID.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(10)
        prior = prior[ids.str.startswith('002')]
    prior = prior.sort_values(['GAME_DATE', 'GAME_ID']).drop_duplicates(['TEAM_ID', 'GAME_ID'], keep='last')
    rows = []
    for team in teams.get_teams():
        team_games = prior[prior.TEAM_ID == team['id']]
        wins, losses = int((team_games.WL == 'W').sum()), int((team_games.WL == 'L').sum())
        streak = 0
        if not team_games.empty:
            last = team_games.WL.iloc[-1]
            for result in reversed(team_games.WL.tolist()):
                if result != last:
                    break
                streak += 1 if result == 'W' else -1
        rows.append({'TEAM_ID': team['id'], 'CONF': 'East' if team['id'] in EASTERN_IDS else 'West',
                     'WINS': wins, 'LOSSES': losses, 'WIN_PCT': wins / (wins + losses) if wins + losses else .5,
                     'STREAK': streak})
    table = pd.DataFrame(rows)
    table['CONF_RANK'] = table.groupby('CONF').WIN_PCT.rank(ascending=False, method='min').astype(int)
    table['LEAGUE_RANK'] = table.WIN_PCT.rank(ascending=False, method='min').astype(int)
    for _, group in table.groupby('CONF'):
        leader = group.sort_values(['WIN_PCT', 'WINS'], ascending=False).iloc[0]
        table.loc[group.index, 'GAMES_BACK'] = ((leader.WINS - group.WINS) + (group.LOSSES - leader.LOSSES)) / 2
    return {int(row.TEAM_ID): {key: row[key] for key in
            ('WINS', 'LOSSES', 'WIN_PCT', 'CONF_RANK', 'LEAGUE_RANK', 'GAMES_BACK', 'STREAK')}
            for _, row in table.iterrows()}
