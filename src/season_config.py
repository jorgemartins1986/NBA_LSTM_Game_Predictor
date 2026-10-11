"""Shared active-season setting for prediction and cache updates."""

import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from .paths import get_config_path


def _read_config():
    try:
        with open(get_config_path('season.json'), encoding='utf-8') as stream:
            return json.load(stream)
    except FileNotFoundError:
        return {}


def get_configured_season():
    """Read the season override; null or a missing file enables automatic dates."""
    season = _read_config().get('current_season')
    if season is None:
        return None
    if not isinstance(season, str) or not re.fullmatch(r'\d{4}-\d{2}', season):
        raise ValueError('config/season.json: current_season must be YYYY-YY or null')
    if int(season[-2:]) != (int(season[:4]) + 1) % 100:
        raise ValueError('config/season.json: current_season must span consecutive years')
    return season


def include_preseason():
    """Whether preseason records may contribute to prediction statistics."""
    enabled = _read_config().get('include_preseason', True)
    if not isinstance(enabled, bool):
        raise ValueError('config/season.json: include_preseason must be true or false')
    return enabled


def filter_prediction_games(games_df, preseason_enabled):
    """Filter all input sources equally; retain postseason when preseason is off."""
    if preseason_enabled or games_df.empty:
        return games_df
    if 'SEASON_ID' in games_df.columns:
        preseason = games_df['SEASON_ID'].astype(str).str.startswith('1')
    elif 'GAME_ID' in games_df.columns:
        # CSV parsing can remove the leading zeroes from NBA game IDs.
        ids = games_df['GAME_ID'].astype(str).str.replace(r'\.0$', '', regex=True)
        preseason = ids.str.zfill(10).str.startswith('001')
    else:
        raise ValueError('Cannot exclude preseason: game data has no SEASON_ID or GAME_ID')
    return games_df.loc[~preseason].copy()


def get_current_season():
    """Use the configured season, otherwise roll over in October (Eastern time)."""
    season = get_configured_season()
    if season is not None:
        return season
    now = datetime.now(ZoneInfo('America/New_York'))
    year = now.year if now.month >= 10 else now.year - 1
    return f'{year}-{(year + 1) % 100:02d}'
