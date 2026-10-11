"""Build upcoming-game statistics without crossing season boundaries for form."""

import pandas as pd

from ..season_config import include_preseason, filter_prediction_games, use_previous_season


def previous_season(season):
    year = int(season[:4]) - 1
    return f'{year}-{(year + 1) % 100:02d}'


def load_prediction_history(current_games, season, verbose=True):
    """Load only the immediately previous season when the fallback is enabled."""
    from .nba_data import fetch_season_games

    current = current_games.copy()
    current['SEASON'] = season
    if not use_previous_season():
        return current
    prior_season = previous_season(season)
    try:
        prior = fetch_season_games(prior_season, verbose=verbose).copy()
    except RuntimeError as error:
        if verbose:
            print(f'   Previous-season history unavailable: {error}')
        return current
    prior['SEASON'] = prior_season
    return pd.concat([prior, current], ignore_index=True)


def upcoming_team_stats(team_id, history, season, prediction_date, is_home,
                        window_size=20, carry_previous=None, preseason_enabled=None):
    """Use completed games strictly before the date, up to one rolling window.

    Season form and fatigue use only the current season. Counts describe the
    actual rolling window, not the entire dataframe supplied by the caller.
    """
    from ..nba_predictor import FeatureEngineering

    if carry_previous is None:
        carry_previous = use_previous_season()
    if preseason_enabled is None:
        preseason_enabled = include_preseason()
    cutoff = pd.Timestamp(prediction_date).normalize()
    games = filter_prediction_games(history.copy(), preseason_enabled)
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
    games = games[(games['TEAM_ID'] == team_id)
                  & (games['GAME_DATE'] < cutoff)
                  & games['WL'].isin(['W', 'L'])]
    allowed = [season, previous_season(season)] if carry_previous else [season]
    games = games[games['SEASON'].isin(allowed)].sort_values('GAME_DATE')
    games = games.drop_duplicates(['TEAM_ID', 'GAME_ID'], keep='last')
    current = games[games['SEASON'] == season]
    selected = games.tail(window_size)
    if selected.empty:
        return None

    fe = FeatureEngineering(window_size=window_size)

    def features_for_next(rows):
        # This placeholder cannot affect shifted rolling statistics.
        target = selected.iloc[-1].copy()
        target['GAME_DATE'] = cutoff
        target['SEASON'] = season
        target['WL'] = None
        target['MATCHUP'] = 'TEAM vs. OPP' if is_home else 'TEAM @ OPP'
        frame = pd.concat([rows, target.to_frame().T], ignore_index=True)
        frame['GAME_DATE'] = pd.to_datetime(frame['GAME_DATE'])
        frame = fe.calculate_four_factors(frame)
        return fe.create_rolling_features(frame, team_id).iloc[-1]

    rolling = features_for_next(selected)
    form = features_for_next(current.tail(window_size))
    form_defaults = {
        'WIN_STREAK': 0.0, 'RECENT_FORM': 0.5, 'WIN_PATTERN_3GAME': 3.5,
        'MOMENTUM_TREND': 0.0, 'WIN_CONSISTENCY': 0.5,
    }
    values = {col: rolling[col] for col in rolling.index if 'ROLL' in col}
    for key, default in form_defaults.items():
        value = form[key]
        values[key] = default if pd.isna(value) else float(value)
    values['IS_HOME'] = int(is_home)
    if current.empty:
        values.update(DAYS_REST=3, IS_BACK_TO_BACK=0, IS_3_IN_4=0)
    else:
        values['DAYS_REST'] = min((cutoff - current.iloc[-1]['GAME_DATE']).days, 10)
        values['IS_BACK_TO_BACK'] = int(values['DAYS_REST'] == 1)
        values['IS_3_IN_4'] = int(len(current) >= 2 and
                                (cutoff - current.iloc[-2]['GAME_DATE']).days <= 4)
    values['HISTORY_CURRENT_GAMES'] = int((selected['SEASON'] == season).sum())
    values['HISTORY_PREVIOUS_GAMES'] = int((selected['SEASON'] != season).sum())
    return values
