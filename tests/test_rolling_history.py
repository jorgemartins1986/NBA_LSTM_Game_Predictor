import pandas as pd
import pytest
from unittest.mock import patch

from src.prediction.rolling_history import upcoming_team_stats, load_prediction_history


def games(season, start, count, pts=100, result='W', preseason=False):
    frame = pd.DataFrame({
        'TEAM_ID': 1, 'GAME_ID': [f'{season}-{i}' for i in range(count)],
        'SEASON': season, 'SEASON_ID': 12025 if preseason else 22025,
        'GAME_DATE': pd.date_range(start, periods=count, freq='2D'),
        'MATCHUP': 'AAA vs. BBB', 'WL': result, 'PTS': pts,
        'FGM': 40, 'FG3M': 10, 'FGA': 85, 'FTA': 20, 'TOV': 12,
        'OREB': 10, 'DREB': 30, 'REB': 40, 'AST': 25,
        'FG3A': 30, 'FG_PCT': .47, 'FG3_PCT': .33, 'FT_PCT': .8,
        'STL': 7, 'BLK': 5,
    })
    return frame


def stats(history, date='2025-10-20', **kwargs):
    return upcoming_team_stats(1, history, '2025-26', date, True,
                               preseason_enabled=True, **kwargs)


def test_opener_uses_last_twenty_previous_games_and_resets_form():
    history = games('2024-25', '2025-03-01', 25)
    history.loc[history.index[-1], 'PTS'] = 140
    result = stats(history, carry_previous=True)
    assert result['PTS_ROLL_MEAN'] == 102  # Includes newest completed game.
    assert result['HISTORY_CURRENT_GAMES'] == 0
    assert result['HISTORY_PREVIOUS_GAMES'] == 20
    assert result['WIN_STREAK'] == 0
    assert result['RECENT_FORM'] == .5
    assert result['DAYS_REST'] == 3
    assert result['IS_BACK_TO_BACK'] == result['IS_3_IN_4'] == 0
    assert stats(history, carry_previous=False) is None


@pytest.mark.parametrize('count', [1, 5, 19, 20, 25])
def test_current_results_replace_previous_and_form_is_current_only(count):
    prior = games('2024-25', '2025-03-01', 25, pts=100)
    current = games('2025-26', '2025-10-01', count, pts=120, result='L')
    date = current['GAME_DATE'].max() + pd.Timedelta(days=1)
    result = stats(pd.concat([prior, current]), date, carry_previous=True)
    n = min(count, 20)
    assert result['PTS_ROLL_MEAN'] == (n * 120 + (20 - n) * 100) / 20
    assert result['HISTORY_CURRENT_GAMES'] == n
    assert result['HISTORY_PREVIOUS_GAMES'] == 20 - n
    assert result['WIN_STREAK'] == 0
    assert result['RECENT_FORM'] == 0
    assert result['DAYS_REST'] == 1
    assert result['IS_BACK_TO_BACK'] == 1
    if count >= 20:
        baseline = stats(pd.concat([prior, current]), date, carry_previous=False)
        assert result['PTS_ROLL_MEAN'] == baseline['PTS_ROLL_MEAN']


def test_cutoff_duplicates_unfinished_and_older_seasons_are_excluded():
    current = games('2025-26', '2025-10-01', 5)
    same_day = games('2025-26', '2025-10-20', 1, pts=999)
    future = games('2025-26', '2025-10-22', 1, pts=999)
    unfinished = games('2025-26', '2025-10-19', 1, pts=999, result=None)
    older = games('2023-24', '2024-03-01', 20, pts=999)
    combined = pd.concat([current, current, same_day, future, unfinished, older])
    # Give nonduplicate future rows distinct IDs.
    combined.loc[combined['PTS'] == 999, 'GAME_ID'] = combined.loc[
        combined['PTS'] == 999, 'GAME_DATE'].astype(str)
    result = stats(combined, carry_previous=True)
    assert result['PTS_ROLL_MEAN'] == 100
    assert result['HISTORY_CURRENT_GAMES'] == 5
    assert result['HISTORY_PREVIOUS_GAMES'] == 0


def test_preseason_flag_applies_to_previous_season_too():
    prior = games('2024-25', '2025-03-01', 20, preseason=True)
    result = upcoming_team_stats(1, prior, '2025-26', '2025-10-20', True,
                                 carry_previous=True, preseason_enabled=False)
    assert result is None


def test_previous_history_is_not_loaded_when_disabled():
    current = games('2025-26', '2025-10-01', 1)
    with patch('src.prediction.rolling_history.use_previous_season', return_value=False), \
            patch('src.prediction.nba_data.fetch_season_games') as fetch:
        result = load_prediction_history(current, '2025-26')
    fetch.assert_not_called()
    assert len(result) == 1


def test_loader_requests_only_immediately_previous_season():
    current = games('2025-26', '2025-10-01', 1)
    with patch('src.prediction.rolling_history.use_previous_season', return_value=True), \
            patch('src.prediction.nba_data.fetch_season_games') as fetch:
        fetch.return_value = games('2024-25', '2025-03-01', 20)
        result = load_prediction_history(current, '2025-26', verbose=False)
    fetch.assert_called_once_with('2024-25', verbose=False)
    assert set(result['SEASON']) == {'2024-25', '2025-26'}


def test_loader_survives_unavailable_previous_season(capsys):
    current = games('2025-26', '2025-10-01', 1)
    with patch('src.prediction.rolling_history.use_previous_season', return_value=True), \
            patch('src.prediction.nba_data.fetch_season_games', side_effect=RuntimeError('offline')):
        result = load_prediction_history(current, '2025-26')
    assert len(result) == 1
    assert 'unavailable' in capsys.readouterr().out
