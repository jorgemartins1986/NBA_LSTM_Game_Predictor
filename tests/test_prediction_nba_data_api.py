"""
Tests for NBA Data Module - API-dependent functions
====================================================
Tests for get_live_standings, get_todays_games, fetch_season_games
from src/prediction/nba_data.py using mocked NBA API responses.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.prediction.nba_data import (
    get_live_standings,
    get_todays_games,
    fetch_season_games,
    get_current_season,
)


def _df_to_api_result(df):
    """Convert a DataFrame to the dict format returned by _fetch_nba_stats."""
    return {
        'resultSets': [{
            'headers': list(df.columns),
            'rowSet': df.values.tolist(),
        }]
    }


# ---------------------------------------------------------------------------
# get_live_standings
# ---------------------------------------------------------------------------

class TestGetLiveStandings:
    """Tests for get_live_standings with mocked NBA API"""

    def _make_standings_df(self, rows):
        return pd.DataFrame(rows)

    @patch('time.sleep')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    def test_parses_win_streak(self, mock_fetch, _sleep):
        df = self._make_standings_df([{
            'TeamID': 1, 'WINS': 30, 'LOSSES': 10, 'WinPCT': 0.75,
            'PlayoffRank': 1, 'LeagueRank': 1,
            'ConferenceGamesBack': 0.0, 'strCurrentStreak': 'W 5',
        }])
        mock_fetch.return_value = _df_to_api_result(df)

        standings = get_live_standings(verbose=False)

        assert standings[1]['STREAK'] == 5
        assert standings[1]['WINS'] == 30

    @patch('time.sleep')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    def test_parses_loss_streak(self, mock_fetch, _sleep):
        df = self._make_standings_df([{
            'TeamID': 2, 'WINS': 10, 'LOSSES': 30, 'WinPCT': 0.25,
            'PlayoffRank': 15, 'LeagueRank': 30,
            'ConferenceGamesBack': 20.0, 'strCurrentStreak': 'L 3',
        }])
        mock_fetch.return_value = _df_to_api_result(df)

        standings = get_live_standings(verbose=False)

        assert standings[2]['STREAK'] == -3

    @patch('time.sleep')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    def test_handles_malformed_streak(self, mock_fetch, _sleep):
        df = self._make_standings_df([{
            'TeamID': 3, 'WINS': 20, 'LOSSES': 20, 'WinPCT': 0.5,
            'PlayoffRank': 8, 'LeagueRank': 15,
            'ConferenceGamesBack': 5.0, 'strCurrentStreak': 'INVALID',
        }])
        mock_fetch.return_value = _df_to_api_result(df)

        standings = get_live_standings(verbose=False)

        assert standings[3]['STREAK'] == 0

    @patch('time.sleep')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    def test_handles_missing_league_rank(self, mock_fetch, _sleep):
        df = self._make_standings_df([{
            'TeamID': 4, 'WINS': 15, 'LOSSES': 25, 'WinPCT': 0.375,
            'PlayoffRank': 10, 'LeagueRank': np.nan,
            'ConferenceGamesBack': np.nan, 'strCurrentStreak': 'W 1',
        }])
        mock_fetch.return_value = _df_to_api_result(df)

        standings = get_live_standings(verbose=False)

        assert standings[4]['LEAGUE_RANK'] == 10
        assert standings[4]['GAMES_BACK'] == 0.0

    @patch('time.sleep')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    def test_multiple_teams(self, mock_fetch, _sleep):
        df = self._make_standings_df([
            {'TeamID': 10, 'WINS': 40, 'LOSSES': 5, 'WinPCT': 0.889,
             'PlayoffRank': 1, 'LeagueRank': 1,
             'ConferenceGamesBack': 0.0, 'strCurrentStreak': 'W 10'},
            {'TeamID': 20, 'WINS': 20, 'LOSSES': 25, 'WinPCT': 0.444,
             'PlayoffRank': 9, 'LeagueRank': 18,
             'ConferenceGamesBack': 20.0, 'strCurrentStreak': 'L 2'},
        ])
        mock_fetch.return_value = _df_to_api_result(df)

        standings = get_live_standings(verbose=False)

        assert len(standings) == 2
        assert standings[10]['WIN_PCT'] == pytest.approx(0.889)
        assert standings[20]['CONF_RANK'] == 9

    @patch('time.sleep')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    def test_api_failure_returns_empty(self, mock_fetch, _sleep):
        mock_fetch.side_effect = Exception("API down")

        standings = get_live_standings(verbose=False)

        assert standings == {}

    @patch('time.sleep')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    def test_verbose_prints(self, mock_fetch, _sleep, capsys):
        df = self._make_standings_df([{
            'TeamID': 1, 'WINS': 30, 'LOSSES': 10, 'WinPCT': 0.75,
            'PlayoffRank': 1, 'LeagueRank': 1,
            'ConferenceGamesBack': 0.0, 'strCurrentStreak': 'W 5',
        }])
        mock_fetch.return_value = _df_to_api_result(df)

        get_live_standings(verbose=True)

        captured = capsys.readouterr()
        assert 'Fetching live standings' in captured.out
        assert '1 teams' in captured.out


# ---------------------------------------------------------------------------
# get_todays_games
# ---------------------------------------------------------------------------

class TestGetTodaysGames:
    """Tests for get_todays_games with mocked NBA API"""

    @patch('time.sleep')
    @patch('src.prediction.nba_data.get_eastern_date', return_value='2026-02-13')
    @patch('nba_api.stats.static.teams.get_teams')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    @patch('nba_api.live.nba.endpoints.scoreboard.ScoreBoard')
    def test_parses_games(self, mock_live, mock_fetch, mock_teams, mock_date, _sleep):
        mock_teams.return_value = [
            {'id': 100, 'nickname': 'Lakers'},
            {'id': 200, 'nickname': 'Celtics'},
        ]
        games_df = pd.DataFrame({
            'GAME_ID': ['001'],
            'HOME_TEAM_ID': [100],
            'VISITOR_TEAM_ID': [200],
            'GAME_STATUS_TEXT': ['7:00 PM ET'],
        })
        mock_fetch.return_value = _df_to_api_result(games_df)
        mock_live.side_effect = Exception("no live data")

        games, date_str, all_finished = get_todays_games(verbose=False)

        assert len(games) == 1
        assert games[0]['home_team'] == 'Lakers'
        assert games[0]['away_team'] == 'Celtics'
        assert games[0]['game_status'] == '7:00 PM ET'
        assert date_str == '2026-02-13'
        assert all_finished is False

    @patch('time.sleep')
    @patch('src.prediction.nba_data.get_eastern_date', return_value='2026-02-13')
    @patch('nba_api.stats.static.teams.get_teams', return_value=[])
    @patch('src.prediction.nba_data._fetch_nba_stats')
    @patch('nba_api.live.nba.endpoints.scoreboard.ScoreBoard')
    def test_empty_schedule(self, mock_live, mock_fetch, mock_teams, mock_date, _sleep):
        mock_fetch.return_value = _df_to_api_result(pd.DataFrame())
        mock_live.side_effect = Exception("no live")

        games, date_str, all_finished = get_todays_games(verbose=False)

        assert games == []
        assert all_finished is False

    @patch('time.sleep')
    @patch('src.prediction.nba_data.get_eastern_date', return_value='2026-02-13')
    @patch('nba_api.stats.static.teams.get_teams')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    @patch('nba_api.live.nba.endpoints.scoreboard.ScoreBoard')
    def test_deduplicates_games(self, mock_live, mock_fetch, mock_teams, mock_date, _sleep):
        mock_teams.return_value = [{'id': 1, 'nickname': 'A'}, {'id': 2, 'nickname': 'B'}]
        dup_df = pd.DataFrame({
            'GAME_ID': ['001', '001'],
            'HOME_TEAM_ID': [1, 1],
            'VISITOR_TEAM_ID': [2, 2],
            'GAME_STATUS_TEXT': ['Scheduled', 'Scheduled'],
        })
        mock_fetch.return_value = _df_to_api_result(dup_df)
        mock_live.side_effect = Exception("no live")

        games, _, _ = get_todays_games(verbose=False)

        assert len(games) == 1

    @patch('time.sleep')
    @patch('src.prediction.nba_data.get_eastern_date', return_value='2026-02-13')
    @patch('nba_api.stats.static.teams.get_teams')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    @patch('nba_api.live.nba.endpoints.scoreboard.ScoreBoard')
    def test_all_finished(self, mock_live, mock_fetch, mock_teams, mock_date, _sleep):
        mock_teams.return_value = [{'id': 1, 'nickname': 'A'}, {'id': 2, 'nickname': 'B'}]
        df = pd.DataFrame({
            'GAME_ID': ['001'],
            'HOME_TEAM_ID': [1],
            'VISITOR_TEAM_ID': [2],
            'GAME_STATUS_TEXT': ['Final'],
        })
        mock_fetch.return_value = _df_to_api_result(df)
        mock_live.side_effect = Exception("no live")

        _, _, all_finished = get_todays_games(verbose=False)

        assert all_finished is True

    @patch('time.sleep')
    @patch('src.prediction.nba_data.get_eastern_date', return_value='2026-02-13')
    @patch('nba_api.stats.static.teams.get_teams')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    @patch('nba_api.live.nba.endpoints.scoreboard.ScoreBoard')
    def test_live_status_overlay(self, mock_live, mock_fetch, mock_teams, mock_date, _sleep):
        mock_teams.return_value = [{'id': 1, 'nickname': 'A'}, {'id': 2, 'nickname': 'B'}]
        df = pd.DataFrame({
            'GAME_ID': ['001'],
            'HOME_TEAM_ID': [1],
            'VISITOR_TEAM_ID': [2],
            'GAME_STATUS_TEXT': ['Scheduled'],
        })
        mock_fetch.return_value = _df_to_api_result(df)

        mock_board = MagicMock()
        mock_board.get_dict.return_value = {
            'scoreboard': {'gameDate': '2026-02-13'}
        }
        mock_board.games.get_dict.return_value = [
            {'gameId': '001', 'gameStatusText': 'Q3 5:42'}
        ]
        mock_live.return_value = mock_board

        games, _, _ = get_todays_games(verbose=False)

        assert games[0]['game_status'] == 'Q3 5:42'

    @patch('time.sleep')
    @patch('src.prediction.nba_data.get_eastern_date', return_value='2026-02-13')
    @patch('nba_api.stats.static.teams.get_teams', return_value=[])
    @patch('nba_api.stats.endpoints.scoreboardv2.ScoreboardV2')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    def test_api_failure_returns_empty(self, mock_fetch, mock_sb, mock_teams, mock_date, _sleep):
        mock_fetch.side_effect = Exception("API failure")
        mock_sb.side_effect = Exception("API failure")

        games, date_str, all_finished = get_todays_games(verbose=False)

        assert games == []
        assert date_str == '2026-02-13'
        assert all_finished is False

    @patch('time.sleep')
    @patch('src.prediction.nba_data.get_eastern_date', return_value='2026-02-13')
    @patch('nba_api.stats.static.teams.get_teams')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    @patch('nba_api.live.nba.endpoints.scoreboard.ScoreBoard')
    def test_unknown_team_id_uses_fallback(self, mock_live, mock_fetch, mock_teams, mock_date, _sleep):
        mock_teams.return_value = [{'id': 1, 'nickname': 'A'}]
        df = pd.DataFrame({
            'GAME_ID': ['001'],
            'HOME_TEAM_ID': [1],
            'VISITOR_TEAM_ID': [9999],
            'GAME_STATUS_TEXT': ['Scheduled'],
        })
        mock_fetch.return_value = _df_to_api_result(df)
        mock_live.side_effect = Exception("no live")

        games, _, _ = get_todays_games(verbose=False)

        assert games[0]['home_team'] == 'A'
        assert games[0]['away_team'] == 9999


# ---------------------------------------------------------------------------
# fetch_season_games
# ---------------------------------------------------------------------------

class TestFetchSeasonGames:
    """Tests for fetch_season_games with mocked NBA API"""

    @patch('time.sleep')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    def test_returns_sorted_dataframe(self, mock_fetch, _sleep):
        df = pd.DataFrame({
            'GAME_DATE': ['2026-01-15', '2026-01-10', '2026-01-20'],
            'PTS': [110, 105, 115],
        })
        mock_fetch.return_value = _df_to_api_result(df)

        result = fetch_season_games(season='2025-26', verbose=False)

        assert isinstance(result, pd.DataFrame)
        dates = result['GAME_DATE'].tolist()
        assert dates == sorted(dates)

    @patch('time.sleep')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    def test_uses_current_season_by_default(self, mock_fetch, _sleep):
        df = pd.DataFrame({'GAME_DATE': ['2026-01-10'], 'PTS': [100]})
        mock_fetch.return_value = _df_to_api_result(df)

        fetch_season_games(verbose=False)

        call_args = mock_fetch.call_args
        assert call_args[0][1]['Season'] == get_current_season()

    @patch('time.sleep')
    @patch('src.prediction.nba_data._fetch_nba_stats')
    def test_verbose_prints(self, mock_fetch, _sleep, capsys):
        df = pd.DataFrame({'GAME_DATE': ['2026-01-10'], 'PTS': [100]})
        mock_fetch.return_value = _df_to_api_result(df)

        fetch_season_games(season='2025-26', verbose=True)

        captured = capsys.readouterr()
        assert '2025-26' in captured.out


# ---------------------------------------------------------------------------
# get_current_season edge case
# ---------------------------------------------------------------------------

class TestGetCurrentSeasonBranch:
    """Test the October+ branch of get_current_season."""

    @patch('src.prediction.nba_data.datetime')
    def test_october_returns_current_year(self, mock_dt):
        from src.prediction.nba_data import ZoneInfo
        mock_now = MagicMock()
        mock_now.month = 11
        mock_now.year = 2025
        mock_dt.now.return_value = mock_now

        result = get_current_season()

        assert result == '2025-26'
