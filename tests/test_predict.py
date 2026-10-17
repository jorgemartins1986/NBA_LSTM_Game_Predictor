"""
Tests for predict.py
====================
Tests for the prediction entry point, especially game filtering logic.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import filter_regular_games
from types import SimpleNamespace
from unittest.mock import Mock


# --- Fixtures ---

FAKE_TEAMS = [
    {'id': 1, 'full_name': 'Boston Celtics'},
    {'id': 2, 'full_name': 'Los Angeles Lakers'},
    {'id': 3, 'full_name': 'Golden State Warriors'},
]


def _game(home_id, away_id, **extra):
    """Helper to build a minimal game dict."""
    return {'home_team_id': home_id, 'away_team_id': away_id, **extra}


@pytest.mark.parametrize('unavailable', [False, True])
def test_cli_distinguishes_empty_schedule_from_failed_lookup(monkeypatch, capsys, unavailable):
    import predict
    from src.nba_data_manager import NBADataManager
    from src.prediction import ScheduleUnavailableError

    monkeypatch.setattr(predict, 'ModelLoader', lambda: SimpleNamespace(load_ensemble=lambda **_: None))
    monkeypatch.setattr(predict, 'PredictionPipeline', lambda _: None)
    monkeypatch.setattr(NBADataManager, 'update_current_season', lambda _: None)
    monkeypatch.setattr(predict, 'fetch_season_games', lambda **_: None)
    monkeypatch.setattr(predict, 'get_eastern_date', lambda: '2026-10-17')
    lookup = Mock(return_value=([], '2026-10-20', False))
    if unavailable:
        lookup.side_effect = ScheduleUnavailableError('All schedule sources failed')
    monkeypatch.setattr(predict, 'get_todays_games', lookup)
    save = Mock()
    monkeypatch.setattr(predict, 'save_predictions_to_history', save)
    predict.predict_todays_games(date_override='2026-10-20')
    output = capsys.readouterr().out
    assert 'Fetching games for 2026-10-20' in output
    assert 'off-season' not in output
    if unavailable:
        assert 'Schedule unavailable for 2026-10-20' in output
        assert 'No games scheduled' not in output
        assert '--manual-games' in output
    else:
        assert 'No games scheduled for 2026-10-20' in output
        assert 'Schedule unavailable' not in output
    save.assert_not_called()


# --- Tests ---

class TestFilterRegularGames:
    """Tests for filter_regular_games()"""

    def test_regular_game_passes_through(self):
        """A game between two known NBA teams should be kept."""
        games = [_game(1, 2)]
        regular, skipped = filter_regular_games(games, FAKE_TEAMS)

        assert len(regular) == 1
        assert len(skipped) == 0
        assert regular[0]['home_team_id'] == 1

    def test_special_game_both_unknown_ids(self):
        """A game with both team IDs unknown (e.g. All-Star) should be skipped."""
        games = [_game(9999, 8888)]
        regular, skipped = filter_regular_games(games, FAKE_TEAMS)

        assert len(regular) == 0
        assert len(skipped) == 1
        assert 'Team #9999' in skipped[0]['home_name']
        assert 'Team #8888' in skipped[0]['away_name']

    def test_special_game_none_ids(self):
        """A game with None team IDs (observed on All-Star day) should be skipped."""
        games = [_game(None, None)]
        regular, skipped = filter_regular_games(games, FAKE_TEAMS)

        assert len(regular) == 0
        assert len(skipped) == 1
        assert 'Team #None' in skipped[0]['home_name']

    def test_special_game_one_known_one_unknown(self):
        """A game with one real team and one unknown should be skipped."""
        games = [_game(1, 7777)]
        regular, skipped = filter_regular_games(games, FAKE_TEAMS)

        assert len(regular) == 0
        assert len(skipped) == 1
        # The known team should resolve its name
        assert skipped[0]['away_name'] == 'Team #7777'

    def test_mixed_list_filters_correctly(self):
        """Regular games kept, special games skipped, order preserved."""
        games = [
            _game(1, 2),       # regular
            _game(None, None), # special
            _game(2, 3),       # regular
            _game(5555, 6666), # special
        ]
        regular, skipped = filter_regular_games(games, FAKE_TEAMS)

        assert len(regular) == 2
        assert len(skipped) == 2
        assert regular[0]['home_team_id'] == 1
        assert regular[1]['home_team_id'] == 2

    def test_empty_game_list(self):
        """Empty input should return two empty lists."""
        regular, skipped = filter_regular_games([], FAKE_TEAMS)

        assert regular == []
        assert skipped == []

    def test_all_special_games(self):
        """When every game is special, regular list is empty."""
        games = [_game(None, None), _game(9999, 8888)]
        regular, skipped = filter_regular_games(games, FAKE_TEAMS)

        assert len(regular) == 0
        assert len(skipped) == 2

    def test_extra_game_keys_preserved(self):
        """Extra fields on the game dict should survive filtering."""
        games = [_game(1, 2, game_status='Final', game_id='001')]
        regular, skipped = filter_regular_games(games, FAKE_TEAMS)

        assert regular[0]['game_status'] == 'Final'
        assert regular[0]['game_id'] == '001'

    def test_skipped_game_retains_original_keys(self):
        """Skipped games should carry through original keys plus names."""
        games = [_game(9999, 8888, game_status='Scheduled')]
        regular, skipped = filter_regular_games(games, FAKE_TEAMS)

        assert skipped[0]['game_status'] == 'Scheduled'
        assert 'home_name' in skipped[0]
        assert 'away_name' in skipped[0]
