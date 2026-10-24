from types import SimpleNamespace

import pandas as pd
import pytest

from tests.test_rolling_history import games


def test_web_distinguishes_failed_lookup_from_empty_schedule(monkeypatch):
    from src.web import predict_service as service
    from src.prediction import ScheduleUnavailableError
    from unittest.mock import Mock

    monkeypatch.setattr(service, '_load_pipeline', lambda _: None)
    monkeypatch.setattr(service, 'fetch_season_games', lambda **_: pd.DataFrame())
    monkeypatch.setattr(service, 'get_eastern_date', lambda: '2026-10-17')
    monkeypatch.setattr(service, 'get_todays_games', Mock(side_effect=ScheduleUnavailableError('offline')))
    result = service.run_todays_predictions()
    assert result['game_date'] == '2026-10-17'
    assert result['schedule_status'] == 'unavailable'
    assert result['predictions'] == []
    assert result['skipped_reason'] == 'offline'
    monkeypatch.setattr(service, 'get_todays_games', lambda **_: ([], '2026-10-17', False))
    result = service.run_todays_predictions()
    assert result['schedule_status'] == 'available'
    assert result['skipped_reason'] == 'No games scheduled for 2026-10-17 in the retrieved NBA schedule.'


def test_web_prediction_exposes_history_counts_without_network(monkeypatch):
    from src.web import predict_service as service
    from src.prediction.features import FeatureComputer

    prior = games('2024-25', '2025-03-01', 25)
    current = games('2025-26', '2025-10-01', 1)
    raw = pd.concat([prior, current], ignore_index=True)
    away = raw.copy()
    away['TEAM_ID'] = 2
    raw = pd.concat([raw, away], ignore_index=True)
    pipeline = SimpleNamespace(
        ensemble=SimpleNamespace(feature_cols=['HOME_PTS_ROLL_MEAN', 'AWAY_PTS_ROLL_MEAN']),
        feature_computer=FeatureComputer(),
        predict_from_features=lambda _: SimpleNamespace(
            predicted_winner='HOME', confidence=.2, home_win_probability=.6,
            away_win_probability=.4, model_agreement=.9),
    )
    monkeypatch.setattr(service, '_load_pipeline', lambda _: pipeline)
    monkeypatch.setattr(service, 'fetch_season_games', lambda **_: current)
    monkeypatch.setattr(service, 'load_prediction_history', lambda *_, **__: raw)
    monkeypatch.setattr(service, 'get_todays_games', lambda **_: (
        [{'home_team_id': 1, 'away_team_id': 2}], '2025-10-20', False))
    monkeypatch.setattr(service.teams, 'get_teams', lambda: [
        {'id': 1, 'full_name': 'AAA'}, {'id': 2, 'full_name': 'BBB'}])
    monkeypatch.setattr(service, 'get_live_odds', lambda **_: {})
    monkeypatch.setattr(service, 'match_game_to_odds', lambda *_: None)
    result = service.run_todays_predictions()
    assert result['num_games'] == 1
    assert result['predictions'][0]['history']['home'] == {
        'current_season_games': 1, 'previous_season_games': 19}


def test_cli_passes_prediction_date_and_history_to_feature_builder(monkeypatch):
    import predict
    from src.nba_data_manager import NBADataManager

    current = games('2025-26', '2025-10-01', 1)
    monkeypatch.setattr(predict, 'ModelLoader', lambda: SimpleNamespace(load_ensemble=lambda **_: None))
    monkeypatch.setattr(predict, 'PredictionPipeline', lambda _: None)
    monkeypatch.setattr(predict, 'validate_historical_model', lambda *_: {})
    monkeypatch.setattr(predict, 'historical_standings', lambda *_: {})
    monkeypatch.setattr(NBADataManager, 'update_current_season', lambda _: None)
    monkeypatch.setattr(predict, 'fetch_season_games', lambda **_: current)
    monkeypatch.setattr(predict, 'load_prediction_history', lambda *_: current)
    monkeypatch.setattr(predict, 'get_live_odds', lambda: {})
    monkeypatch.setattr(predict.teams, 'get_teams', lambda: [
        {'id': 1, 'full_name': 'AAA'}, {'id': 2, 'full_name': 'BBB'}])

    class ReachedFeatureBuilder(Exception):
        pass

    def capture(team_id, history, season, date, is_home):
        assert team_id == 1 and history is current
        assert season == '2025-26' and date == '2025-10-20' and is_home
        raise ReachedFeatureBuilder

    monkeypatch.setattr(predict, 'upcoming_team_stats', capture)
    with pytest.raises(ReachedFeatureBuilder):
        predict.predict_todays_games(manual_games=[{
            'home_team_id': 1, 'away_team_id': 2, 'game_date': '2025-10-20',
            'game_status': 'Scheduled'}])
