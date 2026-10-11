import json
from unittest.mock import patch

import pandas as pd
import pytest

from src import season_config
from src.nba_data_manager import NBADataManager
from src.prediction import nba_data


def test_prediction_and_cache_share_configured_season(tmp_path, monkeypatch):
    config = tmp_path / 'season.json'
    config.write_text(json.dumps({'current_season': '2026-27'}), encoding='utf-8')
    monkeypatch.setattr(season_config, 'get_config_path', lambda _: str(config))
    assert nba_data.get_current_season() == '2026-27'
    manager = NBADataManager(cache_file=str(tmp_path / 'games.csv'))
    with patch('src.nba_data_manager.leaguegamefinder.LeagueGameFinder') as finder:
        finder.return_value.get_data_frames.return_value = [
            pd.DataFrame({'GAME_DATE': ['2026-10-05'], 'TEAM_ID': [1]})
        ]
        result = manager.update_current_season()
    assert finder.call_args.kwargs['season_nullable'] == '2026-27'
    assert finder.call_args.kwargs['season_type_nullable'] == ''
    assert result['SEASON'].tolist() == ['2026-27']


@pytest.mark.parametrize('value', ['2026-28', '26-27', 2026])
def test_invalid_season_fails_clearly(tmp_path, monkeypatch, value):
    config = tmp_path / 'season.json'
    config.write_text(json.dumps({'current_season': value}), encoding='utf-8')
    monkeypatch.setattr(season_config, 'get_config_path', lambda _: str(config))
    with pytest.raises(ValueError, match='config/season.json'):
        season_config.get_current_season()


@pytest.mark.parametrize('source', ['cache', 'direct', 'fallback'])
@pytest.mark.parametrize('enabled', [True, False])
def test_preseason_flag_applies_to_every_source(tmp_path, monkeypatch, source, enabled):
    config = tmp_path / 'season.json'
    config.write_text(json.dumps({'include_preseason': enabled}), encoding='utf-8')
    monkeypatch.setattr(season_config, 'get_config_path', lambda _: str(config))
    df = pd.DataFrame({
        'GAME_DATE': ['2026-10-05', '2026-10-25', '2027-04-25'],
        'SEASON': ['2026-27'] * 3,
        'SEASON_ID': [12026, 22026, 42026],
        'GAME_ID': [12600001, 22600001, 42600001],
    })
    cache = tmp_path / 'games.csv'
    monkeypatch.setattr(nba_data, 'GAMES_CACHE_FILE', str(cache))
    monkeypatch.delenv('NBA_FORCE_FETCH', raising=False)
    if source == 'cache':
        df.to_csv(cache, index=False)
    response = {'resultSets': [{'headers': list(df.columns), 'rowSet': df.values.tolist()}]}
    with patch('src.prediction.nba_data._fetch_nba_stats') as direct, \
            patch('nba_api.stats.endpoints.leaguegamefinder.LeagueGameFinder') as fallback, \
            patch('time.sleep'):
        direct.return_value = response
        if source == 'fallback':
            direct.side_effect = TimeoutError
        fallback.return_value.get_data_frames.return_value = [df]
        result = nba_data.fetch_season_games('2026-27', verbose=False)
    assert result['SEASON_ID'].tolist() == ([12026, 22026, 42026] if enabled else [22026, 42026])


def test_preseason_filter_handles_cache_ids_without_leading_zeroes():
    df = pd.DataFrame({'GAME_ID': [12600001, 22600001, 42600001]})
    result = season_config.filter_prediction_games(df, False)
    assert result['GAME_ID'].tolist() == [22600001, 42600001]


def test_preseason_flag_requires_boolean(tmp_path, monkeypatch):
    config = tmp_path / 'season.json'
    config.write_text('{"include_preseason": "false"}', encoding='utf-8')
    monkeypatch.setattr(season_config, 'get_config_path', lambda _: str(config))
    with pytest.raises(ValueError, match='include_preseason'):
        season_config.include_preseason()
