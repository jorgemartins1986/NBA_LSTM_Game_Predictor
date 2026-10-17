"""Exercise NBA response parsing and recovery using offline API responses."""

import gzip
import json
import urllib.error
from unittest.mock import Mock, MagicMock

import pandas as pd
import pytest

from src.prediction import nba_data as data


def response(payload, compressed=False):
    body = json.dumps(payload).encode()
    result = MagicMock(read=Mock(return_value=gzip.compress(body) if compressed else body),
                       headers={"Content-Encoding": "gzip" if compressed else "identity"})
    result.__enter__.return_value = result
    return result


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    # Fail immediately if a test accidentally reaches an unconfigured endpoint.
    monkeypatch.setattr(data.urllib.request, "urlopen", Mock(side_effect=AssertionError("unexpected HTTP")))
    monkeypatch.setattr("time.sleep", Mock())
    monkeypatch.setattr(data, 'get_eastern_date', lambda: '2026-10-17')


@pytest.mark.parametrize("compressed", [False, True])
def test_stats_decodes_response_and_encodes_query(monkeypatch, compressed):
    payload = {"resultSets": [{"headers": ["PTS"], "rowSet": [[123]]}]}
    fetch = Mock(return_value=response(payload, compressed))
    monkeypatch.setattr(data.urllib.request, "urlopen", fetch)
    result = data._fetch_nba_stats("leaguegamefinder", {"SeasonType": "Regular Season", "LeagueID": "00"},
                                   timeout=7)
    assert result == payload
    assert fetch.call_args.args[0].full_url.endswith("?LeagueID=00&SeasonType=Regular+Season")
    assert fetch.call_args.kwargs["timeout"] == 7
    assert data._result_set_to_df(result).to_dict("list") == {"PTS": [123]}


def test_stats_retries_transient_errors(monkeypatch):
    fetch = Mock(side_effect=[TimeoutError(), urllib.error.URLError("reset"), response({"ok": True})])
    monkeypatch.setattr(data.urllib.request, "urlopen", fetch)
    assert data._fetch_nba_stats("endpoint", {}, retries=3) == {"ok": True}
    assert fetch.call_count == 3


def test_stats_raises_last_error_when_retries_exhausted(monkeypatch):
    error = ConnectionError("offline")
    fetch = Mock(side_effect=error)
    monkeypatch.setattr(data.urllib.request, "urlopen", fetch)
    with pytest.raises(ConnectionError, match="offline") as caught:
        data._fetch_nba_stats("endpoint", {}, retries=2)
    assert caught.value is error
    assert fetch.call_count == 2


@pytest.mark.parametrize("compressed", [False, True])
def test_cdn_parses_nested_ids_and_default_status(monkeypatch, compressed):
    payload = {"scoreboard": {"gameDate": "2026-10-17", "games": [
        {"gameId": "001", "homeTeam": {"teamId": "100"}, "awayTeam": {"teamId": "200"},
         "gameStatusText": "Q3"},
        {"gameIdStr": "002", "homeTeamId": "invalid", "visitorTeamId": "bad"},
    ]}}
    fetch = Mock(return_value=response(payload, compressed))
    monkeypatch.setattr(data.urllib.request, "urlopen", fetch)
    assert data._fetch_cdn_scoreboard("2026-10-17") == [
        {"game_id": "001", "home_team_id": 100, "away_team_id": 200, "game_status": "Q3"},
        {"game_id": "002", "home_team_id": None, "away_team_id": None, "game_status": "Scheduled"},
    ]
    assert "/20261017/" in fetch.call_args.args[0].full_url


def test_cdn_skips_errors_malformed_payloads_and_other_dates(monkeypatch):
    fetch = Mock(side_effect=[
        TimeoutError(), response([]),
        response({"scoreboard": {"gameDate": "2026-10-16", "games": [{"gameId": "wrong"}]}}),
        response({"scoreboard": {"gameDate": "2026-10-17", "games": [
            {"GAME_ID": "correct", "homeTeamId": 1, "awayTeamId": 2, "status": "Final"}]}}),
    ])
    monkeypatch.setattr(data.urllib.request, "urlopen", fetch)
    assert data._fetch_cdn_scoreboard("2026-10-17")[0]["game_id"] == "correct"
    assert fetch.call_count == 4


@pytest.mark.parametrize("compressed", [False, True])
def test_cdn_falls_back_to_static_schedule_and_filters_date(monkeypatch, compressed):
    payload = {"leagueSchedule": {"gameDates": [{"games": [
        {"gameId": "wrong", "gameDateEst": "2026-10-18T00:00:00Z"},
        {"gameId": "001", "gameDateEst": "2026-10-17T00:00:00Z",
         "homeTeam": {"teamId": "100"}, "awayTeam": {"teamId": "200"}, "gameStatusText": "7 PM"},
        {"gameId": "002", "gameDateEst": "2026-10-17T00:00:00Z",
         "homeTeam": {"teamId": "bad"}, "awayTeam": {"teamId": "bad"}},
    ]}]}}
    fetch = Mock(side_effect=[TimeoutError()] * 4 + [response(payload, compressed)])
    monkeypatch.setattr(data.urllib.request, "urlopen", fetch)
    result = data._fetch_cdn_scoreboard("2026-10-17")
    assert [game["game_id"] for game in result] == ["001", "002"]
    assert result[0]["home_team_id"] == 100
    assert result[0]["away_team_id"] == 200
    assert result[1]["home_team_id"] is None
    assert result[1]["away_team_id"] is None
    assert result[1]["game_status"] == "Scheduled"
    assert fetch.call_count == 5


def test_cdn_exhausted_sources_raise_unavailable(monkeypatch):
    monkeypatch.setattr(data.urllib.request, 'urlopen', Mock(side_effect=TimeoutError('offline')))
    with pytest.raises(data.ScheduleUnavailableError, match='unavailable.*2026-10-17'):
        data._fetch_cdn_scoreboard('2026-10-17')


@pytest.mark.parametrize('date_fields', [
    {'gameDateEst': '10/20/2026 00:00:00'},
    {'gameDateTimeEst': '2026-10-20T21:30:00'},
    {'gameDateTimeUTC': '2026-10-21T01:30:00Z'},
])
def test_future_schedule_uses_official_s3_fallback_and_eastern_date(monkeypatch, date_fields):
    payload = {'leagueSchedule': {'gameDates': [{'games': [
        {'gameId': '0022600003', **date_fields, 'gameStatus': 1,
         'homeTeam': {'teamId': '1610612759'}, 'awayTeam': {'teamId': '1610612760'}},
        {'gameId': 'wrong-date', 'gameDateEst': '2026-10-21',
         'homeTeam': {'teamId': 1}, 'awayTeam': {'teamId': 2}},
    ]}]}}
    denied = urllib.error.HTTPError('https://cdn.nba.com', 403, 'Forbidden', {}, None)
    fetch = Mock(side_effect=[denied, response(payload, compressed=True)])
    monkeypatch.setattr(data.urllib.request, 'urlopen', fetch)
    stats = Mock(side_effect=AssertionError('future dates must not query results endpoints'))
    monkeypatch.setattr('nba_api.stats.endpoints.scoreboardv2.ScoreboardV2', stats)
    monkeypatch.setattr('nba_api.stats.endpoints.leaguegamefinder.LeagueGameFinder', stats)
    games, date, finished = data.get_todays_games(date_str='2026-10-20', verbose=False)
    assert date == '2026-10-20' and not finished
    assert len(games) == 1
    assert games[0]['game_id'] == '0022600003'
    assert games[0]['home_team_id'] == 1610612759
    assert games[0]['away_team_id'] == 1610612760
    assert games[0]['home_team'] == 'Spurs'
    assert games[0]['game_status'] == 'Scheduled'
    assert 's3.amazonaws.com/NBA/staticData/' in fetch.call_args.args[0].full_url
    assert fetch.call_count == 2
    stats.assert_not_called()


def test_future_empty_schedule_is_distinct_from_source_failure(monkeypatch):
    payload = {'leagueSchedule': {'gameDates': [{'gameDate': '10/20/2026 00:00:00', 'games': []}]}}
    monkeypatch.setattr(data.urllib.request, 'urlopen', Mock(return_value=response(payload)))
    assert data.get_todays_games(date_str='2026-10-20', verbose=False) == ([], '2026-10-20', False)
    monkeypatch.setattr(data.urllib.request, 'urlopen', Mock(side_effect=TimeoutError('offline')))
    with pytest.raises(data.ScheduleUnavailableError, match='unavailable.*2026-10-20'):
        data.get_todays_games(date_str='2026-10-20', verbose=False)


@pytest.mark.parametrize('payload', [[], {}, {'leagueSchedule': {'gameDates': []}},
    {'leagueSchedule': {'gameDates': [{'gameDate': '10/20/2025 00:00:00', 'games': []}]}},
    {'leagueSchedule': {'gameDates': [{'games': [{'gameId': '001'}]}]}}])
def test_stale_or_unrecognizable_schedule_is_not_an_empty_future_day(monkeypatch, payload):
    monkeypatch.setattr(data.urllib.request, 'urlopen', Mock(return_value=response(payload)))
    with pytest.raises(data.ScheduleUnavailableError):
        data.get_todays_games(date_str='2026-10-20', verbose=False)


def test_date_between_published_schedule_days_is_confirmed_empty(monkeypatch):
    payload = {'leagueSchedule': {'gameDates': [
        {'gameDate': '10/19/2026 00:00:00', 'games': []},
        {'gameDate': '10/21/2026 00:00:00', 'games': []}]}}
    monkeypatch.setattr(data.urllib.request, 'urlopen', Mock(return_value=response(payload)))
    assert data._fetch_static_schedule('2026-10-20') == []


@pytest.fixture
def schedule_sources(monkeypatch, tmp_path):
    import nba_api.stats.endpoints.scoreboardv2 as scoreboard
    import nba_api.stats.endpoints.leaguegamefinder as finder
    import nba_api.live.nba.endpoints.scoreboard as live

    cdn = Mock(return_value=[])
    primary = Mock()
    primary.return_value.get_data_frames.return_value = [pd.DataFrame()]
    fallback = Mock()
    fallback.return_value.get_data_frames.return_value = [pd.DataFrame()]
    monkeypatch.setattr(data, "_fetch_cdn_scoreboard", cdn)
    monkeypatch.setattr(scoreboard, "ScoreboardV2", primary)
    monkeypatch.setattr(finder, "LeagueGameFinder", fallback)
    monkeypatch.setattr(live, "ScoreBoard", Mock(side_effect=ConnectionError()))
    monkeypatch.setattr(data, "GAMES_CACHE_FILE", str(tmp_path / "games.csv"))
    monkeypatch.setattr("nba_api.stats.static.teams.get_teams", lambda: [
        {"id": 100, "nickname": "Lakers"}, {"id": 200, "nickname": "Celtics"}])
    return cdn, primary, fallback


def test_schedule_prefers_cdn_and_honors_explicit_date(schedule_sources):
    cdn, primary, fallback = schedule_sources
    cdn.return_value = [{"game_id": "001", "home_team_id": 100, "away_team_id": 200,
                         "game_status": "Scheduled"}]
    games, date, finished = data.get_todays_games(date_str="2026-10-17")
    assert date == "2026-10-17"
    assert games[0]["home_team"] == "Lakers"
    assert games[0]["away_team"] == "Celtics"
    assert not finished
    cdn.assert_called_once_with("2026-10-17")
    primary.assert_not_called()
    fallback.assert_not_called()


def test_schedule_recovers_both_missing_ids_from_cache(schedule_sources, tmp_path):
    _, primary, _ = schedule_sources
    primary.return_value.get_data_frames.return_value = [pd.DataFrame({
        "GAME_ID": ["0042500403"], "HOME_TEAM_ID": [None], "VISITOR_TEAM_ID": [None],
        "GAME_STATUS_TEXT": ["Final"],
    })]
    pd.DataFrame({"GAME_ID": [42500403, 42500403], "TEAM_ID": [100, 200],
                  "MATCHUP": ["LAL vs. BOS", "BOS @ LAL"]}).to_csv(tmp_path / "games.csv", index=False)
    games, _, finished = data.get_todays_games(date_str="2026-10-17")
    assert games[0]["home_team_id"] == 100
    assert games[0]["away_team_id"] == 200
    assert finished


def test_schedule_handles_unresolved_ids_and_corrupt_cache(schedule_sources, tmp_path):
    _, primary, _ = schedule_sources
    primary.return_value.get_data_frames.return_value = [pd.DataFrame({
        "GAME_ID": ["001"], "HOME_TEAM_ID": [None], "VISITOR_TEAM_ID": ["invalid"],
        "GAME_STATUS_TEXT": ["Scheduled"],
    })]
    (tmp_path / "games.csv").write_text("invalid cache")
    games, _, finished = data.get_todays_games(date_str="2026-10-17")
    assert games[0]["home_team_id"] is None
    assert games[0]["away_team_id"] is None
    assert not finished


@pytest.mark.parametrize("use_wider_window", [False, True])
def test_schedule_falls_back_to_gamefinder(schedule_sources, use_wider_window):
    _, primary, fallback = schedule_sources
    primary.side_effect = TimeoutError()
    frame = pd.DataFrame({
        "GAME_ID": ["001", "001"], "TEAM_ID": [100, 200],
        "TEAM_NAME": ["Lakers", "Celtics"], "MATCHUP": ["LAL vs. BOS", "BOS @ LAL"],
        "GAME_DATE": ["2026-10-17", "2026-10-17"],
    })
    if use_wider_window:
        endpoints = [Mock(), Mock(), Mock(), Mock()]
        for endpoint in endpoints[:3]:
            endpoint.get_data_frames.return_value = [pd.DataFrame()]
        endpoints[3].get_data_frames.return_value = [frame]
        fallback.side_effect = endpoints
    else:
        fallback.return_value.get_data_frames.return_value = [frame]
    games, date, finished = data.get_todays_games(date_str="2026-10-17")
    assert len(games) == 1
    assert games[0]["home_team"] == "Lakers"
    assert games[0]["away_team"] == "Celtics"
    assert date == "2026-10-17"
    assert not finished
    if use_wider_window:
        assert fallback.call_args.kwargs["date_from_nullable"] == "10/16/2026"
        assert fallback.call_args.kwargs["date_to_nullable"] == "10/18/2026"


def test_schedule_rechecks_cdn_after_stats_sources_fail(schedule_sources):
    cdn, _, fallback = schedule_sources
    fallback.side_effect = TimeoutError()
    cdn.side_effect = [[], [{"game_id": "late", "home_team_id": 100, "away_team_id": 200,
                            "game_status": "Scheduled"}]]
    games, _, _ = data.get_todays_games(date_str="2026-10-17")
    assert games[0]["game_id"] == "late"
    assert cdn.call_count == 2


def test_today_reports_unavailable_when_all_schedule_sources_fail(schedule_sources):
    cdn, primary, fallback = schedule_sources
    cdn.side_effect = data.ScheduleUnavailableError('offline')
    primary.side_effect = TimeoutError()
    fallback.side_effect = TimeoutError()
    with pytest.raises(data.ScheduleUnavailableError, match='Could not retrieve.*2026-10-17'):
        data.get_todays_games(date_str='2026-10-17', verbose=False)


def test_cdn_final_status_is_respected(schedule_sources):
    cdn, _, _ = schedule_sources
    cdn.return_value = [{'game_id': '001', 'home_team_id': 100, 'away_team_id': 200,
                         'game_status': 'Final/OT'}]
    assert data.get_todays_games(date_str='2026-10-17', verbose=False)[2] is True


def test_season_fetch_reports_both_failed_sources(monkeypatch):
    monkeypatch.setenv("NBA_FORCE_FETCH", "1")
    monkeypatch.setattr(data, "_fetch_nba_stats", Mock(side_effect=TimeoutError("primary")))
    monkeypatch.setattr("nba_api.stats.endpoints.leaguegamefinder.LeagueGameFinder",
                        Mock(side_effect=ConnectionError("fallback")))
    with pytest.raises(RuntimeError, match="primary.*fallback"):
        data.fetch_season_games("2026-27")
