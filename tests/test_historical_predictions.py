"""Historical prediction inputs exclude target/future outcomes."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.prediction.historical import historical_standings, season_for_date, validate_historical_model
from src.prediction.history import save_predictions_to_history


@pytest.fixture
def model_set(tmp_path):
    names = ['nba_ensemble_logistic_1.pkl', 'ensemble_scalers.pkl', 'ensemble_features.pkl',
             'ensemble_model_types.pkl', 'ensemble_stack_meta_lr.pkl', 'ensemble_stack_platt.pkl']
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    columns = ['HOME_PTS_ROLL_MEAN', 'AWAY_PTS_ROLL_MEAN', 'HOME_WINS', 'AWAY_WINS', 'HOME_WIN_PCT']
    manifest = {'protocol': 'chronological-five-period-v1', 'method_selected_using_final_test': False,
        'created_at_utc': '2026-10-17T16:00:00+00:00', 'model_types': ['logistic'],
        'feature_cols': columns, 'periods': {stage: {'end': end} for stage, end in zip(
            ['base_train', 'early_stopping', 'stacker_fit', 'calibrator_fit', 'final_test'],
            ['2020-02-04', '2022-04-02', '2023-04-08', '2024-04-28', '2026-10-16'])},
        'artifact_sha256': {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in names}}
    path = tmp_path / 'ensemble_training_manifest.json'
    path.write_text(json.dumps(manifest))
    ensemble = SimpleNamespace(models=[object()], model_types=['logistic'], feature_cols=columns)
    return tmp_path, path, manifest, ensemble


def test_valid_model_passes_historical_checks_without_csv_metadata(model_set):
    root, _, _, ensemble = model_set
    assert validate_historical_model(ensemble, '2026-10-21', root) is None


@pytest.mark.parametrize('fault', ['missing', 'corrupt', 'future_created', 'same_day_created', 'future_fit',
                                  'future_test', 'bad_date', 'bad_hash', 'missing_hash', 'weights',
                                  'partial_models', 'schema', 'protocol', 'selected_on_test'])
def test_unverifiable_or_later_models_cannot_replay_past_games(model_set, fault):
    root, path, manifest, ensemble = model_set
    if fault == 'future_created':
        manifest['created_at_utc'] = '2026-10-22T12:00:00Z'
    elif fault == 'same_day_created':
        manifest['created_at_utc'] = '2026-10-21T05:00:00Z'
    elif fault == 'future_fit':
        manifest['periods']['calibrator_fit']['end'] = '2026-10-21'
    elif fault == 'future_test':
        manifest['periods']['final_test']['end'] = '2026-10-22'
    elif fault == 'bad_date':
        manifest['periods']['final_test']['end'] = None
    elif fault == 'bad_hash':
        (root / 'nba_ensemble_logistic_1.pkl').write_bytes(b'replaced')
    elif fault == 'missing_hash':
        manifest['artifact_sha256'].pop('ensemble_scalers.pkl')
    elif fault == 'weights':
        (root / 'ensemble_weights.pkl').write_bytes(b'unverified')
    elif fault == 'partial_models':
        ensemble.models = []
    elif fault == 'schema':
        ensemble.feature_cols = ['other']
    elif fault == 'protocol':
        manifest['protocol'] = 'legacy'
    elif fault == 'selected_on_test':
        manifest['method_selected_using_final_test'] = True
    path.write_text(json.dumps(manifest))
    if fault == 'missing':
        path.unlink()
    elif fault == 'corrupt':
        path.write_text('{')
    with pytest.raises(ValueError, match='Historical prediction blocked'):
        validate_historical_model(ensemble, '2026-10-21', root)


@pytest.mark.parametrize('date,expected', [('2026-10-21', '2026-27'), ('2026-01-21', '2025-26')])
def test_replay_season_comes_from_date(date, expected):
    assert season_for_date(date) == expected


def test_standings_ignore_target_future_duplicates_preseason_and_other_seasons():
    home, away = 1610612738, 1610612752
    frame = pd.DataFrame({'TEAM_ID': [home, away, home, home, home, home, home],
        'GAME_ID': ['0022600001', '0022600001', '0022600001', '0022600002', '0022600003', '0012600004', '0022500001'],
        'GAME_DATE': pd.to_datetime(['2026-10-19'] * 3 + ['2026-10-21', '2026-10-22', '2026-10-18', '2025-10-19']),
        'WL': ['W', 'L', 'W', 'L', 'L', 'L', 'L'], 'SEASON': ['2026-27'] * 6 + ['2025-26']})
    before = historical_standings(frame, '2026-10-21', '2026-27')
    assert before[home]['WINS'] == 1 and before[home]['LOSSES'] == 0
    assert before[away]['WINS'] == 0 and before[away]['LOSSES'] == 1
    frame.loc[frame.GAME_DATE >= '2026-10-21', 'WL'] = 'W'
    assert historical_standings(frame.sample(frac=1), '2026-10-21', '2026-27') == before


def test_cli_saves_finished_yesterday_games_without_live_inputs(model_set, monkeypatch, tmp_path):
    import predict
    from src.prediction import history
    from src.prediction.features import FeatureComputer
    from tests.test_rolling_history import games
    from src.nba_data_manager import NBADataManager

    root, _, _, ensemble = model_set
    home, away = 1610612738, 1610612752
    prior = games('2026-27', '2026-10-09', 7)
    prior['TEAM_ID'] = home
    prior['TEAM_ABBREVIATION'] = 'BOS'
    prior['MATCHUP'] = 'BOS vs. NYK'
    visitors = prior.copy()
    visitors['TEAM_ID'] = away
    visitors['TEAM_ABBREVIATION'] = 'NYK'
    visitors['MATCHUP'] = 'NYK @ BOS'
    visitors['WL'] = 'L'
    raw = pd.concat([prior, visitors], ignore_index=True)
    vectors = []
    def score(vector):
        vectors.append(vector.copy())
        return SimpleNamespace(predicted_winner='HOME', confidence=.2, home_win_probability=.6,
                               model_agreement=1., individual_predictions=[.6])
    pipeline = SimpleNamespace(feature_computer=FeatureComputer(), predict_from_features=score)
    monkeypatch.setattr(predict, 'ModelLoader', lambda: SimpleNamespace(load_ensemble=lambda **_: ensemble))
    monkeypatch.setattr(predict, 'PredictionPipeline', lambda _: pipeline)
    monkeypatch.setattr(predict, 'get_eastern_date', lambda: '2026-10-22')
    monkeypatch.setattr(predict, 'validate_historical_model', lambda e, d: validate_historical_model(e, d, root))
    forbidden = Mock(side_effect=AssertionError('live/post-date source used'))
    monkeypatch.setattr(predict, 'get_live_odds', forbidden)
    monkeypatch.setattr(NBADataManager, 'update_current_season', forbidden)
    fetched = []
    def fetch(season, **kwargs):
        fetched.append(season)
        return raw.copy()
    monkeypatch.setattr(predict, 'fetch_season_games', fetch)
    monkeypatch.setattr(predict, 'load_prediction_history', lambda frame, _: frame)
    monkeypatch.setattr(predict, 'get_todays_games', lambda **_: ([{
        'home_team_id': home, 'away_team_id': away, 'game_status': 'Final'}], '2026-10-21', True))
    output = tmp_path / 'history.csv'
    monkeypatch.setattr(history, 'PREDICTION_HISTORY_FILE', str(output))
    predict.predict_todays_games(date_override='2026-10-21')
    stored = pd.read_csv(output)
    assert len(stored) == 1
    assert len(stored.columns) == 10
    assert stored.iloc[0].date == '2026-10-21'
    assert stored.iloc[0].prediction == 'Boston Celtics'
    assert pd.isna(stored.iloc[0]['winner']) and pd.isna(stored.iloc[0]['correct'])
    raw.loc[raw.GAME_DATE >= '2026-10-21', ['PTS', 'WL']] = [9999, 'L']
    predict.predict_todays_games(date_override='2026-10-21')
    np.testing.assert_array_equal(vectors[0], vectors[1])
    assert len(pd.read_csv(output)) == 1  # Replays never overwrite an existing pick.
    assert fetched == ['2026-27', '2026-27']
    forbidden.assert_not_called()

    # On the target date, the CLI must build the exact same vector as replay.
    monkeypatch.setattr(predict, 'get_eastern_date', lambda: '2026-10-21')
    monkeypatch.setattr(NBADataManager, 'update_current_season', lambda _: None)
    monkeypatch.setattr(predict, 'get_live_odds', lambda: {})
    monkeypatch.setattr('main.get_profitable_filters', lambda: [])
    predict.predict_todays_games(date_override='2026-10-21')
    np.testing.assert_array_equal(vectors[0], vectors[2])

    # The dashboard must use that same input vector and date/season policy.
    from src.web import predict_service as service
    monkeypatch.setattr(service, '_load_pipeline', lambda _: pipeline)
    monkeypatch.setattr(service, 'fetch_season_games', fetch)
    monkeypatch.setattr(service, 'load_prediction_history', lambda frame, *_, **__: frame)
    monkeypatch.setattr(service, 'get_todays_games', lambda **_: ([{
        'home_team_id': home, 'away_team_id': away, 'game_status': 'Scheduled'}], '2026-10-21', False))
    monkeypatch.setattr(service, 'get_live_odds', lambda **_: {})
    pipeline.ensemble = ensemble
    # The web payload includes both sides of the same probability.
    base_score = pipeline.predict_from_features
    def web_score(vector):
        result = base_score(vector)
        result.away_win_probability = 1 - result.home_win_probability
        return result
    pipeline.predict_from_features = web_score
    payload = service.run_todays_predictions()
    np.testing.assert_array_equal(vectors[0], vectors[3])
    assert payload['predictions'][0]['home_win_prob'] == .6
    assert len(pd.read_csv(output)) == 1


def test_history_preserves_all_original_values_without_extra_columns(tmp_path, monkeypatch):
    from src.prediction import history
    output = tmp_path / 'history.csv'
    monkeypatch.setattr(history, 'PREDICTION_HISTORY_FILE', str(output))
    old = pd.DataFrame([{'date': '2026-10-21', 'away_team': 'A', 'home_team': 'B', 'prediction': 'A',
        'winner': 'A', 'confidence': .2, 'home_win_prob': .4, 'model_agreement': .9, 'tier': 'MODERATE', 'correct': 1}])
    old.to_csv(output, index=False)
    pred = {'away_team': 'C', 'home_team': 'D', 'predicted_winner': 'C', 'confidence': .2,
            'home_win_prob': .4, 'model_agreement': .9, 'prediction_mode': 'historical_replay'}
    save_predictions_to_history([pred], '2026-10-21')
    stored = pd.read_csv(output)
    assert len(stored) == 2 and stored.iloc[0].prediction == 'A'
    pd.testing.assert_frame_equal(stored.iloc[:1][old.columns].reset_index(drop=True), old, check_dtype=False)
    assert stored.columns.tolist() == old.columns.tolist()
