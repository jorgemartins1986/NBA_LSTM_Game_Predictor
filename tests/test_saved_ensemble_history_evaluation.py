import numpy as np
import pandas as pd

from scripts.evaluate_saved_ensemble_history import predict_batch, summary, target_games
from src.prediction import PredictionPipeline, LoadedEnsemble


class Scaler:
    def transform(self, matrix):
        return np.asarray(matrix) / 2


class Classifier:
    def predict_proba(self, matrix):
        p = .5 + matrix[:, 0] / 10
        return np.column_stack([1 - p, p])


class KerasModel:
    def predict(self, matrix, **kwargs):
        return (.4 + matrix[:, 0] / 10).reshape(-1, 1)


def test_batched_inference_matches_actual_pipeline_with_selection_and_weights():
    forest = Classifier()
    forest._top_feature_indices = [1]
    ensemble = LoadedEnsemble(
        models=[Classifier(), forest, Classifier(), KerasModel()],
        scalers=[Scaler()] * 4, model_types=['xgboost', 'random_forest', 'logistic', 'lstm'],
        feature_cols=['a', 'b'], ensemble_weights=[1, 2, 1, 3])
    matrix = np.array([[1., 2.], [2., 3.]])
    result = predict_batch(ensemble, matrix)
    pipeline = PredictionPipeline(ensemble)
    expected = [pipeline.predict_from_features(row.reshape(1, -1)).home_win_probability
                for row in matrix]
    np.testing.assert_allclose(result, expected)


def test_summary_compares_only_common_games_and_counts_skips_separately():
    rows = [
        {'policy': 'carry_previous', 'game_id': 1, 'home_win': 1,
         'probability': .8, 'predicted_home': 1},
        {'policy': 'carry_previous', 'game_id': 2, 'home_win': 0,
         'probability': .7, 'predicted_home': 1},
    ]
    assert summary(rows, 'carry_previous')['accuracy'] == .5
    assert summary(rows, 'carry_previous', {1})['accuracy'] == 1
    assert summary(rows, 'current_only') == {'games': 0}


def test_targets_exclude_model_date_today_unknown_opponents_and_unfinished_games():
    rows = []
    for i, date in enumerate(['2026-10-06', '2026-10-07', '2026-10-11']):
        for team, matchup, wl in [(1610612738, 'BOS vs. NYK', 'W'),
                                  (1610612752, 'NYK @ BOS', 'L')]:
            rows.append({'SEASON': '2026-27', 'SEASON_ID': 12026, 'TEAM_ID': team,
                         'GAME_DATE': pd.Timestamp(date), 'GAME_ID': i,
                         'MATCHUP': matchup, 'WL': wl})
    frame = pd.DataFrame(rows)
    result = target_games(frame, '2026-27', start='2026-10-06', end='2026-10-11')
    assert [game['game_id'] for game in result] == [1]
