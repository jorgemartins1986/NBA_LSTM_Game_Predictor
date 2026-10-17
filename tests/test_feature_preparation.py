"""Regression checks for process workers and cached feature preparation."""

import os
import subprocess
import sys

import pandas as pd


def _games(count=8):
    rows = []
    for i in range(count):
        for team_id in (1, 2):
            rows.append({
                "GAME_ID": f"001260{i:04d}", "TEAM_ID": team_id,
                "TEAM_ABBREVIATION": "AAA" if team_id == 1 else "BBB",
                "GAME_DATE": pd.Timestamp("2026-10-01") + pd.Timedelta(days=i),
                "MATCHUP": "AAA vs. BBB" if team_id == 1 else "BBB @ AAA",
                "WL": "W" if team_id == 1 else "L", "PTS": 90 + i + team_id,
                "FGM": 40, "FG3M": 10, "FGA": 85, "FTA": 20, "TOV": 12,
                "OREB": 10, "DREB": 30, "REB": 40, "AST": 25,
                "FG3A": 30, "FG_PCT": .47, "FG3_PCT": .33, "FT_PCT": .8,
                "STL": 7, "BLK": 5,
            })
    return pd.DataFrame(rows)


def test_process_workers_do_not_import_tensorflow(tmp_path):
    # Start fresh processes so imports from other tests cannot hide the problem.
    input_path = tmp_path / "games.pkl"
    _games().to_pickle(input_path)
    script = '''
import sys
import pandas as pd
from joblib import Parallel, delayed
from src.feature_engineering import FeatureEngineering, _compute_team_features_parallel

def compute(team_id, games):
    result = _compute_team_features_parallel(team_id, games, 20)
    assert "tensorflow" not in sys.modules
    return result

games = pd.read_pickle(sys.argv[1])
fe = FeatureEngineering()
games = fe.calculate_four_factors(games)
results = Parallel(n_jobs=2)(delayed(compute)(tid, games) for tid in (1, 2))
for tid, result in zip((1, 2), results):
    pd.testing.assert_frame_equal(result, fe.create_rolling_features(games, tid))
assert "tensorflow" not in sys.modules
print("workers OK")
'''
    env = os.environ.copy()
    env.pop("TF_CPP_MIN_LOG_LEVEL", None)
    result = subprocess.run([sys.executable, "-X", "utf8", "-c", script, str(input_path)],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "workers OK"
    assert "oneDNN" not in result.stderr


def test_cached_matchups_equal_fresh_and_incremental_results(tmp_path, monkeypatch):
    import src.nba_predictor as module

    for name, filename in (
        ("FEATURE_CACHE_FILE", "features.pkl"), ("MATCHUP_CACHE_FILE", "matchups.pkl"),
        ("ENRICHED_GAMES_CSV", "enriched.csv"), ("MATCHUPS_CSV", "matchups.csv"),
    ):
        monkeypatch.setattr(module, name, str(tmp_path / filename))
    predictor = module.NBAPredictor()
    first = predictor.prepare_matchup_data(_games(), n_jobs=1)
    assert len(first) == 3
    cached = predictor.prepare_matchup_data(_games(), n_jobs=1)
    pd.testing.assert_frame_equal(cached, first)
    incremental = predictor.prepare_matchup_data(_games(9), n_jobs=1)
    fresh = predictor.prepare_matchup_data(_games(9), use_cache=False, n_jobs=1)
    assert len(incremental) == 4
    pd.testing.assert_frame_equal(incremental, fresh)
    # The current game's score must not enter its own rolling feature.
    assert first.iloc[0]["HOME_PTS_ROLL_MEAN"] == 93.0
