import pandas as pd
import numpy as np

from scripts.evaluate_rolling_history import build_dataset
from src.prediction.rolling_history import upcoming_team_stats
from tests.test_rolling_history import games


def test_vectorized_evaluation_matches_live_features_at_each_cutoff():
    prior = games('2024-25', '2025-03-01', 25)
    current = games('2025-26', '2025-10-01', 25, pts=120, result='L')
    home = pd.concat([prior, current], ignore_index=True)
    away = home.copy()
    away['TEAM_ID'] = 2
    away['MATCHUP'] = 'BBB @ AAA'
    away['WL'] = 'W'
    raw = pd.concat([home, away], ignore_index=True)
    dataset, cols = build_dataset(raw, True)
    for carry in [True, False]:
        policy = 'carry_previous' if carry else 'current_only'
        rows = dataset[(dataset['SEASON'] == '2025-26') & (dataset['policy'] == policy)]
        for offset in [0, 1, 2, 5, 19, 20, 24]:
            row = rows.iloc[offset]
            actual = upcoming_team_stats(1, raw, '2025-26', row['GAME_DATE_home'], True,
                                         carry_previous=carry, preseason_enabled=True)
            if actual is None:
                assert not row['eligible']
                continue
            assert actual['HISTORY_CURRENT_GAMES'] == min(row['current_count_home'], 20)
            assert actual['HISTORY_PREVIOUS_GAMES'] == row['prior_count_home']
            for key, value in actual.items():
                col = f'{key}_home'
                if col in cols:
                    expected = 0 if pd.isna(value) else value
                    assert np.isclose(row[col], expected), (offset, carry, key, row[col], value)
