"""
Pytest Configuration and Shared Fixtures
=========================================
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_games_df():
    """Create a sample games DataFrame for testing"""
    np.random.seed(42)
    n_games = 100
    
    data = {
        'GAME_ID': [f'00221{i:05d}' for i in range(n_games * 2)],
        'TEAM_ID': list(range(1, n_games * 2 + 1)),
        'TEAM_NAME': ['Team A', 'Team B'] * n_games,
        'GAME_DATE': pd.date_range('2024-01-01', periods=n_games).repeat(2),
        'MATCHUP': ['Team A vs. Team B', 'Team B @ Team A'] * n_games,
        'WL': ['W', 'L'] * n_games,
        'PTS': np.random.randint(85, 130, n_games * 2),
        'FG_PCT': np.random.uniform(0.4, 0.55, n_games * 2),
        'FG3_PCT': np.random.uniform(0.3, 0.45, n_games * 2),
        'FT_PCT': np.random.uniform(0.7, 0.9, n_games * 2),
        'REB': np.random.randint(35, 55, n_games * 2),
        'AST': np.random.randint(18, 32, n_games * 2),
        'STL': np.random.randint(5, 15, n_games * 2),
        'BLK': np.random.randint(2, 10, n_games * 2),
        'TOV': np.random.randint(8, 20, n_games * 2),
        'SEASON': ['2023-24'] * (n_games * 2),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_matchup_df():
    """Create a sample matchup DataFrame for testing"""
    np.random.seed(42)
    n_matchups = 200
    
    # Create realistic feature names
    feature_cols = [
        'HOME_PTS_ROLLING', 'AWAY_PTS_ROLLING',
        'HOME_FG_PCT_ROLLING', 'AWAY_FG_PCT_ROLLING',
        'HOME_REB_ROLLING', 'AWAY_REB_ROLLING',
        'ELO_HOME', 'ELO_AWAY', 'ELO_DIFF',
        'HOME_REST_DAYS', 'AWAY_REST_DAYS',
        'HOME_H2H_WIN_PCT', 'HOME_WIN_PCT', 'AWAY_WIN_PCT'
    ]
    
    data = {
        'GAME_ID': [f'00221{i:05d}' for i in range(n_matchups)],
        'GAME_DATE': pd.date_range('2024-01-01', periods=n_matchups),
        'HOME_TEAM_ID': np.random.randint(1610612737, 1610612766, n_matchups),
        'AWAY_TEAM_ID': np.random.randint(1610612737, 1610612766, n_matchups),
        'HOME_WIN': np.random.randint(0, 2, n_matchups),
    }
    
    # Add feature columns with random data
    for col in feature_cols:
        if 'ELO' in col:
            data[col] = np.random.uniform(1400, 1600, n_matchups)
        elif 'PCT' in col:
            data[col] = np.random.uniform(0.3, 0.6, n_matchups)
        elif 'REST' in col:
            data[col] = np.random.randint(0, 5, n_matchups)
        else:
            data[col] = np.random.uniform(90, 120, n_matchups)
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary directory for cache files"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return str(cache_dir)


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create a temporary directory for model files"""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return str(models_dir)


@pytest.fixture
def mock_api_response():
    """Sample API response for odds testing"""
    return [
        {
            'id': 'abc123',
            'sport_key': 'basketball_nba',
            'sport_title': 'NBA',
            'commence_time': '2026-02-05T00:00:00Z',
            'home_team': 'Boston Celtics',
            'away_team': 'Los Angeles Lakers',
            'bookmakers': [
                {
                    'key': 'pinnacle',
                    'title': 'Pinnacle',
                    'markets': [
                        {
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Boston Celtics', 'price': 1.50},
                                {'name': 'Los Angeles Lakers', 'price': 2.60}
                            ]
                        }
                    ]
                }
            ]
        }
    ]


@pytest.fixture
def sample_prediction_history():
    """Sample prediction history DataFrame"""
    return pd.DataFrame({
        'date': ['2026-01-15', '2026-01-15', '2026-01-16', '2026-01-16'],
        'matchup': [
            'Lakers vs Celtics',
            'Heat vs Bulls',
            'Warriors vs Suns',
            'Nets vs Knicks'
        ],
        'prediction': ['Celtics', 'Heat', 'Warriors', 'Knicks'],
        'probability': [0.65, 0.72, 0.58, 0.81],
        'confidence': [0.30, 0.44, 0.16, 0.62],
        'tier': ['GOOD', 'STRONG', 'RISKY', 'EXCELLENT'],
        'actual_winner': ['Celtics', 'Bulls', 'Warriors', 'Knicks'],
        'correct': [1, 0, 1, 1]
    })


# ============================================================================
# Configuration for test environment
# ============================================================================

@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables"""
    # Disable GPU for testing
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setenv('TF_CPP_MIN_LOG_LEVEL', '3')  # Suppress TF warnings
