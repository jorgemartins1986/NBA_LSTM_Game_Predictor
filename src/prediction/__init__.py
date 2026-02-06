"""
Prediction Module
=================
Modular prediction components for the NBA ensemble predictor.
Designed for testability with dependency injection.
"""

from .features import FeatureComputer, GameFeatures
from .pipeline import PredictionPipeline, PredictionResult
from .loader import ModelLoader, LoadedEnsemble
from .nba_data import (
    get_eastern_date,
    get_current_season,
    get_live_standings,
    get_todays_games,
    fetch_season_games,
    get_recent_team_stats,
    compute_head_to_head,
    get_default_standings,
)
from .odds import get_live_odds, match_game_to_odds, format_odds_display
from .history import save_predictions_to_history, get_confidence_tier

__all__ = [
    # Core prediction
    'FeatureComputer',
    'GameFeatures',
    'PredictionPipeline',
    'PredictionResult',
    'ModelLoader',
    'LoadedEnsemble',
    # NBA data
    'get_eastern_date',
    'get_current_season',
    'get_live_standings',
    'get_todays_games',
    'fetch_season_games',
    'get_recent_team_stats',
    'compute_head_to_head',
    'get_default_standings',
    # Odds
    'get_live_odds',
    'match_game_to_odds',
    'format_odds_display',
    # History
    'save_predictions_to_history',
    'get_confidence_tier',
]
