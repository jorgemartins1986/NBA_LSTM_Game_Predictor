"""
Path configuration for NBA Predictor
====================================
Centralizes all file paths used throughout the project.
"""

import os

# Get the project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')

# Ensure directories exist
for dir_path in [MODELS_DIR, CACHE_DIR, CONFIG_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model files
def get_model_path(filename):
    return os.path.join(MODELS_DIR, filename)

# Cache files
def get_cache_path(filename):
    return os.path.join(CACHE_DIR, filename)

# Config files
def get_config_path(filename):
    return os.path.join(CONFIG_DIR, filename)

# Report files
def get_report_path(filename):
    return os.path.join(REPORTS_DIR, filename)

# Specific file paths (for convenience)
GAMES_CACHE_FILE = get_cache_path('nba_games_cache.csv')
ELO_CACHE_FILE = get_cache_path('nba_elo_cache.pkl')
FEATURE_CACHE_FILE = get_cache_path('feature_cache.pkl')
MATCHUP_CACHE_FILE = get_cache_path('matchup_cache.pkl')
PREDICTION_HISTORY_FILE = get_report_path('prediction_history.csv')

XGB_PARAMS_FILE = get_config_path('best_xgb_params.json')

# Ensemble model files
ENSEMBLE_TYPES_FILE = get_model_path('ensemble_model_types.pkl')
ENSEMBLE_SCALERS_FILE = get_model_path('ensemble_scalers.pkl')
ENSEMBLE_FEATURES_FILE = get_model_path('ensemble_features.pkl')
ENSEMBLE_META_LR_FILE = get_model_path('ensemble_stack_meta_lr.pkl')
ENSEMBLE_PLATT_FILE = get_model_path('ensemble_stack_platt.pkl')
ENSEMBLE_WEIGHTS_FILE = get_model_path('ensemble_weights.pkl')
