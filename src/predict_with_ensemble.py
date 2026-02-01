"""
Predict Today's Games Using Ensemble
====================================
Uses the trained ensemble (XGBoost + Random Forest + Logistic + LSTM) to predict NBA games.

Usage:
    python -m src.predict_with_ensemble
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import leaguegamefinder, scoreboardv2, leaguestandings
from nba_api.stats.static import teams
from .nba_predictor import FeatureEngineering, SumPooling1D
from .paths import (
    get_model_path,
    ENSEMBLE_TYPES_FILE, ENSEMBLE_SCALERS_FILE, ENSEMBLE_FEATURES_FILE,
    ENSEMBLE_META_LR_FILE, ENSEMBLE_PLATT_FILE, ENSEMBLE_WEIGHTS_FILE,
    PREDICTION_HISTORY_FILE
)
from datetime import datetime
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, will use OS environment variables

# Optional odds API integration
try:
    from .odds_api import OddsAPIClient, get_live_odds_for_predictions
    ODDS_API_AVAILABLE = True
except ImportError:
    ODDS_API_AVAILABLE = False
    print("ℹ️  Odds API not available (optional)")

def get_eastern_date():
    """Get current date in Eastern Time (NBA's timezone)"""
    eastern = ZoneInfo('America/New_York')
    return datetime.now(eastern).strftime('%Y-%m-%d')


def get_current_season():
    """Get current NBA season string (e.g., '2024-25')"""
    eastern = ZoneInfo('America/New_York')
    now = datetime.now(eastern)
    # NBA season starts in October
    if now.month >= 10:
        return f"{now.year}-{str(now.year + 1)[-2:]}"
    else:
        return f"{now.year - 1}-{str(now.year)[-2:]}"


def get_live_standings():
    """Fetch current standings from NBA API.
    
    Returns:
        dict: {team_id: {WINS, LOSSES, WIN_PCT, CONF_RANK, LEAGUE_RANK, GAMES_BACK, STREAK}}
    """
    import time
    
    season = get_current_season()
    print(f"📊 Fetching live standings for {season}...")
    
    try:
        time.sleep(0.6)  # Rate limiting
        ls = leaguestandings.LeagueStandings(season=season)
        df = ls.get_data_frames()[0]
        
        standings = {}
        for _, row in df.iterrows():
            team_id = row['TeamID']
            
            # Parse streak (format like "W 5" or "L 3")
            streak_str = row.get('strCurrentStreak', 'W 0')
            try:
                streak_parts = streak_str.split()
                streak_val = int(streak_parts[1]) if len(streak_parts) > 1 else 0
                streak = streak_val if streak_parts[0] == 'W' else -streak_val
            except:
                streak = 0
            
            standings[team_id] = {
                'WINS': int(row['WINS']),
                'LOSSES': int(row['LOSSES']),
                'WIN_PCT': float(row['WinPCT']),
                'CONF_RANK': int(row['PlayoffRank']),
                'LEAGUE_RANK': int(row['LeagueRank']) if pd.notna(row.get('LeagueRank')) else int(row['PlayoffRank']),
                'GAMES_BACK': float(row['ConferenceGamesBack']) if pd.notna(row.get('ConferenceGamesBack')) else 0.0,
                'STREAK': streak
            }
        
        print(f"   ✓ Got standings for {len(standings)} teams")
        return standings
        
    except Exception as e:
        print(f"   ⚠️ Could not fetch standings: {e}")
        return {}


def get_live_odds():
    """
    Fetch live betting odds for today's NBA games.
    
    Returns:
        dict: {game_key: odds_features} where game_key is "home_team_vs_away_team" (lowercase)
        
    Note: Requires ODDS_API_KEY environment variable to be set.
          Get free API key at: https://the-odds-api.com/
    """
    if not ODDS_API_AVAILABLE:
        return {}
    
    # Check if API key is configured
    api_key = os.environ.get('ODDS_API_KEY')
    if not api_key:
        return {}
    
    print("💰 Fetching live betting odds...")
    try:
        odds = get_live_odds_for_predictions()
        if odds:
            print(f"   ✓ Got odds for {len(odds)} games")
        return odds
    except Exception as e:
        print(f"   ⚠️ Could not fetch odds: {e}")
        return {}


def match_game_to_odds(home_team_name: str, away_team_name: str, odds_dict: dict) -> dict:
    """
    Match a game to its odds using fuzzy team name matching.
    
    Args:
        home_team_name: Full home team name (e.g., "Los Angeles Lakers")
        away_team_name: Full away team name
        odds_dict: Dict from get_live_odds()
        
    Returns:
        Odds features dict or empty dict if not found
    """
    if not odds_dict:
        return {}
    
    home_lower = home_team_name.lower()
    away_lower = away_team_name.lower()
    
    # Try exact match first
    key = f"{home_lower}_vs_{away_lower}"
    if key in odds_dict:
        return odds_dict[key]
    
    # Try partial matching
    for game_key, features in odds_dict.items():
        parts = game_key.split('_vs_')
        if len(parts) != 2:
            continue
            
        odds_home, odds_away = parts
        
        # Check if key parts of team names match
        home_words = set(home_lower.split())
        away_words = set(away_lower.split())
        odds_home_words = set(odds_home.split())
        odds_away_words = set(odds_away.split())
        
        # Remove common words
        common = {'the', 'a', 'an'}
        home_words -= common
        away_words -= common
        odds_home_words -= common
        odds_away_words -= common
        
        # Check for overlap
        if (len(home_words & odds_home_words) >= 1 and
            len(away_words & odds_away_words) >= 1):
            return features
            
    return {}


def format_odds_display(odds_features: dict) -> str:
    """Format odds for display in predictions."""
    if not odds_features:
        return ""
    
    home_odds = odds_features.get('HOME_AVG_ODDS', 0)
    away_odds = odds_features.get('AWAY_AVG_ODDS', 0)
    home_prob = odds_features.get('HOME_IMPLIED_PROB', 0)
    away_prob = odds_features.get('AWAY_IMPLIED_PROB', 0)
    spread = odds_features.get('HOME_ODDS_SPREAD', 0)
    
    if home_odds and away_odds:
        return (f"   💰 Bookmaker Odds: Home {home_odds:.2f} ({home_prob*100:.0f}%) | "
                f"Away {away_odds:.2f} ({away_prob*100:.0f}%)")
    return ""


# Enable unsafe deserialization for Lambda layers with Python lambdas
keras.config.enable_unsafe_deserialization()


def load_ensemble():
    """Load all ensemble models and artifacts"""
    print("📥 Loading ensemble models...")
    
    # Load model types
    with open(ENSEMBLE_TYPES_FILE, 'rb') as f:
        model_types = pickle.load(f)
    
    # Custom objects for model deserialization
    custom_objects = {
        'SumPooling1D': SumPooling1D,
        'reduce_sum_axis1': lambda x: tf.reduce_sum(x, axis=1)
    }
    
    # Load models based on type
    models = []
    loaded_types = []  # Track which models actually loaded
    loaded_indices = []  # Track indices for scaler matching
    for i, model_type in enumerate(model_types):
        if model_type == 'xgboost':
            model = xgb.XGBClassifier()
            model.load_model(get_model_path(f'nba_ensemble_xgboost_{i+1}.json'))
            print(f"✓ Loaded XGBoost model {i+1}")
            models.append(model)
            loaded_types.append(model_type)
            loaded_indices.append(i)
        elif model_type == 'random_forest':
            with open(get_model_path(f'nba_ensemble_rf_{i+1}.pkl'), 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Loaded Random Forest model {i+1}")
            models.append(model)
            loaded_types.append(model_type)
            loaded_indices.append(i)
        elif model_type == 'logistic':
            with open(get_model_path(f'nba_ensemble_logistic_{i+1}.pkl'), 'rb') as f:
                model = pickle.load(f)
            print(f"✓ Loaded Logistic Regression model {i+1}")
            models.append(model)
            loaded_types.append(model_type)
            loaded_indices.append(i)
        else:  # keras
            try:
                model = keras.models.load_model(
                    get_model_path(f'nba_ensemble_model_{i+1}.keras'), 
                    safe_mode=False,
                    custom_objects=custom_objects
                )
                print(f"✓ Loaded Keras model {i+1}")
                models.append(model)
                loaded_types.append(model_type)
                loaded_indices.append(i)
            except Exception as e:
                print(f"⚠ Skipping Keras model {i+1} (needs retraining): {type(e).__name__}")
                print(f"  Run 'python train.py' to retrain with fixed LSTM architecture")
                continue  # Skip this model
    
    # Update model_types to only include loaded models
    model_types = loaded_types
    
    # Load scalers and features
    with open(ENSEMBLE_SCALERS_FILE, 'rb') as f:
        all_scalers = pickle.load(f)
    
    # Filter scalers to only loaded models
    scalers = [all_scalers[i] for i in loaded_indices]
    
    with open(ENSEMBLE_FEATURES_FILE, 'rb') as f:
        feature_cols = pickle.load(f)
    
    print(f"✓ Loaded {len(models)} models")
    print(f"✓ Loaded {len(feature_cols)} features")
    
    # Try to load stacking artifacts (optional)
    meta_clf = None
    platt = None
    try:
        with open(ENSEMBLE_META_LR_FILE, 'rb') as f:
            meta_clf = pickle.load(f)
        with open(ENSEMBLE_PLATT_FILE, 'rb') as f:
            platt = pickle.load(f)
        print('✓ Loaded stacking meta-model and Platt calibrator')
    except Exception:
        pass

    # Try to load ensemble weights/threshold (optional)
    ensemble_weights = None
    ensemble_threshold = None
    try:
        with open(ENSEMBLE_WEIGHTS_FILE, 'rb') as f:
            w = pickle.load(f)
            ensemble_weights = w.get('weights')
            ensemble_threshold = w.get('threshold')
        print('✓ Loaded ensemble weights and threshold')
    except Exception:
        pass

    return models, scalers, feature_cols, model_types, meta_clf, platt, ensemble_weights, ensemble_threshold


def save_predictions_to_history(predictions, date_str=None):
    """Save predictions to history CSV file (creates or appends)
    
    Args:
        predictions: List of prediction dicts with keys:
            - away_team, home_team, predicted_winner, confidence, home_win_prob, model_agreement
        date_str: Optional date string (YYYY-MM-DD), defaults to Eastern time today
    """
    import os
    
    if date_str is None:
        date_str = get_eastern_date()
    
    # Prepare rows
    rows = []
    for pred in predictions:
        # Map confidence to bet tier
        conf = pred['confidence']
        if conf >= 0.50:
            tier = "EXCELLENT"
        elif conf >= 0.40:
            tier = "STRONG"
        elif conf >= 0.30:
            tier = "GOOD"
        elif conf >= 0.20:
            tier = "MODERATE"
        elif conf >= 0.10:
            tier = "RISKY"
        else:
            tier = "SKIP"
        
        rows.append({
            'date': date_str,
            'away_team': pred['away_team'],
            'home_team': pred['home_team'],
            'prediction': pred['predicted_winner'],
            'winner': '',  # To be filled later
            'confidence': round(pred['confidence'], 3),
            'home_win_prob': round(pred['home_win_prob'], 3),
            'model_agreement': round(pred.get('model_agreement', 0), 3),
            'tier': tier,
            'correct': ''  # To be filled later (1 or 0)
        })
    
    # Check if file exists
    file_exists = os.path.exists(PREDICTION_HISTORY_FILE)
    
    # Create DataFrame with column order
    columns = ['date', 'away_team', 'home_team', 'prediction', 'winner', 
               'confidence', 'home_win_prob', 'model_agreement', 'tier', 'correct']
    new_df = pd.DataFrame(rows, columns=columns)
    
    if file_exists:
        # Read existing and check for duplicates
        existing_df = pd.read_csv(PREDICTION_HISTORY_FILE)
        
        # Check if predictions for this date already exist (match by away_team + home_team)
        existing_df['match_key'] = existing_df['away_team'] + ' vs ' + existing_df['home_team']
        new_df['match_key'] = new_df['away_team'] + ' vs ' + new_df['home_team']
        existing_matches_today = existing_df[existing_df['date'] == date_str]['match_key'].tolist()
        
        # Filter out duplicates
        new_df = new_df[~new_df['match_key'].isin(existing_matches_today)]
        new_df = new_df.drop(columns=['match_key'])
        existing_df = existing_df.drop(columns=['match_key'])
        
        if len(new_df) == 0:
            print(f"\n📋 All predictions for {date_str} already in history. Skipping save.")
            return
        
        # Append new predictions
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(PREDICTION_HISTORY_FILE, index=False)
        print(f"\n📋 Appended {len(new_df)} new predictions to history ({PREDICTION_HISTORY_FILE})")
    else:
        # Create new file
        new_df.to_csv(PREDICTION_HISTORY_FILE, index=False)
        print(f"\n📋 Created prediction history with {len(new_df)} predictions ({PREDICTION_HISTORY_FILE})")


def get_todays_games():
    """Fetch today's NBA games
    
    Returns:
        tuple: (list of games, game_date string in YYYY-MM-DD format, all_finished bool)
        
    Uses ScoreboardV2 (stats API) which shows scheduled games for today,
    with live scoreboard as fallback for game status.
    """
    game_date_str = get_eastern_date()
    all_teams_list = teams.get_teams()
    team_id_to_info = {t['id']: t for t in all_teams_list}
    
    try:
        # Use ScoreboardV2 which has scheduled games (not just live/recent)
        sb = scoreboardv2.ScoreboardV2(game_date=game_date_str)
        games_df = sb.get_data_frames()[0]  # GameHeader dataframe
        
        if len(games_df) == 0:
            print(f"   No games scheduled for {game_date_str}")
            return [], game_date_str, False
        
        # IMPORTANT: NBA API sometimes returns duplicate rows - deduplicate by GAME_ID
        games_df = games_df.drop_duplicates(subset=['GAME_ID'], keep='first')
        
        # Try to get live status from live scoreboard
        live_status = {}
        try:
            board = scoreboard.ScoreBoard()
            live_data = board.get_dict()
            live_date = live_data.get('scoreboard', {}).get('gameDate', '')
            # Only use live status if it matches today's date
            if live_date == game_date_str:
                for game in board.games.get_dict():
                    live_status[game['gameId']] = game['gameStatusText']
        except:
            pass
        
        today_games = []
        finished_count = 0
        
        for _, row in games_df.iterrows():
            game_id = row['GAME_ID']
            home_team_id = row['HOME_TEAM_ID']
            visitor_team_id = row['VISITOR_TEAM_ID']
            
            home_info = team_id_to_info.get(home_team_id, {})
            visitor_info = team_id_to_info.get(visitor_team_id, {})
            
            # Get status from live board if available, else from schedule
            status = live_status.get(game_id, row.get('GAME_STATUS_TEXT', 'Scheduled'))
            
            today_games.append({
                'game_id': game_id,
                'home_team': home_info.get('nickname', row.get('HOME_TEAM_ID', 'Unknown')),
                'away_team': visitor_info.get('nickname', row.get('VISITOR_TEAM_ID', 'Unknown')),
                'home_team_id': home_team_id,
                'away_team_id': visitor_team_id,
                'game_status': status
            })
            
            if status == 'Final':
                finished_count += 1
        
        all_finished = len(today_games) > 0 and finished_count == len(today_games)
        return today_games, game_date_str, all_finished
        
    except Exception as e:
        print(f"   Error fetching games: {e}")
        return [], game_date_str, False


def get_recent_team_stats(team_id, games_df, window_size=20):
    """Get recent statistics for a team including fatigue features"""
    team_games = games_df[games_df['TEAM_ID'] == team_id].sort_values('GAME_DATE')
    
    if len(team_games) == 0:
        return None
    
    fe = FeatureEngineering(window_size=window_size)
    
    # CRITICAL: Calculate four factors on the FULL games_df before creating rolling features
    games_df_with_factors = fe.calculate_four_factors(games_df.copy())
    
    recent_with_features = fe.create_rolling_features(games_df_with_factors, team_id)
    
    if len(recent_with_features) == 0:
        return None
    
    latest = recent_with_features.iloc[-1]
    # Include new fatigue features
    roll_cols = [col for col in recent_with_features.columns 
                 if 'ROLL' in col or col in ['WIN_STREAK', 'RECENT_FORM', 'WIN_PATTERN_3GAME', 
                                              'MOMENTUM_TREND', 'WIN_CONSISTENCY', 'IS_HOME',
                                              'DAYS_REST', 'IS_BACK_TO_BACK', 'IS_3_IN_4']]
    features = {col: latest[col] for col in roll_cols if col in latest.index}
    
    return features


def compute_head_to_head(games_df, team_id, opponent_id, window=10):
    """Compute head-to-head record against specific opponent (last N meetings)"""
    # Get all completed games for this team
    team_games = games_df[games_df['TEAM_ID'] == team_id].copy()
    
    # Get opponent abbreviation
    opp_games = games_df[games_df['TEAM_ID'] == opponent_id]
    if len(opp_games) == 0:
        return {'H2H_WIN_RATE': 0.5, 'H2H_GAMES': 0, 'H2H_PTS_DIFF': 0}
    
    opp_abbrev = opp_games.iloc[0].get('TEAM_ABBREVIATION', '')
    
    # Find games against this opponent
    h2h_games = team_games[team_games['MATCHUP'].str.contains(opp_abbrev, na=False)].tail(window)
    
    if len(h2h_games) == 0:
        return {'H2H_WIN_RATE': 0.5, 'H2H_GAMES': 0, 'H2H_PTS_DIFF': 0}
    
    wins = (h2h_games['WL'] == 'W').sum()
    games_played = len(h2h_games)
    pts_diff = h2h_games['PLUS_MINUS'].mean() if 'PLUS_MINUS' in h2h_games.columns else 0
    
    return {
        'H2H_WIN_RATE': wins / games_played,
        'H2H_GAMES': min(games_played, window),
        'H2H_PTS_DIFF': pts_diff if not pd.isna(pts_diff) else 0
    }


def predict_game_ensemble(models, scalers, feature_cols, model_types, home_features, away_features, meta_clf=None, platt=None, ensemble_weights=None, ensemble_threshold=None, h2h_home=None, h2h_away=None, home_standings=None, away_standings=None, home_odds=None, away_odds=None):
    """Predict game using ensemble
    
    Args:
        models: List of trained models
        scalers: List of scalers for each model
        feature_cols: List of feature column names
        model_types: List of model type strings
        home_features: Dict of home team rolling features
        away_features: Dict of away team rolling features
        meta_clf: Optional stacking meta-classifier
        platt: Optional Platt calibration model
        ensemble_weights: Optional weights for ensemble averaging
        ensemble_threshold: Optional threshold for weighted ensemble
        h2h_home: Head-to-head features from home team perspective
        h2h_away: Head-to-head features from away team perspective
        home_standings: Current standings for home team
        away_standings: Current standings for away team
        home_odds: Odds features for home team (from odds API)
        away_odds: Odds features for away team (from odds API)
        
    Returns:
        Dict with prediction results including probability, confidence, etc.
    """
    
    # Combine features in correct order
    feature_dict = {}
    for col in feature_cols:
        if col.startswith('HOME_'):
            base_col = col.replace('HOME_', '')
            # Check for H2H features
            if base_col.startswith('H2H_') and h2h_home:
                feature_dict[col] = h2h_home.get(base_col, 0)
            # Check for standings features
            elif home_standings and base_col in home_standings:
                feature_dict[col] = home_standings.get(base_col, 0)
            # Check for odds features (e.g., HOME_AVG_ODDS, HOME_IMPLIED_PROB)
            elif home_odds and base_col in ['AVG_ODDS', 'BEST_ODDS', 'WORST_ODDS', 'ODDS_SPREAD', 'IMPLIED_PROB']:
                feature_dict[col] = home_odds.get(f'HOME_{base_col}', 0)
            else:
                val = home_features.get(base_col, None)
                feature_dict[col] = val if val is not None else 0
        elif col.startswith('AWAY_'):
            base_col = col.replace('AWAY_', '')
            # Check for H2H features
            if base_col.startswith('H2H_') and h2h_away:
                feature_dict[col] = h2h_away.get(base_col, 0)
            # Check for standings features
            elif away_standings and base_col in away_standings:
                feature_dict[col] = away_standings.get(base_col, 0)
            # Check for odds features
            elif away_odds and base_col in ['AVG_ODDS', 'BEST_ODDS', 'WORST_ODDS', 'ODDS_SPREAD', 'IMPLIED_PROB']:
                feature_dict[col] = away_odds.get(f'AWAY_{base_col}', 0)
            else:
                val = away_features.get(base_col, None)
                feature_dict[col] = val if val is not None else 0
        elif col == 'RANK_DIFF':
            if home_standings and away_standings:
                feature_dict[col] = home_standings.get('CONF_RANK', 8) - away_standings.get('CONF_RANK', 8)
            else:
                feature_dict[col] = 0
        elif col == 'WIN_PCT_DIFF':
            if home_standings and away_standings:
                feature_dict[col] = home_standings.get('WIN_PCT', 0.5) - away_standings.get('WIN_PCT', 0.5)
            else:
                feature_dict[col] = 0
        elif col == 'BOOKMAKER_COUNT':
            # Odds feature - number of bookmakers
            feature_dict[col] = home_odds.get('BOOKMAKER_COUNT', 0) if home_odds else 0
        else:
            # Unknown column type - default to 0
            feature_dict[col] = 0
    
    # Convert to array and handle any remaining NaN
    features = np.array([feature_dict.get(col, 0) for col in feature_cols]).reshape(1, -1)
    
    # Replace any remaining NaN with 0
    features = np.nan_to_num(features, nan=0.0)
    
    # Get predictions from each model
    predictions = []
    for model, scaler, model_type in zip(models, scalers, model_types):
        features_scaled = scaler.transform(features)
        
        if model_type == 'xgboost':
            pred = model.predict_proba(features_scaled)[0][1]
        elif model_type == 'random_forest':
            # Use top features if feature selection was applied
            if hasattr(model, '_top_feature_indices'):
                features_rf = features_scaled[:, model._top_feature_indices]
            else:
                features_rf = features_scaled
            pred = model.predict_proba(features_rf)[0][1]
        elif model_type == 'logistic':
            pred = model.predict_proba(features_scaled)[0][1]
        else:  # keras
            pred = model.predict(features_scaled, verbose=0)[0][0]
        
        predictions.append(float(pred))
    
    # If stacking meta-model is available, build meta feature vector and use it
    if meta_clf is not None and platt is not None:
        raw = np.array(predictions).reshape(1, -1)
        conf = np.abs(raw - 0.5)
        meta_feats = [raw, conf]
        if 'ELO_DIFF' in feature_cols:
            elo_diff = home_features.get('ELO_DIFF', 0) if isinstance(home_features, dict) else 0
            meta_feats.append(np.array([[elo_diff]]))
        meta_input = np.hstack(meta_feats)

        # Try to predict with the saved meta_clf; if feature mismatch, fall back to raw-only
        try:
            meta_prob = meta_clf.predict_proba(meta_input)[:, 1][0]
        except Exception:
            try:
                meta_prob = meta_clf.predict_proba(raw)[:, 1][0]
            except Exception:
                # give up on stacking, fallback to averaged predictions
                meta_prob = None

        if meta_prob is not None:
            try:
                ensemble_pred = float(platt.predict_proba(np.array([[meta_prob]]))[:, 1][0])
            except Exception:
                ensemble_pred = float(meta_prob)
        else:
            # fallback to simple average
            ensemble_pred = np.mean(predictions)
    else:
        # If ensemble weights are provided, use weighted average with threshold if available
        if ensemble_weights is not None:
            w = np.array(ensemble_weights)
            # normalize weights if not normalized
            if not np.isclose(w.sum(), 1.0):
                w = w / w.sum()
            ensemble_prob = float(np.dot(w, np.array(predictions)))
            if ensemble_threshold is not None:
                ensemble_pred = float(1.0 if ensemble_prob >= ensemble_threshold else 0.0)
            else:
                ensemble_pred = ensemble_prob
        else:
            # Average predictions
            ensemble_pred = np.mean(predictions)
    
    return {
        'home_win_probability': float(ensemble_pred),
        'away_win_probability': float(1 - ensemble_pred),
        'predicted_winner': 'HOME' if ensemble_pred > 0.5 else 'AWAY',
        'confidence': abs(ensemble_pred - 0.5) * 2,
        'individual_predictions': predictions,
        'model_agreement': 1 - np.std(predictions)  # High = models agree
    }


def main(single_model=None):
    """Main prediction function
    
    Args:
        single_model: If specified, use only this model ('lstm', 'xgboost', 'random_forest', 'logistic')
                      If None, use full ensemble
    """
    print("="*70)
    if single_model:
        print(f"NBA GAME PREDICTIONS - {single_model.upper()} ONLY")
    else:
        print("NBA GAME PREDICTIONS - ENSEMBLE MODE")
        print("XGBoost + Random Forest + Logistic + LSTM")
    print("="*70)
    
    # Load ensemble
    models, scalers, feature_cols, model_types, meta_clf, platt, ensemble_weights, ensemble_threshold = load_ensemble()
    
    # Filter to single model if specified
    if single_model:
        # Find the model matching the requested type
        indices = [i for i, mt in enumerate(model_types) if mt == single_model]
        if not indices:
            print(f"❌ Model type '{single_model}' not found in ensemble!")
            print(f"   Available: {model_types}")
            return
        idx = indices[0]
        models = [models[idx]]
        scalers = [scalers[idx]]
        model_types = [model_types[idx]]
        # Disable stacking for single model
        meta_clf = None
        platt = None
        ensemble_weights = None
        print(f"✓ Using {single_model.upper()} model only")
    
    # Fetch recent data
    print("\n📊 Fetching recent team statistics...")
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable='2025-26',
        league_id_nullable='00'
    )
    games_df = gamefinder.get_data_frames()[0]
    games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
    games_df = games_df.sort_values('GAME_DATE')
    print(f"✓ Loaded {len(games_df)} games from 2025-26 season")
    
    # Get today's games
    print("\n🏀 Fetching today's games...")
    todays_games, game_date, all_finished = get_todays_games()
    eastern_today = get_eastern_date()
    print(f"   Game date from API: {game_date}")
    print(f"   Current date (Eastern): {eastern_today}")
    
    if len(todays_games) == 0:
        print("\n❌ No games found for today.")
        print("   The NBA schedule might be empty, or it's the off-season.")
        return
    
    # Warn if showing old games
    if all_finished and game_date != eastern_today:
        print(f"\n⚠️  WARNING: All games shown are FINAL (from {game_date})")
        print(f"   Today is {eastern_today} - today's games may not have started yet.")
        print("   Predictions will still run but won't be saved to history (already recorded).")
    
    print(f"✓ Found {len(todays_games)} games for {game_date}")
    
    # Fetch live standings for standings features
    live_standings = get_live_standings()
    
    # Fetch live odds (optional - requires API key)
    live_odds = get_live_odds()
    
    # Predict each game
    print("\n" + "="*70)
    print("ENSEMBLE PREDICTIONS")
    print("="*70)
    
    all_teams = teams.get_teams()
    predictions = []
    
    for i, game in enumerate(todays_games):
        print(f"\n🏀 Game {i+1}/{len(todays_games)}")
        print("-"*70)
        
        # Get team names
        home_team = [t for t in all_teams if t['id'] == game['home_team_id']][0]
        away_team = [t for t in all_teams if t['id'] == game['away_team_id']][0]
        
        print(f"   {away_team['full_name']} @ {home_team['full_name']}")
        print(f"   Status: {game['game_status']}")
        
        # Get recent stats
        home_features = get_recent_team_stats(game['home_team_id'], games_df)
        away_features = get_recent_team_stats(game['away_team_id'], games_df)
        
        if home_features is None or away_features is None:
            print("   ⚠️  Not enough data for prediction")
            continue
        
        # Compute head-to-head
        h2h_home = compute_head_to_head(games_df, game['home_team_id'], game['away_team_id'])
        h2h_away = compute_head_to_head(games_df, game['away_team_id'], game['home_team_id'])
        
        # Get standings for this game
        home_standings = live_standings.get(game['home_team_id'], {
            'WINS': 0, 'LOSSES': 0, 'WIN_PCT': 0.5,
            'CONF_RANK': 8, 'LEAGUE_RANK': 15, 'GAMES_BACK': 0, 'STREAK': 0
        })
        away_standings = live_standings.get(game['away_team_id'], {
            'WINS': 0, 'LOSSES': 0, 'WIN_PCT': 0.5,
            'CONF_RANK': 8, 'LEAGUE_RANK': 15, 'GAMES_BACK': 0, 'STREAK': 0
        })
        
        # Get odds for this game (if available)
        game_odds = match_game_to_odds(home_team['full_name'], away_team['full_name'], live_odds)
        
        # Predict with ensemble
        try:
            prediction = predict_game_ensemble(
                models, scalers, feature_cols, model_types,
                home_features, away_features,
                meta_clf=meta_clf, platt=platt,
                ensemble_weights=ensemble_weights, ensemble_threshold=ensemble_threshold,
                h2h_home=h2h_home, h2h_away=h2h_away,
                home_standings=home_standings, away_standings=away_standings,
                home_odds=game_odds, away_odds=game_odds  # Same dict has both HOME_* and AWAY_* features
            )
            
            winner = home_team['full_name'] if prediction['predicted_winner'] == 'HOME' else away_team['full_name']
            
            print(f"\n   🏆 Predicted Winner: {winner}")
            print(f"   📊 Confidence: {prediction['confidence']*100:.1f}%")
            print(f"   🏠 Home Win Prob: {prediction['home_win_probability']*100:.1f}%")
            print(f"   ✈️  Away Win Prob: {prediction['away_win_probability']*100:.1f}%")
            
            # Show bookmaker odds (if available)
            odds_display = format_odds_display(game_odds)
            if odds_display:
                print(odds_display)
                # Show value comparison (model vs market)
                model_prob = prediction['home_win_probability']
                market_prob = game_odds.get('HOME_IMPLIED_PROB', 0)
                if market_prob > 0:
                    value_diff = (model_prob - market_prob) * 100
                    value_emoji = "🔥" if abs(value_diff) > 5 else "🤔"
                    print(f"   {value_emoji} Value: Model {model_prob*100:.1f}% vs Market {market_prob*100:.1f}% ({value_diff:+.1f}%)")
            
            # Show fatigue info if available
            home_rest = home_features.get('DAYS_REST', 'N/A')
            away_rest = away_features.get('DAYS_REST', 'N/A')
            home_b2b = "⚠️ B2B" if home_features.get('IS_BACK_TO_BACK', 0) == 1 else ""
            away_b2b = "⚠️ B2B" if away_features.get('IS_BACK_TO_BACK', 0) == 1 else ""
            print(f"\n   😴 Rest: Home {home_rest}d {home_b2b} | Away {away_rest}d {away_b2b}")
            print(f"   🆚 H2H: Home {h2h_home['H2H_WIN_RATE']*100:.0f}% ({h2h_home['H2H_GAMES']} games)")
            
            # Show standings info
            home_record = f"{home_standings['WINS']}-{home_standings['LOSSES']}"
            away_record = f"{away_standings['WINS']}-{away_standings['LOSSES']}"
            print(f"   📈 Standings: Home #{home_standings['CONF_RANK']} ({home_record}) | Away #{away_standings['CONF_RANK']} ({away_record})")
            
            # Show individual model predictions
            print(f"\n   🤖 Model Agreement: {prediction['model_agreement']*100:.1f}%")
            indiv = prediction['individual_predictions']
            # Map model types to display names
            model_type_labels = {
                'xgboost': 'XGBoost',
                'random_forest': 'RF', 
                'logistic': 'Logistic',
                'keras': 'LSTM'
            }
            for j, mtype in enumerate(model_types[:len(indiv)]):
                label = model_type_labels.get(mtype, mtype)
                print(f"      {label}:".ljust(16) + f"{indiv[j]*100:.1f}%")
            
            predictions.append({
                'away_team': away_team['full_name'],
                'home_team': home_team['full_name'],
                'predicted_winner': winner,
                'confidence': prediction['confidence'],
                'home_win_prob': prediction['home_win_probability'],
                'model_agreement': prediction['model_agreement']
            })
            
        except Exception as e:
            print(f"   ❌ Error predicting: {e}")
    
    # Summary
    if predictions:
        print("\n" + "="*70)
        print("SUMMARY - ENSEMBLE PREDICTIONS")
        print("="*70)
        
        # Bet quality tiers based on confidence (probability distance from 50%)
        # 50% confidence = 75% probability, 40% = 70%, 30% = 65%, 20% = 60%
        for pred in predictions:
            conf = pred['confidence']
            
            # Bet quality tier
            if conf >= 0.50:  # 75%+ probability
                bet_tier = "🔥 EXCELLENT BET"
                tier_color = "🔥"
            elif conf >= 0.40:  # 70%+ probability
                bet_tier = "💰 STRONG BET"
                tier_color = "💰"
            elif conf >= 0.30:  # 65%+ probability
                bet_tier = "⚡ GOOD BET"
                tier_color = "⚡"
            elif conf >= 0.20:  # 60%+ probability
                bet_tier = "📊 MODERATE BET"
                tier_color = "📊"
            elif conf >= 0.10:  # 55%+ probability
                bet_tier = "❓ RISKY - LOW EDGE"
                tier_color = "❓"
            else:  # <55% probability
                bet_tier = "⛔ SKIP - COIN FLIP"
                tier_color = "⛔"
            
            # Agreement icon
            agree_icon = "✅" if pred['model_agreement'] > 0.90 else "⚠️" if pred['model_agreement'] > 0.80 else "❗"
            
            print(f"\n{tier_color} {pred['away_team']} @ {pred['home_team']}")
            print(f"   → {pred['predicted_winner']} ({pred['home_win_prob']*100:.1f}% prob, {conf*100:.0f}% confidence)")
            print(f"   {agree_icon} Model agreement: {pred['model_agreement']*100:.0f}%")
            print(f"   📋 {bet_tier}")
        
        # Legend
        print("\n" + "-"*70)
        print("BET QUALITY LEGEND:")
        print("   🔥 EXCELLENT (75%+) | 💰 STRONG (70-75%) | ⚡ GOOD (65-70%)")
        print("   📊 MODERATE (60-65%) | ❓ RISKY (55-60%) | ⛔ SKIP (<55%)")
        print("-"*70)
        
        # Save predictions to history (skip if showing old finished games)
        if all_finished and game_date != eastern_today:
            print(f"\n📋 Skipping history save (showing old games from {game_date})")
        else:
            save_predictions_to_history(predictions, date_str=game_date)
    
    print("\n" + "="*70)
    print("Predictions complete! Good luck! 🍀")
    print("="*70)


def predict_with_single_model(model_name, matchup=None):
    """Wrapper function for single model predictions from CLI
    
    Args:
        model_name: Model to use ('xgboost', 'random_forest', 'logistic', 'lstm')
        matchup: Optional tuple of (away_team, home_team) abbreviations
    """
    # Map lstm to keras (internal name)
    if model_name == 'lstm':
        model_name = 'keras'
    
    display_names = {'keras': 'LSTM', 'xgboost': 'XGBoost', 
                     'random_forest': 'Random Forest', 'logistic': 'Logistic'}
    print(f"🎯 Using SINGLE MODEL: {display_names.get(model_name, model_name.upper())}")
    
    main(single_model=model_name)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    single_model = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        # Map aliases
        if arg == 'lstm':
            arg = 'keras'  # LSTM is stored as 'keras' type
        if arg == 'rf':
            arg = 'random_forest'  # RF alias
        if arg in ['keras', 'xgboost', 'random_forest', 'logistic']:
            single_model = arg
            display_names = {'keras': 'LSTM', 'xgboost': 'XGBoost', 
                           'random_forest': 'Random Forest', 'logistic': 'Logistic'}
            print(f"🎯 Using SINGLE MODEL: {display_names.get(arg, arg.upper())}")
        elif arg == '--help' or arg == '-h':
            print("Usage: python predict_with_ensemble.py [model]")
            print("\nOptions:")
            print("  (no args)     - Use full ensemble (XGBoost + RF + Logistic + LSTM)")
            print("  lstm          - Use only LSTM model")
            print("  xgboost       - Use only XGBoost model")
            print("  random_forest - Use only Random Forest model")
            print("  rf            - Alias for random_forest")
            print("  logistic      - Use only Logistic Regression model")
            sys.exit(0)
    
    main(single_model=single_model)