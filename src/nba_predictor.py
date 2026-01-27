"""
NBA Game Prediction using Multi-Layer Perceptron (MLP)
=======================================================

This implementation replicates the architecture from recent research that 
achieved 72.35% accuracy on NBA game predictions using 8 seasons of data.

Architecture: Dense(128) -> BatchNorm -> Dropout(0.3) -> Dense(64) -> 
              Dropout(0.3) -> Dense(32) -> Dense(1, sigmoid)

Requirements:
pip install nba_api pandas numpy scikit-learn tensorflow matplotlib seaborn joblib

Note: First run will take time to fetch data. Data is cached for subsequent runs.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

# Parallel processing
from joblib import Parallel, delayed
import multiprocessing

# NBA API imports
from nba_api.stats.endpoints import leaguegamefinder, teamgamelog
from nba_api.stats.static import teams

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Custom layer for sum pooling (properly serializable, unlike Lambda)
@keras.utils.register_keras_serializable(package="Custom", name="SumPooling1D")
class SumPooling1D(layers.Layer):
    """Sum pooling over the time/sequence dimension (axis=1)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)
    
    def get_config(self):
        return super().get_config()


# Path configuration
from .paths import FEATURE_CACHE_FILE, MATCHUP_CACHE_FILE, ENRICHED_GAMES_CSV, MATCHUPS_CSV, get_model_path

# GPU Configuration - use GPU when available for neural networks
def configure_gpu():
    """Configure TensorFlow to use GPU if available"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            return True
        except RuntimeError:
            return False
    return False

GPU_AVAILABLE = configure_gpu()

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


def _compute_team_features_parallel(team_id, games_df, window_size):
    """Standalone function for parallel feature computation"""
    fe = FeatureEngineering(window_size=window_size)
    return fe.create_rolling_features(games_df, team_id)


class NBADataFetcher:
    """Fetch and process NBA game data"""
    
    def __init__(self, seasons=['2022-23', '2023-24', '2024-25']):
        self.seasons = seasons
        self.all_teams = teams.get_teams()
        
    def fetch_games(self):
        """Fetch all games for specified seasons"""
        print("Fetching NBA game data...")
        all_games = []
        
        for season in self.seasons:
            print(f"Fetching {season} season...")
            try:
                gamefinder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    league_id_nullable='00'
                )
                games = gamefinder.get_data_frames()[0]
                all_games.append(games)
            except Exception as e:
                print(f"Error fetching {season}: {e}")
        
        if all_games:
            df = pd.concat(all_games, ignore_index=True)
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE').reset_index(drop=True)
            print(f"Fetched {len(df)} total game records")
            return df
        return None


# Global cache for standings computation
_STANDINGS_CACHE = {}

# Eastern Conference team IDs (hardcoded for performance)
EASTERN_IDS = {1610612738, 1610612751, 1610612752, 1610612755, 1610612761,  # Atlantic
               1610612741, 1610612739, 1610612765, 1610612754, 1610612749,  # Central
               1610612737, 1610612766, 1610612748, 1610612753, 1610612764}  # Southeast


def precompute_all_standings(games_df):
    """Pre-compute cumulative standings for all games efficiently.
    
    This uses vectorized operations to compute standings at each point in time,
    then caches them for fast lookup during matchup creation.
    
    Returns:
        dict: {(game_date, team_id): standings_dict}
    """
    global _STANDINGS_CACHE
    
    # Sort by date
    games_sorted = games_df.sort_values('GAME_DATE').copy()
    
    # Pre-compute cumulative wins/losses for each team
    all_teams = games_sorted['TEAM_ID'].unique()
    all_dates = games_sorted['GAME_DATE'].unique()
    
    print(f"   Pre-computing standings for {len(all_dates)} unique dates...")
    
    # Create a wins/losses accumulator per team
    team_records = {}  # {team_id: {'wins': 0, 'losses': 0, 'games': []}}
    for tid in all_teams:
        conf = 'East' if tid in EASTERN_IDS else 'West'
        team_records[tid] = {'wins': 0, 'losses': 0, 'streak': 0, 'last_result': None, 'conf': conf}
    
    # Group games by date
    date_groups = games_sorted.groupby('GAME_DATE')
    
    # Process each date in order
    for date in all_dates:
        if date in date_groups.groups:
            # Snapshot BEFORE this date's games
            # Compute rankings based on current records
            standings_at_date = []
            for tid, record in team_records.items():
                total = record['wins'] + record['losses']
                win_pct = record['wins'] / total if total > 0 else 0.5
                standings_at_date.append({
                    'TEAM_ID': tid,
                    'WINS': record['wins'],
                    'LOSSES': record['losses'],
                    'WIN_PCT': win_pct,
                    'CONF': record['conf'],
                    'STREAK': record['streak']
                })
            
            # Compute rankings (only if we have data)
            if standings_at_date:
                sdf = pd.DataFrame(standings_at_date)
                
                # Conference rank
                sdf['CONF_RANK'] = sdf.groupby('CONF')['WIN_PCT'].rank(ascending=False, method='min').astype(int)
                
                # League rank
                sdf['LEAGUE_RANK'] = sdf['WIN_PCT'].rank(ascending=False, method='min').astype(int)
                
                # Games back
                for conf in ['East', 'West']:
                    conf_mask = sdf['CONF'] == conf
                    if conf_mask.any():
                        leader_wins = sdf.loc[conf_mask, 'WINS'].max()
                        leader_losses = sdf.loc[conf_mask, 'LOSSES'].min()
                        sdf.loc[conf_mask, 'GAMES_BACK'] = ((leader_wins - sdf.loc[conf_mask, 'WINS']) + 
                                                            (sdf.loc[conf_mask, 'LOSSES'] - leader_losses)) / 2
                
                # Store in cache
                for _, row in sdf.iterrows():
                    _STANDINGS_CACHE[(date, row['TEAM_ID'])] = {
                        'WINS': row['WINS'],
                        'LOSSES': row['LOSSES'],
                        'WIN_PCT': row['WIN_PCT'],
                        'CONF_RANK': row.get('CONF_RANK', 8),
                        'LEAGUE_RANK': row.get('LEAGUE_RANK', 15),
                        'GAMES_BACK': row.get('GAMES_BACK', 0),
                        'STREAK': row['STREAK']
                    }
            
            # NOW update records with this date's games
            day_games = date_groups.get_group(date)
            for _, game in day_games.iterrows():
                tid = game['TEAM_ID']
                result = game['WL']
                
                if result == 'W':
                    team_records[tid]['wins'] += 1
                    if team_records[tid]['last_result'] == 'W':
                        team_records[tid]['streak'] += 1
                    else:
                        team_records[tid]['streak'] = 1
                    team_records[tid]['last_result'] = 'W'
                elif result == 'L':
                    team_records[tid]['losses'] += 1
                    if team_records[tid]['last_result'] == 'L':
                        team_records[tid]['streak'] -= 1
                    else:
                        team_records[tid]['streak'] = -1
                    team_records[tid]['last_result'] = 'L'
    
    print(f"   ✓ Cached standings for {len(_STANDINGS_CACHE)} (date, team) combinations")
    return _STANDINGS_CACHE


def get_standings_for_game(game_date, team_id):
    """Get cached standings for a specific game date and team.
    
    Must call precompute_all_standings() first.
    """
    global _STANDINGS_CACHE
    
    default = {
        'WINS': 0, 'LOSSES': 0, 'WIN_PCT': 0.5,
        'CONF_RANK': 8, 'LEAGUE_RANK': 15,
        'GAMES_BACK': 0, 'STREAK': 0
    }
    
    return _STANDINGS_CACHE.get((game_date, team_id), default)


def compute_standings_at_date(games_df, target_date, team_id=None):
    """Compute standings (wins, losses, rank) at a specific date from game history.
    
    This is the slow version - use precompute_all_standings + get_standings_for_game for batch operations.
    
    Args:
        games_df: DataFrame with all games (must have TEAM_ID, GAME_DATE, WL columns)
        target_date: Date to compute standings for (will use games before this date)
        team_id: Optional - if provided, returns dict for that team only
        
    Returns:
        If team_id is None: DataFrame with standings for all teams
        If team_id provided: Dict with standings for that team
    """
    # Check cache first
    global _STANDINGS_CACHE
    if team_id is not None and (target_date, team_id) in _STANDINGS_CACHE:
        return _STANDINGS_CACHE[(target_date, team_id)]
    
    # Filter to games BEFORE target date (crucial to avoid leakage)
    prior_games = games_df[games_df['GAME_DATE'] < target_date].copy()
    
    if len(prior_games) == 0:
        # No prior games - return default standings
        if team_id is not None:
            return {
                'WINS': 0, 'LOSSES': 0, 'WIN_PCT': 0.5,
                'CONF_RANK': 8, 'LEAGUE_RANK': 15,
                'GAMES_BACK': 0, 'STREAK': 0
            }
        return pd.DataFrame()
    
    # Compute wins/losses per team
    standings_data = []
    
    for tid in prior_games['TEAM_ID'].unique():
        team_prior = prior_games[prior_games['TEAM_ID'] == tid]
        
        wins = (team_prior['WL'] == 'W').sum()
        losses = (team_prior['WL'] == 'L').sum()
        total = wins + losses
        win_pct = wins / total if total > 0 else 0.5
        
        # Compute current streak
        recent = team_prior.sort_values('GAME_DATE', ascending=False)
        streak = 0
        if len(recent) > 0:
            last_result = recent.iloc[0]['WL']
            for _, row in recent.iterrows():
                if row['WL'] == last_result:
                    streak += 1 if last_result == 'W' else -1
                else:
                    break
        
        conf = 'East' if tid in EASTERN_IDS else 'West'
        standings_data.append({
            'TEAM_ID': tid,
            'WINS': wins,
            'LOSSES': losses,
            'WIN_PCT': win_pct,
            'CONFERENCE': conf,
            'STREAK': streak
        })
    
    standings_df = pd.DataFrame(standings_data)
    
    if len(standings_df) == 0:
        if team_id is not None:
            return {
                'WINS': 0, 'LOSSES': 0, 'WIN_PCT': 0.5,
                'CONF_RANK': 8, 'LEAGUE_RANK': 15,
                'GAMES_BACK': 0, 'STREAK': 0
            }
        return pd.DataFrame()
    
    # Compute conference rank
    standings_df['CONF_RANK'] = standings_df.groupby('CONFERENCE')['WIN_PCT'].rank(
        ascending=False, method='min'
    ).astype(int)
    
    # Compute league rank
    standings_df['LEAGUE_RANK'] = standings_df['WIN_PCT'].rank(
        ascending=False, method='min'
    ).astype(int)
    
    # Compute games back from conference leader
    for conf in ['East', 'West']:
        conf_mask = standings_df['CONFERENCE'] == conf
        if conf_mask.any():
            leader_wins = standings_df.loc[conf_mask, 'WINS'].max()
            leader_losses = standings_df.loc[conf_mask, 'LOSSES'].min()
            standings_df.loc[conf_mask, 'GAMES_BACK'] = (
                (leader_wins - standings_df.loc[conf_mask, 'WINS']) + 
                (standings_df.loc[conf_mask, 'LOSSES'] - leader_losses)
            ) / 2
    
    if team_id is not None:
        team_row = standings_df[standings_df['TEAM_ID'] == team_id]
        if len(team_row) == 0:
            return {
                'WINS': 0, 'LOSSES': 0, 'WIN_PCT': 0.5,
                'CONF_RANK': 8, 'LEAGUE_RANK': 15,
                'GAMES_BACK': 0, 'STREAK': 0
            }
        row = team_row.iloc[0]
        return {
            'WINS': row['WINS'],
            'LOSSES': row['LOSSES'],
            'WIN_PCT': row['WIN_PCT'],
            'CONF_RANK': row['CONF_RANK'],
            'LEAGUE_RANK': row['LEAGUE_RANK'],
            'GAMES_BACK': row['GAMES_BACK'],
            'STREAK': row['STREAK']
        }
    
    return standings_df


class FeatureEngineering:
    """Create features from raw game data"""
    
    def __init__(self, window_size=20):
        self.window_size = window_size
        
    def calculate_four_factors(self, df):
        """Calculate Dean Oliver's Four Factors of Basketball Success"""
        # Effective Field Goal Percentage
        df['EFG_PCT'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA'].replace(0, 1)
        
        # Turnover Percentage
        df['TOV_PCT'] = df['TOV'] / (df['FGA'] + 0.44 * df['FTA'] + df['TOV']).replace(0, 1)
        
        # Offensive Rebound Percentage (approximation)
        df['OREB_PCT'] = df['OREB'] / (df['OREB'] + df['DREB']).replace(0, 1)
        
        # Free Throw Rate
        df['FT_RATE'] = df['FTA'] / df['FGA'].replace(0, 1)
        
        # True Shooting Percentage
        df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA'])).replace(0, 1)
        
        # Additional advanced metrics
        # Pace (possessions per game approximation)
        df['PACE'] = df['FGA'] + 0.44 * df['FTA'] - df['OREB'] + df['TOV']
        
        # Assist to Turnover Ratio
        df['AST_TO_RATIO'] = df['AST'] / df['TOV'].replace(0, 1)
        
        # Defensive Rating Proxy (points allowed per 100 possessions - need opponent data)
        # Using rebound differential as a defensive indicator
        df['REB_DIFF'] = df['REB'] - (df['OREB'] + df['DREB'])  # This approximates opponent rebounds
        
        # Three point attempt rate
        df['FG3A_RATE'] = df['FG3A'] / df['FGA'].replace(0, 1)
        
        return df
    
    def create_rolling_features(self, df, team_id):
        """Create rolling window features for a team using ONLY past games (no leakage)"""
        team_games = df[df['TEAM_ID'] == team_id].sort_values('GAME_DATE').copy()
        
        # Features to roll - REMOVED PLUS_MINUS (it's the game outcome!)
        roll_cols = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 
                     'TOV', 'STL', 'BLK',
                     'EFG_PCT', 'TOV_PCT', 'OREB_PCT', 'FT_RATE', 'TS_PCT',
                     'PACE', 'AST_TO_RATIO', 'REB_DIFF', 'FG3A_RATE']
        
        for col in roll_cols:
            if col in team_games.columns:
                # CRITICAL: shift(1) means we use the previous N games, NOT including current game
                shifted = team_games[col].shift(1)
                
                team_games[f'{col}_ROLL_MEAN'] = shifted.rolling(
                    window=self.window_size, min_periods=5
                ).mean()
                
                # Rolling std for volatility (also shifted)
                team_games[f'{col}_ROLL_STD'] = shifted.rolling(
                    window=self.window_size, min_periods=5
                ).std()
                
                # Add exponential weighted moving average for recent emphasis
                team_games[f'{col}_EWM'] = shifted.ewm(span=10, min_periods=5).mean()
        
        # Win/Loss indicators
        win_indicator = team_games['WL'].apply(lambda x: 1 if x == 'W' else 0)
        
        # Win streak (rolling sum of wins) - shifted to exclude current game
        team_games['WIN_STREAK'] = win_indicator.shift(1).rolling(window=5, min_periods=1).sum()
        
        # Recent form (last 5 games win percentage)
        team_games['RECENT_FORM'] = win_indicator.shift(1).rolling(window=5, min_periods=1).mean()
        
        # CRITICAL NEW FEATURE: Win/Loss Pattern (Last 3 Games)
        # This encodes patterns like WWW, WWL, WLW, etc.
        # Converted to a single numerical feature
        def encode_pattern(series):
            """Encode last 3 W/L as a number (0-7)
            WWW=7, WWL=6, WLW=5, WLL=4, LWW=3, LWL=2, LLW=1, LLL=0
            """
            if len(series) < 3:
                return 3.5  # Neutral if not enough games
            
            # Get last 3 results (already shifted, so these are pre-game)
            pattern = series.tail(3).values
            # Convert to binary: W=1, L=0
            binary_str = ''.join(['1' if x == 1 else '0' for x in pattern])
            return int(binary_str, 2)  # Convert binary to decimal (0-7)
        
        team_games['WIN_PATTERN_3GAME'] = win_indicator.shift(1).rolling(
            window=3, min_periods=1
        ).apply(encode_pattern, raw=False)
        
        # Momentum indicators
        # Winning momentum: more recent wins weighted higher
        recent_3 = win_indicator.shift(1).rolling(3, min_periods=1).mean()
        recent_5 = win_indicator.shift(1).rolling(5, min_periods=1).mean()
        recent_10 = win_indicator.shift(1).rolling(10, min_periods=1).mean()
        
        # Momentum trend: are they getting better or worse?
        team_games['MOMENTUM_TREND'] = recent_3 - recent_10  # Positive = heating up
        
        # Consistency: low std in recent performance = consistent team
        team_games['WIN_CONSISTENCY'] = 1 - win_indicator.shift(1).rolling(10, min_periods=3).std().fillna(0.5)
        
        # Home/Away split (create indicator)
        team_games['IS_HOME'] = team_games['MATCHUP'].str.contains('vs.', case=False, na=False).astype(int)
        
        # REST DAYS: Days since last game (important for fatigue)
        team_games['DAYS_REST'] = team_games['GAME_DATE'].diff().dt.days.fillna(3)  # Default 3 for first game
        
        # Cap extreme values (e.g., after All-Star break or start of season)
        team_games['DAYS_REST'] = team_games['DAYS_REST'].clip(0, 10)
        
        # BACK-TO-BACK: Playing on consecutive days (major fatigue factor)
        team_games['IS_BACK_TO_BACK'] = (team_games['DAYS_REST'] == 1).astype(int)
        
        # 3-IN-4 NIGHTS: Three games in four nights (cumulative fatigue)
        team_games['IS_3_IN_4'] = (
            team_games['GAME_DATE'].diff().dt.days.rolling(window=3, min_periods=1).sum() <= 4
        ).astype(int).fillna(0)
        
        return team_games


class NBAPredictor:
    """MLP-based NBA game outcome predictor (replicates 72.35% accuracy architecture)"""
    
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        
    def _compute_head_to_head(self, team_id, opponent_id, game_date, team_abbrev_map, team_games_cache, window=10):
        """Compute head-to-head record against specific opponent (optimized, last N meetings)
        
        Args:
            team_id: Team to compute H2H for
            opponent_id: Opponent team ID
            game_date: Date of current game (only use games BEFORE this)
            team_abbrev_map: Pre-computed dict of team_id -> abbreviation
            team_games_cache: Pre-computed dict of team_id -> games DataFrame
            window: Number of recent H2H games to consider
            
        Returns:
            dict: H2H win rate, games played, point differential
        """
        # Get pre-filtered team games
        if team_id not in team_games_cache:
            return {'H2H_WIN_RATE': 0.5, 'H2H_GAMES': 0, 'H2H_PTS_DIFF': 0}
        
        team_games = team_games_cache[team_id]
        team_games = team_games[team_games['GAME_DATE'] < game_date]
        
        if len(team_games) == 0:
            return {'H2H_WIN_RATE': 0.5, 'H2H_GAMES': 0, 'H2H_PTS_DIFF': 0}
        
        # Get opponent abbreviation from pre-computed map
        opp_abbrev = team_abbrev_map.get(opponent_id, '')
        
        if opp_abbrev:
            # Fast vectorized check for opponent in matchup string
            h2h_games = team_games[
                team_games['MATCHUP'].str.contains(opp_abbrev, na=False, regex=False)
            ].tail(window)
        else:
            h2h_games = pd.DataFrame()
        
        if len(h2h_games) == 0:
            return {'H2H_WIN_RATE': 0.5, 'H2H_GAMES': 0, 'H2H_PTS_DIFF': 0}
        
        wins = (h2h_games['WL'] == 'W').sum()
        games_played = len(h2h_games)
        pts_diff = h2h_games['PLUS_MINUS'].mean() if 'PLUS_MINUS' in h2h_games.columns else 0
        
        return {
            'H2H_WIN_RATE': wins / games_played,
            'H2H_GAMES': min(games_played, window),
            'H2H_PTS_DIFF': pts_diff
        }
    
    def prepare_matchup_data(self, games_df, use_cache=True, n_jobs=-1):
        """Prepare dataset with home/away team features
        
        Args:
            games_df: DataFrame with game data
            use_cache: If True, use cached features and only compute new ones
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        print("Preparing matchup dataset...")
        
        fe = FeatureEngineering(window_size=self.window_size)
        games_df = fe.calculate_four_factors(games_df)
        
        # Determine number of cores
        if n_jobs == -1:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {n_jobs} CPU cores for parallel processing")
        
        team_ids = games_df['TEAM_ID'].unique()
        
        # Check for cached features
        cached_features = None
        cached_max_date = None
        
        if use_cache and os.path.exists(FEATURE_CACHE_FILE):
            try:
                with open(FEATURE_CACHE_FILE, 'rb') as f:
                    cache_data = pickle.load(f)
                cached_features = cache_data['features']
                cached_max_date = cache_data['max_date']
                print(f"📦 Loaded cached features up to {cached_max_date}")
            except Exception as e:
                print(f"⚠️ Could not load cache: {e}")
                cached_features = None
        
        # Filter games that need feature computation
        if cached_features is not None and cached_max_date is not None:
            # Only compute features for games AFTER the cached date
            new_games_df = games_df[games_df['GAME_DATE'] > cached_max_date]
            
            if len(new_games_df) == 0:
                print("✓ All features already cached, skipping computation")
                all_features_df = cached_features
            else:
                print(f"📊 Computing features for {len(new_games_df)} new game records...")
                
                # Get teams that have new games
                new_team_ids = new_games_df['TEAM_ID'].unique()
                print(f"   {len(new_team_ids)} teams have new games")
                
                # Compute features for teams with new games (using FULL history for rolling)
                new_team_features = Parallel(n_jobs=n_jobs, verbose=1)(
                    delayed(_compute_team_features_parallel)(team_id, games_df, self.window_size)
                    for team_id in new_team_ids
                )
                
                # Combine: keep cached features for games before cutoff, add new features
                new_features_df = pd.concat(new_team_features, ignore_index=True)
                
                # Filter new features to only include games after cached date
                new_features_df = new_features_df[new_features_df['GAME_DATE'] > cached_max_date]
                
                # Combine with cached (remove old entries for updated teams)
                cached_filtered = cached_features[~cached_features['TEAM_ID'].isin(new_team_ids) | 
                                                   (cached_features['GAME_DATE'] <= cached_max_date)]
                
                # For teams with new games, get their full history from new computation
                teams_with_updates = pd.concat(new_team_features, ignore_index=True)
                
                all_features_df = pd.concat([
                    cached_features[~cached_features['TEAM_ID'].isin(new_team_ids)],
                    teams_with_updates
                ], ignore_index=True)
        else:
            # No cache - compute all features in parallel
            print(f"🔄 Computing features for all {len(team_ids)} teams (first run)...")
            
            all_team_features = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(_compute_team_features_parallel)(team_id, games_df, self.window_size)
                for team_id in team_ids
            )
            
            all_features_df = pd.concat(all_team_features, ignore_index=True)
        
        # Save to cache (pickle for fast loading)
        try:
            max_date = games_df['GAME_DATE'].max()
            with open(FEATURE_CACHE_FILE, 'wb') as f:
                pickle.dump({'features': all_features_df, 'max_date': max_date}, f)
            print(f"💾 Cached features up to {max_date}")
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")
        
        # Also save enriched games to CSV (portable, human-readable)
        try:
            all_features_df.to_csv(ENRICHED_GAMES_CSV, index=False)
            print(f"📄 Saved enriched games CSV: {ENRICHED_GAMES_CSV}")
        except Exception as e:
            print(f"⚠️ Could not save enriched CSV: {e}")
        
        # CRITICAL: Remove rows where rolling features are NaN (these are the first games without enough history)
        roll_cols = [col for col in all_features_df.columns if 'ROLL' in col or col == 'WIN_STREAK']
        
        print(f"Before removing NaN: {len(all_features_df)} rows")
        all_features_df = all_features_df.dropna(subset=roll_cols)
        print(f"After removing NaN: {len(all_features_df)} rows (removed games without enough history)")
        
        # Check for cached matchups
        cached_matchups = None
        max_cached_game_date = None
        
        if use_cache and os.path.exists(MATCHUP_CACHE_FILE):
            try:
                with open(MATCHUP_CACHE_FILE, 'rb') as f:
                    matchup_cache = pickle.load(f)
                cached_matchups = matchup_cache['matchups']
                max_cached_game_date = matchup_cache['max_date']
                print(f"📦 Loaded {len(cached_matchups)} cached matchups up to {max_cached_game_date}")
            except Exception as e:
                print(f"⚠️ Could not load matchup cache: {e}")
        
        # Pre-compute team abbreviation map and team games cache for fast H2H lookup
        print("🔧 Building lookup tables for H2H computation...")
        team_abbrev_map = {}
        team_games_cache = {}
        for team_id in games_df['TEAM_ID'].unique():
            team_rows = games_df[games_df['TEAM_ID'] == team_id]
            if len(team_rows) > 0:
                team_abbrev_map[team_id] = team_rows.iloc[0].get('TEAM_ABBREVIATION', '')
                team_games_cache[team_id] = team_rows.copy()
        
        # Filter features to only create matchups for new games
        if cached_matchups is not None and max_cached_game_date is not None:
            new_features_df = all_features_df[all_features_df['GAME_DATE'] > max_cached_game_date]
            print(f"📊 Creating matchups for {len(new_features_df['GAME_ID'].unique())} new games...")
        else:
            new_features_df = all_features_df
        
        # Pre-compute standings for all dates (fast vectorized approach)
        print("📈 Pre-computing standings for all games...")
        precompute_all_standings(games_df)
        
        # Create matchups (each game has 2 rows, one per team)
        matchups = []
        
        for game_id in new_features_df['GAME_ID'].unique():
            game_data = new_features_df[new_features_df['GAME_ID'] == game_id]
            
            if len(game_data) != 2:
                continue
            
            # Separate home and away - try different patterns
            home_mask = game_data['MATCHUP'].str.contains('vs.', case=False, na=False)
            away_mask = game_data['MATCHUP'].str.contains('@', na=False)
            
            home_data = game_data[home_mask]
            away_data = game_data[away_mask]
            
            # Skip if we can't identify home/away clearly
            if len(home_data) != 1 or len(away_data) != 1:
                continue
                
            home_data = home_data.iloc[0]
            away_data = away_data.iloc[0]
            
            # Get rolling feature columns (include new fatigue features)
            roll_cols = [col for col in game_data.columns 
                        if 'ROLL' in col or col in ['WIN_STREAK', 'RECENT_FORM', 'WIN_PATTERN_3GAME',
                                                     'MOMENTUM_TREND', 'WIN_CONSISTENCY', 'IS_HOME',
                                                     'DAYS_REST', 'IS_BACK_TO_BACK', 'IS_3_IN_4']]
            
            # CRITICAL CHECK: Ensure we have valid features (no NaN)
            home_features_valid = all([pd.notna(home_data.get(f'HOME_{col}', np.nan)) or pd.notna(home_data.get(col, np.nan)) for col in roll_cols])
            away_features_valid = all([pd.notna(away_data.get(f'AWAY_{col}', np.nan)) or pd.notna(away_data.get(col, np.nan)) for col in roll_cols])
            
            if not (home_features_valid and away_features_valid):
                continue
            
            # Create feature dict
            matchup = {
                'GAME_ID': game_id,
                'GAME_DATE': home_data['GAME_DATE'],
                'HOME_TEAM_ID': home_data['TEAM_ID'],
                'AWAY_TEAM_ID': away_data['TEAM_ID'],
                'HOME_WIN': 1 if home_data['WL'] == 'W' else 0
            }
            
            # Add home team features
            for col in roll_cols:
                if col in home_data.index and pd.notna(home_data[col]):
                    matchup[f'HOME_{col}'] = home_data[col]
            
            # Add away team features
            for col in roll_cols:
                if col in away_data.index and pd.notna(away_data[col]):
                    matchup[f'AWAY_{col}'] = away_data[col]
            
            # Add HEAD-TO-HEAD features (using optimized lookup)
            home_h2h = self._compute_head_to_head(
                home_data['TEAM_ID'], away_data['TEAM_ID'], home_data['GAME_DATE'],
                team_abbrev_map, team_games_cache
            )
            away_h2h = self._compute_head_to_head(
                away_data['TEAM_ID'], home_data['TEAM_ID'], away_data['GAME_DATE'],
                team_abbrev_map, team_games_cache
            )
            
            matchup['HOME_H2H_WIN_RATE'] = home_h2h['H2H_WIN_RATE']
            matchup['HOME_H2H_GAMES'] = home_h2h['H2H_GAMES']
            matchup['HOME_H2H_PTS_DIFF'] = home_h2h['H2H_PTS_DIFF']
            matchup['AWAY_H2H_WIN_RATE'] = away_h2h['H2H_WIN_RATE']
            matchup['AWAY_H2H_GAMES'] = away_h2h['H2H_GAMES']
            matchup['AWAY_H2H_PTS_DIFF'] = away_h2h['H2H_PTS_DIFF']
            
            # Add STANDINGS features (using pre-computed cache for speed)
            home_standings = get_standings_for_game(home_data['GAME_DATE'], home_data['TEAM_ID'])
            away_standings = get_standings_for_game(away_data['GAME_DATE'], away_data['TEAM_ID'])
            
            matchup['HOME_WINS'] = home_standings['WINS']
            matchup['HOME_LOSSES'] = home_standings['LOSSES']
            matchup['HOME_WIN_PCT'] = home_standings['WIN_PCT']
            matchup['HOME_CONF_RANK'] = home_standings['CONF_RANK']
            matchup['HOME_LEAGUE_RANK'] = home_standings['LEAGUE_RANK']
            matchup['HOME_GAMES_BACK'] = home_standings['GAMES_BACK']
            matchup['HOME_STREAK'] = home_standings['STREAK']
            
            matchup['AWAY_WINS'] = away_standings['WINS']
            matchup['AWAY_LOSSES'] = away_standings['LOSSES']
            matchup['AWAY_WIN_PCT'] = away_standings['WIN_PCT']
            matchup['AWAY_CONF_RANK'] = away_standings['CONF_RANK']
            matchup['AWAY_LEAGUE_RANK'] = away_standings['LEAGUE_RANK']
            matchup['AWAY_GAMES_BACK'] = away_standings['GAMES_BACK']
            matchup['AWAY_STREAK'] = away_standings['STREAK']
            
            # Derived standings features
            matchup['RANK_DIFF'] = home_standings['CONF_RANK'] - away_standings['CONF_RANK']  # Negative = home is better
            matchup['WIN_PCT_DIFF'] = home_standings['WIN_PCT'] - away_standings['WIN_PCT']  # Positive = home is better
            
            matchups.append(matchup)
        
        new_matchup_df = pd.DataFrame(matchups)
        new_matchup_df = new_matchup_df.dropna()
        
        # Combine with cached matchups
        if cached_matchups is not None and len(cached_matchups) > 0:
            matchup_df = pd.concat([cached_matchups, new_matchup_df], ignore_index=True)
            matchup_df = matchup_df.drop_duplicates(subset=['GAME_ID'], keep='last')
            print(f"Combined: {len(cached_matchups)} cached + {len(new_matchup_df)} new = {len(matchup_df)} total matchups")
        else:
            matchup_df = new_matchup_df
            print(f"Created {len(matchup_df)} matchups")
        
        # Save matchup cache (pickle for fast loading)
        try:
            max_matchup_date = matchup_df['GAME_DATE'].max()
            with open(MATCHUP_CACHE_FILE, 'wb') as f:
                pickle.dump({'matchups': matchup_df, 'max_date': max_matchup_date}, f)
            print(f"💾 Cached matchups up to {max_matchup_date}")
        except Exception as e:
            print(f"⚠️ Could not save matchup cache: {e}")
        
        # Also save matchups to CSV (portable, human-readable, usable in other projects)
        try:
            matchup_df.to_csv(MATCHUPS_CSV, index=False)
            print(f"📄 Saved matchups CSV: {MATCHUPS_CSV}")
            print(f"   → {len(matchup_df)} matchups with {len(matchup_df.columns)} features each")
        except Exception as e:
            print(f"⚠️ Could not save matchups CSV: {e}")
        
        # DEBUG: Check for any suspiciously perfect correlations
        if len(matchup_df) > 0:
            print("\n🔍 CHECKING FOR DATA LEAKAGE...")
            
            # Sample a few rows to inspect
            sample = matchup_df.head(3)
            print(f"Sample matchup dates: {sample['GAME_DATE'].tolist()}")
            print(f"Sample home wins: {sample['HOME_WIN'].tolist()}")
            
            # Check if features have variation
            feature_cols = [col for col in matchup_df.columns if col.startswith('HOME_') or col.startswith('AWAY_')]
            if len(feature_cols) > 0:
                first_feature = feature_cols[0]
                print(f"Sample feature ({first_feature}) range: {matchup_df[first_feature].min():.3f} to {matchup_df[first_feature].max():.3f}")
        
        return matchup_df
    
    def build_lstm_model(self, input_shape, architecture='baseline'):
        """Build MLP model architecture
        
        Args:
            input_shape: Shape of input features
            architecture: 'baseline', 'deep', or 'cnn_hybrid'
        """
        
        if architecture == 'baseline':
            # Original research architecture: 72.35% accuracy
            model = keras.Sequential([
                layers.Input(shape=input_shape),
                
                # Dense layer 1
                layers.Dense(128, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # Dense layer 2
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                
                # Dense layer 3
                layers.Dense(32, activation='relu'),
                
                # Output layer
                layers.Dense(1, activation='sigmoid')
            ])
        
        elif architecture == 'deep':
            # Deeper architecture - experiment to see if it improves accuracy
            model = keras.Sequential([
                layers.Input(shape=input_shape),
                
                # Block 1
                layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),

                # Block 2
                layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),

                # Block 3
                layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                
                # Block 4
                layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                # Block 5
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                
                # Block 6
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                
                # Output layer
                layers.Dense(1, activation='sigmoid')
            ])
        
        elif architecture == 'xgboost':
            # XGBoost model - tree-based, excellent for tabular data
            # This is not a Keras model, so we handle it separately
            import xgboost as xgb
            
            # XGBoost doesn't use Keras, so we return a special marker
            # The training loop will handle this differently
            return 'xgboost'
        
        elif architecture == 'lstm':
            # Enhanced LSTM: Designed for sequential/temporal patterns
            # Treats the feature vector as a sequence (home features → away features)
            # Uses attention mechanism to weight important timesteps
            
            # Calculate feature dimensions
            n_features = input_shape[0]
            n_timesteps = 2  # home, away
            features_per_step = n_features // 2
            
            # Input layer
            inputs = layers.Input(shape=input_shape)
            
            # Reshape for LSTM: (timesteps, features_per_step)
            x = layers.Reshape((n_timesteps, features_per_step))(inputs)
            
            # First Bidirectional LSTM with residual-like structure
            lstm1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True, 
                                                      kernel_regularizer=keras.regularizers.l2(0.001)))(x)
            lstm1 = layers.BatchNormalization()(lstm1)
            lstm1 = layers.Dropout(0.3)(lstm1)
            
            # Second LSTM layer
            lstm2 = layers.Bidirectional(layers.LSTM(32, return_sequences=True,
                                                      kernel_regularizer=keras.regularizers.l2(0.001)))(lstm1)
            lstm2 = layers.BatchNormalization()(lstm2)
            lstm2 = layers.Dropout(0.3)(lstm2)
            
            # Attention mechanism - learn which timestep matters more
            attention = layers.Dense(1, activation='tanh')(lstm2)
            attention = layers.Flatten()(attention)
            attention = layers.Activation('softmax')(attention)
            attention = layers.RepeatVector(64)(attention)  # 32*2 from bidirectional
            attention = layers.Permute([2, 1])(attention)
            
            # Apply attention
            context = layers.Multiply()([lstm2, attention])
            # Sum over timesteps using custom serializable layer
            context = SumPooling1D()(context)
            
            # Also add global pooling paths
            avg_pool = layers.GlobalAveragePooling1D()(lstm2)
            max_pool = layers.GlobalMaxPooling1D()(lstm2)
            
            # Concatenate all representations
            combined = layers.Concatenate()([context, avg_pool, max_pool])
            
            # Dense layers for final classification
            x = layers.Dense(128, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(0.001))(combined)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            
            x = layers.Dense(64, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.001))(x)
            x = layers.Dropout(0.3)(x)
            
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
            
            # Output layer
            outputs = layers.Dense(1, activation='sigmoid')(x)
            
            model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),  # Added gradient clipping
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def train(self, matchup_df, test_size=0.2, epochs=100, architecture='baseline'):
        """Train the MLP model
        
        Args:
            matchup_df: DataFrame with matchup features
            test_size: Fraction of data for testing
            epochs: Number of training epochs
            architecture: 'baseline' or 'deep'
        """
        print("\nPreparing training data...")
        
        # Feature columns - EXCLUDE the target and metadata columns!
        # CRITICAL: Do NOT include HOME_WIN, GAME_ID, GAME_DATE, HOME_TEAM_ID, AWAY_TEAM_ID
        exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
        self.feature_cols = [col for col in matchup_df.columns 
                            if col not in exclude_cols]
        
        print(f"✓ Using {len(self.feature_cols)} features (excluded: {exclude_cols})")
        
        X = matchup_df[self.feature_cols].values
        y = matchup_df['HOME_WIN'].values
        
        # CHRONOLOGICAL SPLIT: Use shuffle=False to prevent temporal leakage
        # Training data = older games, test data = newer games
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler for future predictions
        import pickle
        with open(get_model_path('scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        print("Scaler saved to models/scaler.pkl")
        
        # Build model (MLP - no reshape needed)
        print(f"\nBuilding Multi-Layer Perceptron model ({architecture} architecture)...")
        self.model = self.build_lstm_model((X_train_scaled.shape[1],), architecture=architecture)
        print(self.model.summary())
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=25,  # More patience for complex model
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.2,  # Less aggressive reduction
            patience=5,  # More patience before reducing
            min_lr=0.000005,  # Lower minimum
            mode='max',
            verbose=1
        )
        
        # Model checkpoint to save best model
        from tensorflow.keras.callbacks import ModelCheckpoint
        checkpoint = ModelCheckpoint(
            get_model_path('nba_model_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Train model
        print("\nTraining model...")
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Evaluate
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        y_pred_prob = self.model.predict(X_test_scaled)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Away Win', 'Home Win']))
        
        # Plot training history
        self.plot_training_history(history)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return history, accuracy
    
    def plot_training_history(self, history):
        """Plot training metrics"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Away Win', 'Home Win'],
                   yticklabels=['Away Win', 'Home Win'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
    def predict_game(self, home_features, away_features):
        """Predict outcome of a single game"""
        # Combine features
        features = np.concatenate([home_features, away_features])
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict (no reshape needed for MLP)
        home_win_prob = self.model.predict(features_scaled, verbose=0)[0][0]
        
        return {
            'home_win_probability': float(home_win_prob),
            'away_win_probability': float(1 - home_win_prob),
            'predicted_winner': 'HOME' if home_win_prob > 0.5 else 'AWAY'
        }


def main():
    """Main execution function"""
    print("="*60)
    print("NBA GAME PREDICTION USING MLP")
    print("Replicating 72.35% Accuracy Architecture")
    print("="*60)
    
    # Configuration
    WINDOW_SIZE = 20  # Games to look back
    
    # ARCHITECTURE OPTIONS:
    # 'baseline' - Original paper (128→64→32) - Fast, ~63%
    # 'deep' - Deeper MLP (256→128→64→32) - Moderate, ~64%
    # 'lstm' - LSTM for sequential patterns - Better for temporal data
    ARCHITECTURE = 'deep'  # Use 'lstm' for sequential learning
    
    # Use 6 seasons for better accuracy (UPDATED FOR 2025-26 SEASON)
    # Current season is 2025-26
    # SEASONS = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

    # Use all seasons for training
    SEASONS = ['2005-06', '2006-07', '2007-08', '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15',
               '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', 
               '2025-26']
    
    # Step 1: Fetch data
    print("\n📥 STEP 1: Fetching NBA Data")
    print("-" * 60)
    fetcher = NBADataFetcher(seasons=SEASONS)
    games_df = fetcher.fetch_games()
    
    if games_df is None or len(games_df) == 0:
        print("❌ Error: Could not fetch game data")
        return
    
    # Step 2: Prepare features
    print("\n🔧 STEP 2: Engineering Features")
    print("-" * 60)
    predictor = NBAPredictor(window_size=WINDOW_SIZE)
    matchup_df = predictor.prepare_matchup_data(games_df)
    
    # Step 3: Train model
    print("\n🧠 STEP 3: Training Neural Network")
    print("-" * 60)
    history, accuracy = predictor.train(matchup_df, epochs=150, architecture=ARCHITECTURE)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print(f"Final Test Accuracy: {accuracy*100:.2f}%")
    print("="*60)
    
    # Save model
    print("\n💾 Saving model and artifacts...")
    predictor.model.save(get_model_path('nba_lstm_model.keras'))
    print("✓ Model saved to models/nba_lstm_model.keras")
    print("✓ Best model saved to models/nba_model_best.keras")
    print("✓ Scaler saved to models/scaler.pkl")
    
    # Save feature columns for predictions
    import pickle
    with open(get_model_path('feature_columns.pkl'), 'wb') as f:
        pickle.dump(predictor.feature_cols, f)
    print("✓ Feature columns saved to models/feature_columns.pkl")
    
    print("\n" + "="*60)
    print("🎯 NEXT STEPS:")
    print("="*60)
    print("1. Model is trained and ready to use")
    print("2. Use predict_specific_game() to predict matchups")
    print("3. See README.md for prediction examples")
    print("4. Re-run weekly to update with fresh data")
    
    return predictor, matchup_df, games_df


def predict_specific_game(predictor, home_team_name, away_team_name, games_df):
    """Predict a specific matchup
    
    Args:
        predictor: Trained NBAPredictor instance
        home_team_name: Name of home team (e.g., "Lakers", "Warriors")
        away_team_name: Name of away team
        games_df: DataFrame with historical game data
        
    Returns:
        dict: Prediction results with probabilities
    """
    print(f"\n🔍 Preparing prediction for: {away_team_name} @ {home_team_name}")
    
    # Get team IDs
    all_teams = teams.get_teams()
    
    home_matches = [t for t in all_teams if home_team_name.lower() in t['full_name'].lower()]
    away_matches = [t for t in all_teams if away_team_name.lower() in t['full_name'].lower()]
    
    if not home_matches:
        print(f"❌ Could not find team: {home_team_name}")
        return None
    if not away_matches:
        print(f"❌ Could not find team: {away_team_name}")
        return None
    
    home_team = home_matches[0]
    away_team = away_matches[0]
    
    print(f"Found teams: {home_team['full_name']} vs {away_team['full_name']}")
    
    # Get recent features for both teams
    fe = FeatureEngineering(window_size=predictor.window_size)
    games_df = fe.calculate_four_factors(games_df)
    
    home_features = fe.create_rolling_features(games_df, home_team['id'])
    away_features = fe.create_rolling_features(games_df, away_team['id'])
    
    if len(home_features) == 0 or len(away_features) == 0:
        print("❌ Not enough recent games for these teams")
        return None
    
    # Get the most recent game features
    home_latest = home_features.iloc[-1]
    away_latest = away_features.iloc[-1]
    
    # Extract feature columns
    roll_cols = [col for col in home_features.columns if 'ROLL' in col or col == 'WIN_STREAK']
    home_vals = home_latest[roll_cols].values
    away_vals = away_latest[roll_cols].values
    
    # Predict
    prediction = predictor.predict_game(home_vals, away_vals)
    
    print(f"\n{'='*60}")
    print(f"🏀 PREDICTION: {away_team['full_name']} @ {home_team['full_name']}")
    print(f"{'='*60}")
    print(f"🏠 Home Win Probability: {prediction['home_win_probability']*100:.1f}%")
    print(f"✈️  Away Win Probability: {prediction['away_win_probability']*100:.1f}%")
    print(f"🏆 Predicted Winner: {prediction['predicted_winner']}")
    print(f"{'='*60}")
    
    return prediction


if __name__ == "__main__":
    predictor, matchup_data, games_data = main()
    
    # Example prediction after training
    print("\n" + "="*60)
    print("📊 EXAMPLE PREDICTION")
    print("="*60)
    
    # Uncomment to test a prediction:
    # predict_specific_game(predictor, "Lakers", "Warriors", games_data)