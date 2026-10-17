"""Rolling feature calculations used by training and prediction.

Keep this module free of TensorFlow so process workers only load data tools.
"""

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


def _compute_team_features_parallel(team_id, games_df, window_size):
    """Standalone function for parallel feature computation"""
    fe = FeatureEngineering(window_size=window_size)
    return fe.create_rolling_features(games_df, team_id)
