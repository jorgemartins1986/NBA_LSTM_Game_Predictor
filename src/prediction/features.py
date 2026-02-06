"""
Feature Computation
===================
Pure functions for computing game features for prediction.
Separated from I/O for testability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class TeamFeatures:
    """Rolling features for a team"""
    team_id: int
    features: Dict[str, float] = field(default_factory=dict)
    
    def get(self, key: str, default: float = 0.0) -> float:
        return self.features.get(key, default)
    
    def to_dict(self) -> Dict[str, float]:
        return self.features.copy()


@dataclass
class HeadToHeadFeatures:
    """Head-to-head features between two teams"""
    win_rate: float = 0.5
    games_played: int = 0
    pts_diff: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'H2H_WIN_RATE': self.win_rate,
            'H2H_GAMES': self.games_played,
            'H2H_PTS_DIFF': self.pts_diff
        }


@dataclass
class StandingsFeatures:
    """Team standings features"""
    wins: int = 0
    losses: int = 0
    win_pct: float = 0.5
    conf_rank: int = 8
    league_rank: int = 15
    games_back: float = 0.0
    streak: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'WINS': self.wins,
            'LOSSES': self.losses,
            'WIN_PCT': self.win_pct,
            'CONF_RANK': self.conf_rank,
            'LEAGUE_RANK': self.league_rank,
            'GAMES_BACK': self.games_back,
            'STREAK': self.streak
        }


@dataclass
class OddsFeatures:
    """Betting odds features"""
    avg_odds: float = 0.0
    implied_prob: float = 0.5
    spread: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'AVG_ODDS': self.avg_odds,
            'IMPLIED_PROB': self.implied_prob,
            'ODDS_SPREAD': self.spread
        }


@dataclass
class GameFeatures:
    """Complete features for a game prediction"""
    home_team_id: int
    away_team_id: int
    home_features: TeamFeatures
    away_features: TeamFeatures
    home_h2h: Optional[HeadToHeadFeatures] = None
    away_h2h: Optional[HeadToHeadFeatures] = None
    home_standings: Optional[StandingsFeatures] = None
    away_standings: Optional[StandingsFeatures] = None
    home_odds: Optional[OddsFeatures] = None
    away_odds: Optional[OddsFeatures] = None


class FeatureComputer:
    """
    Computes features for prediction.
    
    All methods are pure functions with no side effects.
    External data is passed in rather than fetched.
    """
    
    # Default feature values when data is missing
    DEFAULT_ROLLING_FEATURES = {
        'ROLL_AVG_PTS': 110.0,
        'ROLL_AVG_FG_PCT': 0.46,
        'ROLL_AVG_FG3_PCT': 0.36,
        'ROLL_AVG_REB': 44.0,
        'ROLL_AVG_AST': 25.0,
        'ROLL_AVG_STL': 7.5,
        'ROLL_AVG_BLK': 5.0,
        'ROLL_AVG_TOV': 13.5,
        'ROLL_AVG_PLUS_MINUS': 0.0,
        'WIN_STREAK': 0,
        'RECENT_FORM': 0.5,
        'MOMENTUM_TREND': 0.0,
        'DAYS_REST': 2,
        'IS_BACK_TO_BACK': 0,
        'IS_3_IN_4': 0,
    }
    
    def __init__(self, window_size: int = 20):
        """
        Initialize feature computer.
        
        Args:
            window_size: Window size for rolling features
        """
        self.window_size = window_size
    
    def compute_head_to_head(self,
                            games_df: pd.DataFrame,
                            team_id: int,
                            opponent_id: int,
                            window: int = 10) -> HeadToHeadFeatures:
        """
        Compute head-to-head record between teams.
        
        Args:
            games_df: Historical games dataframe
            team_id: Team to compute for
            opponent_id: Opponent team
            window: Number of recent games to consider
            
        Returns:
            HeadToHeadFeatures
        """
        # Get team's games
        team_games = games_df[games_df['TEAM_ID'] == team_id].copy()
        
        if len(team_games) == 0:
            return HeadToHeadFeatures()
        
        # Get opponent abbreviation
        opp_games = games_df[games_df['TEAM_ID'] == opponent_id]
        if len(opp_games) == 0:
            return HeadToHeadFeatures()
        
        opp_abbrev = opp_games.iloc[0].get('TEAM_ABBREVIATION', '')
        if not opp_abbrev:
            return HeadToHeadFeatures()
        
        # Find games against this opponent
        h2h_games = team_games[
            team_games['MATCHUP'].str.contains(opp_abbrev, na=False)
        ].tail(window)
        
        if len(h2h_games) == 0:
            return HeadToHeadFeatures()
        
        wins = (h2h_games['WL'] == 'W').sum()
        games_played = len(h2h_games)
        pts_diff = h2h_games['PLUS_MINUS'].mean() if 'PLUS_MINUS' in h2h_games.columns else 0
        
        return HeadToHeadFeatures(
            win_rate=wins / games_played,
            games_played=min(games_played, window),
            pts_diff=pts_diff if not pd.isna(pts_diff) else 0
        )
    
    def build_feature_vector(self,
                            home_features: Dict[str, float],
                            away_features: Dict[str, float],
                            feature_cols: List[str],
                            h2h_home: Optional[Dict] = None,
                            h2h_away: Optional[Dict] = None,
                            home_standings: Optional[Dict] = None,
                            away_standings: Optional[Dict] = None,
                            home_odds: Optional[Dict] = None,
                            away_odds: Optional[Dict] = None) -> np.ndarray:
        """
        Build feature vector for model prediction.
        
        Combines all feature sources into the format expected by trained models.
        
        Args:
            home_features: Home team rolling features
            away_features: Away team rolling features
            feature_cols: List of feature column names (defines order)
            *_standings: Standings features
            *_odds: Odds features
            
        Returns:
            Numpy array of features in correct order
        """
        # Start with combined features dict
        feature_dict = {}
        
        # Add home team features with HOME_ prefix
        for key, value in home_features.items():
            col_name = f"HOME_{key}" if not key.startswith('HOME_') else key
            feature_dict[col_name] = value
        
        # Add away team features with AWAY_ prefix  
        for key, value in away_features.items():
            col_name = f"AWAY_{key}" if not key.startswith('AWAY_') else key
            feature_dict[col_name] = value
        
        # Add head-to-head features
        if h2h_home:
            for key, value in h2h_home.items():
                feature_dict[f'HOME_{key}'] = value
        if h2h_away:
            for key, value in h2h_away.items():
                feature_dict[f'AWAY_{key}'] = value
        
        # Add standings features
        if home_standings:
            for key, value in home_standings.items():
                feature_dict[f'HOME_{key}'] = value
        if away_standings:
            for key, value in away_standings.items():
                feature_dict[f'AWAY_{key}'] = value
        
        # Add odds features (handle both prefixed and unprefixed keys)
        if home_odds:
            for key, value in home_odds.items():
                if 'HOME' not in key:
                    feature_dict[f'HOME_{key}'] = value
                else:
                    feature_dict[key] = value
            # Also add BOOKMAKER_COUNT if present
            if 'BOOKMAKER_COUNT' in home_odds:
                feature_dict['BOOKMAKER_COUNT'] = home_odds['BOOKMAKER_COUNT']
        if away_odds:
            for key, value in away_odds.items():
                if 'AWAY' not in key:
                    feature_dict[f'AWAY_{key}'] = value
                else:
                    feature_dict[key] = value
        
        # Compute differential features from standings
        if home_standings and away_standings:
            feature_dict['RANK_DIFF'] = home_standings.get('CONF_RANK', 8) - away_standings.get('CONF_RANK', 8)
            feature_dict['WIN_PCT_DIFF'] = home_standings.get('WIN_PCT', 0.5) - away_standings.get('WIN_PCT', 0.5)
        
        # Build vector in correct order
        vector = []
        for col in feature_cols:
            if col in feature_dict:
                val = feature_dict[col]
                # Handle NaN values
                vector.append(val if val is not None and not (isinstance(val, float) and np.isnan(val)) else 0.0)
            else:
                # Use default or 0
                vector.append(self._get_default_value(col))
        
        return np.array(vector, dtype=np.float64).reshape(1, -1)
    
    def _get_default_value(self, col_name: str) -> float:
        """Get default value for a missing feature"""
        # Try to match column name patterns
        for key, value in self.DEFAULT_ROLLING_FEATURES.items():
            if key in col_name:
                return value
        
        # Default to 0 for unknown features
        return 0.0
    
    def compute_differentials(self,
                             home_features: Dict[str, float],
                             away_features: Dict[str, float]) -> Dict[str, float]:
        """
        Compute differential features (home - away).
        
        Args:
            home_features: Home team features
            away_features: Away team features
            
        Returns:
            Dict of differential features
        """
        diffs = {}
        
        # Find matching features between home and away
        home_keys = set(home_features.keys())
        away_keys = set(away_features.keys())
        
        for home_key in home_keys:
            # Try to find corresponding away key
            away_key = home_key  # Might be the same
            if home_key.startswith('HOME_'):
                away_key = 'AWAY_' + home_key[5:]
            
            if away_key in away_keys:
                diff_key = f"DIFF_{home_key.replace('HOME_', '').replace('AWAY_', '')}"
                diffs[diff_key] = home_features[home_key] - away_features[away_key]
        
        return diffs
    
    @staticmethod
    def compute_fatigue_score(days_rest: int, 
                              is_back_to_back: bool,
                              is_3_in_4: bool) -> float:
        """
        Compute fatigue score (0 = rested, 1 = exhausted).
        
        Args:
            days_rest: Days since last game
            is_back_to_back: Playing second night of B2B
            is_3_in_4: Third game in 4 nights
            
        Returns:
            Fatigue score (0 to 1)
        """
        score = 0.0
        
        if days_rest == 0:
            score += 0.4
        elif days_rest == 1:
            score += 0.2
        
        if is_back_to_back:
            score += 0.3
        
        if is_3_in_4:
            score += 0.3
        
        return min(score, 1.0)
