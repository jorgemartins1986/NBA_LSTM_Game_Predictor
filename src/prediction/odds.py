"""
Odds Module
===========
Functions for fetching and matching betting odds.
"""

import os
from typing import Dict, Optional


def get_live_odds(verbose: bool = True) -> Dict[str, Dict]:
    """
    Fetch live betting odds for today's NBA games.
    
    Returns:
        dict: {game_key: odds_features} where game_key is "home_team_vs_away_team" (lowercase)
        
    Note: Requires ODDS_API_KEY environment variable to be set.
          Get free API key at: https://the-odds-api.com/
    """
    # Try to import odds API
    try:
        from ..odds_api import get_live_odds_for_predictions
        ODDS_API_AVAILABLE = True
    except ImportError:
        ODDS_API_AVAILABLE = False
        return {}
    
    # Check if API key is configured
    api_key = os.environ.get('ODDS_API_KEY')
    if not api_key:
        return {}
    
    if verbose:
        print("💰 Fetching live betting odds...")
    try:
        odds = get_live_odds_for_predictions()
        if odds and verbose:
            print(f"   ✓ Got odds for {len(odds)} games")
        return odds
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Could not fetch odds: {e}")
        return {}


def match_game_to_odds(home_team_name: str, away_team_name: str, odds_dict: Dict) -> Dict:
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


def format_odds_display(odds_features: Dict) -> str:
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
