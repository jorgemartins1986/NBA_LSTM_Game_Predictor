"""
NBA Odds API Integration
========================

Fetches odds data from The-Odds-API for both:
1. Historical odds (for training data enrichment)
2. Live odds (for real-time predictions)

Supports European decimal odds format (1.XX).

API Documentation: https://the-odds-api.com/liveapi/guides/v4/

Usage:
    # Set API key in .env file (recommended):
    # ODDS_API_KEY=your_api_key_here
    
    # Or set environment variable:
    # export ODDS_API_KEY="your_api_key_here"
    
    # Or pass directly in Python:
    # client = OddsAPIClient(api_key="your_api_key")
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Find .env file in project root
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, will use OS environment variables

from .paths import get_cache_path, CACHE_DIR

# Cache file for historical odds
ODDS_CACHE_FILE = get_cache_path('nba_historical_odds.csv')
ODDS_API_CONFIG_FILE = get_cache_path('odds_api_config.json')


class OddsAPIClient:
    """Client for The-Odds-API with caching and rate limiting"""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT_KEY = "basketball_nba"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Odds API client.
        
        Args:
            api_key: API key for The-Odds-API. If not provided, reads from
                     ODDS_API_KEY environment variable or .env file.
        """
        self.api_key = api_key or os.environ.get('ODDS_API_KEY')
        if not self.api_key:
            print("⚠️  No API key provided. Set ODDS_API_KEY environment variable")
            print("   or pass api_key parameter to OddsAPIClient()")
            print("   Get free key at: https://the-odds-api.com/")
        
        self.requests_remaining = None
        self.requests_used = None
        
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make API request with rate limiting and error handling"""
        if not self.api_key:
            print("❌ No API key configured")
            return None
            
        params['apiKey'] = self.api_key
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            # Track quota usage
            self.requests_remaining = response.headers.get('x-requests-remaining')
            self.requests_used = response.headers.get('x-requests-used')
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print("❌ Invalid API key")
                return None
            elif response.status_code == 429:
                print("⚠️  Rate limit exceeded. Waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            else:
                print(f"❌ API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return None
    
    def get_quota_status(self) -> Tuple[Optional[str], Optional[str]]:
        """Return remaining and used quota credits"""
        return self.requests_remaining, self.requests_used
    
    def print_quota_status(self):
        """Print current quota usage"""
        if self.requests_remaining and self.requests_used:
            print(f"📊 API Quota: {self.requests_remaining} remaining, {self.requests_used} used")
    
    # =========================================================================
    # LIVE ODDS (for predictions)
    # =========================================================================
    
    def get_live_odds(
        self,
        regions: str = 'eu',
        markets: str = 'h2h',
        odds_format: str = 'decimal'
    ) -> Optional[List[Dict]]:
        """
        Get live odds for upcoming NBA games.
        
        Args:
            regions: Bookmaker regions (eu, us, uk, au)
            markets: Betting markets (h2h = moneyline, spreads, totals)
            odds_format: 'decimal' (1.XX) or 'american' (+150, -200)
            
        Returns:
            List of games with bookmaker odds
            
        Cost: 1 credit per region per market
        """
        params = {
            'regions': regions,
            'markets': markets,
            'oddsFormat': odds_format,
            'dateFormat': 'iso'
        }
        
        data = self._make_request(f"sports/{self.SPORT_KEY}/odds", params)
        self.print_quota_status()
        return data
    
    def get_live_odds_for_game(
        self,
        home_team: str,
        away_team: str,
        regions: str = 'eu',
        odds_format: str = 'decimal'
    ) -> Optional[Dict]:
        """
        Get live odds for a specific game by team names.
        
        Args:
            home_team: Home team name (e.g., "Los Angeles Lakers")
            away_team: Away team name (e.g., "Boston Celtics")
            regions: Bookmaker regions
            odds_format: Odds format
            
        Returns:
            Dict with game odds or None if not found
        """
        all_odds = self.get_live_odds(regions=regions, odds_format=odds_format)
        if not all_odds:
            return None
            
        # Fuzzy match team names
        for game in all_odds:
            if (self._match_team(home_team, game.get('home_team', '')) and
                self._match_team(away_team, game.get('away_team', ''))):
                return game
                
        return None
    
    def _match_team(self, search: str, api_name: str) -> bool:
        """Fuzzy match team names"""
        search_lower = search.lower()
        api_lower = api_name.lower()
        
        # Check if key parts match
        for word in search_lower.split():
            if word in ['the', 'a', 'an']:
                continue
            if word in api_lower:
                return True
        return False
    
    # =========================================================================
    # HISTORICAL ODDS (for training data enrichment)
    # =========================================================================
    
    def get_historical_odds(
        self,
        date: datetime,
        regions: str = 'eu',
        markets: str = 'h2h',
        odds_format: str = 'decimal'
    ) -> Optional[Dict]:
        """
        Get historical odds snapshot for a specific date/time.
        
        Args:
            date: Date/time to get odds for (closest available snapshot)
            regions: Bookmaker regions
            markets: Betting markets
            odds_format: Odds format
            
        Returns:
            Dict with timestamp info and odds data
            
        Cost: 10 credits per region per market
        
        Note: Historical data available from June 2020, snapshots every 5-10 minutes
        """
        date_str = date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        params = {
            'date': date_str,
            'regions': regions,
            'markets': markets,
            'oddsFormat': odds_format,
            'dateFormat': 'iso'
        }
        
        data = self._make_request(f"historical/sports/{self.SPORT_KEY}/odds", params)
        self.print_quota_status()
        return data
    
    # =========================================================================
    # ODDS FEATURE EXTRACTION
    # =========================================================================
    
    @staticmethod
    def extract_odds_features(game_odds: Dict) -> Dict:
        """
        Extract machine learning features from game odds.
        
        Args:
            game_odds: Single game odds data from API
            
        Returns:
            Dict with extracted features:
            - HOME_AVG_ODDS: Average decimal odds for home team across bookmakers
            - AWAY_AVG_ODDS: Average decimal odds for away team across bookmakers
            - HOME_BEST_ODDS: Best (highest) odds for home team
            - AWAY_BEST_ODDS: Best odds for away team
            - ODDS_SPREAD: Max - Min odds (uncertainty signal)
            - HOME_IMPLIED_PROB: Implied probability from average odds
            - AWAY_IMPLIED_PROB: Implied probability from average odds
            - BOOKMAKER_COUNT: Number of bookmakers with odds
        """
        features = {
            'HOME_AVG_ODDS': np.nan,
            'AWAY_AVG_ODDS': np.nan,
            'HOME_BEST_ODDS': np.nan,
            'AWAY_BEST_ODDS': np.nan,
            'HOME_WORST_ODDS': np.nan,
            'AWAY_WORST_ODDS': np.nan,
            'HOME_ODDS_SPREAD': np.nan,
            'AWAY_ODDS_SPREAD': np.nan,
            'HOME_IMPLIED_PROB': np.nan,
            'AWAY_IMPLIED_PROB': np.nan,
            'BOOKMAKER_COUNT': 0
        }
        
        if not game_odds or 'bookmakers' not in game_odds:
            return features
            
        home_team = game_odds.get('home_team', '')
        away_team = game_odds.get('away_team', '')
        
        home_odds_list = []
        away_odds_list = []
        
        for bookmaker in game_odds.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'h2h':
                    for outcome in market.get('outcomes', []):
                        name = outcome.get('name', '')
                        price = outcome.get('price')
                        
                        if price and price > 0:
                            if name == home_team:
                                home_odds_list.append(price)
                            elif name == away_team:
                                away_odds_list.append(price)
        
        if home_odds_list:
            features['HOME_AVG_ODDS'] = np.mean(home_odds_list)
            features['HOME_BEST_ODDS'] = max(home_odds_list)
            features['HOME_WORST_ODDS'] = min(home_odds_list)
            features['HOME_ODDS_SPREAD'] = features['HOME_BEST_ODDS'] - features['HOME_WORST_ODDS']
            features['HOME_IMPLIED_PROB'] = 1 / features['HOME_AVG_ODDS']
            
        if away_odds_list:
            features['AWAY_AVG_ODDS'] = np.mean(away_odds_list)
            features['AWAY_BEST_ODDS'] = max(away_odds_list)
            features['AWAY_WORST_ODDS'] = min(away_odds_list)
            features['AWAY_ODDS_SPREAD'] = features['AWAY_BEST_ODDS'] - features['AWAY_WORST_ODDS']
            features['AWAY_IMPLIED_PROB'] = 1 / features['AWAY_AVG_ODDS']
            
        features['BOOKMAKER_COUNT'] = len(game_odds.get('bookmakers', []))
        
        return features
    
    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """
        Convert American odds to decimal format.
        
        Examples:
            -200 -> 1.50 (bet $200 to win $100 profit)
            +150 -> 2.50 (bet $100 to win $150 profit)
        """
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    @staticmethod
    def decimal_to_implied_probability(decimal_odds: float) -> float:
        """
        Convert decimal odds to implied probability.
        
        Example: 1.50 -> 0.667 (66.7% implied probability)
        """
        if decimal_odds <= 0:
            return np.nan
        return 1 / decimal_odds


class OddsDataEnricher:
    """Enrich games DataFrame with historical odds data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OddsAPIClient(api_key=api_key)
        self.odds_cache = self._load_cache()
        
    def _load_cache(self) -> pd.DataFrame:
        """Load cached odds data"""
        if os.path.exists(ODDS_CACHE_FILE):
            return pd.read_csv(ODDS_CACHE_FILE, parse_dates=['GAME_DATE'])
        return pd.DataFrame()
    
    def _save_cache(self):
        """Save odds cache to file"""
        if not self.odds_cache.empty:
            self.odds_cache.to_csv(ODDS_CACHE_FILE, index=False)
            print(f"💾 Saved {len(self.odds_cache)} odds records to cache")
    
    def enrich_games_with_odds(
        self,
        games_df: pd.DataFrame,
        batch_size: int = 10,
        delay_between_batches: float = 1.0
    ) -> pd.DataFrame:
        """
        Add odds features to games DataFrame.
        
        Args:
            games_df: DataFrame with columns GAME_DATE, HOME_TEAM, AWAY_TEAM
            batch_size: Number of games to fetch per batch (to manage API quota)
            delay_between_batches: Seconds to wait between API calls
            
        Returns:
            DataFrame with added odds columns
            
        Note: Historical API costs 10 credits per call. With 500 free credits/month,
              you can fetch ~50 dates of historical data.
        """
        if not self.client.api_key:
            print("❌ No API key - returning original DataFrame")
            return games_df
        
        # Initialize odds columns
        odds_columns = [
            'HOME_AVG_ODDS', 'AWAY_AVG_ODDS',
            'HOME_BEST_ODDS', 'AWAY_BEST_ODDS',
            'HOME_WORST_ODDS', 'AWAY_WORST_ODDS',
            'HOME_ODDS_SPREAD', 'AWAY_ODDS_SPREAD',
            'HOME_IMPLIED_PROB', 'AWAY_IMPLIED_PROB',
            'BOOKMAKER_COUNT'
        ]
        
        for col in odds_columns:
            if col not in games_df.columns:
                games_df[col] = np.nan
        
        # Group games by date for efficient API calls
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        unique_dates = games_df['GAME_DATE'].dt.date.unique()
        
        print(f"📊 Enriching {len(games_df)} games across {len(unique_dates)} dates...")
        
        # Check which dates are already cached
        cached_dates = set()
        if not self.odds_cache.empty:
            cached_dates = set(self.odds_cache['GAME_DATE'].dt.date.unique())
        
        dates_to_fetch = [d for d in unique_dates if d not in cached_dates]
        print(f"   {len(cached_dates)} dates in cache, {len(dates_to_fetch)} to fetch")
        
        # Fetch new dates
        for i, game_date in enumerate(dates_to_fetch):
            if i > 0 and i % batch_size == 0:
                time.sleep(delay_between_batches)
                
            print(f"   Fetching odds for {game_date}... ({i+1}/{len(dates_to_fetch)})")
            
            # Get odds for date (use 12:00 UTC as snapshot time - before most games)
            snapshot_time = datetime.combine(game_date, datetime.min.time().replace(hour=12))
            
            odds_data = self.client.get_historical_odds(snapshot_time)
            
            if odds_data and 'data' in odds_data:
                for game in odds_data['data']:
                    features = self.client.extract_odds_features(game)
                    features['GAME_DATE'] = pd.Timestamp(game_date)
                    features['HOME_TEAM'] = game.get('home_team', '')
                    features['AWAY_TEAM'] = game.get('away_team', '')
                    
                    self.odds_cache = pd.concat([
                        self.odds_cache,
                        pd.DataFrame([features])
                    ], ignore_index=True)
        
        # Save updated cache
        self._save_cache()
        
        # Merge odds into games DataFrame
        if not self.odds_cache.empty:
            # Match by date and team names
            games_df = self._merge_odds(games_df, self.odds_cache)
        
        return games_df
    
    def _merge_odds(
        self,
        games_df: pd.DataFrame,
        odds_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge odds data into games DataFrame by matching date and teams"""
        
        games_df = games_df.copy()
        games_df['_date'] = pd.to_datetime(games_df['GAME_DATE']).dt.date
        odds_df = odds_df.copy()
        odds_df['_date'] = pd.to_datetime(odds_df['GAME_DATE']).dt.date
        
        # Normalize team names for matching
        games_df['_home_norm'] = games_df['HOME_TEAM'].str.lower().str.strip()
        games_df['_away_norm'] = games_df['AWAY_TEAM'].str.lower().str.strip()
        odds_df['_home_norm'] = odds_df['HOME_TEAM'].str.lower().str.strip()
        odds_df['_away_norm'] = odds_df['AWAY_TEAM'].str.lower().str.strip()
        
        # Merge on date and normalized team names
        merged = games_df.merge(
            odds_df.drop(columns=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM']),
            on=['_date', '_home_norm', '_away_norm'],
            how='left',
            suffixes=('', '_odds')
        )
        
        # Update odds columns
        odds_columns = [
            'HOME_AVG_ODDS', 'AWAY_AVG_ODDS',
            'HOME_BEST_ODDS', 'AWAY_BEST_ODDS',
            'HOME_WORST_ODDS', 'AWAY_WORST_ODDS',
            'HOME_ODDS_SPREAD', 'AWAY_ODDS_SPREAD',
            'HOME_IMPLIED_PROB', 'AWAY_IMPLIED_PROB',
            'BOOKMAKER_COUNT'
        ]
        
        for col in odds_columns:
            if f'{col}_odds' in merged.columns:
                merged[col] = merged[f'{col}_odds'].combine_first(merged.get(col))
                merged.drop(columns=[f'{col}_odds'], inplace=True, errors='ignore')
        
        # Clean up temp columns
        merged.drop(columns=['_date', '_home_norm', '_away_norm'], inplace=True, errors='ignore')
        
        matched = merged[odds_columns[0]].notna().sum()
        print(f"✅ Matched odds for {matched}/{len(merged)} games ({100*matched/len(merged):.1f}%)")
        
        return merged


def get_live_odds_for_predictions() -> Dict[str, Dict]:
    """
    Get live odds for today's NBA games (for use in predictions).
    
    Returns:
        Dict mapping game key (home_vs_away) to odds features
        
    Example:
        odds = get_live_odds_for_predictions()
        lakers_odds = odds.get('los angeles lakers_vs_boston celtics', {})
    """
    client = OddsAPIClient()
    games = client.get_live_odds(regions='eu', odds_format='decimal')
    
    if not games:
        return {}
    
    result = {}
    for game in games:
        home = game.get('home_team', '').lower()
        away = game.get('away_team', '').lower()
        key = f"{home}_vs_{away}"
        
        result[key] = client.extract_odds_features(game)
        result[key]['commence_time'] = game.get('commence_time')
        
    return result


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='NBA Odds API Tools')
    parser.add_argument('command', choices=['test', 'live', 'historical', 'quota'],
                        help='Command to run')
    parser.add_argument('--date', type=str, help='Date for historical odds (YYYY-MM-DD)')
    parser.add_argument('--teams', type=str, help='Teams to filter (home_vs_away)')
    
    args = parser.parse_args()
    
    client = OddsAPIClient()
    
    if args.command == 'test':
        print("Testing API connection...")
        # Test with sports endpoint (free, doesn't cost credits)
        response = requests.get(
            f"{client.BASE_URL}/sports",
            params={'apiKey': client.api_key}
        )
        if response.status_code == 200:
            sports = response.json()
            nba = next((s for s in sports if s['key'] == 'basketball_nba'), None)
            if nba:
                print(f"✅ API connected! NBA active: {nba.get('active', False)}")
            else:
                print("✅ API connected, but NBA not in active sports list")
        else:
            print(f"❌ API test failed: {response.status_code}")
            
    elif args.command == 'live':
        print("Fetching live NBA odds...")
        odds = client.get_live_odds()
        if odds:
            print(f"\n📊 {len(odds)} upcoming games with odds:\n")
            for game in odds[:5]:  # Show first 5
                features = client.extract_odds_features(game)
                print(f"  {game['away_team']} @ {game['home_team']}")
                print(f"    Home: {features['HOME_AVG_ODDS']:.2f} "
                      f"(implied: {features['HOME_IMPLIED_PROB']*100:.1f}%)")
                print(f"    Away: {features['AWAY_AVG_ODDS']:.2f} "
                      f"(implied: {features['AWAY_IMPLIED_PROB']*100:.1f}%)")
                print(f"    Spread: {features['HOME_ODDS_SPREAD']:.3f} "
                      f"({features['BOOKMAKER_COUNT']} bookmakers)")
                print()
                
    elif args.command == 'historical':
        if not args.date:
            print("❌ --date required for historical command")
        else:
            date = datetime.strptime(args.date, '%Y-%m-%d')
            print(f"Fetching historical odds for {args.date}...")
            print("⚠️  Note: This costs 10 API credits!")
            odds = client.get_historical_odds(date)
            if odds and 'data' in odds:
                print(f"\n📊 Snapshot from {odds.get('timestamp')}:")
                print(f"   {len(odds['data'])} games found\n")
                for game in odds['data'][:5]:
                    features = client.extract_odds_features(game)
                    print(f"  {game['away_team']} @ {game['home_team']}")
                    print(f"    Home: {features['HOME_AVG_ODDS']:.2f}")
                    print()
                    
    elif args.command == 'quota':
        # Make a free request to check quota
        response = requests.get(
            f"{client.BASE_URL}/sports",
            params={'apiKey': client.api_key}
        )
        print(f"\n📊 API Quota Status:")
        print(f"   Remaining: {response.headers.get('x-requests-remaining', 'N/A')}")
        print(f"   Used: {response.headers.get('x-requests-used', 'N/A')}")
