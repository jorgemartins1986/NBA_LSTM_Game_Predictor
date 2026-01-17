"""
NBA Data Manager with Local Caching
===================================
Downloads data once, then incrementally updates.
Calculates ELO ratings for all teams.

Usage:
    from nba_data_manager import NBADataManager
    
    manager = NBADataManager()
    games_df = manager.get_games()  # Fast! Uses cache
    games_df = manager.update_current_season()  # Only fetches new games
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import pickle
from tqdm import tqdm
from .paths import GAMES_CACHE_FILE, ELO_CACHE_FILE
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams


class ELORatingSystem:
    """Calculate and track ELO ratings for NBA teams"""
    
    def __init__(self, k_factor=20, initial_rating=1500):
        """
        Args:
            k_factor: How much ratings change per game (10-40, higher = more volatile)
            initial_rating: Starting ELO for all teams
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.team_ratings = {}
        
    def get_rating(self, team_id):
        """Get current ELO rating for team"""
        if team_id not in self.team_ratings:
            self.team_ratings[team_id] = self.initial_rating
        return self.team_ratings[team_id]
    
    def expected_score(self, rating_a, rating_b):
        """Calculate expected win probability for team A vs team B"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, team_a_id, team_b_id, team_a_won, home_advantage=100):
        """
        Update ELO ratings after a game
        
        Args:
            team_a_id: Home team ID
            team_b_id: Away team ID
            team_a_won: Did home team win? (1=yes, 0=no)
            home_advantage: ELO points added to home team (typical: 100)
        
        Returns:
            dict: Updated ratings and changes
        """
        # Get current ratings
        rating_a = self.get_rating(team_a_id)
        rating_b = self.get_rating(team_b_id)
        
        # Add home advantage to team A
        rating_a_adjusted = rating_a + home_advantage
        
        # Calculate expected scores
        expected_a = self.expected_score(rating_a_adjusted, rating_b)
        expected_b = 1 - expected_a
        
        # Actual scores (1 for win, 0 for loss)
        actual_a = 1 if team_a_won else 0
        actual_b = 1 - actual_a
        
        # Calculate rating changes
        change_a = self.k_factor * (actual_a - expected_a)
        change_b = self.k_factor * (actual_b - expected_b)
        
        # Update ratings
        new_rating_a = rating_a + change_a
        new_rating_b = rating_b + change_b
        
        self.team_ratings[team_a_id] = new_rating_a
        self.team_ratings[team_b_id] = new_rating_b
        
        return {
            'team_a_rating': new_rating_a,
            'team_b_rating': new_rating_b,
            'team_a_change': change_a,
            'team_b_change': change_b,
            'expected_a': expected_a
        }
    
    def reset_season(self, regression_factor=0.75):
        """
        Reset ratings at start of new season (with regression to mean)
        
        Args:
            regression_factor: How much to regress (0.75 = regress 25% to mean)
        """
        for team_id in self.team_ratings:
            current_rating = self.team_ratings[team_id]
            # Regress toward mean
            self.team_ratings[team_id] = (
                regression_factor * current_rating + 
                (1 - regression_factor) * self.initial_rating
            )


class NBADataManager:
    """Manage NBA game data with local caching and ELO ratings"""
    
    def __init__(self, cache_file=None, elo_file=None):
        self.cache_file = cache_file or GAMES_CACHE_FILE
        self.elo_file = elo_file or ELO_CACHE_FILE
        self.all_teams = teams.get_teams()
        self.elo_system = ELORatingSystem(k_factor=20, initial_rating=1500)
        
    def download_all_seasons(self, seasons):
        """Download data for multiple seasons (one-time operation)"""
        print("="*70)
        print("DOWNLOADING ALL HISTORICAL DATA (This will take a while...)")
        print("="*70)
        
        all_games = []
        
        for i, season in enumerate(seasons):
            print(f"\n[{i+1}/{len(seasons)}] Fetching {season} season...")
            try:
                gamefinder = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    league_id_nullable='00'
                )
                games = gamefinder.get_data_frames()[0]
                games['SEASON'] = season
                all_games.append(games)
                print(f"✓ Fetched {len(games)} records")
            except Exception as e:
                print(f"❌ Error fetching {season}: {e}")
        
        if all_games:
            df = pd.concat(all_games, ignore_index=True)
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE').reset_index(drop=True)
            
            # Save to cache
            df.to_csv(self.cache_file, index=False)
            print(f"\n✅ Saved {len(df)} records to {self.cache_file}")
            print(f"💾 Cache size: {os.path.getsize(self.cache_file) / 1024 / 1024:.1f} MB")
            
            return df
        return None
    
    def load_cache(self):
        """Load games from cache file"""
        if not os.path.exists(self.cache_file):
            return None
        
        print(f"📂 Loading from cache: {self.cache_file}")
        df = pd.read_csv(self.cache_file)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        print(f"✓ Loaded {len(df)} cached records")
        return df
    
    def update_current_season(self, current_season='2025-26'):
        """Fetch only the current season's games and append to cache"""
        print(f"\n🔄 Updating {current_season} season...")
        
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=current_season,
                league_id_nullable='00'
            )
            new_games = gamefinder.get_data_frames()[0]
            new_games['SEASON'] = current_season
            new_games['GAME_DATE'] = pd.to_datetime(new_games['GAME_DATE'])
            
            # Load existing cache
            cached_df = self.load_cache()
            
            if cached_df is not None:
                # Remove old current season data
                cached_df = cached_df[cached_df['SEASON'] != current_season]
                
                # Append new data
                updated_df = pd.concat([cached_df, new_games], ignore_index=True)
                updated_df = updated_df.sort_values('GAME_DATE').reset_index(drop=True)
                
                # Save back to cache
                updated_df.to_csv(self.cache_file, index=False)
                print(f"✓ Updated cache with {len(new_games)} new records")
                
                return updated_df
            else:
                # No cache exists, just save current season
                new_games.to_csv(self.cache_file, index=False)
                return new_games
                
        except Exception as e:
            print(f"❌ Error updating: {e}")
            return self.load_cache()  # Return cached data if update fails
    
    def calculate_elo_ratings(self, games_df):
        """Calculate ELO ratings for all games chronologically (optimized)"""
        print("\n⚖️  Calculating ELO ratings...")
        
        # Sort by date to process chronologically
        games_df = games_df.sort_values('GAME_DATE').copy()
        
        # Initialize ELO columns
        games_df['ELO_HOME'] = 0.0
        games_df['ELO_AWAY'] = 0.0
        games_df['ELO_DIFF'] = 0.0
        games_df['ELO_PROB_HOME'] = 0.0
        
        # Reset ELO system
        self.elo_system = ELORatingSystem(k_factor=20, initial_rating=1500)
        
        # Track season changes for regression
        current_season = None
        
        # Pre-build index for fast lookups
        games_df = games_df.reset_index(drop=True)
        game_id_to_indices = games_df.groupby('GAME_ID').indices
        
        # Identify home/away once for all games (vectorized)
        is_home = games_df['MATCHUP'].str.contains('vs.', case=False, na=False)
        is_away = games_df['MATCHUP'].str.contains('@', na=False)
        
        # Store results in arrays (faster than df.loc updates)
        elo_home_arr = np.zeros(len(games_df))
        elo_away_arr = np.zeros(len(games_df))
        elo_diff_arr = np.zeros(len(games_df))
        elo_prob_arr = np.zeros(len(games_df))
        
        # Get unique games in chronological order
        unique_games = games_df.drop_duplicates('GAME_ID')[['GAME_ID', 'GAME_DATE', 'SEASON']]
        
        # Process each game with progress bar
        game_ids_processed = set()
        
        for _, game_row in tqdm(unique_games.iterrows(), total=len(unique_games), desc="Computing ELO"):
            game_id = game_row['GAME_ID']
            
            if game_id in game_ids_processed:
                continue
            
            # Get indices for this game
            indices = game_id_to_indices.get(game_id, [])
            if len(indices) != 2:
                continue
            
            # Get home and away from pre-computed masks
            home_idx = [i for i in indices if is_home.iloc[i]]
            away_idx = [i for i in indices if is_away.iloc[i]]
            
            if len(home_idx) != 1 or len(away_idx) != 1:
                continue
            
            home_idx = home_idx[0]
            away_idx = away_idx[0]
            
            home_team_id = games_df.iloc[home_idx]['TEAM_ID']
            away_team_id = games_df.iloc[away_idx]['TEAM_ID']
            home_won = (games_df.iloc[home_idx]['WL'] == 'W')
            
            # Check for season change (apply regression)
            if current_season != game_row['SEASON']:
                if current_season is not None:
                    self.elo_system.reset_season(regression_factor=0.75)
                current_season = game_row['SEASON']
            
            # Get ELO before game
            elo_home_before = self.elo_system.get_rating(home_team_id)
            elo_away_before = self.elo_system.get_rating(away_team_id)
            
            # Calculate expected probability
            elo_prob_home = self.elo_system.expected_score(
                elo_home_before + 100,  # Home advantage
                elo_away_before
            )
            
            # Update ELO ratings
            self.elo_system.update_ratings(home_team_id, away_team_id, home_won)
            
            # Store in arrays (much faster than df.loc)
            for idx in indices:
                elo_home_arr[idx] = elo_home_before
                elo_away_arr[idx] = elo_away_before
                elo_diff_arr[idx] = elo_home_before - elo_away_before
                elo_prob_arr[idx] = elo_prob_home
            
            game_ids_processed.add(game_id)
        
        # Assign arrays to dataframe (single operation)
        games_df['ELO_HOME'] = elo_home_arr
        games_df['ELO_AWAY'] = elo_away_arr
        games_df['ELO_DIFF'] = elo_diff_arr
        games_df['ELO_PROB_HOME'] = elo_prob_arr
        
        print(f"✓ Calculated ELO for {len(game_ids_processed)} games")
        
        # Save ELO system state
        with open(self.elo_file, 'wb') as f:
            pickle.dump(self.elo_system.team_ratings, f)
        print(f"✓ Saved ELO ratings to {self.elo_file}")
        
        return games_df
    
    def get_games(self, seasons=None, force_download=False):
        """
        Get NBA games data (uses cache if available)
        
        Args:
            seasons: List of seasons to download (if cache doesn't exist)
            force_download: Force re-download even if cache exists
        
        Returns:
            DataFrame with games including ELO ratings
        """
        if force_download or not os.path.exists(self.cache_file):
            if seasons is None:
                seasons = [
                    '2005-06', '2006-07', '2007-08', '2008-09', '2009-10',
                    '2010-11', '2011-12', '2012-13', '2013-14', '2014-15',
                    '2015-16', '2016-17', '2017-18', '2018-19', '2019-20',
                    '2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26'
                ]
            
            games_df = self.download_all_seasons(seasons)
        else:
            # Try to update current season first
            games_df = self.update_current_season()
            
            if games_df is None:
                games_df = self.load_cache()
        
        if games_df is not None:
            # Calculate ELO ratings
            games_df = self.calculate_elo_ratings(games_df)
        
        return games_df
    
    def get_current_elo_ratings(self):
        """Get current ELO ratings for all teams"""
        import pickle
        
        if os.path.exists(self.elo_file):
            with open(self.elo_file, 'rb') as f:
                ratings = pickle.load(f)
            return ratings
        return {}


# Convenience function for backward compatibility
class NBADataFetcher:
    """Wrapper to maintain compatibility with existing code"""
    
    def __init__(self, seasons=None):
        self.seasons = seasons
        self.manager = NBADataManager()
        
    def fetch_games(self):
        """Fetch games using the new caching system"""
        return self.manager.get_games(seasons=self.seasons)


if __name__ == "__main__":
    # Example usage
    print("NBA Data Manager Example")
    print("="*70)
    
    manager = NBADataManager()
    
    # First time: Downloads all data (slow)
    # Subsequent times: Uses cache (fast!)
    games_df = manager.get_games()
    
    print(f"\n📊 Dataset Info:")
    print(f"   Total games: {len(games_df)}")
    print(f"   Date range: {games_df['GAME_DATE'].min()} to {games_df['GAME_DATE'].max()}")
    print(f"   Seasons: {sorted(games_df['SEASON'].unique())}")
    
    print(f"\n⚖️  ELO Rating Stats:")
    print(f"   ELO range: {games_df['ELO_HOME'].min():.0f} to {games_df['ELO_HOME'].max():.0f}")
    print(f"   Mean ELO: {games_df['ELO_HOME'].mean():.0f}")
    
    # Show current ELO ratings
    current_ratings = manager.get_current_elo_ratings()
    if current_ratings:
        print(f"\n🏆 Top 5 Teams by Current ELO:")
        sorted_ratings = sorted(current_ratings.items(), key=lambda x: x[1], reverse=True)
        all_teams = teams.get_teams()
        for i, (team_id, rating) in enumerate(sorted_ratings[:5]):
            team = [t for t in all_teams if t['id'] == team_id][0]
            print(f"   {i+1}. {team['full_name']}: {rating:.0f}")