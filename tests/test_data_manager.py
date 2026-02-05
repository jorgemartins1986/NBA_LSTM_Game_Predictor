"""
Tests for src/nba_data_manager.py
=================================
Tests for NBADataManager class and related functionality.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nba_data_manager import NBADataManager, NBADataFetcher


class TestNBADataManagerInit:
    """Tests for NBADataManager initialization"""
    
    def test_default_initialization(self):
        """Manager should initialize with default paths"""
        manager = NBADataManager()
        assert manager.cache_file is not None
        assert manager.elo_file is not None
        assert manager.elo_system is not None
    
    def test_custom_cache_file(self, tmp_path):
        """Manager should accept custom cache file path"""
        cache_file = str(tmp_path / "custom_cache.csv")
        manager = NBADataManager(cache_file=cache_file)
        assert manager.cache_file == cache_file
    
    def test_custom_elo_file(self, tmp_path):
        """Manager should accept custom ELO file path"""
        elo_file = str(tmp_path / "custom_elo.pkl")
        manager = NBADataManager(elo_file=elo_file)
        assert manager.elo_file == elo_file
    
    def test_teams_loaded(self):
        """Manager should load NBA teams on init"""
        manager = NBADataManager()
        assert manager.all_teams is not None
        assert len(manager.all_teams) > 0


class TestLoadCache:
    """Tests for cache loading functionality"""
    
    def test_load_cache_returns_none_if_no_file(self, tmp_path):
        """load_cache should return None if cache file doesn't exist"""
        cache_file = str(tmp_path / "nonexistent.csv")
        manager = NBADataManager(cache_file=cache_file)
        result = manager.load_cache()
        assert result is None
    
    def test_load_cache_reads_csv(self, tmp_path, sample_games_df):
        """load_cache should read existing CSV file"""
        cache_file = str(tmp_path / "cache.csv")
        sample_games_df.to_csv(cache_file, index=False)
        
        manager = NBADataManager(cache_file=cache_file)
        result = manager.load_cache()
        
        assert result is not None
        assert len(result) == len(sample_games_df)
    
    def test_load_cache_converts_dates(self, tmp_path, sample_games_df):
        """load_cache should convert GAME_DATE to datetime"""
        cache_file = str(tmp_path / "cache.csv")
        sample_games_df.to_csv(cache_file, index=False)
        
        manager = NBADataManager(cache_file=cache_file)
        result = manager.load_cache()
        
        assert pd.api.types.is_datetime64_any_dtype(result['GAME_DATE'])


class TestCalculateELORatings:
    """Tests for ELO rating calculation"""
    
    def test_adds_elo_columns(self, sample_games_df, tmp_path):
        """calculate_elo_ratings should add ELO columns"""
        elo_file = str(tmp_path / "elo.pkl")
        manager = NBADataManager(elo_file=elo_file)
        
        result = manager.calculate_elo_ratings(sample_games_df)
        
        assert 'ELO_HOME' in result.columns
        assert 'ELO_AWAY' in result.columns
        assert 'ELO_DIFF' in result.columns
        assert 'ELO_PROB_HOME' in result.columns
    
    def test_elo_values_reasonable(self, sample_games_df, tmp_path):
        """ELO values should be in reasonable range"""
        elo_file = str(tmp_path / "elo.pkl")
        manager = NBADataManager(elo_file=elo_file)
        
        result = manager.calculate_elo_ratings(sample_games_df)
        
        # Filter out zero values (unmatched games)
        non_zero_elo = result[result['ELO_HOME'] > 0]['ELO_HOME']
        
        if len(non_zero_elo) > 0:
            assert non_zero_elo.min() > 1000
            assert non_zero_elo.max() < 2500
    
    def test_elo_prob_in_valid_range(self, sample_games_df, tmp_path):
        """ELO probability should be between 0 and 1"""
        elo_file = str(tmp_path / "elo.pkl")
        manager = NBADataManager(elo_file=elo_file)
        
        result = manager.calculate_elo_ratings(sample_games_df)
        
        # Filter out zero values
        valid_probs = result[result['ELO_PROB_HOME'] > 0]['ELO_PROB_HOME']
        
        if len(valid_probs) > 0:
            assert valid_probs.min() >= 0
            assert valid_probs.max() <= 1
    
    def test_saves_elo_cache(self, sample_games_df, tmp_path):
        """calculate_elo_ratings should save ELO cache file"""
        elo_file = str(tmp_path / "elo.pkl")
        manager = NBADataManager(elo_file=elo_file)
        
        manager.calculate_elo_ratings(sample_games_df)
        
        assert os.path.exists(elo_file)


class TestGetCurrentELORatings:
    """Tests for retrieving current ELO ratings"""
    
    def test_returns_empty_dict_if_no_cache(self, tmp_path):
        """Should return empty dict if no ELO cache exists"""
        elo_file = str(tmp_path / "nonexistent.pkl")
        manager = NBADataManager(elo_file=elo_file)
        
        result = manager.get_current_elo_ratings()
        
        assert result == {}
    
    def test_loads_ratings_from_cache(self, tmp_path, sample_games_df):
        """Should load ratings from cache file"""
        elo_file = str(tmp_path / "elo.pkl")
        manager = NBADataManager(elo_file=elo_file)
        
        # First calculate ELO to create the cache
        manager.calculate_elo_ratings(sample_games_df)
        
        # Then retrieve from cache
        result = manager.get_current_elo_ratings()
        
        assert isinstance(result, dict)


class TestNBADataFetcher:
    """Tests for NBADataFetcher wrapper class"""
    
    def test_initialization(self):
        """NBADataFetcher should initialize correctly"""
        fetcher = NBADataFetcher()
        assert fetcher.manager is not None
        assert fetcher.seasons is None
    
    def test_initialization_with_seasons(self):
        """NBADataFetcher should accept seasons parameter"""
        seasons = ['2023-24', '2024-25']
        fetcher = NBADataFetcher(seasons=seasons)
        assert fetcher.seasons == seasons
    
    def test_has_fetch_games_method(self):
        """NBADataFetcher should have fetch_games method"""
        fetcher = NBADataFetcher()
        assert hasattr(fetcher, 'fetch_games')
        assert callable(fetcher.fetch_games)


class TestDataFrameOperations:
    """Tests for DataFrame operations in NBADataManager"""
    
    def test_games_sorted_by_date(self, sample_games_df, tmp_path):
        """Games should be sorted by date after ELO calculation"""
        elo_file = str(tmp_path / "elo.pkl")
        manager = NBADataManager(elo_file=elo_file)
        
        result = manager.calculate_elo_ratings(sample_games_df)
        
        # Verify sorted by date
        dates = result['GAME_DATE'].values
        assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1))
    
    def test_handles_missing_matchup_info(self, tmp_path):
        """Should handle games with incomplete matchup info"""
        # Create a game with only one team record
        df = pd.DataFrame({
            'GAME_ID': ['001'],
            'TEAM_ID': [1],
            'TEAM_NAME': ['Team A'],
            'GAME_DATE': pd.to_datetime(['2024-01-01']),
            'MATCHUP': ['Team A vs. Team B'],
            'WL': ['W'],
            'PTS': [100],
            'SEASON': ['2023-24'],
        })
        
        elo_file = str(tmp_path / "elo.pkl")
        manager = NBADataManager(elo_file=elo_file)
        
        # Should not raise an exception
        result = manager.calculate_elo_ratings(df)
        assert result is not None


class TestEdgeCases:
    """Edge case tests for NBADataManager"""
    
    def test_empty_dataframe(self, tmp_path):
        """Should handle empty DataFrame"""
        df = pd.DataFrame(columns=[
            'GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE', 
            'MATCHUP', 'WL', 'PTS', 'SEASON'
        ])
        
        elo_file = str(tmp_path / "elo.pkl")
        manager = NBADataManager(elo_file=elo_file)
        
        result = manager.calculate_elo_ratings(df)
        assert result is not None
        assert len(result) == 0
    
    def test_single_game(self, tmp_path):
        """Should handle single game"""
        df = pd.DataFrame({
            'GAME_ID': ['001', '001'],
            'TEAM_ID': [1, 2],
            'TEAM_NAME': ['Team A', 'Team B'],
            'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'MATCHUP': ['Team A vs. Team B', 'Team B @ Team A'],
            'WL': ['W', 'L'],
            'PTS': [100, 95],
            'SEASON': ['2023-24', '2023-24'],
        })
        
        elo_file = str(tmp_path / "elo.pkl")
        manager = NBADataManager(elo_file=elo_file)
        
        result = manager.calculate_elo_ratings(df)
        assert result is not None
        assert len(result) == 2
