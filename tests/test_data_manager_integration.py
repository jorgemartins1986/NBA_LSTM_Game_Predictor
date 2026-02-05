"""
Additional tests for src/nba_data_manager.py
=============================================
Integration tests and more coverage for data manager functions.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nba_data_manager import NBADataManager, NBADataFetcher, ELORatingSystem


class TestNBADataManagerGetGames:
    """Tests for get_games method"""
    
    def test_get_games_uses_cache(self, tmp_path, sample_games_df):
        """get_games should use cache if available"""
        cache_file = str(tmp_path / "cache.csv")
        elo_file = str(tmp_path / "elo.pkl")
        sample_games_df.to_csv(cache_file, index=False)
        
        manager = NBADataManager(cache_file=cache_file, elo_file=elo_file)
        
        with patch.object(manager, 'update_current_season') as mock_update:
            mock_update.return_value = sample_games_df
            result = manager.get_games()
            
            assert result is not None
    
    def test_get_games_force_download(self, tmp_path, sample_games_df):
        """get_games with force_download should re-download"""
        cache_file = str(tmp_path / "cache.csv")
        elo_file = str(tmp_path / "elo.pkl")
        
        manager = NBADataManager(cache_file=cache_file, elo_file=elo_file)
        
        with patch.object(manager, 'download_all_seasons') as mock_download:
            mock_download.return_value = sample_games_df
            result = manager.get_games(force_download=True, seasons=['2024-25'])
            
            mock_download.assert_called_once()


class TestNBADataManagerUpdateCurrentSeason:
    """Tests for update_current_season method"""
    
    @patch('src.nba_data_manager.leaguegamefinder.LeagueGameFinder')
    def test_update_appends_to_cache(self, mock_gamefinder, tmp_path, sample_games_df):
        """update_current_season should append new games to cache"""
        cache_file = str(tmp_path / "cache.csv")
        elo_file = str(tmp_path / "elo.pkl")
        
        # Save initial cache with older season
        old_games = sample_games_df.copy()
        old_games['SEASON'] = '2023-24'
        old_games.to_csv(cache_file, index=False)
        
        # Mock API response for new season
        new_games = sample_games_df.copy()
        new_games['SEASON'] = '2025-26'
        
        mock_df = Mock()
        mock_df.get_data_frames.return_value = [new_games]
        mock_gamefinder.return_value = mock_df
        
        manager = NBADataManager(cache_file=cache_file, elo_file=elo_file)
        result = manager.update_current_season('2025-26')
        
        assert result is not None
        assert '2025-26' in result['SEASON'].unique()


class TestELOCalculationEdgeCases:
    """Edge cases for ELO calculation"""
    
    def test_elo_with_season_transitions(self, tmp_path):
        """ELO should regress at season transitions"""
        # Create games spanning two seasons
        n_games = 20
        
        data = {
            'GAME_ID': [f'00221{i:05d}' for i in range(n_games * 2)],
            'TEAM_ID': [1, 2] * n_games,
            'TEAM_NAME': ['Team A', 'Team B'] * n_games,
            'GAME_DATE': pd.date_range('2024-01-01', periods=n_games).repeat(2),
            'MATCHUP': ['Team A vs. Team B', 'Team B @ Team A'] * n_games,
            'WL': ['W', 'L'] * n_games,
            'PTS': [100, 95] * n_games,
            'SEASON': ['2023-24'] * (n_games // 2 * 2) + ['2024-25'] * (n_games // 2 * 2),
        }
        
        df = pd.DataFrame(data)
        elo_file = str(tmp_path / "elo.pkl")
        manager = NBADataManager(elo_file=elo_file)
        
        result = manager.calculate_elo_ratings(df)
        
        assert result is not None
        assert 'ELO_HOME' in result.columns


class TestNBADataFetcherIntegration:
    """Integration tests for NBADataFetcher"""
    
    def test_fetch_games_delegates_to_manager(self):
        """fetch_games should use manager.get_games"""
        fetcher = NBADataFetcher(seasons=['2024-25'])
        
        with patch.object(fetcher.manager, 'get_games') as mock_get:
            mock_get.return_value = pd.DataFrame()
            fetcher.fetch_games()
            mock_get.assert_called_once()
