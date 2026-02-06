"""
Tests for NBA Data Module
=========================
Tests for src/prediction/nba_data.py
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta

from src.prediction.nba_data import (
    get_eastern_date,
    get_current_season,
    get_default_standings,
    compute_head_to_head,
)


class TestGetEasternDate:
    """Tests for get_eastern_date function"""
    
    def test_returns_string(self):
        """Test that function returns a string date"""
        result = get_eastern_date()
        assert isinstance(result, str)
        # Should be in YYYY-MM-DD format
        assert len(result) == 10
        assert result[4] == '-'
        assert result[7] == '-'
    
    def test_consistent_within_call(self):
        """Test that multiple calls return same-format date"""
        result1 = get_eastern_date()
        result2 = get_eastern_date()
        # Format should be consistent
        assert len(result1) == len(result2)


class TestGetCurrentSeason:
    """Tests for get_current_season function"""
    
    def test_returns_season_string(self):
        """Test season string format"""
        result = get_current_season()
        assert isinstance(result, str)
        assert '-' in result
        assert len(result) == 7  # YYYY-YY format
    
    def test_season_format_valid(self):
        """Test season has valid format like 2024-25"""
        result = get_current_season()
        parts = result.split('-')
        assert len(parts) == 2
        assert len(parts[0]) == 4
        assert len(parts[1]) == 2
        # Second part should be first_year + 1 mod 100
        first_year = int(parts[0])
        second_part = int(parts[1])
        assert second_part == (first_year + 1) % 100


class TestGetDefaultStandings:
    """Tests for get_default_standings function"""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary"""
        result = get_default_standings()
        assert isinstance(result, dict)
    
    def test_has_required_keys(self):
        """Test that all required keys are present"""
        result = get_default_standings()
        
        expected_keys = ['WINS', 'LOSSES', 'WIN_PCT', 'CONF_RANK', 'LEAGUE_RANK', 'GAMES_BACK', 'STREAK']
        for key in expected_keys:
            assert key in result
    
    def test_default_values(self):
        """Test default values are sensible"""
        result = get_default_standings()
        
        assert result['WINS'] == 0
        assert result['LOSSES'] == 0
        assert result['WIN_PCT'] == 0.5  # Neutral assumption
        assert result['CONF_RANK'] == 8  # Middle of conference
        assert result['LEAGUE_RANK'] == 15  # Middle of league
    
    def test_returns_fresh_dict(self):
        """Test that each call returns a new dict"""
        result1 = get_default_standings()
        result2 = get_default_standings()
        
        result1['WINS'] = 50
        assert result2['WINS'] == 0  # Should not be affected


class TestComputeHeadToHead:
    """Tests for compute_head_to_head function"""
    
    def test_with_h2h_games(self):
        """Test with existing head-to-head games"""
        games_df = pd.DataFrame({
            'TEAM_ID': [100, 200, 100, 200, 100, 200],
            'TEAM_ABBREVIATION': ['LAL', 'BOS', 'LAL', 'BOS', 'LAL', 'BOS'],
            'MATCHUP': ['LAL vs. BOS', 'BOS @ LAL', 'LAL @ BOS', 'BOS vs. LAL', 'LAL vs. BOS', 'BOS @ LAL'],
            'WL': ['W', 'L', 'L', 'W', 'W', 'L'],
            'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-02-01', '2024-02-01', '2024-03-01', '2024-03-01']),
            'PLUS_MINUS': [10, -10, -5, 5, 8, -8]
        })
        
        result = compute_head_to_head(games_df, team_id=100, opponent_id=200)
        
        assert 'H2H_WIN_RATE' in result
        assert 'H2H_GAMES' in result
        assert 'H2H_PTS_DIFF' in result
        assert result['H2H_GAMES'] == 3
        assert result['H2H_WIN_RATE'] == pytest.approx(2/3, rel=0.01)
    
    def test_no_h2h_games(self):
        """Test when teams haven't played each other"""
        games_df = pd.DataFrame({
            'TEAM_ID': [100, 300],
            'TEAM_ABBREVIATION': ['LAL', 'MIA'],
            'MATCHUP': ['LAL vs. MIA', 'MIA @ LAL'],
            'WL': ['W', 'L'],
            'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'PLUS_MINUS': [10, -10]
        })
        
        result = compute_head_to_head(games_df, team_id=100, opponent_id=200)
        
        assert result['H2H_WIN_RATE'] == 0.5
        assert result['H2H_GAMES'] == 0
        assert result['H2H_PTS_DIFF'] == 0
    
    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        games_df = pd.DataFrame(columns=['TEAM_ID', 'TEAM_ABBREVIATION', 'MATCHUP', 'WL', 'GAME_DATE', 'PLUS_MINUS'])
        
        result = compute_head_to_head(games_df, team_id=100, opponent_id=200)
        
        assert result['H2H_GAMES'] == 0
    
    def test_window_limit(self):
        """Test window limits number of games considered"""
        # Create 10 H2H games (5 matches where both teams play)
        dates = pd.to_datetime([f'2024-{i+1:02d}-01' for i in range(10)])
        games_df = pd.DataFrame({
            'TEAM_ID': [100] * 10,
            'TEAM_ABBREVIATION': ['LAL'] * 10,
            'MATCHUP': ['LAL vs. BOS'] * 10,
            'WL': ['W'] * 10,
            'GAME_DATE': dates,
            'PLUS_MINUS': [5] * 10
        })
        
        # Add opponent games to get abbreviation
        opp_df = pd.DataFrame({
            'TEAM_ID': [200],
            'TEAM_ABBREVIATION': ['BOS'],
            'MATCHUP': ['BOS @ LAL'],
            'WL': ['L'],
            'GAME_DATE': [dates[0]],
            'PLUS_MINUS': [-5]
        })
        games_df = pd.concat([games_df, opp_df], ignore_index=True)
        
        result = compute_head_to_head(games_df, team_id=100, opponent_id=200, window=5)
        
        assert result['H2H_GAMES'] <= 5
    
    def test_returns_required_keys(self):
        """Test that result always has required keys"""
        games_df = pd.DataFrame({
            'TEAM_ID': [100],
            'TEAM_ABBREVIATION': ['LAL'],
            'MATCHUP': ['LAL vs. BOS'],
            'WL': ['W'],
            'GAME_DATE': pd.to_datetime(['2024-01-01']),
            'PLUS_MINUS': [10]
        })
        
        result = compute_head_to_head(games_df, team_id=100, opponent_id=200)
        
        assert 'H2H_WIN_RATE' in result
        assert 'H2H_GAMES' in result
        assert 'H2H_PTS_DIFF' in result


class TestHeadToHeadEdgeCases:
    """Edge cases for head-to-head computation"""
    
    def test_missing_plus_minus_column(self):
        """Test handling when PLUS_MINUS column is missing"""
        games_df = pd.DataFrame({
            'TEAM_ID': [100, 200],
            'TEAM_ABBREVIATION': ['LAL', 'BOS'],
            'MATCHUP': ['LAL vs. BOS', 'BOS @ LAL'],
            'WL': ['W', 'L'],
            'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-01-01'])
        })
        
        result = compute_head_to_head(games_df, team_id=100, opponent_id=200)
        
        # Should handle gracefully
        assert 'H2H_WIN_RATE' in result
        assert 'H2H_PTS_DIFF' in result
        # PTS_DIFF should be 0 when column missing
        assert result['H2H_PTS_DIFF'] == 0
    
    def test_all_losses(self):
        """Test when team has lost all H2H games"""
        games_df = pd.DataFrame({
            'TEAM_ID': [100, 200, 100, 200],
            'TEAM_ABBREVIATION': ['LAL', 'BOS', 'LAL', 'BOS'],
            'MATCHUP': ['LAL vs. BOS', 'BOS @ LAL', 'LAL @ BOS', 'BOS vs. LAL'],
            'WL': ['L', 'W', 'L', 'W'],
            'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-02-01', '2024-02-01']),
            'PLUS_MINUS': [-10, 10, -5, 5]
        })
        
        result = compute_head_to_head(games_df, team_id=100, opponent_id=200)
        
        assert result['H2H_WIN_RATE'] == 0.0
        assert result['H2H_GAMES'] == 2

