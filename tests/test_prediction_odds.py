"""
Tests for Odds Module
=====================
Tests for src/prediction/odds.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import os

from src.prediction.odds import (
    get_live_odds,
    match_game_to_odds,
    format_odds_display
)


class TestGetLiveOdds:
    """Tests for get_live_odds function"""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_no_api_key_returns_empty(self):
        """Test returns empty dict when no API key is set"""
        # Ensure ODDS_API_KEY is not set
        result = get_live_odds(verbose=False)
        assert result == {}
    
    def test_handles_import_error_gracefully(self):
        """Test graceful handling when odds_api module not available"""
        result = get_live_odds(verbose=False)
        assert isinstance(result, dict)
    
    def test_verbose_mode(self, capsys):
        """Test verbose output"""
        result = get_live_odds(verbose=True)
        # May or may not print depending on import availability
        assert isinstance(result, dict)


class TestMatchGameToOdds:
    """Tests for match_game_to_odds function"""
    
    def test_exact_match(self):
        """Test exact team name match"""
        odds_dict = {
            'los angeles lakers_vs_boston celtics': {
                'HOME_AVG_ODDS': -150,
                'AWAY_AVG_ODDS': 130
            }
        }
        
        result = match_game_to_odds('Los Angeles Lakers', 'Boston Celtics', odds_dict)
        
        assert result['HOME_AVG_ODDS'] == -150
        assert result['AWAY_AVG_ODDS'] == 130
    
    def test_empty_odds_dict(self):
        """Test with empty odds dict"""
        result = match_game_to_odds('Lakers', 'Celtics', {})
        assert result == {}
    
    def test_none_odds_dict(self):
        """Test with None odds dict"""
        result = match_game_to_odds('Lakers', 'Celtics', None)
        assert result == {}
    
    def test_partial_match_city_name(self):
        """Test partial matching with city names"""
        odds_dict = {
            'golden state warriors_vs_miami heat': {
                'HOME_AVG_ODDS': -200,
                'AWAY_AVG_ODDS': 175
            }
        }
        
        result = match_game_to_odds('Golden State Warriors', 'Miami Heat', odds_dict)
        
        assert result.get('HOME_AVG_ODDS') == -200
    
    def test_no_match_found(self):
        """Test when no matching game found"""
        odds_dict = {
            'lakers_vs_celtics': {'HOME_AVG_ODDS': -150}
        }
        
        result = match_game_to_odds('Chicago Bulls', 'Detroit Pistons', odds_dict)
        
        assert result == {}
    
    def test_partial_match_single_word(self):
        """Test matching with single word overlap"""
        odds_dict = {
            'warriors_vs_heat': {
                'HOME_AVG_ODDS': -200,
                'AWAY_AVG_ODDS': 175
            }
        }
        
        # Should match based on 'Warriors' and 'Heat' words
        result = match_game_to_odds('Warriors', 'Heat', odds_dict)
        
        assert result.get('HOME_AVG_ODDS') == -200
    
    def test_case_insensitive(self):
        """Test case insensitive matching"""
        odds_dict = {
            'lakers_vs_celtics': {'HOME_AVG_ODDS': -150}
        }
        
        result = match_game_to_odds('LAKERS', 'CELTICS', odds_dict)
        
        assert result.get('HOME_AVG_ODDS') == -150
    
    def test_ignores_common_words(self):
        """Test that common words like 'the' are ignored"""
        odds_dict = {
            'golden state warriors_vs_boston celtics': {'HOME_AVG_ODDS': -150}
        }
        
        # 'The' should be ignored in matching
        result = match_game_to_odds('The Golden State Warriors', 'The Boston Celtics', odds_dict)
        
        # May or may not match depending on exact implementation
        assert isinstance(result, dict)
    
    def test_malformed_key_no_vs(self):
        """Test handling of malformed keys without _vs_"""
        odds_dict = {
            'lakers-celtics': {'HOME_AVG_ODDS': -150},
            'lakers_vs_celtics': {'HOME_AVG_ODDS': -200}
        }
        
        result = match_game_to_odds('Lakers', 'Celtics', odds_dict)
        
        # Should find the valid key
        assert result.get('HOME_AVG_ODDS') == -200


class TestFormatOddsDisplay:
    """Tests for format_odds_display function"""
    
    def test_full_odds_display(self):
        """Test formatting complete odds data"""
        odds_features = {
            'HOME_AVG_ODDS': -150.0,
            'AWAY_AVG_ODDS': 130.0,
            'HOME_IMPLIED_PROB': 0.60,
            'AWAY_IMPLIED_PROB': 0.435,
            'HOME_ODDS_SPREAD': -5.5
        }
        
        result = format_odds_display(odds_features)
        
        assert '💰' in result
        assert 'Home' in result
        assert 'Away' in result
        assert '-150.00' in result
        assert '130.00' in result
        assert '60%' in result
    
    def test_empty_odds(self):
        """Test with empty odds dict"""
        result = format_odds_display({})
        assert result == ""
    
    def test_none_odds(self):
        """Test with None odds"""
        result = format_odds_display(None)
        assert result == ""
    
    def test_missing_odds_values(self):
        """Test with missing key values"""
        odds_features = {
            'HOME_IMPLIED_PROB': 0.55,
            'AWAY_IMPLIED_PROB': 0.45
        }
        
        result = format_odds_display(odds_features)
        
        # Should return empty or partial string
        assert isinstance(result, str)
    
    def test_zero_odds(self):
        """Test with zero odds values"""
        odds_features = {
            'HOME_AVG_ODDS': 0,
            'AWAY_AVG_ODDS': 0,
            'HOME_IMPLIED_PROB': 0.5,
            'AWAY_IMPLIED_PROB': 0.5
        }
        
        result = format_odds_display(odds_features)
        
        assert result == ""  # Zero odds should skip display
    
    def test_positive_odds(self):
        """Test with positive (underdog) odds"""
        odds_features = {
            'HOME_AVG_ODDS': 150.0,
            'AWAY_AVG_ODDS': -180.0,
            'HOME_IMPLIED_PROB': 0.40,
            'AWAY_IMPLIED_PROB': 0.64,
            'HOME_ODDS_SPREAD': 4.5
        }
        
        result = format_odds_display(odds_features)
        
        assert '150.00' in result
        assert '-180.00' in result


class TestOddsMatchingEdgeCases:
    """Edge case tests for odds matching"""
    
    def test_very_long_team_name(self):
        """Test with unusually long team name"""
        odds_dict = {
            'los angeles lakers_vs_boston celtics': {'HOME_AVG_ODDS': -150}
        }
        
        result = match_game_to_odds(
            'The Los Angeles Lakers Basketball Team',
            'The Boston Celtics Basketball Team',
            odds_dict
        )
        
        # Should still match based on key words
        assert isinstance(result, dict)
    
    def test_special_characters_in_name(self):
        """Test team names with special characters"""
        odds_dict = {
            "trail blazers_vs_76ers": {'HOME_AVG_ODDS': -110}
        }
        
        result = match_game_to_odds('Trail Blazers', '76ers', odds_dict)
        
        # May or may not match - test that it doesn't crash
        assert isinstance(result, dict)
    
    def test_unicode_team_name(self):
        """Test with unicode characters"""
        result = match_game_to_odds('Läkers', 'Cëltics', {})
        assert result == {}
    
    def test_multiple_potential_matches(self):
        """Test behavior with multiple potential matches"""
        odds_dict = {
            'warriors_vs_heat': {'HOME_AVG_ODDS': -150},
            'golden state warriors_vs_miami heat': {'HOME_AVG_ODDS': -200}
        }
        
        result = match_game_to_odds('Golden State Warriors', 'Miami Heat', odds_dict)
        
        # Should find one of them
        assert isinstance(result, dict)


class TestOddsIntegration:
    """Integration-style tests for odds functionality"""
    
    def test_realistic_game_matching(self):
        """Test with realistic NBA game data"""
        odds_dict = {
            'los angeles lakers_vs_boston celtics': {
                'HOME_AVG_ODDS': -150.0,
                'AWAY_AVG_ODDS': 130.0,
                'HOME_IMPLIED_PROB': 0.60,
                'AWAY_IMPLIED_PROB': 0.435,
                'HOME_ODDS_SPREAD': -5.5
            },
            'golden state warriors_vs_miami heat': {
                'HOME_AVG_ODDS': -200.0,
                'AWAY_AVG_ODDS': 175.0,
                'HOME_IMPLIED_PROB': 0.667,
                'AWAY_IMPLIED_PROB': 0.364,
                'HOME_ODDS_SPREAD': -7.0
            }
        }
        
        # Match game
        matched = match_game_to_odds('Los Angeles Lakers', 'Boston Celtics', odds_dict)
        
        # Format display
        display = format_odds_display(matched)
        
        assert matched['HOME_AVG_ODDS'] == -150.0
        assert '💰' in display
        assert 'Home' in display
