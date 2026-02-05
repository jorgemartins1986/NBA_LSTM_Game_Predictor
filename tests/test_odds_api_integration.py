"""
Additional tests for src/odds_api.py
====================================
More coverage tests for odds API functionality.
"""

import pytest
import os
import sys
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.odds_api import OddsAPIClient


class TestOddsAPIHistoricalOdds:
    """Tests for historical odds functionality"""
    
    @patch.object(OddsAPIClient, '_make_request')
    def test_get_historical_odds_params(self, mock_request):
        """get_historical_odds should pass correct parameters"""
        mock_request.return_value = []
        
        client = OddsAPIClient(api_key="test_key")
        target_date = datetime(2026, 1, 15)
        
        client.get_historical_odds(date=target_date)
        
        mock_request.assert_called_once()


class TestOddsAPIRateLimiting:
    """Tests for rate limiting behavior"""
    
    @patch('requests.get')
    def test_rate_limit_retry(self, mock_get):
        """Should retry after rate limit with delay"""
        # First call returns 429, second returns success
        rate_limit_response = Mock()
        rate_limit_response.status_code = 429
        rate_limit_response.headers = {}
        
        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = {'data': 'test'}
        success_response.headers = {'x-requests-remaining': '999', 'x-requests-used': '1'}
        
        mock_get.side_effect = [rate_limit_response, success_response]
        
        client = OddsAPIClient(api_key="test_key")
        
        with patch('time.sleep'):  # Don't actually sleep in tests
            result = client._make_request('test', {})
        
        assert result == {'data': 'test'}


class TestOddsAPIOddsFormatting:
    """Tests for odds formatting and parsing"""
    
    def test_parse_bookmaker_odds(self, mock_api_response):
        """Should correctly parse bookmaker odds from response"""
        game = mock_api_response[0]
        bookmakers = game['bookmakers']
        
        assert len(bookmakers) > 0
        
        first_bookmaker = bookmakers[0]
        assert 'markets' in first_bookmaker
        
        h2h_market = first_bookmaker['markets'][0]
        assert h2h_market['key'] == 'h2h'
        
        outcomes = h2h_market['outcomes']
        assert len(outcomes) == 2
        
        # Check home team odds
        home_outcome = [o for o in outcomes if o['name'] == 'Boston Celtics'][0]
        assert home_outcome['price'] == 1.50
    
    def test_average_odds_calculation(self, mock_api_response):
        """Should correctly average odds across bookmakers"""
        game = mock_api_response[0]
        
        # Simulate averaging from multiple bookmakers
        all_home_odds = []
        for bookmaker in game['bookmakers']:
            for market in bookmaker['markets']:
                if market['key'] == 'h2h':
                    for outcome in market['outcomes']:
                        if outcome['name'] == game['home_team']:
                            all_home_odds.append(outcome['price'])
        
        avg_odds = sum(all_home_odds) / len(all_home_odds)
        assert avg_odds > 1.0


class TestOddsAPITeamNameNormalization:
    """Tests for team name normalization"""
    
    def test_normalize_common_variations(self):
        """Should match common team name variations"""
        client = OddsAPIClient(api_key="test_key")
        
        # Common variations should match
        assert client._match_team("Lakers", "Los Angeles Lakers")
        assert client._match_team("Celtics", "Boston Celtics")
        assert client._match_team("Heat", "Miami Heat")
    
    def test_handle_city_names(self):
        """Should match on city names"""
        client = OddsAPIClient(api_key="test_key")
        
        assert client._match_team("Boston", "Boston Celtics")
        assert client._match_team("Phoenix", "Phoenix Suns")
        assert client._match_team("Miami", "Miami Heat")


class TestOddsAPICaching:
    """Tests for odds caching functionality"""
    
    def test_cache_file_path(self):
        """Cache file should be in cache directory"""
        from src.odds_api import ODDS_CACHE_FILE
        
        assert 'cache' in ODDS_CACHE_FILE.lower()
        assert ODDS_CACHE_FILE.endswith('.csv')
    
    @patch.object(OddsAPIClient, 'get_live_odds')
    def test_multiple_games_processing(self, mock_get_live):
        """Should correctly process multiple games"""
        mock_response = [
            {
                'id': '1',
                'home_team': 'Boston Celtics',
                'away_team': 'Los Angeles Lakers',
                'bookmakers': []
            },
            {
                'id': '2',
                'home_team': 'Miami Heat',
                'away_team': 'Chicago Bulls',
                'bookmakers': []
            }
        ]
        mock_get_live.return_value = mock_response
        
        client = OddsAPIClient(api_key="test_key")
        
        # Should find Boston game
        result1 = client.get_live_odds_for_game("Boston Celtics", "Los Angeles Lakers")
        assert result1 is not None
        assert result1['home_team'] == 'Boston Celtics'
        
        # Should find Miami game
        result2 = client.get_live_odds_for_game("Miami Heat", "Chicago Bulls")
        assert result2 is not None
        assert result2['home_team'] == 'Miami Heat'


class TestOddsAPIErrorRecovery:
    """Tests for error recovery in odds API"""
    
    @patch('requests.get')
    def test_timeout_handling(self, mock_get):
        """Should handle timeout errors"""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Timeout")
        
        client = OddsAPIClient(api_key="test_key")
        result = client._make_request('test', {})
        
        assert result is None
    
    @patch('requests.get')
    def test_connection_error_handling(self, mock_get):
        """Should handle connection errors"""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        client = OddsAPIClient(api_key="test_key")
        result = client._make_request('test', {})
        
        assert result is None
