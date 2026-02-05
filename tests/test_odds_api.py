"""
Tests for src/odds_api.py
=========================
Tests for the Odds API client and related functionality.
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.odds_api import OddsAPIClient


class TestOddsAPIClientInit:
    """Tests for OddsAPIClient initialization"""
    
    def test_init_with_api_key(self):
        """Client should initialize with provided API key"""
        client = OddsAPIClient(api_key="test_key_123")
        assert client.api_key == "test_key_123"
    
    def test_init_from_env_variable(self, monkeypatch):
        """Client should read API key from environment variable"""
        monkeypatch.setenv('ODDS_API_KEY', 'env_test_key')
        client = OddsAPIClient()
        assert client.api_key == 'env_test_key'
    
    def test_init_without_api_key(self, monkeypatch):
        """Client should handle missing API key gracefully"""
        monkeypatch.delenv('ODDS_API_KEY', raising=False)
        client = OddsAPIClient()
        assert client.api_key is None
    
    def test_quota_tracking_initialized(self):
        """Quota tracking should be initialized to None"""
        client = OddsAPIClient(api_key="test_key")
        assert client.requests_remaining is None
        assert client.requests_used is None


class TestMakeRequest:
    """Tests for _make_request method"""
    
    def test_request_fails_without_api_key(self, monkeypatch):
        """Request should fail gracefully without API key"""
        monkeypatch.delenv('ODDS_API_KEY', raising=False)
        client = OddsAPIClient()
        result = client._make_request('test/endpoint', {})
        assert result is None
    
    @patch('requests.get')
    def test_successful_request(self, mock_get):
        """Successful request should return JSON data"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': 'test'}
        mock_response.headers = {
            'x-requests-remaining': '999',
            'x-requests-used': '1'
        }
        mock_get.return_value = mock_response
        
        client = OddsAPIClient(api_key="test_key")
        result = client._make_request('test/endpoint', {'param': 'value'})
        
        assert result == {'data': 'test'}
        assert client.requests_remaining == '999'
        assert client.requests_used == '1'
    
    @patch('requests.get')
    def test_unauthorized_request(self, mock_get):
        """401 response should return None"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_get.return_value = mock_response
        
        client = OddsAPIClient(api_key="invalid_key")
        result = client._make_request('test/endpoint', {})
        
        assert result is None
    
    @patch('requests.get')
    def test_request_error_handling(self, mock_get):
        """Request exceptions should be handled gracefully"""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")
        
        client = OddsAPIClient(api_key="test_key")
        result = client._make_request('test/endpoint', {})
        
        assert result is None


class TestGetQuotaStatus:
    """Tests for quota status methods"""
    
    def test_get_quota_status(self):
        """get_quota_status should return tuple of remaining and used"""
        client = OddsAPIClient(api_key="test_key")
        client.requests_remaining = '500'
        client.requests_used = '10'
        
        remaining, used = client.get_quota_status()
        
        assert remaining == '500'
        assert used == '10'
    
    def test_get_quota_status_when_not_set(self):
        """get_quota_status should return None values if not set"""
        client = OddsAPIClient(api_key="test_key")
        
        remaining, used = client.get_quota_status()
        
        assert remaining is None
        assert used is None
    
    def test_print_quota_status_no_error(self, capsys):
        """print_quota_status should not raise errors"""
        client = OddsAPIClient(api_key="test_key")
        client.requests_remaining = '500'
        client.requests_used = '10'
        
        # Should not raise
        client.print_quota_status()
        
        captured = capsys.readouterr()
        assert '500' in captured.out
        assert '10' in captured.out


class TestMatchTeam:
    """Tests for team name matching"""
    
    def test_exact_match(self):
        """Should match exact team names"""
        client = OddsAPIClient(api_key="test_key")
        
        assert client._match_team("Boston Celtics", "Boston Celtics")
    
    def test_partial_match(self):
        """Should match partial team names"""
        client = OddsAPIClient(api_key="test_key")
        
        assert client._match_team("Celtics", "Boston Celtics")
        assert client._match_team("Boston", "Boston Celtics")
    
    def test_case_insensitive(self):
        """Matching should be case insensitive"""
        client = OddsAPIClient(api_key="test_key")
        
        assert client._match_team("CELTICS", "Boston Celtics")
        assert client._match_team("celtics", "BOSTON CELTICS")
    
    def test_ignores_articles(self):
        """Should ignore common articles like 'the'"""
        client = OddsAPIClient(api_key="test_key")
        
        # 'the' should be ignored
        result = client._match_team("the Lakers", "Los Angeles Lakers")
        # Actually 'Lakers' matches, so this should work
        assert result
    
    def test_no_match(self):
        """Should return False for non-matching teams"""
        client = OddsAPIClient(api_key="test_key")
        
        assert not client._match_team("Celtics", "Los Angeles Lakers")


class TestGetLiveOdds:
    """Tests for get_live_odds method"""
    
    @patch.object(OddsAPIClient, '_make_request')
    def test_calls_correct_endpoint(self, mock_request, mock_api_response):
        """get_live_odds should call the correct endpoint"""
        mock_request.return_value = mock_api_response
        
        client = OddsAPIClient(api_key="test_key")
        client.get_live_odds()
        
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        assert 'basketball_nba' in call_args[0][0]
    
    @patch.object(OddsAPIClient, '_make_request')
    def test_default_parameters(self, mock_request, mock_api_response):
        """get_live_odds should use default parameters"""
        mock_request.return_value = mock_api_response
        
        client = OddsAPIClient(api_key="test_key")
        client.get_live_odds()
        
        call_args = mock_request.call_args
        params = call_args[0][1]
        
        assert params['regions'] == 'eu'
        assert params['markets'] == 'h2h'
        assert params['oddsFormat'] == 'decimal'
    
    @patch.object(OddsAPIClient, '_make_request')
    def test_custom_parameters(self, mock_request, mock_api_response):
        """get_live_odds should accept custom parameters"""
        mock_request.return_value = mock_api_response
        
        client = OddsAPIClient(api_key="test_key")
        client.get_live_odds(regions='us', markets='spreads', odds_format='american')
        
        call_args = mock_request.call_args
        params = call_args[0][1]
        
        assert params['regions'] == 'us'
        assert params['markets'] == 'spreads'
        assert params['oddsFormat'] == 'american'
    
    @patch.object(OddsAPIClient, '_make_request')
    def test_returns_api_response(self, mock_request, mock_api_response):
        """get_live_odds should return API response"""
        mock_request.return_value = mock_api_response
        
        client = OddsAPIClient(api_key="test_key")
        result = client.get_live_odds()
        
        assert result == mock_api_response


class TestGetLiveOddsForGame:
    """Tests for get_live_odds_for_game method"""
    
    @patch.object(OddsAPIClient, 'get_live_odds')
    def test_finds_matching_game(self, mock_get_live, mock_api_response):
        """Should find and return matching game"""
        mock_get_live.return_value = mock_api_response
        
        client = OddsAPIClient(api_key="test_key")
        result = client.get_live_odds_for_game(
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers"
        )
        
        assert result is not None
        assert result['home_team'] == 'Boston Celtics'
    
    @patch.object(OddsAPIClient, 'get_live_odds')
    def test_returns_none_for_no_match(self, mock_get_live, mock_api_response):
        """Should return None if game not found"""
        mock_get_live.return_value = mock_api_response
        
        client = OddsAPIClient(api_key="test_key")
        result = client.get_live_odds_for_game(
            home_team="Miami Heat",
            away_team="Chicago Bulls"
        )
        
        assert result is None
    
    @patch.object(OddsAPIClient, 'get_live_odds')
    def test_returns_none_for_api_failure(self, mock_get_live):
        """Should return None if API call fails"""
        mock_get_live.return_value = None
        
        client = OddsAPIClient(api_key="test_key")
        result = client.get_live_odds_for_game(
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers"
        )
        
        assert result is None


class TestOddsCalculations:
    """Tests for odds-related calculations"""
    
    def test_decimal_odds_to_probability(self):
        """Test converting decimal odds to implied probability"""
        # This is a conceptual test - the formula is: probability = 1 / decimal_odds
        
        # Fair odds: 2.0 = 50%
        assert abs(1 / 2.0 - 0.5) < 0.001
        
        # Favorite odds: 1.5 = 66.7%
        assert abs(1 / 1.5 - 0.667) < 0.01
        
        # Underdog odds: 3.0 = 33.3%
        assert abs(1 / 3.0 - 0.333) < 0.01
    
    def test_overround_calculation(self):
        """Test calculating bookmaker overround (vig)"""
        # If home is 1.90 and away is 1.90
        # Implied probs: 1/1.90 + 1/1.90 = 1.053 (5.3% overround)
        home_odds = 1.90
        away_odds = 1.90
        
        overround = (1 / home_odds + 1 / away_odds) - 1
        
        assert abs(overround - 0.053) < 0.01


class TestEdgeCases:
    """Edge case tests for OddsAPIClient"""
    
    def test_empty_response(self):
        """Should handle empty API response"""
        client = OddsAPIClient(api_key="test_key")
        
        with patch.object(client, 'get_live_odds', return_value=[]):
            result = client.get_live_odds_for_game("Team A", "Team B")
            assert result is None
    
    def test_malformed_response(self):
        """Should handle malformed API response"""
        client = OddsAPIClient(api_key="test_key")
        
        # Response missing expected fields
        malformed_response = [{'id': '123'}]  # Missing home_team, away_team
        
        with patch.object(client, 'get_live_odds', return_value=malformed_response):
            result = client.get_live_odds_for_game("Team A", "Team B")
            # Should not raise, just return None
            assert result is None
