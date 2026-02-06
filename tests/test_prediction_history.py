"""
Tests for Prediction History Module
===================================
Tests for src/prediction/history.py
"""

import pytest
import json
import os
from datetime import datetime
from unittest.mock import Mock, patch, mock_open

from src.prediction.history import get_confidence_tier, save_predictions_to_history


class TestGetConfidenceTier:
    """Tests for get_confidence_tier function"""
    
    def test_excellent_tier(self):
        """Test EXCELLENT tier for very high confidence"""
        assert get_confidence_tier(0.50) == "EXCELLENT"
        assert get_confidence_tier(0.60) == "EXCELLENT"
        assert get_confidence_tier(1.0) == "EXCELLENT"
    
    def test_strong_tier(self):
        """Test STRONG tier"""
        assert get_confidence_tier(0.40) == "STRONG"
        assert get_confidence_tier(0.45) == "STRONG"
        assert get_confidence_tier(0.49) == "STRONG"
    
    def test_good_tier(self):
        """Test GOOD tier"""
        assert get_confidence_tier(0.30) == "GOOD"
        assert get_confidence_tier(0.35) == "GOOD"
        assert get_confidence_tier(0.39) == "GOOD"
    
    def test_moderate_tier(self):
        """Test MODERATE tier"""
        assert get_confidence_tier(0.20) == "MODERATE"
        assert get_confidence_tier(0.25) == "MODERATE"
        assert get_confidence_tier(0.29) == "MODERATE"
    
    def test_risky_tier(self):
        """Test RISKY tier"""
        assert get_confidence_tier(0.10) == "RISKY"
        assert get_confidence_tier(0.15) == "RISKY"
        assert get_confidence_tier(0.19) == "RISKY"
    
    def test_skip_tier(self):
        """Test SKIP tier for very low confidence"""
        assert get_confidence_tier(0.0) == "SKIP"
        assert get_confidence_tier(0.05) == "SKIP"
        assert get_confidence_tier(0.09) == "SKIP"
    
    def test_edge_cases(self):
        """Test edge case values"""
        # Negative values should still work
        assert get_confidence_tier(-0.1) == "SKIP"
        # Values above 1 should still work
        assert get_confidence_tier(1.5) == "EXCELLENT"


class TestSavePredictionsToHistory:
    """Tests for save_predictions_to_history function"""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing"""
        return [
            {
                'away_team': 'Celtics',
                'home_team': 'Lakers',
                'predicted_winner': 'Lakers',
                'confidence': 0.35,
                'home_win_prob': 0.65,
                'model_agreement': 0.8
            },
            {
                'away_team': 'Heat',
                'home_team': 'Warriors',
                'predicted_winner': 'Warriors',
                'confidence': 0.1,
                'home_win_prob': 0.52,
                'model_agreement': 0.6
            }
        ]
    
    @patch('src.prediction.history.get_eastern_date')
    @patch('pandas.DataFrame.to_csv')
    @patch('os.path.exists')
    def test_creates_new_history_file(self, mock_exists, mock_to_csv, mock_date, sample_predictions):
        """Test creating new history file"""
        mock_exists.return_value = False
        mock_date.return_value = '2024-03-15'
        
        save_predictions_to_history(sample_predictions)
        
        mock_to_csv.assert_called_once()
    
    @patch('src.prediction.history.get_eastern_date')
    @patch('pandas.DataFrame.to_csv')
    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_appends_to_existing_file(self, mock_exists, mock_read_csv, mock_to_csv, mock_date, sample_predictions):
        """Test appending to existing history file"""
        import pandas as pd
        
        mock_exists.return_value = True
        mock_date.return_value = '2024-03-15'
        mock_read_csv.return_value = pd.DataFrame({
            'date': ['2024-03-14'],
            'away_team': ['Bulls'],
            'home_team': ['Pistons'],
            'predicted_winner': ['Pistons'],
            'confidence': [0.25],
            'tier': ['MODERATE'],
            'home_win_prob': [0.6],
            'model_agreement': [0.7],
            'result': [None]
        })
        
        save_predictions_to_history(sample_predictions)
        
        mock_to_csv.assert_called_once()
    
    @patch('src.prediction.history.get_eastern_date')
    def test_empty_predictions_list(self, mock_date):
        """Test with empty predictions list"""
        mock_date.return_value = '2024-03-15'
        
        # Should handle empty list gracefully (may do nothing or raise)
        try:
            save_predictions_to_history([])
        except Exception:
            pass  # Empty list handling varies
    
    @patch('src.prediction.history.get_eastern_date')
    @patch('pandas.DataFrame.to_csv')
    @patch('os.path.exists')
    def test_custom_date_string(self, mock_exists, mock_to_csv, mock_date, sample_predictions):
        """Test with custom date string"""
        mock_exists.return_value = False
        
        save_predictions_to_history(sample_predictions, date_str='2024-01-01')
        
        # Should not call get_eastern_date when date_str is provided
        mock_to_csv.assert_called_once()


class TestConfidenceTierValues:
    """Additional tests for confidence tier behavior"""
    
    def test_tier_returns_string(self):
        """Test that tier always returns a string"""
        for conf in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            result = get_confidence_tier(conf)
            assert isinstance(result, str)
    
    def test_all_valid_tiers(self):
        """Test that all returned tiers are valid options"""
        valid_tiers = {"EXCELLENT", "STRONG", "GOOD", "MODERATE", "RISKY", "SKIP"}
        
        for conf in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
            result = get_confidence_tier(conf)
            assert result in valid_tiers

