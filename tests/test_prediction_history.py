"""
Tests for Prediction History Module
===================================
Tests for src/prediction/history.py
"""

import pytest

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
    
    @pytest.fixture(autouse=True)
    def temporary_history(self, tmp_path, monkeypatch):
        from src.prediction import history
        self.path = tmp_path / 'history.csv'
        monkeypatch.setattr(history, 'PREDICTION_HISTORY_FILE', str(self.path))
        monkeypatch.setattr(history, 'get_eastern_date', lambda: '2024-03-15')

    def test_creates_new_history_file(self, sample_predictions):
        from src.history_store import read_history
        save_predictions_to_history(sample_predictions)
        saved = read_history(self.path)
        assert len(saved.rows) == 2
        assert saved.rows[0]['date'] == '2024-03-15'
        assert saved.rows[0]['winner'] == saved.rows[0]['correct'] == ''

    def test_appends_to_existing_file(self, sample_predictions):
        from src.history_store import read_history
        old = (b'date,away_team,home_team,prediction,winner,confidence,home_win_prob,model_agreement,tier,correct\r\n'
               b'2024-03-14,Bulls,Pistons,Pistons,Pistons,0.250,0.600,0.700,MODERATE,1.0\r\n')
        self.path.write_bytes(old)
        previous = read_history(self.path)
        save_predictions_to_history(sample_predictions)
        saved = read_history(self.path)
        assert len(saved.rows) == 3
        assert {col: saved.rows[0][col] for col in previous.columns} == previous.rows[0]
        assert self.path.with_name('history.csv.bak').read_bytes() == old

    def test_empty_predictions_list(self):
        save_predictions_to_history([])
        assert not self.path.exists()

    def test_custom_date_string(self, sample_predictions):
        from src.history_store import read_history
        save_predictions_to_history(sample_predictions, date_str='2024-01-01')
        saved = read_history(self.path)
        assert all(row['date'] == '2024-01-01' for row in saved.rows)
        assert saved.columns == ['date', 'away_team', 'home_team', 'prediction', 'winner',
                                 'confidence', 'home_win_prob', 'model_agreement', 'tier', 'correct']

    def test_duplicates_leave_file_byte_identical(self, sample_predictions):
        save_predictions_to_history(sample_predictions)
        original = self.path.read_bytes()
        # A later prediction of the same games must never clear their outcomes.
        save_predictions_to_history(sample_predictions + sample_predictions)
        assert self.path.read_bytes() == original
        assert not self.path.with_name('history.csv.bak').exists()


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

