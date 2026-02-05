"""
Tests for ELO Rating System
===========================
Tests for the ELORatingSystem class in nba_data_manager.py
"""

import pytest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nba_data_manager import ELORatingSystem


class TestELORatingSystemInit:
    """Tests for ELORatingSystem initialization"""
    
    def test_default_initialization(self):
        """ELO system should initialize with default values"""
        elo = ELORatingSystem()
        assert elo.k_factor == 20
        assert elo.initial_rating == 1500
        assert elo.team_ratings == {}
    
    def test_custom_k_factor(self):
        """ELO system should accept custom k_factor"""
        elo = ELORatingSystem(k_factor=32)
        assert elo.k_factor == 32
    
    def test_custom_initial_rating(self):
        """ELO system should accept custom initial rating"""
        elo = ELORatingSystem(initial_rating=1200)
        assert elo.initial_rating == 1200
    
    def test_custom_both_params(self):
        """ELO system should accept both custom parameters"""
        elo = ELORatingSystem(k_factor=16, initial_rating=1000)
        assert elo.k_factor == 16
        assert elo.initial_rating == 1000


class TestGetRating:
    """Tests for get_rating method"""
    
    def test_new_team_gets_initial_rating(self):
        """New team should get initial rating"""
        elo = ELORatingSystem()
        rating = elo.get_rating(12345)
        assert rating == 1500
    
    def test_new_team_stored_in_dict(self):
        """Getting rating for new team should store it"""
        elo = ELORatingSystem()
        _ = elo.get_rating(12345)
        assert 12345 in elo.team_ratings
        assert elo.team_ratings[12345] == 1500
    
    def test_existing_team_returns_stored_rating(self):
        """Existing team should return stored rating"""
        elo = ELORatingSystem()
        elo.team_ratings[12345] = 1600
        rating = elo.get_rating(12345)
        assert rating == 1600
    
    def test_multiple_teams(self):
        """Should handle multiple teams correctly"""
        elo = ELORatingSystem()
        r1 = elo.get_rating(1)
        r2 = elo.get_rating(2)
        r3 = elo.get_rating(3)
        
        assert r1 == r2 == r3 == 1500
        assert len(elo.team_ratings) == 3


class TestExpectedScore:
    """Tests for expected_score method"""
    
    def test_equal_ratings(self):
        """Equal ratings should give 50% win probability"""
        elo = ELORatingSystem()
        expected = elo.expected_score(1500, 1500)
        assert abs(expected - 0.5) < 0.001
    
    def test_higher_rating_favored(self):
        """Higher rated team should have >50% expected score"""
        elo = ELORatingSystem()
        expected = elo.expected_score(1600, 1400)
        assert expected > 0.5
    
    def test_lower_rating_underdog(self):
        """Lower rated team should have <50% expected score"""
        elo = ELORatingSystem()
        expected = elo.expected_score(1400, 1600)
        assert expected < 0.5
    
    def test_400_point_difference(self):
        """400 point difference should give ~90% expected score"""
        elo = ELORatingSystem()
        expected = elo.expected_score(1900, 1500)
        # At 400 point difference, expected score ≈ 0.909
        assert abs(expected - 0.909) < 0.01
    
    def test_expected_scores_sum_to_one(self):
        """Expected scores for both teams should sum to 1"""
        elo = ELORatingSystem()
        expected_a = elo.expected_score(1550, 1450)
        expected_b = elo.expected_score(1450, 1550)
        assert abs((expected_a + expected_b) - 1.0) < 0.001
    
    def test_symmetry(self):
        """Swapping teams should give complementary probabilities"""
        elo = ELORatingSystem()
        expected_a = elo.expected_score(1600, 1400)
        expected_b = elo.expected_score(1400, 1600)
        assert abs(expected_a + expected_b - 1.0) < 0.001


class TestUpdateRatings:
    """Tests for update_ratings method"""
    
    def test_home_win_increases_home_rating(self):
        """Home team win should increase home team rating"""
        elo = ELORatingSystem()
        elo.team_ratings[1] = 1500
        elo.team_ratings[2] = 1500
        
        result = elo.update_ratings(1, 2, team_a_won=True)
        
        assert result['team_a_rating'] > 1500
    
    def test_home_win_decreases_away_rating(self):
        """Home team win should decrease away team rating"""
        elo = ELORatingSystem()
        elo.team_ratings[1] = 1500
        elo.team_ratings[2] = 1500
        
        result = elo.update_ratings(1, 2, team_a_won=True)
        
        assert result['team_b_rating'] < 1500
    
    def test_away_win_decreases_home_rating(self):
        """Away team win should decrease home team rating"""
        elo = ELORatingSystem()
        elo.team_ratings[1] = 1500
        elo.team_ratings[2] = 1500
        
        result = elo.update_ratings(1, 2, team_a_won=False)
        
        assert result['team_a_rating'] < 1500
    
    def test_away_win_increases_away_rating(self):
        """Away team win should increase away team rating"""
        elo = ELORatingSystem()
        elo.team_ratings[1] = 1500
        elo.team_ratings[2] = 1500
        
        result = elo.update_ratings(1, 2, team_a_won=False)
        
        assert result['team_b_rating'] > 1500
    
    def test_ratings_stored_after_update(self):
        """Ratings should be stored in team_ratings dict after update"""
        elo = ELORatingSystem()
        elo.team_ratings[1] = 1500
        elo.team_ratings[2] = 1500
        
        result = elo.update_ratings(1, 2, team_a_won=True)
        
        assert elo.team_ratings[1] == result['team_a_rating']
        assert elo.team_ratings[2] == result['team_b_rating']
    
    def test_rating_changes_sum_to_zero(self):
        """Rating changes should sum to approximately zero (conservation)"""
        elo = ELORatingSystem()
        elo.team_ratings[1] = 1500
        elo.team_ratings[2] = 1500
        
        result = elo.update_ratings(1, 2, team_a_won=True)
        
        # Changes should roughly cancel out (not exactly due to home advantage)
        total_change = result['team_a_change'] + result['team_b_change']
        assert abs(total_change) < 0.01
    
    def test_upset_causes_bigger_change(self):
        """Upset win should cause bigger rating change"""
        elo = ELORatingSystem()
        
        # Favorite wins (expected)
        elo.team_ratings[1] = 1600
        elo.team_ratings[2] = 1400
        result_expected = elo.update_ratings(1, 2, team_a_won=True)
        change_expected = abs(result_expected['team_a_change'])
        
        # Reset and test upset
        elo.team_ratings[1] = 1600
        elo.team_ratings[2] = 1400
        result_upset = elo.update_ratings(1, 2, team_a_won=False)
        change_upset = abs(result_upset['team_a_change'])
        
        # Upset should cause bigger rating change
        assert change_upset > change_expected
    
    def test_result_contains_expected_probability(self):
        """Result should contain expected probability"""
        elo = ELORatingSystem()
        elo.team_ratings[1] = 1500
        elo.team_ratings[2] = 1500
        
        result = elo.update_ratings(1, 2, team_a_won=True)
        
        assert 'expected_a' in result
        assert 0 < result['expected_a'] < 1
    
    def test_home_advantage_effect(self):
        """Home advantage should increase home team expected score"""
        elo = ELORatingSystem()
        
        # With default 100 point home advantage
        expected_with_adv = elo.expected_score(1500 + 100, 1500)
        expected_without = elo.expected_score(1500, 1500)
        
        assert expected_with_adv > expected_without


class TestResetSeason:
    """Tests for reset_season method"""
    
    def test_regression_to_mean(self):
        """Ratings should regress toward mean after season reset"""
        elo = ELORatingSystem()
        elo.team_ratings[1] = 1700  # Above mean
        elo.team_ratings[2] = 1300  # Below mean
        
        elo.reset_season(regression_factor=0.75)
        
        # High rating should decrease
        assert elo.team_ratings[1] < 1700
        assert elo.team_ratings[1] > 1500  # But still above mean
        
        # Low rating should increase
        assert elo.team_ratings[2] > 1300
        assert elo.team_ratings[2] < 1500  # But still below mean
    
    def test_regression_factor_strength(self):
        """Different regression factors should have different effects"""
        elo1 = ELORatingSystem()
        elo1.team_ratings[1] = 1700
        elo1.reset_season(regression_factor=0.9)  # Weak regression
        
        elo2 = ELORatingSystem()
        elo2.team_ratings[1] = 1700
        elo2.reset_season(regression_factor=0.5)  # Strong regression
        
        # Stronger regression should move rating closer to mean
        assert elo2.team_ratings[1] < elo1.team_ratings[1]
    
    def test_full_regression(self):
        """Regression factor of 0 should reset to initial rating"""
        elo = ELORatingSystem()
        elo.team_ratings[1] = 1700
        
        elo.reset_season(regression_factor=0)
        
        assert elo.team_ratings[1] == 1500
    
    def test_no_regression(self):
        """Regression factor of 1 should keep ratings unchanged"""
        elo = ELORatingSystem()
        elo.team_ratings[1] = 1700
        
        elo.reset_season(regression_factor=1.0)
        
        assert elo.team_ratings[1] == 1700


class TestELOIntegration:
    """Integration tests for ELO system"""
    
    def test_simulate_season(self):
        """Simulate a small season and verify rating dynamics"""
        elo = ELORatingSystem(k_factor=20)
        
        # Create two teams
        team_a = 1
        team_b = 2
        
        # Team A is stronger, wins 7 out of 10 games
        results = [True, True, False, True, True, True, False, True, False, True]
        
        for home_win in results:
            elo.update_ratings(team_a, team_b, team_a_won=home_win)
        
        # Team A should have higher rating after winning more
        assert elo.team_ratings[team_a] > elo.team_ratings[team_b]
    
    def test_ratings_bounded(self):
        """Ratings should stay within reasonable bounds"""
        elo = ELORatingSystem(k_factor=20)
        
        team_a = 1
        team_b = 2
        
        # Team A wins 100 straight games
        for _ in range(100):
            elo.update_ratings(team_a, team_b, team_a_won=True)
        
        # Ratings should be bounded (not infinite)
        assert elo.team_ratings[team_a] < 3000
        assert elo.team_ratings[team_b] > 0
