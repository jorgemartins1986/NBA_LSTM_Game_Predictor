"""
Tests for Prediction Features Module
====================================
Tests for src/prediction/features.py
"""

import pytest
import numpy as np
import pandas as pd

from src.prediction.features import (
    TeamFeatures,
    HeadToHeadFeatures,
    StandingsFeatures,
    OddsFeatures,
    GameFeatures,
    FeatureComputer
)


class TestTeamFeatures:
    """Tests for TeamFeatures dataclass"""
    
    def test_basic_creation(self):
        """Test basic creation"""
        features = TeamFeatures(team_id=1, features={'PTS': 110.5, 'REB': 45.0})
        
        assert features.team_id == 1
        assert features.get('PTS') == 110.5
    
    def test_get_default(self):
        """Test get with default value"""
        features = TeamFeatures(team_id=1)
        
        assert features.get('MISSING', 0.0) == 0.0
        assert features.get('MISSING', 99.0) == 99.0
    
    def test_to_dict(self):
        """Test conversion to dict"""
        features = TeamFeatures(team_id=1, features={'A': 1, 'B': 2})
        
        d = features.to_dict()
        
        assert d == {'A': 1, 'B': 2}
        # Should be a copy
        d['C'] = 3
        assert 'C' not in features.features


class TestHeadToHeadFeatures:
    """Tests for HeadToHeadFeatures dataclass"""
    
    def test_defaults(self):
        """Test default values"""
        h2h = HeadToHeadFeatures()
        
        assert h2h.win_rate == 0.5
        assert h2h.games_played == 0
        assert h2h.pts_diff == 0.0
    
    def test_to_dict(self):
        """Test conversion to dict"""
        h2h = HeadToHeadFeatures(win_rate=0.6, games_played=5, pts_diff=3.5)
        
        d = h2h.to_dict()
        
        assert d['H2H_WIN_RATE'] == 0.6
        assert d['H2H_GAMES'] == 5
        assert d['H2H_PTS_DIFF'] == 3.5


class TestStandingsFeatures:
    """Tests for StandingsFeatures dataclass"""
    
    def test_defaults(self):
        """Test default values"""
        standings = StandingsFeatures()
        
        assert standings.wins == 0
        assert standings.win_pct == 0.5
        assert standings.conf_rank == 8
    
    def test_to_dict(self):
        """Test conversion to dict"""
        standings = StandingsFeatures(wins=30, losses=20, win_pct=0.6)
        
        d = standings.to_dict()
        
        assert d['WINS'] == 30
        assert d['LOSSES'] == 20
        assert d['WIN_PCT'] == 0.6


class TestOddsFeatures:
    """Tests for OddsFeatures dataclass"""
    
    def test_defaults(self):
        """Test default values"""
        odds = OddsFeatures()
        
        assert odds.avg_odds == 0.0
        assert odds.implied_prob == 0.5
    
    def test_to_dict(self):
        """Test conversion to dict"""
        odds = OddsFeatures(avg_odds=1.91, implied_prob=0.52, spread=-2.5)
        
        d = odds.to_dict()
        
        assert d['AVG_ODDS'] == 1.91
        assert d['IMPLIED_PROB'] == 0.52


class TestFeatureComputer:
    """Tests for FeatureComputer class"""
    
    @pytest.fixture
    def computer(self):
        return FeatureComputer(window_size=20)
    
    @pytest.fixture
    def sample_games_df(self):
        """Create sample games dataframe for H2H testing"""
        return pd.DataFrame({
            'TEAM_ID': [1, 1, 1, 2, 2, 2, 1, 1],
            'TEAM_ABBREVIATION': ['LAL', 'LAL', 'LAL', 'BOS', 'BOS', 'BOS', 'LAL', 'LAL'],
            'MATCHUP': ['LAL vs. BOS', 'LAL @ GSW', 'LAL vs. BOS', 'BOS @ LAL', 'BOS vs. GSW', 'BOS @ LAL', 'LAL @ BOS', 'LAL vs. BOS'],
            'WL': ['W', 'L', 'W', 'L', 'W', 'L', 'W', 'L'],
            'PLUS_MINUS': [5, -3, 8, -5, 10, -8, 6, -2],
            'GAME_DATE': pd.date_range('2024-01-01', periods=8)
        })
    
    def test_compute_head_to_head_basic(self, computer, sample_games_df):
        """Test head-to-head computation"""
        h2h = computer.compute_head_to_head(
            sample_games_df,
            team_id=1,  # LAL
            opponent_id=2  # BOS
        )
        
        assert isinstance(h2h, HeadToHeadFeatures)
        # LAL played 4 games vs BOS: W, W, W, L
        assert h2h.games_played == 4
        assert h2h.win_rate == 0.75  # 3 wins out of 4
    
    def test_compute_head_to_head_no_games(self, computer):
        """Test H2H when teams haven't played"""
        df = pd.DataFrame({
            'TEAM_ID': [1, 2],
            'TEAM_ABBREVIATION': ['LAL', 'BOS'],
            'MATCHUP': ['LAL vs. GSW', 'BOS vs. MIA'],
            'WL': ['W', 'W'],
            'PLUS_MINUS': [5, 5]
        })
        
        h2h = computer.compute_head_to_head(df, team_id=1, opponent_id=2)
        
        assert h2h.win_rate == 0.5  # Default
        assert h2h.games_played == 0
    
    def test_compute_head_to_head_empty_df(self, computer):
        """Test H2H with empty dataframe"""
        df = pd.DataFrame(columns=['TEAM_ID', 'TEAM_ABBREVIATION', 'MATCHUP', 'WL'])
        
        h2h = computer.compute_head_to_head(df, team_id=1, opponent_id=2)
        
        assert h2h.win_rate == 0.5
        assert h2h.games_played == 0
    
    def test_build_feature_vector_basic(self, computer):
        """Test feature vector building"""
        home_features = {'ROLL_AVG_PTS': 110.0, 'ROLL_AVG_REB': 45.0}
        away_features = {'ROLL_AVG_PTS': 105.0, 'ROLL_AVG_REB': 42.0}
        feature_cols = ['HOME_ROLL_AVG_PTS', 'HOME_ROLL_AVG_REB', 
                       'AWAY_ROLL_AVG_PTS', 'AWAY_ROLL_AVG_REB']
        
        vector = computer.build_feature_vector(
            home_features, away_features, feature_cols
        )
        
        assert vector.shape == (1, 4)
        assert vector[0, 0] == 110.0  # HOME_ROLL_AVG_PTS
        assert vector[0, 2] == 105.0  # AWAY_ROLL_AVG_PTS
    
    def test_build_feature_vector_with_h2h(self, computer):
        """Test feature vector with H2H features"""
        home_features = {'PTS': 110}
        away_features = {'PTS': 105}
        h2h_home = {'H2H_WIN_RATE': 0.6, 'H2H_GAMES': 5}
        
        feature_cols = ['HOME_PTS', 'AWAY_PTS', 'HOME_H2H_WIN_RATE', 'HOME_H2H_GAMES']
        
        vector = computer.build_feature_vector(
            home_features, away_features, feature_cols,
            h2h_home=h2h_home
        )
        
        assert vector[0, 2] == 0.6  # H2H_WIN_RATE
        assert vector[0, 3] == 5    # H2H_GAMES
    
    def test_build_feature_vector_missing_features(self, computer):
        """Test feature vector handles missing features"""
        home_features = {'PTS': 110}
        away_features = {'PTS': 105}
        feature_cols = ['HOME_PTS', 'AWAY_PTS', 'HOME_MISSING_FEATURE']
        
        vector = computer.build_feature_vector(
            home_features, away_features, feature_cols
        )
        
        # Missing features should get default value
        assert vector.shape == (1, 3)
    
    def test_get_default_value(self, computer):
        """Test default value lookup"""
        # Should find matching pattern
        assert computer._get_default_value('HOME_ROLL_AVG_PTS') == 110.0
        assert computer._get_default_value('AWAY_ROLL_AVG_FG_PCT') == 0.46
        assert computer._get_default_value('HOME_WIN_STREAK') == 0
        
        # Unknown should return 0
        assert computer._get_default_value('UNKNOWN_FEATURE') == 0.0
    
    def test_compute_differentials(self, computer):
        """Test differential feature computation"""
        home = {'HOME_PTS': 110, 'HOME_REB': 45}
        away = {'AWAY_PTS': 105, 'AWAY_REB': 42}
        
        diffs = computer.compute_differentials(home, away)
        
        assert 'DIFF_PTS' in diffs
        assert diffs['DIFF_PTS'] == 5  # 110 - 105
        assert diffs['DIFF_REB'] == 3  # 45 - 42
    
    def test_compute_fatigue_score_rested(self):
        """Test fatigue for rested team"""
        score = FeatureComputer.compute_fatigue_score(
            days_rest=3, is_back_to_back=False, is_3_in_4=False
        )
        assert score == 0.0
    
    def test_compute_fatigue_score_back_to_back(self):
        """Test fatigue for back-to-back"""
        score = FeatureComputer.compute_fatigue_score(
            days_rest=1, is_back_to_back=True, is_3_in_4=False
        )
        assert score == 0.5  # 0.2 (1 day rest) + 0.3 (b2b)
    
    def test_compute_fatigue_score_exhausted(self):
        """Test fatigue for exhausted team"""
        score = FeatureComputer.compute_fatigue_score(
            days_rest=0, is_back_to_back=True, is_3_in_4=True
        )
        assert score == 1.0  # Capped at 1.0
