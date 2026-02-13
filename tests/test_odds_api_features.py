"""
Tests for Odds API - static methods and OddsDataEnricher
=========================================================
Covers extract_odds_features, american_to_decimal,
decimal_to_implied_probability, and OddsDataEnricher._merge_odds.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.odds_api import OddsAPIClient, OddsDataEnricher


# ---------------------------------------------------------------------------
# extract_odds_features (static, pure)
# ---------------------------------------------------------------------------

class TestExtractOddsFeatures:
    """Tests for OddsAPIClient.extract_odds_features"""

    def test_returns_nan_defaults_for_none(self):
        features = OddsAPIClient.extract_odds_features(None)
        assert np.isnan(features['HOME_AVG_ODDS'])
        assert np.isnan(features['AWAY_AVG_ODDS'])
        assert features['BOOKMAKER_COUNT'] == 0

    def test_returns_nan_defaults_for_missing_bookmakers(self):
        features = OddsAPIClient.extract_odds_features({'id': '123'})
        assert np.isnan(features['HOME_AVG_ODDS'])
        assert features['BOOKMAKER_COUNT'] == 0

    def test_single_bookmaker(self):
        game_odds = {
            'home_team': 'Boston Celtics',
            'away_team': 'LA Lakers',
            'bookmakers': [{
                'key': 'pinnacle',
                'markets': [{
                    'key': 'h2h',
                    'outcomes': [
                        {'name': 'Boston Celtics', 'price': 1.50},
                        {'name': 'LA Lakers', 'price': 2.60},
                    ]
                }]
            }]
        }
        features = OddsAPIClient.extract_odds_features(game_odds)

        assert features['HOME_AVG_ODDS'] == pytest.approx(1.50)
        assert features['AWAY_AVG_ODDS'] == pytest.approx(2.60)
        assert features['HOME_BEST_ODDS'] == pytest.approx(1.50)
        assert features['AWAY_BEST_ODDS'] == pytest.approx(2.60)
        assert features['HOME_WORST_ODDS'] == pytest.approx(1.50)
        assert features['AWAY_WORST_ODDS'] == pytest.approx(2.60)
        assert features['HOME_ODDS_SPREAD'] == pytest.approx(0.0)
        assert features['HOME_IMPLIED_PROB'] == pytest.approx(1 / 1.50)
        assert features['AWAY_IMPLIED_PROB'] == pytest.approx(1 / 2.60)
        assert features['BOOKMAKER_COUNT'] == 1

    def test_multiple_bookmakers(self):
        game_odds = {
            'home_team': 'Team H',
            'away_team': 'Team A',
            'bookmakers': [
                {'key': 'b1', 'markets': [{'key': 'h2h', 'outcomes': [
                    {'name': 'Team H', 'price': 1.40},
                    {'name': 'Team A', 'price': 3.00},
                ]}]},
                {'key': 'b2', 'markets': [{'key': 'h2h', 'outcomes': [
                    {'name': 'Team H', 'price': 1.50},
                    {'name': 'Team A', 'price': 2.80},
                ]}]},
            ]
        }
        features = OddsAPIClient.extract_odds_features(game_odds)

        assert features['HOME_AVG_ODDS'] == pytest.approx(1.45)
        assert features['HOME_BEST_ODDS'] == pytest.approx(1.50)
        assert features['HOME_WORST_ODDS'] == pytest.approx(1.40)
        assert features['HOME_ODDS_SPREAD'] == pytest.approx(0.10)
        assert features['AWAY_BEST_ODDS'] == pytest.approx(3.00)
        assert features['AWAY_WORST_ODDS'] == pytest.approx(2.80)
        assert features['BOOKMAKER_COUNT'] == 2

    def test_ignores_non_h2h_markets(self):
        game_odds = {
            'home_team': 'Team H',
            'away_team': 'Team A',
            'bookmakers': [{
                'key': 'b1',
                'markets': [
                    {'key': 'spreads', 'outcomes': [
                        {'name': 'Team H', 'price': 1.90},
                    ]},
                    {'key': 'h2h', 'outcomes': [
                        {'name': 'Team H', 'price': 1.60},
                        {'name': 'Team A', 'price': 2.40},
                    ]},
                ]
            }]
        }
        features = OddsAPIClient.extract_odds_features(game_odds)

        # Only h2h price should be captured
        assert features['HOME_AVG_ODDS'] == pytest.approx(1.60)

    def test_skips_zero_or_missing_price(self):
        game_odds = {
            'home_team': 'Team H',
            'away_team': 'Team A',
            'bookmakers': [{
                'key': 'b1',
                'markets': [{'key': 'h2h', 'outcomes': [
                    {'name': 'Team H', 'price': 0},    # zero → skipped
                    {'name': 'Team A'},                  # missing → skipped
                ]}]
            }]
        }
        features = OddsAPIClient.extract_odds_features(game_odds)

        assert np.isnan(features['HOME_AVG_ODDS'])
        assert np.isnan(features['AWAY_AVG_ODDS'])


# ---------------------------------------------------------------------------
# american_to_decimal (static, pure)
# ---------------------------------------------------------------------------

class TestAmericanToDecimal:
    """Tests for OddsAPIClient.american_to_decimal"""

    def test_positive_odds(self):
        assert OddsAPIClient.american_to_decimal(150) == pytest.approx(2.50)

    def test_negative_odds(self):
        assert OddsAPIClient.american_to_decimal(-200) == pytest.approx(1.50)

    def test_even_money(self):
        assert OddsAPIClient.american_to_decimal(100) == pytest.approx(2.00)

    def test_heavy_favorite(self):
        assert OddsAPIClient.american_to_decimal(-500) == pytest.approx(1.20)

    def test_big_underdog(self):
        assert OddsAPIClient.american_to_decimal(500) == pytest.approx(6.00)


# ---------------------------------------------------------------------------
# decimal_to_implied_probability (static, pure)
# ---------------------------------------------------------------------------

class TestDecimalToImpliedProbability:
    """Tests for OddsAPIClient.decimal_to_implied_probability"""

    def test_even_odds(self):
        assert OddsAPIClient.decimal_to_implied_probability(2.0) == pytest.approx(0.50)

    def test_favorite(self):
        assert OddsAPIClient.decimal_to_implied_probability(1.50) == pytest.approx(0.6667, rel=1e-3)

    def test_underdog(self):
        assert OddsAPIClient.decimal_to_implied_probability(3.0) == pytest.approx(0.3333, rel=1e-3)

    def test_zero_odds_returns_nan(self):
        assert np.isnan(OddsAPIClient.decimal_to_implied_probability(0))

    def test_negative_odds_returns_nan(self):
        assert np.isnan(OddsAPIClient.decimal_to_implied_probability(-1.5))


# ---------------------------------------------------------------------------
# OddsDataEnricher
# ---------------------------------------------------------------------------

class TestOddsDataEnricher:
    """Tests for OddsDataEnricher.__init__ and cache."""

    @patch('src.odds_api.os.path.exists', return_value=False)
    def test_init_empty_cache(self, _exists):
        enricher = OddsDataEnricher(api_key='test')
        assert enricher.odds_cache.empty

    @patch('src.odds_api.os.path.exists', return_value=False)
    def test_enrich_without_api_key(self, _exists, monkeypatch):
        """Should return original dataframe when no API key."""
        monkeypatch.delenv('ODDS_API_KEY', raising=False)
        enricher = OddsDataEnricher(api_key=None)
        df = pd.DataFrame({'GAME_DATE': ['2026-01-10'], 'PTS': [100]})
        result = enricher.enrich_games_with_odds(df)
        assert len(result) == len(df)


class TestMergeOdds:
    """Tests for OddsDataEnricher._merge_odds"""

    @patch('src.odds_api.os.path.exists', return_value=False)
    def test_merges_matching_games(self, _exists):
        enricher = OddsDataEnricher(api_key='test')

        games_df = pd.DataFrame({
            'GAME_DATE': pd.to_datetime(['2026-01-10']),
            'HOME_TEAM': ['Boston Celtics'],
            'AWAY_TEAM': ['LA Lakers'],
        })
        odds_df = pd.DataFrame({
            'GAME_DATE': pd.to_datetime(['2026-01-10']),
            'HOME_TEAM': ['Boston Celtics'],
            'AWAY_TEAM': ['LA Lakers'],
            'HOME_AVG_ODDS': [1.50],
            'AWAY_AVG_ODDS': [2.60],
            'HOME_BEST_ODDS': [1.55],
            'AWAY_BEST_ODDS': [2.70],
            'HOME_WORST_ODDS': [1.45],
            'AWAY_WORST_ODDS': [2.50],
            'HOME_ODDS_SPREAD': [0.10],
            'AWAY_ODDS_SPREAD': [0.20],
            'HOME_IMPLIED_PROB': [0.667],
            'AWAY_IMPLIED_PROB': [0.385],
            'BOOKMAKER_COUNT': [3],
        })

        result = enricher._merge_odds(games_df, odds_df)

        assert result['HOME_AVG_ODDS'].iloc[0] == pytest.approx(1.50)
        assert result['BOOKMAKER_COUNT'].iloc[0] == 3

    @patch('src.odds_api.os.path.exists', return_value=False)
    def test_no_match_leaves_nan(self, _exists):
        enricher = OddsDataEnricher(api_key='test')

        games_df = pd.DataFrame({
            'GAME_DATE': pd.to_datetime(['2026-01-10']),
            'HOME_TEAM': ['Boston Celtics'],
            'AWAY_TEAM': ['LA Lakers'],
            'HOME_AVG_ODDS': [np.nan],
            'AWAY_AVG_ODDS': [np.nan],
            'HOME_BEST_ODDS': [np.nan],
            'AWAY_BEST_ODDS': [np.nan],
            'HOME_WORST_ODDS': [np.nan],
            'AWAY_WORST_ODDS': [np.nan],
            'HOME_ODDS_SPREAD': [np.nan],
            'AWAY_ODDS_SPREAD': [np.nan],
            'HOME_IMPLIED_PROB': [np.nan],
            'AWAY_IMPLIED_PROB': [np.nan],
            'BOOKMAKER_COUNT': [np.nan],
        })
        # Odds for a different date — should not match
        odds_df = pd.DataFrame({
            'GAME_DATE': pd.to_datetime(['2026-01-11']),
            'HOME_TEAM': ['Boston Celtics'],
            'AWAY_TEAM': ['LA Lakers'],
            'HOME_AVG_ODDS': [1.50],
            'AWAY_AVG_ODDS': [2.60],
            'HOME_BEST_ODDS': [1.55],
            'AWAY_BEST_ODDS': [2.70],
            'HOME_WORST_ODDS': [1.45],
            'AWAY_WORST_ODDS': [2.50],
            'HOME_ODDS_SPREAD': [0.10],
            'AWAY_ODDS_SPREAD': [0.20],
            'HOME_IMPLIED_PROB': [0.667],
            'AWAY_IMPLIED_PROB': [0.385],
            'BOOKMAKER_COUNT': [3],
        })

        result = enricher._merge_odds(games_df, odds_df)

        assert np.isnan(result['HOME_AVG_ODDS'].iloc[0])
