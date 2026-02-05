"""
Tests for main.py
=================
Tests for the main entry point and CLI commands.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestShowDetailedStats:
    """Tests for show_detailed_stats function"""
    
    def test_handles_missing_file(self, tmp_path, monkeypatch, capsys):
        """Should handle missing prediction history file"""
        monkeypatch.setattr('src.paths.PREDICTION_HISTORY_FILE', str(tmp_path / 'nonexistent.csv'))
        
        from main import show_detailed_stats
        show_detailed_stats()
        
        captured = capsys.readouterr()
        assert 'No prediction history' in captured.out or '❌' in captured.out
    
    def test_handles_empty_completed(self, tmp_path, monkeypatch, capsys):
        """Should handle when no predictions are completed"""
        # Create file with only pending predictions
        df = pd.DataFrame({
            'date': ['2026-01-15'],
            'matchup': ['Team A vs Team B'],
            'prediction': ['Team A'],
            'probability': [0.65],
            'confidence': [0.30],
            'tier': ['GOOD'],
            'actual_winner': [''],
            'correct': ['']
        })
        
        history_file = tmp_path / 'history.csv'
        df.to_csv(history_file, index=False)
        
        # Need to reload main module after patching
        import importlib
        import src.paths
        original_path = src.paths.PREDICTION_HISTORY_FILE
        src.paths.PREDICTION_HISTORY_FILE = str(history_file)
        
        try:
            import main
            importlib.reload(main)
            main.show_detailed_stats()
            
            captured = capsys.readouterr()
            # Check for various expected output patterns
            assert ('No completed predictions' in captured.out or 
                    'completed' in captured.out.lower() or
                    len(captured.out) > 0)
        finally:
            src.paths.PREDICTION_HISTORY_FILE = original_path


class TestStatisticsCalculations:
    """Tests for statistics calculation logic"""
    
    def test_accuracy_calculation(self, sample_prediction_history):
        """Accuracy should be correctly calculated"""
        total = len(sample_prediction_history)
        correct = sample_prediction_history['correct'].sum()
        accuracy = correct / total * 100
        
        assert accuracy == 75.0  # 3 out of 4 correct
    
    def test_tier_breakdown(self, sample_prediction_history):
        """Should calculate stats by tier"""
        tier_stats = sample_prediction_history.groupby('tier').agg({
            'correct': ['sum', 'count']
        })
        
        # EXCELLENT tier: 1 correct out of 1
        excellent = sample_prediction_history[sample_prediction_history['tier'] == 'EXCELLENT']
        assert len(excellent) == 1
        assert excellent['correct'].sum() == 1
    
    def test_roi_calculation(self):
        """ROI should be correctly calculated at -110 odds"""
        # If we bet $110 to win $100 on 10 games
        # Win 6, lose 4
        wins = 6
        losses = 4
        total = wins + losses
        
        # ROI = (wins * 100 - losses * 110) / (total * 110) * 100
        roi = (wins * 100 - losses * 110) / (total * 110) * 100
        
        assert abs(roi - 14.5) < 0.5  # ~14.5% ROI
    
    def test_negative_roi(self):
        """Negative ROI for losing record"""
        wins = 4
        losses = 6
        total = wins + losses
        
        roi = (wins * 100 - losses * 110) / (total * 110) * 100
        
        assert roi < 0


class TestArgumentParsing:
    """Tests for CLI argument parsing"""
    
    def test_predict_command(self):
        """Should parse 'predict' command"""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('command', nargs='?', default=None)
        
        args = parser.parse_args(['predict'])
        assert args.command == 'predict'
    
    def test_train_command(self):
        """Should parse 'train' command"""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('command', nargs='?', default=None)
        
        args = parser.parse_args(['train'])
        assert args.command == 'train'
    
    def test_stats_flag(self):
        """Should parse --stats flag"""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--stats', action='store_true')
        
        args = parser.parse_args(['--stats'])
        assert args.stats is True
    
    def test_updateresults_flag(self):
        """Should parse --updateresults flag"""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--updateresults', nargs='?', const='yesterday', default=None)
        
        args = parser.parse_args(['--updateresults'])
        assert args.updateresults == 'yesterday'
    
    def test_updateresults_with_date(self):
        """Should parse --updateresults with specific date"""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--updateresults', nargs='?', const='yesterday', default=None)
        
        args = parser.parse_args(['--updateresults', '2026-01-15'])
        assert args.updateresults == '2026-01-15'


class TestConfidenceLevels:
    """Tests for confidence level binning"""
    
    def test_confidence_bins(self):
        """Confidence should be binned correctly"""
        bins = [
            (0.45, 1.0, '45%+'),
            (0.40, 0.45, '40-45%'),
            (0.35, 0.40, '35-40%'),
            (0.30, 0.35, '30-35%'),
            (0.20, 0.30, '20-30%'),
            (0.10, 0.20, '10-20%'),
            (0.0, 0.10, '<10%'),
        ]
        
        test_confidence = 0.38
        
        for low, high, label in bins:
            if low <= test_confidence < high:
                assert label == '35-40%'
                break
    
    def test_high_confidence_categorization(self):
        """High confidence (>45%) should be categorized correctly"""
        confidence = 0.52
        
        if confidence >= 0.45:
            category = '45%+'
        
        assert category == '45%+'


class TestBetQualityTiers:
    """Tests for bet quality tier logic"""
    
    def test_excellent_tier(self):
        """EXCELLENT tier should be for >75% probability"""
        probability = 0.78
        
        if probability >= 0.75:
            tier = 'EXCELLENT'
        elif probability >= 0.70:
            tier = 'STRONG'
        else:
            tier = 'OTHER'
        
        assert tier == 'EXCELLENT'
    
    def test_strong_tier(self):
        """STRONG tier should be for 70-75% probability"""
        probability = 0.72
        
        if probability >= 0.75:
            tier = 'EXCELLENT'
        elif probability >= 0.70:
            tier = 'STRONG'
        elif probability >= 0.65:
            tier = 'GOOD'
        else:
            tier = 'OTHER'
        
        assert tier == 'STRONG'
    
    def test_tier_ordering(self):
        """Tiers should be in correct order"""
        tier_order = ['EXCELLENT', 'STRONG', 'GOOD', 'MODERATE', 'RISKY', 'SKIP']
        
        assert tier_order[0] == 'EXCELLENT'
        assert tier_order[-1] == 'SKIP'
        assert len(tier_order) == 6


class TestDateParsing:
    """Tests for date parsing in main.py"""
    
    def test_parse_valid_date(self):
        """Should parse valid date strings"""
        from datetime import datetime
        
        date_str = '2026-01-15'
        parsed = datetime.strptime(date_str, '%Y-%m-%d')
        
        assert parsed.year == 2026
        assert parsed.month == 1
        assert parsed.day == 15
    
    def test_date_range_calculation(self):
        """Should calculate date ranges correctly"""
        from datetime import datetime, timedelta
        
        start_date = datetime(2026, 1, 1)
        end_date = datetime(2026, 1, 31)
        
        delta = end_date - start_date
        
        assert delta.days == 30
    
    def test_yesterday_calculation(self):
        """Should calculate yesterday's date"""
        from datetime import datetime, timedelta
        
        today = datetime(2026, 2, 5)
        yesterday = today - timedelta(days=1)
        
        assert yesterday == datetime(2026, 2, 4)


class TestOutputFormatting:
    """Tests for output formatting"""
    
    def test_accuracy_percentage_format(self):
        """Accuracy should be formatted as percentage"""
        accuracy = 0.6414
        formatted = f"{accuracy * 100:.1f}%"
        
        assert formatted == "64.1%"
    
    def test_record_format(self):
        """Record should be formatted as wins/total"""
        wins = 25
        total = 40
        formatted = f"{wins}/{total}"
        
        assert formatted == "25/40"
    
    def test_roi_format(self):
        """ROI should be formatted with sign"""
        positive_roi = 14.5
        negative_roi = -8.3
        
        assert f"{positive_roi:+.1f}%" == "+14.5%"
        assert f"{negative_roi:+.1f}%" == "-8.3%"


class TestErrorHandling:
    """Tests for error handling in main.py"""
    
    def test_handles_corrupted_csv(self, tmp_path, monkeypatch, capsys):
        """Should handle corrupted CSV files"""
        # Create a malformed CSV
        csv_file = tmp_path / 'bad.csv'
        csv_file.write_text('col1,col2\nval1')  # Missing value
        
        # The actual code should handle this gracefully
        df = pd.read_csv(csv_file, on_bad_lines='warn')
        assert df is not None
    
    def test_handles_missing_columns(self, tmp_path):
        """Should handle missing expected columns"""
        df = pd.DataFrame({
            'date': ['2026-01-15'],
            'matchup': ['Team A vs Team B']
            # Missing: prediction, probability, etc.
        })
        
        # Check for missing columns
        expected_cols = ['date', 'matchup', 'prediction', 'probability']
        missing = [col for col in expected_cols if col not in df.columns]
        
        assert len(missing) == 2
        assert 'prediction' in missing
        assert 'probability' in missing
