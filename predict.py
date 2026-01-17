#!/usr/bin/env python
"""
Predict Today's NBA Games
=========================
Entry point script for making predictions on today's games.

Usage:
    python predict.py                          # Ensemble predictions for today
    python predict.py --model xgboost          # Use only XGBoost
    python predict.py --model lightgbm         # Use only LightGBM  
    python predict.py --model logistic         # Use only Logistic Regression
    python predict.py --model lstm             # Use only LSTM
    python predict.py --model ensemble         # Use full ensemble (default)
    python predict.py --matchup LAL BOS        # Predict specific matchup
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description='NBA Game Predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py                    Show today's predictions using ensemble
  python predict.py --model xgboost    Use only XGBoost model
  python predict.py --model lstm       Use only LSTM model
  python predict.py --matchup LAL BOS  Predict Lakers vs Celtics
        """
    )
    parser.add_argument(
        '--model', '-m',
        choices=['xgboost', 'lightgbm', 'logistic', 'lstm', 'ensemble'],
        default='ensemble',
        help='Model to use for predictions (default: ensemble)'
    )
    parser.add_argument(
        '--matchup',
        nargs=2,
        metavar=('AWAY', 'HOME'),
        help='Predict a specific matchup (e.g., --matchup LAL BOS)'
    )
    
    args = parser.parse_args()
    
    if args.model == 'ensemble':
        # Use full ensemble
        from src.predict_with_ensemble import main as ensemble_main
        ensemble_main()
    else:
        # Use single model prediction
        from src.predict_with_ensemble import predict_with_single_model
        predict_with_single_model(args.model, args.matchup)


if __name__ == "__main__":
    main()
