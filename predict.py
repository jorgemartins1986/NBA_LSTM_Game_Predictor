#!/usr/bin/env python
"""
Predict Today's NBA Games
=========================
Entry point script for making predictions on today's games.

Usage:
    python predict.py                    # Show today's predictions
    python predict.py --matchup LAL BOS  # Predict specific matchup
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.predict_with_ensemble import main

if __name__ == "__main__":
    main()
