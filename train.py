#!/usr/bin/env python
"""
Train the NBA Ensemble Model
============================
Entry point script for training all ensemble models.

Usage:
    python train.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.nba_ensemble_predictor import main

if __name__ == "__main__":
    main()
