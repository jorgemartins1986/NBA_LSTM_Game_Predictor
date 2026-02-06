"""
History Module
==============
Functions for saving and managing prediction history.
"""

import os
import pandas as pd
from typing import Dict, List, Optional

from .nba_data import get_eastern_date
from ..paths import PREDICTION_HISTORY_FILE


def get_confidence_tier(confidence: float) -> str:
    """Map confidence score to betting tier label."""
    if confidence >= 0.50:
        return "EXCELLENT"
    elif confidence >= 0.40:
        return "STRONG"
    elif confidence >= 0.30:
        return "GOOD"
    elif confidence >= 0.20:
        return "MODERATE"
    elif confidence >= 0.10:
        return "RISKY"
    else:
        return "SKIP"


def save_predictions_to_history(predictions: List[Dict], date_str: str = None) -> None:
    """Save predictions to history CSV file (creates or appends).
    
    Args:
        predictions: List of prediction dicts with keys:
            - away_team, home_team, predicted_winner, confidence, home_win_prob, model_agreement
        date_str: Optional date string (YYYY-MM-DD), defaults to Eastern time today
    """
    if date_str is None:
        date_str = get_eastern_date()
    
    # Prepare rows
    rows = []
    for pred in predictions:
        conf = pred['confidence']
        tier = get_confidence_tier(conf)
        
        rows.append({
            'date': date_str,
            'away_team': pred['away_team'],
            'home_team': pred['home_team'],
            'prediction': pred['predicted_winner'],
            'winner': '',  # To be filled later
            'confidence': round(pred['confidence'], 3),
            'home_win_prob': round(pred['home_win_prob'], 3),
            'model_agreement': round(pred.get('model_agreement', 0), 3),
            'tier': tier,
            'correct': ''  # To be filled later (1 or 0)
        })
    
    # Check if file exists
    file_exists = os.path.exists(PREDICTION_HISTORY_FILE)
    
    # Create DataFrame with column order
    columns = ['date', 'away_team', 'home_team', 'prediction', 'winner', 
               'confidence', 'home_win_prob', 'model_agreement', 'tier', 'correct']
    new_df = pd.DataFrame(rows, columns=columns)
    
    if file_exists:
        # Read existing and check for duplicates
        existing_df = pd.read_csv(PREDICTION_HISTORY_FILE)
        
        # Check if predictions for this date already exist (match by away_team + home_team)
        existing_df['match_key'] = existing_df['away_team'] + ' vs ' + existing_df['home_team']
        new_df['match_key'] = new_df['away_team'] + ' vs ' + new_df['home_team']
        existing_matches_today = existing_df[existing_df['date'] == date_str]['match_key'].tolist()
        
        # Filter out duplicates
        new_df = new_df[~new_df['match_key'].isin(existing_matches_today)]
        new_df = new_df.drop(columns=['match_key'])
        existing_df = existing_df.drop(columns=['match_key'])
        
        if len(new_df) == 0:
            print(f"\n📋 All predictions for {date_str} already in history. Skipping save.")
            return
        
        # Append new predictions
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(PREDICTION_HISTORY_FILE, index=False)
        print(f"\n📋 Appended {len(new_df)} new predictions to history ({PREDICTION_HISTORY_FILE})")
    else:
        # Create new file
        new_df.to_csv(PREDICTION_HISTORY_FILE, index=False)
        print(f"\n📋 Created prediction history with {len(new_df)} predictions ({PREDICTION_HISTORY_FILE})")
