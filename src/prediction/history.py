"""
History Module
==============
Functions for saving and managing prediction history.
"""

from typing import Dict, List

from .nba_data import get_eastern_date
from ..paths import PREDICTION_HISTORY_FILE
from ..history_store import REMOVABLE_METADATA, read_history, write_history


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

    Existing date/team picks are never replaced by later replays.
    The original ten-column format is retained without replay metadata.
    """
    if date_str is None:
        date_str = get_eastern_date()
    if not predictions:
        return
    
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
            'correct': '',  # To be filled later (1 or 0)
        })
    
    # Preserve existing cell text and column order, including settled results.
    columns = ['date', 'away_team', 'home_team', 'prediction', 'winner', 
               'confidence', 'home_win_prob', 'model_agreement', 'tier', 'correct']
    snapshot = read_history(PREDICTION_HISTORY_FILE)
    key = lambda row: (row['date'], row['away_team'], row['home_team'])
    existing = {key(row) for row in snapshot.rows}
    new_rows = []
    for row in rows:
        if key(row) not in existing:
            new_rows.append(row)
            existing.add(key(row))
    if not new_rows:
        print(f"\n📋 All predictions for {date_str} already in history. Skipping save.")
        return
    columns = [col for col in snapshot.columns if col not in REMOVABLE_METADATA] + [
        col for col in columns if col not in snapshot.columns]
    write_history(snapshot, columns, snapshot.rows + new_rows, drop_metadata=True)
    print(f"\n📋 Saved {len(new_rows)} new predictions to history ({PREDICTION_HISTORY_FILE})")
