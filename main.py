"""
NBA Game Predictor - Main Entry Point
======================================

Usage:
    python main.py                    # Show help
    python main.py predict            # Predict today's games
    python main.py train              # Train ensemble model
    python main.py --stats            # Show detailed prediction statistics
    python main.py --updateresults    # Update results for pending predictions
    python main.py --updateresults 2026-01-15  # Update specific date
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.paths import PREDICTION_HISTORY_FILE


def get_profitable_filters(min_games: int = 5, min_accuracy: float = 70.9):
    """Compute currently profitable filters from prediction history.

    Returns a list of dicts, each with:
        name       : human-readable filter label
        n_games    : sample size
        accuracy   : hit-rate %
        roi        : estimated ROI % at 1.41 decimal odds
        tier       : tier requirement (or 'FAV' / 'ALL')
        min_agree  : minimum model_agreement threshold (0 = no requirement)
        is_fav     : whether only 75%+ favorites qualify
    Sorted by accuracy descending.
    """
    if not os.path.exists(PREDICTION_HISTORY_FILE):
        return []

    df = pd.read_csv(PREDICTION_HISTORY_FILE)
    completed = df[df['correct'].notna() & (df['correct'] != '')].copy()
    if len(completed) == 0:
        return []

    completed['correct'] = completed['correct'].astype(int)
    if 'home_win_prob' in completed.columns:
        completed['home_win_prob'] = pd.to_numeric(completed['home_win_prob'], errors='coerce')
    if 'model_agreement' in completed.columns:
        completed['model_agreement'] = pd.to_numeric(completed['model_agreement'], errors='coerce')

    filters = []

    # --- Tier + agreement combinations ---
    for tier in ['EXCELLENT', 'STRONG', 'GOOD', 'MODERATE', 'RISKY', 'SKIP']:
        for agree_low in [0.90, 0.92, 0.95, 0.98]:
            subset = completed[(completed['tier'] == tier) & (completed['model_agreement'] >= agree_low)]
            if len(subset) >= min_games:
                acc = subset['correct'].sum() / len(subset) * 100
                if acc > min_accuracy:
                    filters.append({
                        'name': f"{tier} + {agree_low*100:.0f}%+ agree",
                        'n_games': len(subset),
                        'accuracy': acc,
                        'tier': tier,
                        'min_agree': agree_low,
                        'is_fav': False,
                    })

    # --- 75%+ favorites ---
    if 'home_win_prob' in completed.columns:
        strong_fav = completed[
            (completed['home_win_prob'] >= 0.75) | (completed['home_win_prob'] <= 0.25)
        ]
        if len(strong_fav) >= min_games:
            acc = strong_fav['correct'].sum() / len(strong_fav) * 100
            if acc > min_accuracy:
                filters.append({
                    'name': "75%+ favorites",
                    'n_games': len(strong_fav),
                    'accuracy': acc,
                    'tier': 'FAV',
                    'min_agree': 0,
                    'is_fav': True,
                })

        # 75%+ fav + agreement
        for agree_low in [0.92, 0.95, 0.98]:
            subset = strong_fav[strong_fav['model_agreement'] >= agree_low]
            if len(subset) >= min_games:
                acc = subset['correct'].sum() / len(subset) * 100
                if acc > min_accuracy:
                    filters.append({
                        'name': f"75%+ fav + {agree_low*100:.0f}%+ agree",
                        'n_games': len(subset),
                        'accuracy': acc,
                        'tier': 'FAV',
                        'min_agree': agree_low,
                        'is_fav': True,
                    })

    # --- Agreement alone (all tiers) ---
    for agree_low in [0.90, 0.92, 0.95, 0.98]:
        subset = completed[completed['model_agreement'] >= agree_low]
        if len(subset) >= min_games:
            acc = subset['correct'].sum() / len(subset) * 100
            if acc > min_accuracy:
                filters.append({
                    'name': f"All tiers + {agree_low*100:.0f}%+ agree",
                    'n_games': len(subset),
                    'accuracy': acc,
                    'tier': 'ALL',
                    'min_agree': agree_low,
                    'is_fav': False,
                })

    # Compute ROI for each
    for f in filters:
        wins = int(round(f['accuracy'] * f['n_games'] / 100))
        losses = f['n_games'] - wins
        f['roi'] = (wins * 0.41 - losses) / f['n_games'] * 100

    return sorted(filters, key=lambda x: -x['accuracy'])


def game_matches_filter(pred: dict, filt: dict) -> bool:
    """Check if a single game prediction matches a profitable filter.

    Args:
        pred: dict with keys confidence, home_win_prob, model_agreement, tier_short
        filt: filter dict from get_profitable_filters()
    """
    # Determine the game's tier label
    conf = pred.get('confidence', 0)
    if conf >= 0.50:
        game_tier = 'EXCELLENT'
    elif conf >= 0.40:
        game_tier = 'STRONG'
    elif conf >= 0.30:
        game_tier = 'GOOD'
    elif conf >= 0.20:
        game_tier = 'MODERATE'
    elif conf >= 0.10:
        game_tier = 'RISKY'
    else:
        game_tier = 'SKIP'

    agree = pred.get('model_agreement', 0)
    home_prob = pred.get('home_win_prob', 0.5)

    # Check tier requirement
    tier_req = filt['tier']
    if tier_req == 'ALL':
        pass  # Any tier qualifies
    elif tier_req == 'FAV':
        # Must be a 75%+ favorite
        if not (home_prob >= 0.75 or home_prob <= 0.25):
            return False
    else:
        # Exact tier match
        if game_tier != tier_req:
            return False

    # Check favourite requirement (for combined filters)
    if filt['is_fav'] and not (home_prob >= 0.75 or home_prob <= 0.25):
        return False

    # Check agreement threshold
    if filt['min_agree'] > 0 and agree < filt['min_agree']:
        return False

    return True


def show_detailed_stats():
    """Display comprehensive prediction accuracy statistics"""
    if not os.path.exists(PREDICTION_HISTORY_FILE):
        print("❌ No prediction history file found.")
        print(f"   Expected: {PREDICTION_HISTORY_FILE}")
        print("   Run predictions first with: python main.py predict")
        return
    
    df = pd.read_csv(PREDICTION_HISTORY_FILE)
    
    # Filter to completed predictions
    completed = df[df['correct'].notna() & (df['correct'] != '')].copy()
    
    if len(completed) == 0:
        print("\n📊 No completed predictions yet.")
        return
    
    completed['correct'] = completed['correct'].astype(int)
    completed['date'] = pd.to_datetime(completed['date'])
    
    print("\n" + "="*70)
    print("NBA PREDICTION STATISTICS - DETAILED BREAKDOWN")
    print("="*70)
    
    # ═══════════════════════════════════════════════════════════════════════
    # OVERALL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    total = len(completed)
    correct_count = int(completed['correct'].sum())
    accuracy = correct_count / total * 100
    
    pending = df[df['correct'].isna() | (df['correct'] == '')]
    
    print(f"\n📊 OVERALL SUMMARY")
    print(f"   Total predictions: {total} completed, {len(pending)} pending")
    print(f"   Accuracy: {correct_count}/{total} ({accuracy:.1f}%)")
    print(f"   Date range: {completed['date'].min().strftime('%Y-%m-%d')} to {completed['date'].max().strftime('%Y-%m-%d')}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # BY BET QUALITY TIER (based on model certainty = |prob - 50%|)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n📈 BY BET QUALITY TIER (based on model certainty)")
    print(f"   EXCELLENT=75%+ prob | STRONG=70%+ | GOOD=65%+ | MODERATE=60%+ | RISKY=55%+ | SKIP<55%")
    print(f"   {'Tier':<12} {'Record':>10} {'Accuracy':>10} {'ROI*':>10}")
    print(f"   {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    
    tier_order = ['EXCELLENT', 'STRONG', 'GOOD', 'MODERATE', 'RISKY', 'SKIP']
    for tier in tier_order:
        tier_games = completed[completed['tier'] == tier]
        if len(tier_games) > 0:
            tier_correct = int(tier_games['correct'].sum())
            tier_total = len(tier_games)
            tier_acc = tier_correct / tier_total * 100
            # ROI at 1.41 decimal odds (average observed odds)
            # Bet €1 to win €0.41 profit. ROI = (wins * 0.41 - losses) / total * 100
            losses = tier_total - tier_correct
            roi = (tier_correct * 0.41 - losses) / tier_total * 100
            roi_str = f"{roi:+.1f}%" if roi != 0 else "0.0%"
            print(f"   {tier:<12} {tier_correct:>4}/{tier_total:<4} {tier_acc:>9.1f}% {roi_str:>10}")
    
    print(f"   * ROI assumes 1.41 decimal odds (breakeven = 70.9% accuracy)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # BY CONFIDENCE LEVEL (same as tier but with finer granularity)
    # ═══════════════════════════════════════════════════════════════════════
    if 'confidence' in completed.columns:
        print(f"\n🎯 BY CONFIDENCE LEVEL (finer buckets of model certainty)")
        print(f"   Conf = |prob - 50%| × 2, so 45%+ conf = prob >95% or <5%")
        print(f"   {'Confidence':<12} {'Record':>10} {'Accuracy':>10} {'Avg Prob':>10}")
        print(f"   {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        
        bins = [
            (0.45, 1.0, '45%+'),
            (0.40, 0.45, '40-45%'),
            (0.35, 0.40, '35-40%'),
            (0.30, 0.35, '30-35%'),
            (0.25, 0.30, '25-30%'),
            (0.20, 0.25, '20-25%'),
            (0.15, 0.20, '15-20%'),
            (0.10, 0.15, '10-15%'),
            (0.05, 0.10, '5-10%'),
            (0.0, 0.05, '<5%')
        ]
        for low, high, label in bins:
            range_games = completed[(completed['confidence'] >= low) & (completed['confidence'] < high)]
            if len(range_games) > 0:
                range_correct = int(range_games['correct'].sum())
                range_total = len(range_games)
                range_acc = range_correct / range_total * 100
                avg_prob = range_games['home_win_prob'].mean() * 100
                print(f"   {label:<12} {range_correct:>4}/{range_total:<4} {range_acc:>9.1f}% {avg_prob:>9.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════
    # BY MODEL AGREEMENT (more granular)
    # ═══════════════════════════════════════════════════════════════════════
    if 'model_agreement' in completed.columns:
        print(f"\n🤝 BY MODEL AGREEMENT")
        print(f"   {'Agreement':<12} {'Record':>10} {'Accuracy':>10}")
        print(f"   {'-'*12} {'-'*10} {'-'*10}")
        
        agree_bins = [
            (0.98, 1.01, '98%+'),
            (0.95, 0.98, '95-98%'),
            (0.92, 0.95, '92-95%'),
            (0.90, 0.92, '90-92%'),
            (0.85, 0.90, '85-90%'),
            (0.0, 0.85, '<85%')
        ]
        for low, high, label in agree_bins:
            subset = completed[(completed['model_agreement'] >= low) & (completed['model_agreement'] < high)]
            if len(subset) > 0:
                sub_correct = int(subset['correct'].sum())
                sub_total = len(subset)
                sub_acc = sub_correct / sub_total * 100
                print(f"   {label:<12} {sub_correct:>4}/{sub_total:<4} {sub_acc:>9.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════
    # BY HOME WIN PROBABILITY
    # ═══════════════════════════════════════════════════════════════════════
    if 'home_win_prob' in completed.columns:
        print(f"\n🏠 BY HOME WIN PROBABILITY")
        print(f"   {'Prob Range':<12} {'Record':>10} {'Home Win%':>10} {'Correct':>10}")
        print(f"   {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        
        # Create a column for actual home win
        def determine_home_win(row):
            if row['winner'] == row['home_team']:
                return 1
            return 0
        
        completed['home_won'] = completed.apply(determine_home_win, axis=1)
        
        prob_bins = [
            (0.75, 1.0, '75%+ home'),
            (0.65, 0.75, '65-75% home'),
            (0.55, 0.65, '55-65% home'),
            (0.45, 0.55, '45-55% toss'),
            (0.35, 0.45, '55-65% away'),
            (0.25, 0.35, '65-75% away'),
            (0.0, 0.25, '75%+ away')
        ]
        for low, high, label in prob_bins:
            subset = completed[(completed['home_win_prob'] >= low) & (completed['home_win_prob'] < high)]
            if len(subset) > 0:
                sub_correct = int(subset['correct'].sum())
                sub_total = len(subset)
                sub_acc = sub_correct / sub_total * 100
                home_wins = int(subset['home_won'].sum())
                home_rate = home_wins / sub_total * 100
                print(f"   {label:<12} {sub_correct:>4}/{sub_total:<4} {home_rate:>9.1f}% {sub_acc:>9.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════
    # TIME-BASED ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n📅 TIME-BASED PERFORMANCE")
    
    now = datetime.now()
    time_periods = [
        (3, 'Last 3 days'),
        (7, 'Last 7 days'),
        (14, 'Last 14 days'),
        (30, 'Last 30 days'),
    ]
    
    print(f"   {'Period':<14} {'Record':>10} {'Accuracy':>10}")
    print(f"   {'-'*14} {'-'*10} {'-'*10}")
    
    for days, label in time_periods:
        cutoff = now - timedelta(days=days)
        recent = completed[completed['date'] >= cutoff]
        if len(recent) > 0:
            recent_correct = int(recent['correct'].sum())
            recent_total = len(recent)
            recent_acc = recent_correct / recent_total * 100
            print(f"   {label:<14} {recent_correct:>4}/{recent_total:<4} {recent_acc:>9.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════
    # DAY OF WEEK ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n📆 BY DAY OF WEEK")
    print(f"   {'Day':<12} {'Record':>10} {'Accuracy':>10}")
    print(f"   {'-'*12} {'-'*10} {'-'*10}")
    
    completed['day_of_week'] = completed['date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for day in day_order:
        day_games = completed[completed['day_of_week'] == day]
        if len(day_games) > 0:
            day_correct = int(day_games['correct'].sum())
            day_total = len(day_games)
            day_acc = day_correct / day_total * 100
            print(f"   {day:<12} {day_correct:>4}/{day_total:<4} {day_acc:>9.1f}%")
    
    # ═══════════════════════════════════════════════════════════════════════
    # BEST/WORST STREAKS
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n🔥 STREAKS")
    
    # Calculate current streak
    sorted_completed = completed.sort_values('date', ascending=False)
    current_streak = 0
    streak_type = None
    
    for _, row in sorted_completed.iterrows():
        if streak_type is None:
            streak_type = row['correct']
            current_streak = 1
        elif row['correct'] == streak_type:
            current_streak += 1
        else:
            break
    
    streak_label = "W" if streak_type == 1 else "L"
    print(f"   Current streak: {streak_label}{current_streak}")
    
    # Calculate best/worst streaks
    best_streak = 0
    worst_streak = 0
    current_w = 0
    current_l = 0
    
    sorted_asc = completed.sort_values('date', ascending=True)
    for _, row in sorted_asc.iterrows():
        if row['correct'] == 1:
            current_w += 1
            current_l = 0
            best_streak = max(best_streak, current_w)
        else:
            current_l += 1
            current_w = 0
            worst_streak = max(worst_streak, current_l)
    
    print(f"   Best win streak: W{best_streak} (most consecutive correct predictions)")
    print(f"   Worst loss streak: L{worst_streak} (most consecutive wrong predictions)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PROFITABLE FILTERS (combining model certainty + model consensus)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n💰 PROFITABLE FILTERS (>70.9% needed at 1.41 odds to profit)")
    print(f"   Filters are cumulative (e.g., 92%+ includes all games ≥92% agreement)")
    
    profitable_combos = get_profitable_filters()
    
    if profitable_combos:
        print(f"   {'Filter':<28} {'Games':>8} {'Accuracy':>10} {'ROI':>10}")
        print(f"   {'-'*28} {'-'*8} {'-'*10} {'-'*10}")
        for f in profitable_combos[:10]:
            print(f"   {f['name']:<28} {f['n_games']:>8} {f['accuracy']:>9.1f}% {f['roi']:>+9.1f}%")
        
        best = profitable_combos[0]
        print(f"\n   ✅ Best filter: {best['name']} ({best['n_games']} games, {best['accuracy']:.1f}% accuracy)")
    else:
        print("   No profitable filters found with sufficient sample size.")
    
    print("\n" + "="*70)


def update_results(date_str=None, update_all=False):
    """Update prediction results - wrapper around update_prediction_results"""
    from src.update_prediction_results import update_results as do_update
    do_update(date_str=date_str, update_all=update_all)


def main():
    parser = argparse.ArgumentParser(
        description='NBA Game Predictor - Ensemble Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py predict              Predict today's games
  python main.py train                Train the ensemble model
  python main.py --stats              Show detailed prediction statistics
  python main.py --updateresults      Update yesterday's results
  python main.py --updateresults --all    Update all pending results
  python main.py --updateresults 2026-01-15   Update specific date
        """
    )
    
    parser.add_argument('command', nargs='?', choices=['predict', 'train'],
                        help='Command to run: predict or train')
    parser.add_argument('--stats', action='store_true',
                        help='Show detailed prediction statistics')
    parser.add_argument('--updateresults', nargs='?', const='yesterday', metavar='DATE',
                        help='Update prediction results (default: yesterday, or specify YYYY-MM-DD)')
    parser.add_argument('--all', action='store_true',
                        help='When used with --updateresults, update all pending results')
    
    args = parser.parse_args()
    
    # Handle commands
    if args.stats:
        show_detailed_stats()
        return
    
    if args.updateresults is not None:
        if args.all:
            update_results(update_all=True)
        elif args.updateresults == 'yesterday':
            update_results()
        else:
            update_results(date_str=args.updateresults)
        return
    
    if args.command == 'predict':
        from predict import predict_todays_games
        predict_todays_games()
        return
    
    if args.command == 'train':
        from train import train_ensemble
        train_ensemble()
        return
    
    # No command given - show help
    parser.print_help()


if __name__ == "__main__":
    main()
