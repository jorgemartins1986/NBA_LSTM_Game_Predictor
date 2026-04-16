"""
Update Prediction Results
=========================
Fetches game results from NBA API and updates the prediction history CSV.

Usage:
    python -m src.update_prediction_results              # Update yesterday's games
    python -m src.update_prediction_results 2026-01-17   # Update specific date
    python -m src.update_prediction_results --all        # Update all pending results
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nba_api.stats.static import teams
from .paths import PREDICTION_HISTORY_FILE
from .prediction.nba_data import _fetch_nba_stats, _result_set_to_df, get_current_season
import os
import sys
import time


def _season_from_date(date_str):
    """Infer NBA season string from a YYYY-MM-DD date."""
    target_date = pd.to_datetime(date_str)
    start_year = target_date.year if target_date.month >= 10 else target_date.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _get_scoreboard_status(date_str):
    """Return lightweight status info for a date from scoreboardv2."""
    try:
        params = {
            'DayOffset': '0',
            'GameDate': date_str,
            'LeagueID': '00',
        }
        result = _fetch_nba_stats('scoreboardv2', params, timeout=60)
        games_df = _result_set_to_df(result, index=0)

        if len(games_df) == 0:
            return {
                'n_games': 0,
                'all_final': False,
                'any_final': False,
                'status_counts': {},
            }

        status_counts = games_df['GAME_STATUS_TEXT'].value_counts(dropna=False).to_dict()
        status_text = games_df['GAME_STATUS_TEXT'].astype(str)
        is_final = status_text.str.contains('Final', case=False, na=False)

        return {
            'n_games': len(games_df),
            'all_final': bool(is_final.all()),
            'any_final': bool(is_final.any()),
            'status_counts': status_counts,
        }
    except Exception:
        return None


def _get_scoreboard_final_results(date_str):
    """Fetch winners from ScoreboardV2 final games using line scores.

    This is a fallback when leaguegamefinder has not yet ingested completed games.
    """
    try:
        params = {
            'DayOffset': '0',
            'GameDate': date_str,
            'LeagueID': '00',
        }
        result = _fetch_nba_stats('scoreboardv2', params, timeout=60)
        game_header_df = _result_set_to_df(result, index=0)
        line_score_df = _result_set_to_df(result, index=1)

        if len(game_header_df) == 0 or len(line_score_df) == 0:
            return {}

        team_name_by_id = {t['id']: t['full_name'] for t in teams.get_teams()}
        results = {}

        for _, game in game_header_df.iterrows():
            status_text = str(game.get('GAME_STATUS_TEXT', ''))
            if 'Final' not in status_text:
                continue

            game_id = game['GAME_ID']
            home_team_id = game['HOME_TEAM_ID']
            away_team_id = game['VISITOR_TEAM_ID']

            game_lines = line_score_df[line_score_df['GAME_ID'] == game_id]
            if len(game_lines) < 2:
                continue

            home_line = game_lines[game_lines['TEAM_ID'] == home_team_id]
            away_line = game_lines[game_lines['TEAM_ID'] == away_team_id]
            if len(home_line) == 0 or len(away_line) == 0:
                continue

            home_pts = pd.to_numeric(home_line.iloc[0].get('PTS'), errors='coerce')
            away_pts = pd.to_numeric(away_line.iloc[0].get('PTS'), errors='coerce')
            if pd.isna(home_pts) or pd.isna(away_pts):
                continue

            if home_pts == away_pts:
                continue

            winner_team_id = home_team_id if home_pts > away_pts else away_team_id
            home_team = team_name_by_id.get(home_team_id, str(home_team_id))
            away_team = team_name_by_id.get(away_team_id, str(away_team_id))
            winner = team_name_by_id.get(winner_team_id, str(winner_team_id))

            match_key = f"{away_team} vs {home_team}"
            results[match_key] = winner

        return results
    except Exception:
        return {}


def get_game_results(date_str):
    """Fetch NBA game results for a specific date
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        
    Returns:
        Dict mapping "Away Team vs Home Team" -> winner team name
    """
    try:
        season = _season_from_date(date_str)
        current_season = get_current_season()
        target_date = pd.to_datetime(date_str)
        date_us = target_date.strftime('%m/%d/%Y')

        if season != current_season:
            print(f"   Checking season {season} for {date_str} (current season is {current_season})")

        # Fetch only the target date (avoids leaguegamefinder season row caps).
        time.sleep(0.6)  # Rate limiting
        params = {
            'Conference': '',
            'DateFrom': date_us,
            'DateTo': date_us,
            'Division': '',
            'DraftNumber': '',
            'DraftRound': '',
            'DraftTeamID': '',
            'DraftYear': '',
            'GameID': '',
            'LeagueID': '00',
            'Location': '',
            'Outcome': '',
            'PORound': '',
            'PlayerID': '',
            'PlayerOrTeam': 'T',
            'RookieYear': '',
            'Season': season,
            'SeasonSegment': '',
            'SeasonType': 'Regular Season',
            'StarterBench': '',
            'TeamID': '',
            'VsConference': '',
            'VsDivision': '',
            'VsTeamID': '',
            'YearsExperience': '',
        }
        result = _fetch_nba_stats('leaguegamefinder', params, timeout=60)
        games_df = _result_set_to_df(result, index=0)
        games_df['GAME_DATE'] = pd.to_datetime(games_df['GAME_DATE'])
        
        # Filter to the specific date
        date_games = games_df[games_df['GAME_DATE'] == target_date].copy()
        
        if len(date_games) == 0:
            print(f"   No completed results found for {date_str} in season {season}.")

            # Look back two weeks to show the latest completed date near target.
            try:
                lookback_start = (target_date - timedelta(days=14)).strftime('%m/%d/%Y')
                lookback_params = {
                    'Conference': '',
                    'DateFrom': lookback_start,
                    'DateTo': date_us,
                    'Division': '',
                    'DraftNumber': '',
                    'DraftRound': '',
                    'DraftTeamID': '',
                    'DraftYear': '',
                    'GameID': '',
                    'LeagueID': '00',
                    'Location': '',
                    'Outcome': '',
                    'PORound': '',
                    'PlayerID': '',
                    'PlayerOrTeam': 'T',
                    'RookieYear': '',
                    'Season': season,
                    'SeasonSegment': '',
                    'SeasonType': 'Regular Season',
                    'StarterBench': '',
                    'TeamID': '',
                    'VsConference': '',
                    'VsDivision': '',
                    'VsTeamID': '',
                    'YearsExperience': '',
                }
                lookback_result = _fetch_nba_stats('leaguegamefinder', lookback_params, timeout=60)
                lookback_df = _result_set_to_df(lookback_result, index=0)
                if len(lookback_df) > 0:
                    lookback_df['GAME_DATE'] = pd.to_datetime(lookback_df['GAME_DATE'])
                    latest_date = lookback_df['GAME_DATE'].max()
                    if pd.notna(latest_date):
                        print(
                            f"   Latest available completed date in leaguegamefinder ({season}) "
                            f"near target: {latest_date.strftime('%Y-%m-%d')}"
                        )
            except Exception:
                pass

            scoreboard_status = _get_scoreboard_status(date_str)
            if scoreboard_status is not None:
                if scoreboard_status['n_games'] == 0:
                    print(f"   Scoreboard check: no NBA games scheduled on {date_str}.")
                elif scoreboard_status['all_final']:
                    print("   Scoreboard check: games are Final, but leaguegamefinder has not updated yet. Try again shortly.")

                    # Fallback: use scoreboard line scores directly for finalized games.
                    scoreboard_results = _get_scoreboard_final_results(date_str)
                    if scoreboard_results:
                        print(f"   Fallback: extracted {len(scoreboard_results)} final result(s) from scoreboard line scores.")
                        print("   Result source: scoreboardv2 line scores fallback")
                        return scoreboard_results
                else:
                    sample_status = ', '.join(list(scoreboard_status['status_counts'].keys())[:3])
                    print(f"   Scoreboard check: {scoreboard_status['n_games']} game(s) listed (statuses: {sample_status}).")
                    print("   Note: scoreboard status text can lag or reflect schedule formatting for historical dates.")

            return {}
        
        # Get team info
        all_teams = {t['id']: t['full_name'] for t in teams.get_teams()}
        
        # Parse matchup and determine winners
        results = {}
        
        # Group by game ID
        for game_id in date_games['GAME_ID'].unique():
            game_rows = date_games[date_games['GAME_ID'] == game_id]
            
            if len(game_rows) != 2:
                continue  # Need both teams
            
            # Find home and away teams
            home_rows = game_rows[game_rows['MATCHUP'].str.contains('vs.', na=False)]
            away_rows = game_rows[game_rows['MATCHUP'].str.contains('@', na=False)]

            if len(home_rows) == 0 or len(away_rows) == 0:
                continue

            home_row = home_rows.iloc[0]
            away_row = away_rows.iloc[0]
            
            home_team = all_teams.get(home_row['TEAM_ID'], home_row['TEAM_NAME'])
            away_team = all_teams.get(away_row['TEAM_ID'], away_row['TEAM_NAME'])
            
            # Determine winner
            if home_row['WL'] == 'W':
                winner = home_team
            else:
                winner = away_team
            
            # Create match key (same format as in prediction)
            match_key = f"{away_team} vs {home_team}"
            results[match_key] = winner
        
        # Optional safety net: if scoreboard has final games that didn't appear yet in
        # leaguegamefinder, merge them in to avoid missing late-ingestion rows.
        source_label = "leaguegamefinder"
        scoreboard_status = _get_scoreboard_status(date_str)
        if scoreboard_status is not None and scoreboard_status['any_final']:
            scoreboard_results = _get_scoreboard_final_results(date_str)
            if scoreboard_results:
                base_count = len(results)
                for k, v in scoreboard_results.items():
                    results.setdefault(k, v)
                if len(results) > base_count:
                    print(f"   Added {len(results) - base_count} result(s) from scoreboard fallback.")
                    source_label = "leaguegamefinder + scoreboardv2 supplement"

        print(f"   Result source: {source_label}")

        return results
        
    except Exception as e:
        print(f"   Error fetching results: {e}")
        return {}


def update_results(date_str=None, update_all=False):
    """Update prediction history with actual game results
    
    Args:
        date_str: Specific date to update (YYYY-MM-DD), or None for yesterday
        update_all: If True, update all rows with missing results
    """
    if not os.path.exists(PREDICTION_HISTORY_FILE):
        print("❌ No prediction history file found.")
        print(f"   Expected: {PREDICTION_HISTORY_FILE}")
        print("   Run predictions first with: python predict.py")
        return
    
    # Load history
    df = pd.read_csv(PREDICTION_HISTORY_FILE)
    print(f"📋 Loaded {len(df)} prediction records")
    
    # Determine which dates to update
    if update_all:
        # Find all dates with missing results
        pending = df[df['winner'].isna() | (df['winner'] == '')]
        dates_to_update = pending['date'].unique().tolist()
        print(f"   Found {len(dates_to_update)} dates with pending results")
    else:
        if date_str is None:
            # Default to yesterday
            date_str = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        dates_to_update = [date_str]
    
    if len(dates_to_update) == 0:
        print("✅ No pending results to update!")
        return
    
    # Update each date
    total_updated = 0
    for date in sorted(dates_to_update):
        print(f"\n📅 Updating results for {date}...")
        
        # Get results for this date
        results = get_game_results(date)
        
        if not results:
            print(f"   ⚠️  No results available yet for {date}")
            continue
        
        # Update matching rows
        date_mask = df['date'] == date
        updated_count = 0
        
        for idx, row in df[date_mask].iterrows():
            # Build match key from separate columns
            match_key = f"{row['away_team']} vs {row['home_team']}"
            if match_key in results:
                winner = results[match_key]
                prediction = row['prediction']
                correct = 1 if prediction == winner else 0
                
                df.at[idx, 'winner'] = winner
                df.at[idx, 'correct'] = correct
                updated_count += 1
                
                status = "✅" if correct == 1 else "❌"
                print(f"   {status} {row['away_team']} @ {row['home_team']}: predicted {prediction}, winner {winner}")
        
        print(f"   Updated {updated_count} games for {date}")
        total_updated += updated_count
        
        # Small delay to avoid API rate limits
        if len(dates_to_update) > 1:
            time.sleep(0.5)
    
    # Save updated history
    df.to_csv(PREDICTION_HISTORY_FILE, index=False)
    print(f"\n💾 Saved {total_updated} updates to {PREDICTION_HISTORY_FILE}")
    
    # Show summary stats
    show_stats(df)


def show_stats(df=None):
    """Display prediction accuracy statistics"""
    if df is None:
        if not os.path.exists(PREDICTION_HISTORY_FILE):
            print("❌ No prediction history file found.")
            return
        df = pd.read_csv(PREDICTION_HISTORY_FILE)
    
    # Filter to completed predictions (correct column is filled)
    completed = df[df['correct'].notna() & (df['correct'] != '')].copy()
    
    if len(completed) == 0:
        print("\n📊 No completed predictions yet.")
        return
    
    completed['correct'] = completed['correct'].astype(int)
    
    print("\n" + "="*60)
    print("PREDICTION HISTORY STATISTICS")
    print("="*60)
    
    # Overall accuracy
    total = len(completed)
    correct_count = completed['correct'].sum()
    accuracy = correct_count / total * 100
    print(f"\n📊 Overall: {correct_count}/{total} correct ({accuracy:.1f}%)")
    
    # By confidence tier
    print("\n📈 By Bet Quality Tier:")
    tier_order = ['EXCELLENT', 'STRONG', 'GOOD', 'MODERATE', 'RISKY', 'SKIP']
    for tier in tier_order:
        tier_games = completed[completed['tier'] == tier]
        if len(tier_games) > 0:
            tier_correct = tier_games['correct'].sum()
            tier_total = len(tier_games)
            tier_acc = tier_correct / tier_total * 100
            print(f"   {tier:12} {tier_correct:3}/{tier_total:3} ({tier_acc:5.1f}%)")
    
    # By confidence ranges (using the actual confidence values)
    if 'confidence' in completed.columns:
        print("\n🎯 By Confidence Level:")
        bins = [(0.4, 1.0, '40%+'), (0.3, 0.4, '30-40%'), (0.2, 0.3, '20-30%'), 
                (0.1, 0.2, '10-20%'), (0.0, 0.1, '<10%')]
        for low, high, label in bins:
            range_games = completed[(completed['confidence'] >= low) & (completed['confidence'] < high)]
            if len(range_games) > 0:
                range_correct = range_games['correct'].sum()
                range_total = len(range_games)
                range_acc = range_correct / range_total * 100
                print(f"   {label:12} {range_correct:3}/{range_total:3} ({range_acc:5.1f}%)")
    
    # By model agreement (if available)
    if 'model_agreement' in completed.columns:
        print("\n🤝 By Model Agreement:")
        high_agree = completed[completed['model_agreement'] >= 0.95]
        med_agree = completed[(completed['model_agreement'] >= 0.90) & (completed['model_agreement'] < 0.95)]
        low_agree = completed[completed['model_agreement'] < 0.90]
        
        for label, subset in [('95%+ agree', high_agree), ('90-95% agree', med_agree), ('<90% agree', low_agree)]:
            if len(subset) > 0:
                sub_correct = subset['correct'].sum()
                sub_total = len(subset)
                sub_acc = sub_correct / sub_total * 100
                print(f"   {label:12} {sub_correct:3}/{sub_total:3} ({sub_acc:5.1f}%)")
    
    # Recent performance (last 7 days)
    completed['date'] = pd.to_datetime(completed['date'])
    week_ago = datetime.now() - timedelta(days=7)
    recent = completed[completed['date'] >= week_ago]
    
    if len(recent) > 0:
        recent_correct = recent['correct'].sum()
        recent_total = len(recent)
        recent_acc = recent_correct / recent_total * 100
        print(f"\n📅 Last 7 days: {recent_correct}/{recent_total} ({recent_acc:.1f}%)")
    
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == '--all':
            update_results(update_all=True)
        elif arg == '--stats':
            show_stats()
        elif arg == '--help' or arg == '-h':
            print("Usage: python -m src.update_prediction_results [date|--all|--stats]")
            print("\nOptions:")
            print("  (no args)     - Update yesterday's results")
            print("  YYYY-MM-DD    - Update specific date")
            print("  --all         - Update all pending results")
            print("  --stats       - Show accuracy statistics only")
        else:
            # Assume it's a date
            update_results(date_str=arg)
    else:
        # Default: update yesterday
        update_results()
