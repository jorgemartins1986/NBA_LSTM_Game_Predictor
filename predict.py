#!/usr/bin/env python
"""
Predict Today's NBA Games
=========================
Entry point script for making predictions on today's games.

Usage:
    python predict.py                          # Ensemble predictions for today
    python predict.py --model xgboost          # Use only XGBoost
    python predict.py --model rf               # Use only Random Forest  
    python predict.py --model logistic         # Use only Logistic Regression
    python predict.py --model lstm             # Use only LSTM
    python predict.py --model ensemble         # Use full ensemble (default)
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nba_api.stats.static import teams

from src.prediction import (
    ModelLoader,
    PredictionPipeline,
    FeatureComputer,
    get_todays_games,
    get_eastern_date,
    get_live_standings,
    fetch_season_games,
    get_recent_team_stats,
    compute_head_to_head,
    get_default_standings,
    get_live_odds,
    match_game_to_odds,
    format_odds_display,
    save_predictions_to_history,
)


def predict_todays_games(single_model: str = None):
    """
    Main prediction function.
    
    Args:
        single_model: If specified, use only this model ('lstm', 'xgboost', 'random_forest', 'logistic')
                      If None, use full ensemble
    """
    print("="*70)
    if single_model:
        print(f"NBA GAME PREDICTIONS - {single_model.upper()} ONLY")
    else:
        print("NBA GAME PREDICTIONS - ENSEMBLE MODE")
        print("XGBoost + Random Forest + Logistic + LSTM")
    print("="*70)
    
    # Load ensemble using new ModelLoader
    loader = ModelLoader()
    ensemble = loader.load_ensemble(verbose=True)
    
    # Filter to single model if specified
    if single_model:
        indices = [i for i, mt in enumerate(ensemble.model_types) if mt == single_model]
        if not indices:
            print(f"❌ Model type '{single_model}' not found in ensemble!")
            print(f"   Available: {ensemble.model_types}")
            return
        idx = indices[0]
        # Create a filtered ensemble
        from src.prediction import LoadedEnsemble
        ensemble = LoadedEnsemble(
            models=[ensemble.models[idx]],
            scalers=[ensemble.scalers[idx]],
            feature_cols=ensemble.feature_cols,
            model_types=[ensemble.model_types[idx]],
            # Disable stacking for single model
            meta_clf=None,
            platt=None,
            ensemble_weights=None,
            ensemble_threshold=None,
        )
        print(f"✓ Using {single_model.upper()} model only")
    
    # Create prediction pipeline
    pipeline = PredictionPipeline(ensemble)
    
    # Fetch recent data
    print("\n📊 Fetching recent team statistics...")
    games_df = fetch_season_games(verbose=True)
    
    # Get today's games
    print("\n🏀 Fetching today's games...")
    todays_games, game_date, all_finished = get_todays_games()
    eastern_today = get_eastern_date()
    print(f"   Game date from API: {game_date}")
    print(f"   Current date (Eastern): {eastern_today}")
    
    if len(todays_games) == 0:
        print("\n❌ No games found for today.")
        print("   The NBA schedule might be empty, or it's the off-season.")
        return
    
    # Warn if showing old games
    if all_finished and game_date != eastern_today:
        print(f"\n⚠️  WARNING: All games shown are FINAL (from {game_date})")
        print(f"   Today is {eastern_today} - today's games may not have started yet.")
        print("   Predictions will still run but won't be saved to history (already recorded).")
    
    print(f"✓ Found {len(todays_games)} games for {game_date}")
    
    # Fetch live standings
    live_standings = get_live_standings()
    
    # Fetch live odds (optional)
    live_odds = get_live_odds()
    
    # Predict each game
    print("\n" + "="*70)
    print("ENSEMBLE PREDICTIONS")
    print("="*70)
    
    all_teams = teams.get_teams()
    predictions = []
    
    for i, game in enumerate(todays_games):
        print(f"\n🏀 Game {i+1}/{len(todays_games)}")
        print("-"*70)
        
        # Get team names
        home_team = [t for t in all_teams if t['id'] == game['home_team_id']][0]
        away_team = [t for t in all_teams if t['id'] == game['away_team_id']][0]
        
        print(f"   {away_team['full_name']} @ {home_team['full_name']}")
        print(f"   Status: {game['game_status']}")
        
        # Get recent stats
        home_features = get_recent_team_stats(game['home_team_id'], games_df)
        away_features = get_recent_team_stats(game['away_team_id'], games_df)
        
        if home_features is None or away_features is None:
            print("   ⚠️  Not enough data for prediction")
            continue
        
        # Compute head-to-head
        h2h_home = compute_head_to_head(games_df, game['home_team_id'], game['away_team_id'])
        h2h_away = compute_head_to_head(games_df, game['away_team_id'], game['home_team_id'])
        
        # Get standings
        home_standings = live_standings.get(game['home_team_id'], get_default_standings())
        away_standings = live_standings.get(game['away_team_id'], get_default_standings())
        
        # Get odds (if available)
        game_odds = match_game_to_odds(home_team['full_name'], away_team['full_name'], live_odds)
        
        # Build feature vector
        features = pipeline.feature_computer.build_feature_vector(
            home_features=home_features,
            away_features=away_features,
            feature_cols=ensemble.feature_cols,
            h2h_home=h2h_home,
            h2h_away=h2h_away,
            home_standings=home_standings,
            away_standings=away_standings,
            home_odds=game_odds,
            away_odds=game_odds,
        )
        
        # Get prediction
        try:
            result = pipeline.predict_from_features(features)
            
            # Display results
            winner_name = home_team['full_name'] if result.predicted_winner == 'HOME' else away_team['full_name']
            conf_pct = result.confidence * 100
            home_prob = result.home_win_probability * 100
            away_prob = (1 - result.home_win_probability) * 100
            
            # Determine confidence tier (5-star system)
            if result.confidence >= 0.50:
                tier = "⭐⭐⭐⭐⭐ EXCELLENT"
                tier_short = "⭐⭐⭐⭐⭐ EXCELLENT BET"
            elif result.confidence >= 0.40:
                tier = "⭐⭐⭐⭐ STRONG"
                tier_short = "⭐⭐⭐⭐ STRONG BET"
            elif result.confidence >= 0.30:
                tier = "⭐⭐⭐ GOOD"
                tier_short = "⭐⭐⭐ GOOD BET"
            elif result.confidence >= 0.20:
                tier = "⭐⭐ MODERATE"
                tier_short = "⭐⭐ MODERATE"
            elif result.confidence >= 0.10:
                tier = "⭐ RISKY"
                tier_short = "⭐ RISKY"
            else:
                tier = "⛔ SKIP"
                tier_short = "⛔ SKIP"
            
            # Main prediction
            print(f"\n   🏆 Predicted Winner: {winner_name}")
            print(f"   📊 Confidence: {conf_pct:.1f}% ({tier})")
            print(f"   🏠 Home Win Prob: {home_prob:.1f}%")
            print(f"   ✈️  Away Win Prob: {away_prob:.1f}%")
            
            # Show odds and value if available
            if game_odds:
                home_odds_val = game_odds.get('HOME_AVG_ODDS', 0)
                away_odds_val = game_odds.get('AWAY_AVG_ODDS', 0)
                if home_odds_val and away_odds_val and home_odds_val > 1 and away_odds_val > 1:
                    market_home = game_odds.get('HOME_IMPLIED_PROB', 0) * 100
                    market_away = game_odds.get('AWAY_IMPLIED_PROB', 0) * 100
                    print(f"   💰 Bookmaker Odds: Home {home_odds_val:.2f} ({market_home:.0f}%) | Away {away_odds_val:.2f} ({market_away:.0f}%)")
                    # Value calculation
                    value_diff = home_prob - market_home
                    if abs(value_diff) > 0.1:
                        print(f"   🔥 Value: Model {home_prob:.1f}% vs Market {market_home:.1f}% ({value_diff:+.1f}%)")
            
            # Rest days
            home_rest = home_features.get('DAYS_REST', 'N/A')
            away_rest = away_features.get('DAYS_REST', 'N/A')
            if home_rest != 'N/A':
                home_rest = f"{home_rest:.0f}"
            if away_rest != 'N/A':
                away_rest = f"{away_rest:.0f}"
            print(f"\n   😴 Rest: Home {home_rest}d | Away {away_rest}d")
            
            # H2H
            h2h_games = h2h_home.get('H2H_GAMES', 0)
            h2h_win_pct = h2h_home.get('H2H_WIN_RATE', 0.5) * 100
            print(f"   🆚 H2H: Home {h2h_win_pct:.0f}% ({h2h_games} games)")
            
            # Standings
            home_rank = home_standings.get('CONF_RANK', '?')
            home_wins = home_standings.get('WINS', 0)
            home_losses = home_standings.get('LOSSES', 0)
            away_rank = away_standings.get('CONF_RANK', '?')
            away_wins = away_standings.get('WINS', 0)
            away_losses = away_standings.get('LOSSES', 0)
            print(f"   📈 Standings: Home #{home_rank} ({home_wins}-{home_losses}) | Away #{away_rank} ({away_wins}-{away_losses})")
            
            # Individual model predictions with names
            model_names = ['XGBoost', 'RF', 'Logistic', 'LSTM']
            print(f"\n   🤖 Model Agreement: {result.model_agreement*100:.1f}%")
            for j, pred in enumerate(result.individual_predictions):
                name = model_names[j] if j < len(model_names) else f"Model{j+1}"
                print(f"      {name:10s} {pred*100:5.1f}%")
            
            predictions.append({
                'away_team': away_team['full_name'],
                'home_team': home_team['full_name'],
                'predicted_winner': winner_name,
                'confidence': result.confidence,
                'home_win_prob': result.home_win_probability,
                'model_agreement': result.model_agreement,
                'tier_short': tier_short,
            })
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            continue
    
    # Save predictions to history (if not all games are finished)
    if predictions and not (all_finished and game_date != eastern_today):
        save_predictions_to_history(predictions, game_date)
    
    # Summary - show GOOD and above bets
    print("\n" + "="*70)
    print("SUMMARY - ENSEMBLE PREDICTIONS")
    print("="*70)
    
    # Sort by confidence (highest first)
    sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    # Show only GOOD and above (confidence >= 30%)
    good_bets = [p for p in sorted_preds if p['confidence'] >= 0.30]
    
    if good_bets:
        for p in good_bets:
            tier_emoji = "⭐⭐⭐⭐⭐" if p['confidence'] >= 0.50 else "⭐⭐⭐⭐" if p['confidence'] >= 0.40 else "⭐⭐⭐"
            print(f"\n{tier_emoji} {p['away_team']} @ {p['home_team']}")
            print(f"   → {p['predicted_winner']} ({p['home_win_prob']*100:.1f}% prob, {p['confidence']*100:.0f}% confidence)")
            print(f"   ✅ Model agreement: {p['model_agreement']*100:.0f}%")
            print(f"   📋 {p['tier_short']}")
    else:
        print("\n   No high-confidence picks today (GOOD or better).")
    
    # Summary stats
    print(f"\n" + "-"*70)
    print(f"   Total games: {len(predictions)}/{len(todays_games)}")
    print(f"   Recommended bets: {len(good_bets)} (⭐⭐⭐ GOOD or better)")


def main():
    parser = argparse.ArgumentParser(
        description='NBA Game Predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py                    Show today's predictions using ensemble
  python predict.py --model xgboost    Use only XGBoost model
  python predict.py --model lstm       Use only LSTM model
        """
    )
    parser.add_argument(
        '--model', '-m',
        choices=['xgboost', 'rf', 'random_forest', 'logistic', 'lstm', 'ensemble'],
        default='ensemble',
        help='Model to use for predictions (default: ensemble)'
    )
    
    args = parser.parse_args()
    
    # Map rf alias to random_forest
    model = args.model
    if model == 'rf':
        model = 'random_forest'
    
    if model == 'ensemble':
        predict_todays_games()
    else:
        predict_todays_games(single_model=model)


if __name__ == "__main__":
    main()
