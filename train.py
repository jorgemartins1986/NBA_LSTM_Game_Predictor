#!/usr/bin/env python
"""
Train the NBA Ensemble Model
============================
Entry point script for training all ensemble models.

Usage:
    python train.py              # Train full ensemble
    python train.py --help       # Show help
"""

import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def train_ensemble():
    """Main ensemble training using refactored modules"""
    from src.nba_data_manager import NBADataFetcher
    from src.nba_predictor import NBAPredictor
    from src.training import EnsembleTrainer, EnsembleConfig
    
    print("="*70)
    print("NBA ENSEMBLE PREDICTOR")
    print("Training multiple models for improved accuracy")
    print("="*70)
    
    # Fetch data
    print("\n📥 Fetching NBA Data...")
    fetcher = NBADataFetcher()
    games_df = fetcher.fetch_games()
    
    # Prepare features
    print("\n🔧 Engineering Features...")
    predictor = NBAPredictor(window_size=20)
    matchup_df = predictor.prepare_matchup_data(games_df)
    
    print(f"✓ Prepared {len(matchup_df)} matchups")
    
    # Configure and train ensemble
    config = EnsembleConfig(
        architectures=['xgboost', 'random_forest', 'logistic', 'lstm'],
        test_size=0.2,
        use_sample_weights=True
    )
    trainer = EnsembleTrainer(config=config)
    
    # Train
    result = trainer.train(matchup_df)
    
    # Evaluate
    trainer.evaluate(result, matchup_df)
    
    # Save
    trainer.save(result)
    
    print("\n" + "="*70)
    print("✅ ENSEMBLE TRAINING COMPLETE!")
    print(f"🎯 Average Accuracy: {result.average_accuracy*100:.2f}%")
    print("="*70)
    print("\nEnsemble composition:")
    print("  1. XGBoost (optimized gradient boosted trees)")
    print("  2. Random Forest (robust ensemble with OOB validation)")
    print("  3. Logistic Regression (linear baseline, well-calibrated)")
    print("  4. LSTM (attention-enhanced neural network)")
    print("\nNew features included:")
    print("  - Rest days / Back-to-back detection")
    print("  - 3-in-4 nights fatigue indicator")
    print("  - Head-to-head historical record")
    print("  - Sample weighting (recent games weighted higher)")
    print("\nTo use ensemble for predictions, run:")
    print("  python main.py predict")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Train the NBA Ensemble Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                Train all ensemble models (XGBoost, RF, Logistic, LSTM)
  
Models trained:
  - XGBoost: Gradient boosted trees with optimized hyperparameters
  - Random Forest: Bagged decision trees with OOB validation
  - Logistic Regression: Linear baseline, well-calibrated probabilities
  - LSTM: Attention-enhanced neural network

Training will:
  1. Fetch latest NBA data from API
  2. Engineer 112+ features including fatigue and H2H
  3. Train all 4 models with chronological split
  4. Build stacking meta-model
  5. Save models to models/ directory
        """
    )
    
    args = parser.parse_args()
    
    # If we get here (no --help), run training
    train_ensemble()


if __name__ == "__main__":
    main()
