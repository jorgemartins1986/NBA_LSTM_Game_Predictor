# NBA Game Predictor - Ensemble Model

## 📋 Overview

An ensemble-based NBA game prediction system combining multiple models with stacking and Platt calibration. The system uses rolling team statistics, ELO ratings, and Dean Oliver's Four Factors to predict game outcomes.

**Current ensemble accuracy: ~64.5%** (realistic for sports prediction; market edge is typically 52-55%)

### Models in the Ensemble
| Model | Type | Accuracy | Purpose |
|-------|------|----------|---------||
| XGBoost | Gradient Boosting | 64.71% | Tree-based learner with tuned hyperparameters |
| Random Forest | Ensemble Trees | 63.68% | Robust ensemble with feature selection (top 40) |
| Logistic Regression | Linear | 63.37% | L2-regularized baseline with balanced class weights |
| LSTM | Neural Network | 65.33% | Attention-enhanced deep learning |

Predictions are combined via **ensemble voting** with optional **stacking meta-learner** and **Platt calibration**.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Predict Today's Games

```bash
python predict.py
```

**Example output:**
```
🏀 Game 1/3
   Chicago Bulls @ Boston Celtics
   🏆 Predicted Winner: Boston Celtics
   📊 Confidence: 42.0%
   🏠 Home Win Prob: 71.0%

   🤖 Model Agreement: 94%
      XGBoost:    72.1%
      RF:         70.3%
      Logistic:   69.8%
      LSTM:       71.5%

   📋 💰 STRONG BET
```

### 3. Train the Ensemble

```bash
python train.py
```

---

## 📁 Project Structure

```
NBA_LSTM_Game_Predictor/
│
├── train.py                      # Entry point: train ensemble
├── predict.py                    # Entry point: make predictions
│
├── src/                          # Core source code
│   ├── __init__.py
│   ├── paths.py                  # Path configuration
│   ├── nba_data_manager.py       # Data fetching & ELO ratings
│   ├── nba_predictor.py          # Feature engineering
│   ├── nba_ensemble_predictor.py # Ensemble training
│   └── predict_with_ensemble.py  # Prediction logic
│
├── models/                       # Trained models (git-ignored)
│   ├── nba_ensemble_xgboost_1.json
│   ├── nba_ensemble_rf_2.pkl
│   ├── nba_ensemble_logistic_3.pkl
│   ├── nba_ensemble_model_4.keras
│   └── ensemble_*.pkl            # Scalers, features, meta-model
│
├── cache/                        # Data caches (git-ignored)
│   ├── nba_games_cache.csv       # Raw historical game data (8 MB)
│   ├── nba_games_enriched.csv    # Games with rolling features (70 MB)
│   ├── nba_matchups.csv          # Full matchup dataset (37 MB, 117 features)
│   ├── nba_elo_cache.pkl         # ELO ratings
│   └── *_cache.pkl               # Feature caches
│
├── config/                       # Configuration files
│   └── best_xgb_params.json      # Tuned XGBoost hyperparameters
│
├── reports/                      # Training reports (git-ignored)
│   └── model_reports_*.txt/.json
│
├── docs/                         # Documentation
│   ├── PREDICTIONGUIDE.md
│   ├── NBAPREDICTIONCEILING.md
│   └── ELOplusCACHINGsummary.md
│
├── requirements.txt
└── README.md
```

---

## 🔄 Training the Ensemble

### Full Training

```bash
python train.py
```

This will:
1. Load/update game data from cache (or download if first run)
2. Prepare matchup features with chronological split (80% train / 20% test)
3. Compute all features: rolling stats, ELO, H2H, standings, fatigue
4. **Export enriched datasets to CSV** (`cache/nba_matchups.csv`)
5. Train all 4 models: XGBoost, Random Forest, Logistic Regression, LSTM
6. Apply sample weighting (recent games weighted more heavily)
7. Build stacking meta-model and Platt calibrator
8. Save all artifacts to `models/`

**Training time:** ~10-20 minutes (depending on hardware)

---

## 🎯 Bet Quality Tiers

The prediction output includes bet-quality tiers based on model confidence:

| Tier | Confidence | Probability | Recommendation |
|------|------------|-------------|----------------|
| 🔥 EXCELLENT | 50%+ | 75%+ | Strong edge |
| 💰 STRONG | 40-50% | 70-75% | Good value |
| ⚡ GOOD | 30-40% | 65-70% | Moderate edge |
| 📊 MODERATE | 20-30% | 60-65% | Small edge |
| ❓ RISKY | 10-20% | 55-60% | Marginal |
| ⛔ SKIP | <10% | <55% | No edge |

**Model Agreement** indicates how much the 4 models agree (higher = more confidence in the prediction).

---

## 📊 Features Used

### Rolling Statistics (20-game window)
- Points, rebounds, assists, steals, blocks, turnovers
- Field goal %, 3-point %, free throw %
- Plus/minus differential

### Dean Oliver's Four Factors
- **eFG%** (Effective Field Goal %)
- **TOV%** (Turnover Rate)
- **ORB%** (Offensive Rebound Rate)
- **FT Rate** (Free Throw Rate)

### Momentum Features
- Win streak
- Recent form (last 5 games)
- Win consistency (std dev)
- 3-game win pattern

### Fatigue Features (NEW)
- **Days of rest** since last game
- **Back-to-back** indicator (played yesterday)
- **3-in-4 nights** indicator (3 games in 4 days)

### Head-to-Head Features
- H2H win rate (last 10 meetings)
- H2H games played
- H2H point differential

### Standings Features (NEW)
- **Conference rank** (1-15)
- **League rank** (1-30)
- **Wins/Losses** at time of game
- **Win percentage**
- **Games back** from conference leader
- **Current streak** (positive = wins, negative = losses)
- **Rank differential** between teams
- **Win % differential** between teams

### ELO Ratings
- Pre-game ELO for home/away
- ELO differential
- Expected win probability from ELO

---

## 🔧 Updating Game Data

### Update Current Season Only

```bash
python scripts/fetch_and_update_games.py --update-current --cache-file nba_games_cache.csv
```

### Download Specific Seasons

```bash
python scripts/fetch_and_update_games.py --download-seasons 2023-24 2024-25 --cache-file nba_games_cache.csv
```

---

## � Prediction History & Tracking

The system automatically tracks all predictions and results for performance analysis.

### Daily Workflow

```bash
# 1. Run predictions (auto-saves to history)
python predict.py

# 2. Next day: update with actual results
python -m src.update_prediction_results

# 3. View accuracy statistics anytime
python -m src.update_prediction_results --stats
```

### History File Format

Predictions are saved to `reports/prediction_history.csv`:

| Column | Description |
|--------|-------------|
| `date` | Prediction date (YYYY-MM-DD) |
| `away_team` | Away team full name |
| `home_team` | Home team full name |
| `prediction` | Predicted winner |
| `winner` | Actual winner (filled after game) |
| `confidence` | Model confidence (0-1, distance from 50%) |
| `home_win_prob` | Home win probability (0-1) |
| `model_agreement` | How much models agreed (0-1) |
| `tier` | Bet quality tier (EXCELLENT/STRONG/GOOD/MODERATE/RISKY/SKIP) |
| `correct` | 1 if correct, 0 if wrong |

### Update Results Commands

```bash
# Update yesterday's results (default)
python -m src.update_prediction_results

# Update specific date
python -m src.update_prediction_results 2026-01-17

# Update all pending results
python -m src.update_prediction_results --all

# View statistics only
python -m src.update_prediction_results --stats
```

### Statistics Output

```
============================================================
PREDICTION HISTORY STATISTICS
============================================================

📊 Overall: 6/9 correct (66.7%)

📈 By Bet Quality Tier:
   EXCELLENT      1/  1 (100.0%)
   GOOD           1/  2 ( 50.0%)
   MODERATE       1/  2 ( 50.0%)
   RISKY          2/  2 (100.0%)
   SKIP           1/  2 ( 50.0%)

🎯 By Confidence Level:
   40%+           1/  1 (100.0%)
   30-40%         1/  2 ( 50.0%)
   ...

📅 Last 7 days: 6/9 (66.7%)
============================================================
```

### Recommended Retraining Schedule

- **Daily**: Run predictions (data is fresh via API)
- **Every 2 weeks**: Full retrain with `python train.py`
- **When accuracy drops**: Check stats and retrain if needed

---

## �🛠️ Troubleshooting

### "No games found for today"
- Check if it's the off-season or no games scheduled
- NBA API live scoreboard only shows games on game days

### "Not enough data for prediction"
- Team needs at least 20 games played in current season
- Early season predictions may be unreliable

### Module import errors
```bash
# Activate virtual environment first
.\venv\Scripts\Activate.ps1   # Windows
source venv/bin/activate       # Mac/Linux

# Then run
python predict_with_ensemble.py
```

### Low accuracy after retraining
- Ensure chronological split is used (shuffle=False)
- Check that all 4 models trained successfully
- Verify stacking artifacts were saved

---

## � Enriched Datasets

Training automatically exports portable CSV datasets you can use in other projects:

| File | Size | Description |
|------|------|-------------|
| `nba_games_cache.csv` | 8 MB | Raw game data from NBA API |
| `nba_games_enriched.csv` | 70 MB | Games with all rolling features |
| `nba_matchups.csv` | 37 MB | **28,000+ matchups with 117 features** |

The `nba_matchups.csv` contains the complete feature-engineered dataset ready for ML:
- Rolling statistics (20-game window)
- ELO ratings and differentials
- Head-to-head historical records
- Standings at time of each game
- Fatigue indicators (rest days, B2B, 3-in-4)
- Target variable: `HOME_WIN` (1/0)

**Use in other projects:**
```python
import pandas as pd
df = pd.read_csv('cache/nba_matchups.csv')
X = df.drop(['GAME_ID', 'GAME_DATE', 'HOME_WIN', 'HOME_TEAM_ID', 'AWAY_TEAM_ID'], axis=1)
y = df['HOME_WIN']
```

---

## �📈 Improving Accuracy

### Feature Ideas (not yet implemented)
- Travel distance / timezone changes
- Injury reports integration
- Referee tendencies

### Model Ideas
- Transformer attention on recent games
- Gradient boosting hyperparameter tuning
- Alternative neural network architectures

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Sports betting involves significant risk. Historical accuracy does not guarantee future results. Always gamble responsibly.

**Realistic expectations:** Professional sports bettors typically achieve 52-55% accuracy. This model aims for 60-65%, providing a potential edge but not guaranteed profits.

---

## 📚 References

- [NBA API](https://github.com/swar/nba_api)
- Dean Oliver's Four Factors of Basketball Success
- ELO Rating System (adapted from chess)

---

Good luck with your predictions! 🏀
