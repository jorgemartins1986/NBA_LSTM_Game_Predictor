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
python main.py predict
```

### 3. View Detailed Statistics

```bash
python main.py --stats
```

### 4. Train the Ensemble

```bash
python main.py train
```

### 5. Update Results After Games

```bash
python main.py --updateresults          # Update yesterday's games
python main.py --updateresults --all    # Update all pending
python main.py --updateresults 2026-01-15  # Specific date
```

### 6. Run Web GUI (FastAPI)

```bash
uvicorn webapp.main:app --host 0.0.0.0 --port 7860
```

Then open: `http://localhost:7860`

### 7. Run with Docker

```bash
docker build -t nba-predictor-web .
docker run --rm -p 7860:7860 nba-predictor-web
```

Or with Compose:

```bash
docker compose up --build
```

### 8. Deploy to Hugging Face Spaces (Docker)

1. Create a new Space with SDK set to `Docker`.
2. Push this repository to the Space repo.
3. The included `Dockerfile` serves FastAPI on port `7860` (required by Spaces).
4. Keep model files and required artifacts in the repo (or pull them from remote storage at startup).

---

## 📁 Project Structure

```
NBA_LSTM_Game_Predictor/
│
├── train.py                      # Entry point: train ensemble
├── predict.py                    # Entry point: make predictions
├── main.py                       # Unified CLI (predict, train, stats, update)
│
├── src/                          # Core source code
│   ├── __init__.py
│   ├── paths.py                  # Path configuration
│   ├── utils.py                  # Shared utilities (GPU config)
│   ├── nba_data_manager.py       # Data fetching & ELO ratings
│   ├── nba_predictor.py          # Feature engineering
│   ├── update_prediction_results.py # Result tracking
│   ├── odds_api.py               # Live betting odds integration
│   │
│   ├── training/                 # Modular training module
│   │   ├── __init__.py           # Module exports
│   │   ├── data_prep.py          # DataPreparation, TrainTestData
│   │   ├── trainers.py           # XGBoostTrainer, RFTrainer, etc.
│   │   ├── evaluation.py         # ModelEvaluator, metrics
│   │   └── ensemble.py           # EnsembleTrainer coordinator
│   │
│   └── prediction/               # Modular prediction module
│       ├── __init__.py           # Module exports
│       ├── features.py           # FeatureComputer, GameFeatures
│       ├── loader.py             # ModelLoader, LoadedEnsemble
│       ├── nba_data.py           # Live NBA API data fetching
│       ├── odds.py               # Odds integration
│       ├── history.py            # Prediction history
│       └── pipeline.py           # PredictionPipeline
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
│   ├── nba_historical_odds.csv   # Historical betting odds (optional)
│   ├── nba_elo_cache.pkl         # ELO ratings
│   └── *_cache.pkl               # Feature caches
│
├── config/                       # Configuration files
│   └── best_xgb_params.json      # Tuned XGBoost hyperparameters
│
├── reports/                      # Training reports (git-ignored)
│   └── model_reports_*.txt/.json
│
├── scripts/                      # Utility scripts
│   └── setup_odds_api.py         # Odds API setup & test
│
├── tests/                        # Test suite (440 tests)
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures
│   ├── test_paths.py
│   ├── test_utils.py
│   ├── test_elo_system.py
│   ├── test_data_manager.py
│   ├── test_data_manager_integration.py
│   ├── test_odds_api.py
│   ├── test_odds_api_integration.py
│   ├── test_odds_api_features.py     # Odds static methods & enricher
│   ├── test_ensemble_predictor.py
│   ├── test_ensemble_integration.py
│   ├── test_main.py
│   ├── test_predict.py               # Game filtering (All-Star handling)
│   ├── test_training_data_prep.py    # DataPreparation tests
│   ├── test_training_trainers.py     # Trainer tests
│   ├── test_training_evaluation.py   # Evaluator tests
│   ├── test_training_ensemble.py     # EnsembleTrainer tests
│   ├── test_training_stacking.py     # Stacking meta-model tests
│   ├── test_prediction_features.py   # FeatureComputer tests
│   ├── test_prediction_loader.py     # ModelLoader tests
│   ├── test_prediction_nba_data.py   # NBA data helpers
│   ├── test_prediction_nba_data_api.py # NBA API mocked tests
│   └── test_prediction_pipeline.py   # Pipeline tests
│
├── docs/                         # Documentation
│   ├── PREDICTIONGUIDE.md
│   ├── NBAPREDICTIONCEILING.md
│   └── ELOplusCACHINGsummary.md
│
├── pytest.ini                    # Pytest configuration
├── .coveragerc                   # Coverage configuration
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

The project includes a comprehensive test suite with **440 tests** covering core functionality at **87% coverage**.

### Run Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_elo_system.py

# Run tests with coverage report
pytest --cov=src --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=src --cov-report=html
# View at: reports/coverage_html/index.html
```

### Test Coverage

| Module | Coverage | Notes |
|--------|----------|-------|
| `paths.py` | 100% | Fully covered |
| `training/data_prep.py` | 100% | Data preparation |
| `prediction/pipeline.py` | 100% | Prediction pipeline |
| `prediction/features.py` | 88% | Feature computation |
| `prediction/nba_data.py` | 85% | NBA API data fetching |
| `training/ensemble.py` | 81% | Ensemble coordinator + stacking |
| `prediction/loader.py` | 85% | Model loading |
| `training/evaluation.py` | 84% | Model evaluation |
| `training/trainers.py` | 84% | Model trainers |
| `nba_data_manager.py` | 82% | Core ELO system |
| `odds_api.py` | 79% | Odds features & enricher |

**Total Coverage: 87%** (target: 80%)

### Architecture (v2.0)

The codebase uses **dependency injection** and **factory patterns** for testability:

```python
# Example: Training with injectable dependencies
from src.training import EnsembleTrainer, EnsembleConfig

config = EnsembleConfig(architectures=['xgboost', 'logistic'])
trainer = EnsembleTrainer(config=config)
result = trainer.train(matchup_df)

# Example: Prediction with injectable loader
from src.prediction import PredictionPipeline, ModelLoader

loader = ModelLoader()
ensemble = loader.load_ensemble()
pipeline = PredictionPipeline(ensemble)
result = pipeline.predict(game_features)
```

---

## 💰 Betting Odds Integration (Optional)

The system can optionally fetch live bookmaker odds to:
- Display odds alongside model predictions
- Show "value" when model disagrees with the market
- (Future) Use odds as training features

### Setup

1. Get a free API key from [The-Odds-API](https://the-odds-api.com/) (500 free credits/month)
2. Set the environment variable:
   ```powershell
   # PowerShell
   $env:ODDS_API_KEY = "your_api_key_here"
   
   # Or permanently in your profile
   [Environment]::SetEnvironmentVariable("ODDS_API_KEY", "your_key", "User")
   ```
3. Run predictions - odds will appear automatically:
   ```bash
   python main.py predict
   ```

### What You'll See

```
🏆 Predicted Winner: Boston Celtics
📊 Confidence: 32.5%
🏠 Home Win Prob: 66.2%
✈️  Away Win Prob: 33.8%
💰 Bookmaker Odds: Home 1.52 (66%) | Away 2.65 (38%)
🔥 Value: Model 66.2% vs Market 65.8% (+0.4%)
```

### API Costs

| Endpoint | Credits | What You Get |
|----------|---------|--------------|
| Live odds | 1/call | Today's games from EU bookmakers |
| Historical | 10/call | One date of historical odds |

With 500 free credits/month: ~500 live calls or ~50 historical dates

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

The prediction output includes tiers based on **model certainty** (how far the probability is from 50%):

| Tier | Confidence | Probability | Meaning |
|------|------------|-------------|-------------|
| ⭐⭐⭐⭐⭐ EXCELLENT | 50%+ | 75%+ | Very confident |
| ⭐⭐⭐⭐ STRONG | 40-50% | 70-75% | Confident |
| ⭐⭐⭐ GOOD | 30-40% | 65-70% | Moderately confident |
| ⭐⭐ MODERATE | 20-30% | 60-65% | Slight lean |
| ⭐ RISKY | 10-20% | 55-60% | Near toss-up |
| ⛔ SKIP | <10% | <55% | No clear winner |

**Note:** Tier and confidence measure the same thing - model certainty. Confidence is the raw percentage, tier is the bucketed label.

**Model Agreement** indicates how much the 4 models agree (different from confidence). High agreement + high confidence = strongest predictions.

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
python main.py predict

# 2. Next day: update with actual results
python main.py --updateresults

# 3. View detailed accuracy statistics
python main.py --stats
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
python main.py --updateresults

# Update specific date
python main.py --updateresults 2026-01-17

# Update all pending results
python main.py --updateresults --all

# View detailed statistics only
python main.py --stats
```

### Statistics Output

The `--stats` command shows comprehensive breakdown:
- **Overall accuracy** with date range
- **By bet quality tier** with estimated ROI
- **By confidence level** (10 granular buckets)
- **By model agreement** (6 buckets)
- **By home win probability** (calibration check)
- **Time-based performance** (3/7/14/30 days)
- **Day of week analysis**
- **Streaks** (current, best win, worst loss)
- **Profitable filters** (combinations that beat the vig)

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
python predict.py
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
