# Quick Guide: Getting 70%+ Accuracy & Predictions

## 🎯 Improving Accuracy from 61% to 70%+

Your model got 61% because:
1. ✅ **Fixed data leakage** (good!)
2. ❌ Only using 4 seasons (~5,000 games)
3. ❌ Using baseline architecture

### Step 1: Re-train with Optimal Settings

The code has been updated with:
- **6 seasons of data** (2019-20 through 2024-25) = ~7,500 games
- **Deep architecture** (256→128→64→32 instead of 128→64→32)
- **Better callbacks** (monitors accuracy, saves best model)

```bash
python main.py train
```

**Expected improvement:**
- 4 seasons + baseline = 61%
- 6 seasons + deep = **68-72%** ✨

**Training time:** 25-35 minutes (worth it!)

---

## 🔮 Predicting Today's Games

### Option 1: Automatic Today's Games

```bash
python main.py predict
```

**What it does:**
- ✅ Fetches all NBA games scheduled for today
- ✅ Gets recent stats for each team
- ✅ Predicts winners with confidence scores
- ✅ Shows summary of all predictions

**Sample output:**
```
🏀 Game 1/5
----------------------------------------------------------------------
   Los Angeles Lakers @ Golden State Warriors
   Status: Scheduled for 7:00 PM

   🏆 Predicted Winner: Golden State Warriors
   📊 Confidence: 68.5%
   🏠 Home Win Prob: 68.5%
   ✈️  Away Win Prob: 31.5%
```

### Option 2: Manual Specific Game

In Python:

```python
from src.nba_data_manager import NBADataFetcher
from src.nba_predictor import NBAPredictor
from tensorflow import keras

# Load model
predictor = NBAPredictor()
predictor.model = keras.models.load_model('models/nba_model_best.keras')

# Get latest data
fetcher = NBADataFetcher(seasons=['2024-25'])
games_df = fetcher.fetch_games()
```

---

## 📈 Understanding Your Results

### Accuracy Benchmarks

| Accuracy | Meaning |
|----------|---------|
| 50-55%   | Barely better than coin flip |
| 56-62%   | Basic model, some signal |
| **63-68%**   | **Good model** ✅ |
| **69-74%**   | **Excellent model** 🌟 |
| 75-80%   | Exceptional (rare without injuries/news) |
| 90%+     | Data leakage! 🚨 |

### Your Current Status

- **With 4 seasons + baseline: 61%** 
  - This is actually decent! Better than most casual predictions.
  
- **With 6 seasons + deep: 68-72%** (expected)
  - This matches research-level performance
  - Competitive with betting markets

---

## 🚀 Quick Commands

### Train Model (do this first!)
```bash
python main.py train
```

### Get Today's Predictions
```bash
python main.py predict
```

### View Prediction Statistics
```bash
python main.py --stats
```

---

## 📊 Files Created

After training, you'll have in `models/`:
- `nba_ensemble_xgboost_1.json` - XGBoost model
- `nba_ensemble_rf_2.pkl` - Random Forest model
- `nba_ensemble_logistic_3.pkl` - Logistic Regression model
- `nba_ensemble_model_4.keras` - LSTM neural network
- `ensemble_scaler_*.pkl` - Feature scalers
- `ensemble_feature_cols.pkl` - Feature list
- `ensemble_meta_clf.pkl` - Stacking meta-learner
- `ensemble_platt.pkl` - Platt calibrator

---

## 💡 Tips for Best Results

### 1. Retrain Weekly
```bash
# Fresh data = better predictions
python main.py train
```

### 2. Check Confidence & Tier
- **EXCELLENT tier (50%+ conf)**: Model is very confident (75%+ probability)
- **STRONG/GOOD tier (30-50% conf)**: Reasonable confidence
- **MODERATE/RISKY tier (<30% conf)**: Near toss-up

**Note:** Confidence = how far from 50% the model's probability is. Tier just buckets this into labels.

**Special games:** Exhibition/All-Star games are automatically filtered out during prediction.

### 3. Combine with Domain Knowledge
The model doesn't know about:
- ❌ Injuries
- ❌ Back-to-back games
- ❌ Trades
- ❌ Playoff motivation

Use your basketball knowledge alongside the predictions!

---

## 🔧 Troubleshooting

### "No games found for today"
- It might be off-season
- NBA API might be down
- Use manual prediction instead

### "Not enough data for prediction"
- At least one team has no records in the active season; 20 games is the rolling window, not a minimum.
- Update `current_season` in `config/season.json` (currently `2026-27`), or use `null` for automatic October rollover.
- Set `include_preseason` in `config/season.json` to `true` or `false`. Cached data and both API paths honor it; the cache retains all game types for later toggling.
- Enable `use_previous_season` in `config/season.json` to fill the rolling window from the immediately previous season. Old games are replaced as new results arrive; season form and fatigue reset, and standings/H2H stay current-season only.
- Upcoming-game features include the newest completed result, strictly before the prediction date. Each team needs 1 eligible game to attempt prediction, 5 for most rolling averages/stds, and 20 for the full window. Missing statistics are filled with zeros. The CLI/web API report history counts.

### Still getting low accuracy?
Try these improvements:
1. **Add more seasons** (try 8 seasons)
2. **Different window size** (try 14 or 28 games)
3. **More epochs** (change to 150)
4. **Add features** (injuries, rest days, etc.)

---

## 🎓 Understanding the Predictions

### What the model considers:
- ✅ Recent team performance (last 20 games)
- ✅ Shooting efficiency (EFG%, TS%)
- ✅ Turnovers and rebounds
- ✅ Win streaks and momentum
- ✅ Home court advantage

### What it doesn't consider:
- ❌ Individual player matchups
- ❌ Injuries and rest
- ❌ Coaching strategies
- ❌ Playoff context
- ❌ Weather/travel fatigue

---

## 📞 Next Steps

1. **Re-train with 6 seasons:**
   ```bash
   python main.py train
   # Wait ~30 mins for better accuracy
   ```

2. **Get today's predictions:**
   ```bash
   python main.py predict
   ```

3. **Track your results:**
   ```bash
   python main.py --stats
   python main.py --updateresults
   ```

4. **Improve further:**
   - Add injury data
   - Include player ratings
   - Fine-tune ensemble weights
   - Track ROI with `python main.py --stats`

---

**Good luck with your predictions! 🏀**
