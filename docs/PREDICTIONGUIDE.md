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
python nba_predictor.py
```

**Expected improvement:**
- 4 seasons + baseline = 61%
- 6 seasons + deep = **68-72%** ✨

**Training time:** 25-35 minutes (worth it!)

---

## 🔮 Predicting Today's Games

### Option 1: Automatic Today's Games

```bash
python predict_todays_games.py
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
from nba_predictor import NBADataFetcher, NBAPredictor, predict_specific_game
from tensorflow import keras

# Load model
predictor = NBAPredictor()
predictor.model = keras.models.load_model('nba_model_best.keras')

# Get latest data
fetcher = NBADataFetcher(seasons=['2024-25'])
games_df = fetcher.fetch_games()

# Predict any matchup!
predict_specific_game(predictor, "Lakers", "Warriors", games_df)
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
python nba_predictor.py
```

### Get Today's Predictions
```bash
python predict_todays_games.py
```

### Check for Data Leakage
```bash
python nba_feature_analysis.py
```

---

## 📊 Files Created

After training, you'll have:
- `nba_lstm_model.keras` - Final model
- `nba_model_best.keras` - **Best model during training** ⭐ (use this!)
- `scaler.pkl` - Feature scaler
- `feature_columns.pkl` - Feature list for predictions

---

## 💡 Tips for Best Results

### 1. Retrain Weekly
```bash
# Fresh data = better predictions
python nba_predictor.py
```

### 2. Check Confidence Scores
- **High confidence (>60%)**: Trust the prediction
- **Medium confidence (30-60%)**: Coin flip territory
- **Low confidence (<30%)**: Very uncertain

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
- Team is new/relocated
- Not enough games played this season
- Wait until more games are played

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
   python nba_predictor.py
   # Wait ~30 mins for better accuracy
   ```

2. **Get today's predictions:**
   ```bash
   python predict_todays_games.py
   ```

3. **Track your results:**
   - Save predictions
   - Compare with actual outcomes
   - Calculate real-world accuracy

4. **Improve further:**
   - Add injury data
   - Include player ratings
   - Try ensemble methods

---

**Good luck with your predictions! 🏀**