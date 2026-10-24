# Prediction guide

Use this guide after installing dependencies from the [README](../README.md).
Commands run from the repository root with the virtual environment active.
Model files, data caches, and prediction history are local generated artifacts.

## Prepare the model and season

If you have no complete compatible ensemble in `models/`, train it:

```powershell
python -m main train
```

Training writes the standard model paths and overwrites earlier models. Existing
saved models can be loaded without retraining. Retrain to apply the corrected
date ordering and separate training/evaluation periods; older artifacts retain
their original protocol. Training now uses approximately 70/10/5/5/10 for base
training, early stopping, stacker fitting, calibration, and final evaluation,
with no shared dates. It saves dates, fingerprints, and final metrics in
`models/ensemble_training_manifest.json`. Final evaluation compares three
methods without automatically changing the averaging method used for picks.
The final table compares all individual models and the three ensemble methods
on identical holdout games, showing correct/game counts, accuracy, Brier score,
and log loss. The mean individual accuracy is labeled separately from the
averaging ensemble's accuracy.
Training duration depends on data
and hardware; a particular runtime or accuracy increase is not guaranteed.

Check `config/season.json`:

```json
{
  "current_season": "2026-27",
  "include_preseason": true,
  "use_previous_season": true
}
```

Use `null` for automatic season selection, which rolls over in October in Eastern
time. An explicit season overrides the calendar. Preseason filtering retains
postseason. Carryover borrows only from the immediately previous season and is
experimental; its benefit to saved-ensemble accuracy is unproven.

A completely empty training cache initially downloads the hardcoded 2005-06 to
2025-26 history. Refresh the active season after that bootstrap. See the
[data guide](ELOplusCACHINGsummary.md) for initial-download and rebuild details.

## Daily CLI workflow

```powershell
# Predict and append new picks to history.
python -m main predict

# After results are available, update yesterday or all pending dates.
python -m main --updateresults
python -m main --updateresults --all

# Review completed predictions.
python -m main --stats
```

Current/future CLI prediction attempts to refresh the active season's game cache. It fetches a
schedule, builds upcoming-game statistics, reconstructs pre-date standings and obtains optional
odds, loads the ensemble, and prints probabilities, tiers, and history counts.
The schedule path tries NBA CDN/static schedule data, ScoreboardV2, and
LeagueGameFinder fallbacks. External services can still return no usable games.

Games whose team IDs are not NBA franchises are skipped, such as All-Star teams.
This team-ID filter does not exclude all preseason or other exhibition games
between real NBA franchises. `include_preseason` filters history inputs, not the
schedule list itself.

`reports/prediction_history.csv` stores one pick per date/away/home combination.
Rerunning a different model does not overwrite that day's already recorded pick.
Past-date predictions are appended even if all games are Final. The CSV keeps
its original ten columns, and completed backfills count in its statistics. Existing picks
for the same date/teams are preserved; inspect the save message for duplicates.

## Select one model

```powershell
python predict.py --model ensemble
python predict.py --model xgboost
python predict.py --model rf
python predict.py --model logistic
python predict.py --model lstm
```

`rf` aliases `random_forest`. Single-model mode disables ensemble weights,
stacking, and calibration for that run. It still loads the ensemble artifacts
first, and shares the same history file and duplicate rules.

## Dates and manual matchups

```powershell
python -m main predict --date 2026-10-17
python predict.py --date 2026-10-17
python predict.py --manual-games matchups.json
```

Example contents of `matchups.json`:

```json
[
  {
    "home_team_id": 1610612738,
    "away_team_id": 1610612747,
    "game_status": "Scheduled",
    "game_date": "2026-10-17"
  }
]
```

These IDs designate Boston at home and the Lakers away; this example does not
assert a real scheduled matchup. Use actual NBA franchise IDs and supply
`game_status`, which the display code expects. With a manual list, the first
object's `game_date` sets the date for the whole list; omitting it uses `--date`
when supplied, otherwise Eastern today.

CLI and web use the same pre-date regular-season standings calculation for all
prediction dates. A past CLI date automatically selects historical replay mode. The season is inferred
from that date, and rolling statistics, H2H, and reconstructed regular-season
standings use only completed earlier dates. Live odds and filter recommendations
learned from today's history are disabled. The historical standings use equal
win-percentage ranks, not the official NBA tiebreaker hierarchy.

The complete saved model set must match its training manifest and artifact hashes.
Its creation date in Eastern time and all fitting/calibration/evaluation cutoffs
must precede the requested day. Otherwise the run stops with a provenance error:
restore a verified earlier model set rather than retraining today for an old date.
This is conservative at calendar-day granularity. Cached stat corrections and
the documented training/live feature differences still limit exact reconstruction
of what was known at the original prediction time.

For example, backfill yesterday and settle the saved replay outcomes:

```powershell
python -m main predict --date 2026-10-21
python -m main --updateresults 2026-10-21
```

No model or replay metadata is added to the history CSV. `winner` and `correct`
stay pending until the result updater runs. The CSV cannot distinguish a later
replay from an original pregame pick: completed replays count in `--stats`, web
statistics, and profitable-filter calculations. Avoid treating mixed history as
proof of performance recorded before games occurred.

Replaying a deleted pick cannot recover its original percentages. Earlier web
versions used live NBA standings; both entry points now reconstruct pre-date
standings with identical rank rules. Identical model files, game data, and
configuration are required for repeatable probabilities. Later source-data
corrections or configuration/model changes can still change a reconstruction.

## Interpret the output

Let `p` be the displayed home-win probability:

- Away-win probability is `1 - p`.
- Confidence is `2 * abs(p - 0.5)`; home 0.70 gives confidence 0.40, or 40%.
- Model agreement is `1 - std(base_model_home_probabilities)`. It measures how
  close the probabilities are, not the fraction of models voting for a winner.
- HOME wins the model decision when `p` exceeds the optional saved threshold;
  the default threshold is 0.5. A tie at the threshold selects AWAY.

The displayed probability comes from averaging the base-model probabilities,
optionally weighted by `models/ensemble_weights.pkl`. The pipeline also computes
a separate stacking/Platt result when artifacts exist; the displayed winner
continues to use the base-model average.

| Tier | Confidence | Larger home/away probability |
| --- | --- | --- |
| EXCELLENT | >= 50% | >= 75% |
| STRONG | 40% to < 50% | 70% to < 75% |
| GOOD | 30% to < 40% | 65% to < 70% |
| MODERATE | 20% to < 30% | 60% to < 65% |
| RISKY | 10% to < 20% | 55% to < 60% |
| SKIP | < 10% | < 55% |

Tiers do not measure historical hit rate or betting profitability. With a custom
threshold, the selected winner may not be the larger-probability team, while the
tier remains based on distance from 0.5. See [evaluation and accuracy](NBAPREDICTIONCEILING.md).

## Which history is used?

Each team's 20-game window uses completed W/L records strictly before the target
date, including the newest eligible completed result. Duplicate team/game rows
are removed. With carryover enabled, previous-season games fill unused places;
only the immediately previous season is eligible.

The CLI prints current/previous-season counts; the web response exposes them in
`history`. One eligible game permits an attempt, five provide most rolling
means/stds, and twenty fill the window. Sparse or unavailable features receive
defaults or zeros, so producing a prediction does not mean the window is complete.

Current-season form, streaks, and fatigue reset independently of borrowed
statistical history. An opener with no current results uses neutral form and
three rest days. The model includes rest days, back-to-back, and 3-in-4 indicators;
it has no dedicated player injury, trade, travel-distance, or roster inputs.
Training's rolling builder does not apply all these same season-boundary policies.

## History and result updates

History stores dates, team names, predicted/actual winners, confidence,
home-win probability, agreement, tier, and correctness. Pending `winner` and
`correct` values are filled after results can be resolved:

```powershell
python -m main --updateresults 2026-10-16
python -m main --updateresults --all --no-cache
python -m main --updateresults --all --no-lgf
```

`--no-cache` bypasses local result-cache lookup; `--no-lgf` skips the
LeagueGameFinder result source. They do not clear training feature caches.

History keeps its original ten columns. Prediction saving never clears an
existing result, and new picks have pending results until the updater runs.

Before replacing existing history, saves validate its old cells and keep its
previous contents in `reports/prediction_history.csv.bak`. The replacement is
atomic. A malformed file or a concurrent change stops the save instead of
discarding history.

The statistics screen groups completed picks and searches historical filters.
Its ROI calculations use fixed 1.41 decimal odds, not actual recorded wagers.
The CSV has no accepted-odds, stake, or payout columns. Use those summaries as
diagnostics, not verified betting returns or proof that a selected filter works
on later games.

## Web workflow

From the repository root, activate your environment and start the server:

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 7860
```

Leave that terminal open, visit [http://127.0.0.1:7860](http://127.0.0.1:7860),
select a model, and click **Run Predictions**. Stop the server with **Ctrl+C**.
The first prediction request loads the saved models and can take longer.

| Page or endpoint | Function |
| --- | --- |
| `/` | Today's Eastern-date predictions using the ensemble or a selected individual model; matchup, winner, confidence, home probability, agreement, and tier. |
| `/stats` | Completed prediction history with overall accuracy, grouped performance, and historical filter/ROI summaries. |
| `/docs`, `/redoc` | Interactive Swagger UI and alternate API reference. |
| `POST /api/predict` | Prediction JSON; use `{"model": "ensemble"}`, `xgboost`, `random_forest`/`rf`, `logistic`, or `lstm`. |
| `/health` | Server response check, returning `{"status": "ok"}`. |

The API also includes away probability, game status, and history-window counts
that the dashboard table does not display. The dashboard selects one model/method
per run and does not show the CLI's individual-model probability breakdown.
Date overrides and manual fixtures are available through the CLI, not the web UI
or prediction request schema.

Web predictions do not append to the history CSV. Run `python -m main predict`
to record picks and `python -m main --updateresults --all` after games finish;
reload `/stats` to see the updated records. ROI uses fixed 1.41 odds, not settled
wager prices. The health endpoint does not check models or NBA service access.

The web service can reuse cached season data without the CLI's initial refresh.
Refresh data explicitly when needed:

```powershell
python -c "from src.nba_data_manager import NBADataManager; NBADataManager().update_current_season()"
```

Loaded pipelines are cached in the web process. Restart the server after replacing
model files. The interface does not train models, settle prediction results, or
edit season configuration. See the [README webapp reference](../README.md#webapp)
for API usage, response details, and [Docker instructions](../README.md#docker).

## Troubleshooting

| Symptom | Check or action |
| --- | --- |
| Missing model/scaler/feature files | Train or restore a matching complete artifact set; verify the loader's reported model count. |
| A model is skipped during loading | Check its filename and library compatibility. A partial model set can also be incompatible with saved stacking artifacts. |
| Training cannot create five periods | Check for missing/invalid dates or game IDs, sufficient distinct dates, and both outcomes in each fitting period. |
| Saving rejects `ensemble_weights.pkl` | Archive/remove old custom weights before saving the new equal-weight ensemble; validate replacement weights separately. |
| Schedule unavailable | The NBA schedule sources failed or do not cover the requested date. Retry later or use confirmed fixtures with `--manual-games`; this does not mean the date has no games. Future dates use the full-season schedule with an official S3 fallback. |
| No games scheduled | A retrieved schedule confirms an empty date. Check the requested Eastern date. Manual input overrides only the schedule. |
| Not enough history | Check season, cache freshness, preseason flag, and carryover setting. Twenty is the window size, not the minimum number required to attempt inference. |
| Stale web predictions | Refresh raw season data and restart the service if models changed. |
| Cache rebuild message | Let feature generation rebuild a legacy/incompatible cache. The raw game cache need not be deleted. |
| Old records or feature formulas corrected | Explicitly rebuild derived caches; their compatibility header does not fingerprint every source row or code change. |
| TensorFlow uses CPU despite an installed GPU | Native Windows TensorFlow 2.11+ lacks CUDA support. Modern GPU use requires a compatible WSL2/Linux environment; hardware presence alone does not enable it. This affects execution speed, not the temporal split. |
| Garbled console symbols | Try UTF-8 mode, for example `python -X utf8 -m main predict`, and a UTF-8-capable terminal. |
| Low reported accuracy | Distinguish developmental scores from final holdout metrics; check sample, input features, dates, model manifest, and training/live differences. Older artifacts can retain the earlier validation reuse. |

There is no automatic retraining schedule. Preserve the old model set and compare
on a fixed later sample when deciding whether a retrain helped.
