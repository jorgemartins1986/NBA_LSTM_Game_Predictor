# ELO and caching

This document describes the current implementation, not expected accuracy gains.
Main sources are `src/nba_data_manager.py`, `src/feature_engineering.py`,
`src/dataframe_cache.py`, and the matchup builder in `src/nba_predictor.py`.
See the [README](../README.md) for setup and the [prediction guide](PREDICTIONGUIDE.md)
for day-to-day commands.

## Raw game data

`NBADataManager.get_games()` downloads historical data when the raw cache is
missing, otherwise attempts an active-season update. The default initial list is
2005-06 through 2025-26. It is a fixed list, not an automatically extended season
range. On a cold start it does not separately fetch the configured 2026-27 season.

`update_current_season()` takes its default from `config/season.json`. It requests
that whole season with an empty season-type filter, removes the cached rows for
that season, inserts the response, sorts by date, and saves the CSV. This replaces
a season; it is not a request restricted to games since the last cached date.
On an update exception, it attempts to return the existing cache.

After bootstrapping historical data, refresh the active season:

```powershell
python -c "from src.nba_data_manager import NBADataManager; NBADataManager().update_current_season()"
```

To choose an initial history explicitly, run Python from the project root:

```python
from src.nba_data_manager import NBADataManager

manager = NBADataManager()
games = manager.get_games(seasons=['2025-26', '2026-27'], force_download=True)
```

`force_download=True` replaces the raw cache with the requested seasons. Without
that flag, a supplied season list is only used when the cache is absent. A short
history changes the training sample; it is not a guaranteed improvement.

## ELO calculation

`get_games()` rebuilds ratings chronologically over the loaded data and saves the
latest team ratings to `cache/nba_elo_cache.pkl`. Defaults are:

| Setting | Value |
| --- | --- |
| Initial rating | 1500 |
| K factor | 20 |
| Home advantage in expected probability | +100 rating points |
| Season transition | Keep 75% of the deviation from 1500 |

The home expectation is `1 / (1 + 10 ** ((away - (home + 100)) / 400))`.
After a home win, the home change is `20 * (1 - expected_home)`; the away change
is its negative. A loss uses `20 * (0 - expected_home)`.

For example, home 1650 versus away 1550 gives about 75.97% home expectation
with the default advantage. A home win adds about 4.81 points; a loss removes
about 15.19 points. These are illustrative rating calculations, not measured
NBA prediction performance.

Both team records receive the same pregame columns:

- `ELO_HOME` and `ELO_AWAY`: ratings before updating that game.
- `ELO_DIFF`: home minus away rating, **without** the extra 100 home points.
- `ELO_PROB_HOME`: expected probability including the home advantage.

Games without exactly one recognizable home and away row are skipped by the
rating loop; initialized ELO values can remain zero for those rows.
`get_current_elo_ratings()` reads the saved dictionary; it does not refresh it.
Calling only `update_current_season()` does not recalculate that ELO file.

## ELO is not currently an ensemble input

The loader adds ELO columns to game data and the enriched CSV can contain them.
However, `FeatureEngineering.create_rolling_features()` has no ELO rolling
columns, and `NBAPredictor.prepare_matchup_data()` does not copy raw ELO columns
into its matchup dataframe. Therefore standard ensemble training does not gain
four ELO inputs merely because the data manager calculated them.

Prediction uses the feature names saved in `models/ensemble_features.pkl`.
The normal CLI/web flow does not supply a separate ELO difference to stacking.
Adding ELO to the predictive model would require consistent training and live
feature construction, a rebuilt feature schema, retraining, and evaluation.
No automatic accuracy uplift is established by the current ELO implementation.

## Cache and export files

All paths are under `cache/`, as defined by `src/paths.py`:

| File | Role |
| --- | --- |
| `nba_games_cache.csv` | Raw team/game records across cached seasons |
| `nba_elo_cache.pkl` | Latest team-rating dictionary |
| `feature_cache.pkl` | Engineered team history and cutoff date |
| `matchup_cache.pkl` | Home/away training rows and cutoff date |
| `nba_games_enriched.csv` | Inspectable export of engineered team rows |
| `nba_matchups.csv` | Inspectable matchup export, including the target |

Training sorts matchup rows itself and creates five date-disjoint periods for
base fitting, early stopping, stacking, calibration, and final evaluation.
Unordered legacy caches do not need rebuilding just for this correction;
existing models need retraining. See the [evaluation guide](NBAPREDICTIONCEILING.md).

Raw records are team rows, usually two per game. Their row count is not the
number of unique games. File sizes, row counts, and timing vary.

Feature and matchup caches have a plain compatibility header recording format
version, pandas version, NumPy version, data kind, and rolling-window size. The
header is checked before unpickling pandas objects. Legacy formats, changed
settings, corrupt payloads, and invalid dates/cutoffs cause a short `Rebuilding`
message and recomputation from raw data. New cache files are written to a
temporary sibling and replace the previous file only after a successful write.

An unchanged compatible cache can skip computation. For newly dated games,
team features are recalculated with full history for the affected teams, while
matchups are created after the cached cutoff. This does not detect every
correction to an older source row. Code/formula changes are not automatically
fingerprinted either; rebuild derived caches when those inputs change.

## Rebuilding derived data

After updating raw data, a full derived-data rebuild can be requested in Python:

```python
from src.nba_data_manager import NBADataFetcher
from src.nba_predictor import NBAPredictor

games = NBADataFetcher().fetch_games()
matchups = NBAPredictor(window_size=20).prepare_matchup_data(games, use_cache=False)
```

This rewrites the standard feature/matchup caches and CSV exports. It does not
train or replace models or rewrite prediction history. No need to delete the raw
game cache for an incompatible feature-cache warning. CSV exports are not used
as an automatic fallback for unreadable feature pickles.

## Prediction freshness and season policies

The CLI attempts an active-season update before predictions. The web prediction
service does not perform that update and can keep reading an older cached season.
`fetch_season_games()` prefers usable local data unless `NBA_FORCE_FETCH=1`.
That environment variable affects prediction season fetching, not all cache types.

`include_preseason` and `use_previous_season` apply to live prediction history.
They do not remove raw cached records or impose identical filtering/reset rules
on the training matchup builder. Turning carryover on does not validate its
accuracy benefit; see [evaluation limits](NBAPREDICTIONCEILING.md).

## Worker startup and speed

Parallel feature workers call a function in `src/feature_engineering.py`, which
does not import TensorFlow. They no longer repeat TensorFlow initialization and
GPU detection for each team worker. The default worker count is CPU count minus
one, with a minimum of one; `n_jobs` can be supplied to `prepare_matchup_data()`.

Cache reuse and avoiding unnecessary imports can reduce runtime, but no fixed
speedup or accuracy increase follows. Logging configuration hides informational
TensorFlow startup messages while retaining warnings/errors; oneDNN remains
configured independently.
