# NBA Game Predictor

A Python project for predicting NBA game winners with XGBoost, Random Forest,
Logistic Regression, and an attention-based LSTM. It provides a command-line
workflow, a FastAPI web interface, local game caches, and prediction tracking.

Quick links: [Setup](#setup) · [CLI commands](#main-commands) ·
[Run the webapp and see its features](#webapp) · [Docker](#docker).

The repository does not establish a single current production accuracy figure.
Training output, historical reports, and recorded live predictions describe
different samples. Read [evaluation and accuracy](docs/NBAPREDICTIONCEILING.md)
before interpreting those numbers.

## Setup

Run commands from the repository root. On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The current environment uses Python 3.12. Dependencies are not fully pinned;
changing pandas, NumPy, TensorFlow, or scikit-learn can affect saved-artifact
compatibility. On macOS/Linux, activate with `source venv/bin/activate`.

A fresh checkout has no trained models or game cache: `models/`, `cache/`, and
`reports/` are ignored. Train locally or supply a compatible complete model set
before predicting. Training overwrites the standard model filenames; keep a
backup if you need to compare model versions.

## Main commands

```powershell
python -m main train
python -m main predict
python -m main --stats
python -m main --updateresults
python -m main --updateresults --all
python -m main --updateresults 2026-10-16
```

To use the browser dashboard instead, start the webapp and open
[http://127.0.0.1:7860](http://127.0.0.1:7860):

```powershell
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 7860
```

The default result update targets yesterday. `--all` processes pending dates.
`--no-cache` and `--no-lgf` apply to result updates, not feature-cache rebuilding:

```powershell
python -m main --updateresults --all --no-cache --no-lgf
```

For model selection, explicit dates, and manual matchups:

```powershell
python predict.py --model xgboost
python predict.py --model rf
python predict.py --model logistic
python predict.py --model lstm
python predict.py --date 2026-10-17
python predict.py --manual-games matchups.json
```

`ensemble` is the default; `rf` aliases `random_forest`. A manual JSON file is an
array of objects with `home_team_id`, `away_team_id`, `game_status`, and optionally
`game_date`. See the [prediction guide](docs/PREDICTIONGUIDE.md) for an example.
`python -m main predict --date YYYY-MM-DD` also overrides the schedule date.

Past dates use a historical replay: the season comes from the requested date,
rolling/H2H inputs and reconstructed regular-season standings exclude that day
and all later results, and live odds/current-history filter recommendations are
disabled. Model artifact fingerprints and the training manifest must establish
that the complete model version existed before that Eastern date, with all
fitting/calibration/evaluation periods ending earlier. Missing or incompatible
provenance blocks the replay; changing `--date` cannot rewind a newer model.
Standings ties use equal win percentages rather than official NBA tiebreakers,
and corrected cached statistics can differ from originally published data.
Future dates use the official NBA season schedule, with its S3 host as a fallback
when the CDN is blocked. An unavailable schedule is reported separately from a
confirmed empty date. Future picks use information available at prediction time.

## Season and early-season history

Current checked-in `config/season.json`:

```json
{
  "current_season": "2026-27",
  "include_preseason": true,
  "use_previous_season": true
}
```

- `current_season` selects training data and active-season cache updates; prediction data follows the requested date;
  past-date replays infer their season from the requested date.
  Set it to `null` for automatic October rollover in America/New_York time.
- `include_preseason` controls preseason records in prediction history inputs.
  Postseason remains eligible. The raw cache retains records for later toggling.
- `use_previous_season` fills the 20-game statistics window from the immediately
  previous season. Current-season results gradually replace that history. The
  option defaults to false when omitted, though the checked-in file enables it.

Upcoming-game features use completed results strictly before the prediction date
and remove duplicate team/game records. One eligible game permits an attempt;
most rolling averages and standard deviations need five. With fewer games,
missing values use defaults or zeros. Twenty games fill the window.

Form, streaks, and fatigue use current-season results only. An opener with no
current-season results uses neutral form and three rest days. H2H uses earlier
current-season results; CLI and web prediction standings use completed regular-season games before the prediction date. CLI output
and web responses report current/previous-season window counts.

Carryover remains experimental: improved accuracy for the saved ensemble has
not been established. These switches configure prediction inputs; the training
matchup builder does not apply the same season filtering and form resets.
Changing them does not retrain saved models.

## Training and features

`python train.py` and `python -m main train` run the same ensemble entry point:

1. Load/download historical data, update the active season when a cache exists,
   and calculate chronological ELO ratings.
2. Build rolling team features and one row per valid home/away matchup. Rows
   without sufficient rolling history are removed from the training dataset.
3. Validate dates and game IDs, sort by calendar date and game ID, and create
   five periods with no shared calendar dates.
4. Train XGBoost, Random Forest, Logistic Regression, and LSTM, then fit stacking
   and Platt calibration on separate later periods.
5. Evaluate averaging, raw stacking, and calibrated stacking on the same final
   holdout, then save models and a training manifest. Averaging remains the
   deployed method; holdout results do not automatically select a method.

| Period | Approximate share | Use |
| --- | --- | --- |
| Base training | 70% | Fit base models and scalers |
| Early stopping | 10% | XGBoost/LSTM validation and developmental model scores |
| Stacker fitting | 5% | Fit the stacking classifier |
| Calibration | 5% | Fit the Platt calibrator |
| Final holdout | 10% | Score the three already fitted methods only |

Whole dates stay together, so exact counts vary. Fractions are configurable in
`EnsembleConfig`; `test_size` is the final holdout fraction. All periods are
prepared once before fitting, and later calls use that same data snapshot.

Recency weights are passed to XGBoost, Random Forest, and LSTM. The Logistic
Regression trainer currently fits without those weights. Random Forest selects
up to 40 features and stores the selected indices with its model.

The split is enforced at training time, so unordered legacy caches do not need
rebuilding to benefit. Missing/invalid dates, missing/duplicate supplied game IDs,
or insufficient dates/classes for the fitting periods cause an error. Training
prints the date ranges. Existing saved models must be retrained to use these
corrections; changing code does not rewrite model files or old reported scores.

Individual model scores printed during fitting describe early-stopping data,
not the final holdout. The final summary reports the accuracy of averaged
predictions. A final comparison table shows each individual model and all three
ensemble methods on the same games, including correct/game counts, accuracy,
Brier score, and log loss. It also distinguishes the mean of individual
accuracies from the accuracy of averaged predictions. The metrics are saved in
`models/ensemble_training_manifest.json`, together with
period dates/counts, an input fingerprint, configuration, package versions, and
artifact hashes. The separate timestamped classification-report exporter is
not automatically called. Repeated tuning against the final period would still
invalidate it as an independent evaluation.

Default trainer settings live in `src/training/trainers.py`. The standard entry
point does not read `config/best_xgb_params.json` automatically.

Current matchup inputs include shooting/counting-stat rolling means/stds,
Four Factors and efficiency proxies, recent form, fatigue, H2H, standings, and
standings differences. Current-game `PLUS_MINUS` is not a rolling model input;
H2H point differential uses earlier meetings. Exact columns and order come from
`models/ensemble_features.pkl`, rather than a fixed documented feature count.

ELO columns are calculated in the enriched game data, but the current matchup
builder does **not** copy them into ensemble inputs. Historical odds enrichment
is also separate from standard training. See [ELO and caching](docs/ELOplusCACHINGsummary.md).

## Saved models and prediction behavior

With the default model order, training writes these files under `models/`:

| File | Purpose |
| --- | --- |
| `nba_ensemble_xgboost_1.json` | XGBoost model |
| `nba_ensemble_rf_2.pkl` | Random Forest and selected feature indices |
| `nba_ensemble_logistic_3.pkl` | Logistic Regression |
| `nba_ensemble_model_4.keras` | LSTM |
| `ensemble_scalers.pkl` | Scalers in model order |
| `ensemble_features.pkl` | Feature names in input order |
| `ensemble_model_types.pkl` | Model types in saved order |
| `ensemble_stack_meta_lr.pkl` | Stacking classifier |
| `ensemble_stack_platt.pkl` | Platt calibrator |
| `ensemble_training_manifest.json` | Training periods, fingerprints, configuration, final metrics |

The loader can also read optional `ensemble_weights.pkl` with `weights` and
`threshold`. Standard training does not create or refresh that file and refuses
to save over a directory containing it: archive/remove the old weights first,
and validate any replacement separately against the new model set.

Displayed home probability is the base-model probability average, weighted when
compatible optional weights exist. The winner is HOME when that probability is
greater than the saved threshold (default 0.5), otherwise AWAY. Stacking/calibration
produce a separate `stacked_probability`; they do not select the displayed winner.

Confidence is `2 * abs(home_probability - 0.5)`. Agreement is
`1 - std(individual_home_probabilities)`, not a voting percentage. Neither is a
measured accuracy or a statistical confidence interval.

| Tier | Confidence | Larger of home/away probabilities |
| --- | --- | --- |
| EXCELLENT | >= 50% | >= 75% |
| STRONG | 40% to < 50% | 70% to < 75% |
| GOOD | 30% to < 40% | 65% to < 70% |
| MODERATE | 20% to < 30% | 60% to < 65% |
| RISKY | 10% to < 20% | 55% to < 60% |
| SKIP | < 10% | < 55% |

A custom winner threshold can select a team other than the one with the larger
probability. Tiers continue to measure distance from 0.5, not betting value.

## Data refresh and caches

Current/future CLI predictions attempt an active-season cache refresh before fetching team
history. The web service reads cached season data when available and does not
perform that same refresh. Refresh explicitly when using the web interface:

```powershell
python -c "from src.nba_data_manager import NBADataManager; NBADataManager().update_current_season()"
```

On an empty cache, the training loader's default historical list runs from
2005-06 through 2025-26. That cold-start branch does not also update 2026-27.
After bootstrapping, refresh the configured season before relying on its results;
the cache guide explains how to request a different initial season list.

| Cache file under `cache/` | Contents |
| --- | --- |
| `nba_games_cache.csv` | Raw team/game records and season labels |
| `nba_elo_cache.pkl` | Latest ELO rating dictionary |
| `feature_cache.pkl` | Versioned engineered team features |
| `matchup_cache.pkl` | Versioned matchup dataframe |
| `nba_games_enriched.csv` | Exported games with engineered columns |
| `nba_matchups.csv` | Exported matchups and `HOME_WIN` target |

Feature/matchup headers record pandas/NumPy versions and window size. Legacy,
unreadable, or mismatched caches rebuild automatically. Atomic replacement
protects the previous file if a cache save fails. These caches are disposable;
models and prediction history are separate.

Caches are not a general source-data fingerprint: corrections to older records
or changed feature formulas may require an explicit rebuild. CSV row counts,
file sizes, and training duration vary with cache contents and hardware.

Feature workers load the lightweight `src/feature_engineering.py` module rather
than importing TensorFlow. Informational TensorFlow startup logs are hidden by
default while warnings/errors remain visible. To show startup details:

```powershell
$env:TF_CPP_MIN_LOG_LEVEL = "0"
```

This logging setting does not disable oneDNN optimizations. CPU fallback is
supported; GPU detection depends on the installed TensorFlow build and platform.
Prediction startup omits routine device announcements and two known Keras
compatibility notices (the ONNX-export NumPy alias probe and the internal
`tf.placeholder` deprecation). Other warnings and errors remain visible.
Native Windows TensorFlow 2.11+ does not support CUDA GPU execution; the GPU
hardware can be present while TensorFlow runs on CPU. Modern TensorFlow GPU
support on Windows uses [WSL2](https://www.tensorflow.org/install/pip).

## Prediction history and results

CLI predictions append to `reports/prediction_history.csv`; duplicate date/away/home
pairs are skipped, so rerunning a model does not replace an existing pick.
Past-date picks are saved even when all games are Final. Existing date/team
picks are preserved rather than replaced by a replay. The web prediction
endpoint returns predictions without writing them to this CSV.

History keeps its original ten columns: `date`, `away_team`, `home_team`,
`prediction`, `winner`, `confidence`, `home_win_prob`, `model_agreement`, `tier`,
and `correct`. No replay or model metadata is added. `winner` and `correct` are
populated by the result updater; new predictions start with pending results.
Dates normally use Eastern time; manual games can supply their own date.

Saves validate existing cells, replace the file atomically, and keep the previous
file in `reports/prediction_history.csv.bak`. Concurrent changes and malformed
history files are rejected instead of overwritten. The six briefly introduced
metadata columns are removed when appending to an existing expanded file;
original prediction/result columns and any other custom columns are preserved.

The CSV does not distinguish original picks from later historical replays.
Completed backfilled rows count in its statistics, so a mixed history cannot
establish accuracy achieved by predictions actually recorded before tipoff.

`--stats` summarizes completed picks by tier, confidence, agreement, date, and
other groups. The displayed ROI and profitable-filter calculations assume fixed
**1.41 decimal odds**; history does not record actual accepted odds, stakes, or
payouts. Those figures are hypothetical, and filters selected on the same history
need later validation. Straight-up accuracy does not imply spread accuracy or profit.

## Optional bookmaker odds

Set `ODDS_API_KEY` in the process environment or a local `.env` file:

```powershell
$env:ODDS_API_KEY = "your_key_here"
python scripts/setup_odds_api.py
python -m main predict
```

`src/odds_api.py` loads `.env`. Keep credentials out of Git. Provider plans and
request costs are not fixed by this repository. Prediction continues without odds
when the key or service is unavailable. Odds influence model inputs only when
matching columns exist in the saved feature schema; standard training does not
add historical odds automatically.

## Webapp

Run from the repository root after the [setup](#setup). Use your existing
compatible models in `models/`; train only if they are missing. The dashboard,
statistics page, and health endpoint can open without loading models, but running
predictions requires the complete ensemble artifacts.

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn webapp.main:app --host 127.0.0.1 --port 7860
```

Keep that terminal running and open [http://127.0.0.1:7860](http://127.0.0.1:7860).
Choose a model and click **Run Predictions**. The first request loads the models
and fetches NBA data, so it can take longer. Stop the server with **Ctrl+C**.
If port 7860 is occupied, use `--port 7861` and open the matching browser URL.

| Function | Where | Available behavior |
| --- | --- | --- |
| Daily predictions | `/` | Choose Ensemble, XGBoost, Random Forest, Logistic Regression, or LSTM and run predictions for today's Eastern date. |
| Prediction results | `/` | View matchup, selected winner, confidence, home-win probability, model agreement, and tier, sorted by confidence. |
| Historical statistics | `/stats` or **Detailed Statistics** | Read completed CLI prediction history: overall record/accuracy/date range; breakdowns by tier, confidence, agreement, home probability, recent periods, and weekday; historical filter summaries. |
| Prediction API | `POST /api/predict` | Return prediction JSON, including away-win probability, game status, and current/previous-season history counts in addition to the dashboard fields. |
| API documentation | `/docs` | Inspect supported request values and use **Try it out** to call the prediction endpoint. `/redoc` provides an alternate API reference. |
| Health check | `/health` | Return `{"status": "ok"}` when the server responds; this does not validate model files or external NBA services. |

For example, call the running API from another PowerShell terminal:

```powershell
Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:7860/api/predict' -ContentType 'application/json' -Body '{"model":"ensemble"}'
```

Accepted `model` values are `ensemble` (the default), `xgboost`, `random_forest`
or `rf`, `logistic`, and `lstm`. Returned probabilities, confidence, and agreement
are fractions from 0 to 1; the dashboard formats them as percentages. Missing
history or an unavailable schedule can produce an empty predictions list; inspect
`skipped_reason` and `schedule_status` when present in the API response.

The webapp currently supports today's predictions. Future/past dates and manual
fixtures use the CLI's `--date` and `--manual-games` options. Individual model
mode shows the selected model's predictions; the dashboard does not show the
CLI's per-model probability breakdown. Ensemble mode uses averaging as described
in [prediction behavior](#saved-models-and-prediction-behavior).

Web predictions are returned without appending to `reports/prediction_history.csv`.
To record picks and populate the statistics page, use `python -m main predict`,
then settle results with `python -m main --updateresults --all` after games finish.
The statistics page reads that local CSV again when reloaded. Its ROI/filter
figures use fixed 1.41 odds, not actual wager returns.

The webapp does not train models, settle results, edit season configuration, or
perform the CLI's initial active-season cache refresh. Before a daily web session,
refresh cached results when needed:

```powershell
python -c "from src.nba_data_manager import NBADataManager; NBADataManager().update_current_season()"
```

Loaded pipelines are cached in the server process; restart the server after
replacing model artifacts. These routes have no authentication; the command
above binds to your local computer.

## Docker

```powershell
docker build -t nba-predictor-web .
docker run --rm -p 7860:7860 nba-predictor-web
# Or use the supplied Compose configuration:
docker compose up --build
```

The Dockerfile starts Uvicorn on `PORT` (default 7860). A deployment must include
compatible model artifacts or mount them into `/app/models`; a source-only clone
cannot predict. The supplied Compose file does not mount persistent data volumes
or pass an odds key. Configure volumes/environment for your deployment. Files
written only inside a disposable container do not persist after removal.

## Tests and coverage

Latest verified suite: **617 passed**, **90.91% line coverage**. Coverage runs
require at least **85%** under `.coveragerc`:

```powershell
python -m pytest
python -m pytest --cov=src --cov-config=.coveragerc --cov-report=term-missing
python -m pytest --cov=src --cov-config=.coveragerc --cov-report=html
```

The HTML report is `reports/coverage_html/index.html`. `.coverage` and worker
coverage data are generated and ignored; `.coveragerc` is tracked configuration.
A plain pytest run does not measure coverage. Existing exclusions are package
initializers, `src/nba_predictor.py`, `src/update_prediction_results.py`, and
`src/utils.py`. The percentage is aggregate line coverage in that scope, not a
per-module guarantee or evidence of predictive accuracy. API recovery tests use
offline mocks, not a check that live NBA endpoints currently work.

## Project map and further reading

| Path | Responsibility |
| --- | --- |
| `main.py`, `train.py`, `predict.py` | CLI entry points |
| `src/training/` | Data preparation, trainers, ensemble evaluation |
| `src/training/partitions.py` | Five chronological training/evaluation periods |
| `src/prediction/` | Model loading, feature vectors, NBA/odds inputs, history |
| `src/feature_engineering.py` | Rolling team calculations and worker function |
| `src/dataframe_cache.py` | Versioned feature/matchup cache I/O |
| `src/nba_data_manager.py` | Historical/current-season data and ELO |
| `src/nba_predictor.py` | Matchup builder and neural-network architectures |
| `src/season_config.py`, `src/paths.py` | Season settings and paths |
| `webapp/`, `src/web/` | FastAPI routes, templates, prediction service |
| `scripts/` | Odds setup and repeatable offline evaluation tools |
| `tests/` | Unit and integration tests |

- [Prediction guide](docs/PREDICTIONGUIDE.md): daily workflow and troubleshooting.
- [ELO and caching](docs/ELOplusCACHINGsummary.md): data lifecycle and actual ELO integration.
- [Evaluation and accuracy](docs/NBAPREDICTIONCEILING.md): metrics and validation limits.
