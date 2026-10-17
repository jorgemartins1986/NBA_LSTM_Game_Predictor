# Understanding prediction accuracy and evaluation

The project has not established a mathematical ceiling for NBA prediction
accuracy or a single verified current production accuracy. Old claims that a
particular architecture would reach 70%, that deeper trees would add a fixed
percentage, or that straight-up accuracy implied spread-betting performance were
not supported by the implementation or a reproducible independent evaluation.

This guide explains the metrics and the current validation limits. For commands,
see the [prediction guide](PREDICTIONGUIDE.md) and [README](../README.md).

## Different numbers answer different questions

| Number | What it describes |
| --- | --- |
| Individual model accuracy during fitting | Developmental accuracy on the early-stopping period |
| Final holdout averaging accuracy | Correct winners from averaged base-model probabilities on the final period |
| Final holdout stacking accuracy | Correct winners from the fitted stacker on that same final period |
| Final holdout calibrated stacking accuracy | Correct winners after separately fitted Platt calibration on that final period |
| History accuracy from `--stats` | Fraction correct among completed records in the local prediction-history CSV |
| Test coverage | Fraction of configured source lines exercised by software tests |

The mean of individual model accuracies need not equal the accuracy of averaged
predictions. The current final summary uses the latter. A timestamped report in
`reports/` describes its original run, not necessarily the currently saved models.
Training saves final metrics in `models/ensemble_training_manifest.json`; it
does not automatically call the separate classification-report export helper.

History also lacks a model fingerprint/training cutoff and does not distinguish
all configuration changes. CLI single-model and ensemble runs share the same
CSV. Date/team duplicate rules keep the first recorded pick. Inspect those facts
before attributing history accuracy to one frozen model version.

## Current validation behavior

Standard ensemble training sorts features and labels together by calendar date
and game ID, then prepares five periods once before fitting:

| Period | Default share | Permitted use |
| --- | --- | --- |
| Base training | 70% | Fit base models and their scalers |
| Early stopping | 10% | XGBoost/LSTM stopping decisions and developmental scores |
| Stacker fitting | 5% | Fit stacking using base-model outputs |
| Calibration | 5% | Fit Platt calibration using the fitted stacker's outputs |
| Final holdout | 10% | Evaluate already fitted averaging, stacking, and calibrated stacking |

Whole dates stay together, and actual counts can differ from those percentages.
Boundaries depend on dates/counts, not labels. Missing/invalid dates,
missing/duplicate supplied game IDs, insufficient distinct dates, nonfinite
features, nonbinary labels, and single-class fitting periods are rejected.
Later stacking/evaluation calls use the original prepared snapshot, even if a
caller supplies a changed dataframe. Refitting stacking after final evaluation
is rejected. The final period is never passed to fitting or stopping callbacks.

Individual training scores remain developmental because they use early-stopping
data. The final table shows correct/game counts, accuracy, Brier score, and log
loss for every individual model and all three ensemble methods on identical
final games, with the date range printed above it. It separately reports the
arithmetic mean of individual holdout accuracies and the averaging ensemble's
difference from that mean in percentage points. Individual and ensemble metrics
are also saved in the manifest. The evaluation does not automatically choose
the method or threshold: deployed predictions continue to use averaging.
Choosing changes after examining this final period requires a fresh later
evaluation before making another independent performance claim.

Both earlier defects are corrected: unordered caches no longer mix future
training dates with past evaluation dates, and stacking/calibration no longer
reuse their fitting data for final scoring. Existing saved models and old
scores are unchanged; retrain to apply the new protocol. The legacy single-model
training method also separates base fitting, early stopping, and final scoring;
it has no stacking/calibration stages.

New standard saves include `ensemble_training_manifest.json`, recording period
dates/counts, input SHA-256, configuration, package versions, final metrics, and
saved artifact hashes. Older models without that manifest have unverified
training provenance; file modification times alone do not prove independence.
The manifest does not establish that every engineered feature is leakage-free.
Historical rolling statistics use shifted results, and upcoming-game history
is restricted to completed games before the target date. Training/live feature
differences still need evaluation as described below.

## Production probability, confidence, and agreement

The normal prediction pipeline reports the mean base-model home probability,
optionally using saved weights. Its winner threshold defaults to 0.5 and can be
loaded from `models/ensemble_weights.pkl`. That file is optional. Standard
training refuses to save a new equal-weight model set into a directory containing
old custom weights; archive/remove them and validate any replacement separately.

Stacking plus Platt calibration can generate `stacked_probability`, but this
separate value does not select the CLI/web winner. Stacked holdout metrics
therefore describe a candidate method, rather than the displayed prediction stream.

For displayed home probability `p`:

```text
away_probability = 1 - p
confidence = 2 * abs(p - 0.5)
agreement = 1 - standard_deviation(base_model_home_probabilities)
```

Confidence is a transformation of model output, not a confidence interval or
observed success rate. Agreement measures probability dispersion, not a fraction
of agreeing votes. High agreement can occur when every model shares the same
bias. Tier labels bucket confidence and do not establish calibration or value.

## Measure accuracy with context

Winner accuracy is `correct / completed`. Always state the number of games,
date/season range, selection policy, input history policy, and model version.
Compare approaches on the same eligible games; report additional predictions
and skipped games separately. Otherwise an apparent improvement can be a change
in sample composition.

A fair baseline depends on the task and sample. A coin flip, always picking the
home team, and picking a pregame market favorite answer different questions.
Measure each on the same dates instead of assuming a universal benchmark.
Preseason and early-season results should be reported separately when they have
different input policies or sample sizes.

Probability quality matters as well as winner accuracy:

- Brier score averages `(p - outcome) ** 2` for binary home outcomes; lower is better.
- Log loss evaluates assigned probabilities and penalizes confident mistakes.
- Calibration compares predicted home probabilities with observed home-win rates.

Standard training prints and saves Brier scores and log loss for all three
methods. `ModelEvaluator` also provides a calibration-error helper. Metrics
alone do not prove independence or remove feature leakage. Changing a
probability threshold can alter accuracy without improving probability quality.

## Carryover and training/live differences

`use_previous_season` fills sparse live 20-game windows using immediately prior
season games. Live form and fatigue reset to the current season, while the
training rolling builder does not implement identical season-boundary filtering
and resets. Assess this training/live difference when evaluating early-season picks.

The option remains experimental; improved accuracy for the saved production
ensemble has not been established. Neither a separate logistic baseline nor a
small replay is sufficient to certify an ensemble improvement. Roster changes
are not represented by a dedicated adjustment.

The current project also calculates ELO in its data loader without copying ELO
columns into standard matchup features. Historical odds enrichment is separate
from the main training path. Refer to the [ELO/cache guide](ELOplusCACHINGsummary.md)
when checking which proposed inputs actually reach the model.

## Repeatable evaluation tools

The repository retains offline comparison scripts:

```powershell
python -X utf8 -m scripts.evaluate_rolling_history --output reports/rolling_history_evaluation.json
python -X utf8 -m scripts.evaluate_saved_ensemble_history --recent-only --output reports/saved_ensemble_history_evaluation.json
python -X utf8 -m scripts.compare_ensemble_methods
python -X utf8 -m scripts.validate_training_protocol
```

Run from the repository root with a suitable local game cache; the saved-ensemble
script also needs compatible saved models. These evaluations can be expensive.
The explicit output paths place generated results under ignored `reports/`.
The two history-policy scripts' default output paths are under `docs/`.

`compare_ensemble_methods` writes its report and per-game probabilities under
`reports/`. It first diagnoses the saved models, then fits fresh experimental
models in memory using date-disjoint training, early-stopping, stacker-fitting,
calibration, and final-test periods (70/10/5/5/10 split by ordered rows, with whole
dates kept together). It compares averaging, raw stacking, and calibrated stacking
on the same final games and also reports NBA-franchise-only samples. Add
`--saved-only` to skip fresh training; that mode does not establish an independent
accuracy result. Neither mode changes production models, prediction behavior,
or history. A single experiment does not certify the current saved artifacts or
automatically select a production method.

`validate_training_protocol` runs the actual current production trainer,
stacking, final evaluation, and saving against the cached matchup dataframe.
It writes isolated models, the training manifest, and a validation result under
`reports/production_protocol_validation/`. It verifies that deployed model files,
the matchup export, and prediction history were not changed. It evaluates all
cached matchups; unlike the NBA-only comparison summaries, this can include
foreign opponents. Its single seeded run does not guarantee future improvement.

`evaluate_rolling_history` trains an independent logistic baseline using earlier
seasons and compares current-only versus carryover inputs. Its result is not the
production four-model ensemble's accuracy.

`evaluate_saved_ensemble_history` replays saved model artifacts, compares input
policies, and records model fingerprints. `--recent-only` filters targets using
saved-model file dates; it still cannot replace an audited training manifest.
Historical standings are reconstructed from earlier results rather than fetched
live, with possible differences from API tie rankings. Neither script establishes
realized betting ROI or modifies production models/prediction history.

## A stronger future comparison

For a new performance claim, freeze data, code/configuration, and model artifacts
with a recorded training cutoff. The standard trainer now implements separate
periods for base training, early stopping, stacking, calibration, and final
evaluation. An explicitly chronological cross-fitting design is another option.
Evaluate all input policies on the same untouched later
games and report uncertainty and coverage alongside accuracy/probability scores.

Retraining, increasing tree depth, adding seasons, changing windows, or adding
injury/player/market inputs are experiments. Measure their effect; fixed additive
accuracy gains are not established. Do not turn development scores into a
current production claim without a corresponding independent sample.

## Accuracy, software coverage, and ROI

The latest verified software suite has 568 passing tests and 90.43% aggregate
line coverage in `.coveragerc`'s scope, with an enforced 85% coverage-run minimum.
That checks software paths; it does not measure or guarantee NBA prediction accuracy.
Legacy scripts, GPU utilities, and package initializers remain excluded.

Straight-up winner accuracy does not determine against-the-spread accuracy.
Actual return requires prices, stakes, and settled payouts. The project's
history CSV does not store those wager details. The CLI/web statistics currently
use fixed 1.41 decimal odds for hypothetical ROI and filter selection; the
break-even hit rate for that fixed price is `1 / 1.41`, about 70.92% before any
other costs. No universal hit rate guarantees profit across different prices.

Filters selected from the same history are exploratory. Test them on later,
independent records before treating the selected subset as evidence of an edge.
