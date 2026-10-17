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
| Individual model accuracy | Correct winner classifications on the trainer's later split |
| Evaluated ensemble accuracy | Correct winners from averaging base-model probabilities on that split |
| Final `Average Accuracy` in training output | Arithmetic mean of the individual model accuracies |
| Printed stacked accuracy | Performance after fitting stacking and calibration on the same later split |
| History accuracy from `--stats` | Fraction correct among completed records in the local prediction-history CSV |
| Test coverage | Fraction of configured source lines exercised by software tests |

The final average of model accuracies need not equal the accuracy of their
averaged predictions. A timestamped report in `reports/` describes its original
run, not necessarily the artifacts currently saved in `models/`. The current
training entry point prints scores but does not automatically call the available
classification-report export helper.

History also lacks a model fingerprint/training cutoff and does not distinguish
all configuration changes. CLI single-model and ensemble runs share the same
CSV. Date/team duplicate rules keep the first recorded pick. Inspect those facts
before attributing history accuracy to one frozen model version.

## Current validation behavior

`DataPreparation` splits input rows 80/20 with `shuffle=False`, and each scaler
is fitted on training rows only. Historical rolling statistics use shifted
results, excluding the target game's statistics. Upcoming-game history is
restricted to completed games before the target date.

Those safeguards do not by themselves make the reported score an untouched
holdout. In the current standard pipeline:

- XGBoost uses the later split as its early-stopping evaluation set.
- LSTM callbacks use that split as validation data.
- The stacking classifier is fitted on base-model predictions for that split.
- The Platt calibrator is fitted on those same labels and meta-model outputs.
- Stacked accuracy is then reported on the sample used to fit those components.

Consequently the printed stacked score is an in-sample development diagnostic,
not evidence of future accuracy. Even base-model scores are influenced by model
selection/early stopping. The split preserves supplied row order; it does not
sort dates itself, so custom or modified matchup data must be checked for date
ordering and overlap.

Saved models do not include an audited training-data manifest/cutoff in the
standard save routine. Historical replay may overlap training, tuning, or
calibration data. File modification times alone do not prove independence.

## Production probability, confidence, and agreement

The normal prediction pipeline reports the mean base-model home probability,
optionally using saved weights. Its winner threshold defaults to 0.5 and can be
loaded from `models/ensemble_weights.pkl`. That file is optional and is not
rewritten by the standard training save routine.

Stacking plus Platt calibration can generate `stacked_probability`, but this
separate value does not select the CLI/web winner. A good stacked development
score therefore does not directly describe the displayed prediction stream.

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

The offline comparison scripts calculate Brier scores and log loss, and
`ModelEvaluator` provides a calibration-error helper. Standard training does not
print all these diagnostics. These metrics do not remove data leakage or
validation reuse. Changing a probability threshold can alter accuracy without
improving probability quality.

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
```

Run from the repository root with a suitable local game cache; the saved-ensemble
script also needs compatible saved models. These evaluations can be expensive.
The explicit output paths place generated results under ignored `reports/`.
The scripts' default output paths are under `docs/`.

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
with a recorded training cutoff. Use separate periods for base training, tuning,
stacking/calibration, and final evaluation, or an explicitly chronological
cross-fitting design. Evaluate all input policies on the same untouched later
games and report uncertainty and coverage alongside accuracy/probability scores.

Retraining, increasing tree depth, adding seasons, changing windows, or adding
injury/player/market inputs are experiments. Measure their effect; fixed additive
accuracy gains are not established. Do not turn development scores into a
current production claim without a corresponding independent sample.

## Accuracy, software coverage, and ROI

The latest verified software suite has 502 passing tests and 88.11% aggregate
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
