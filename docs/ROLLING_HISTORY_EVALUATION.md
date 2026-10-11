# Previous-season rolling history evaluation

The previous-season fallback improves probability estimates in an independent
chronological baseline and is retained as a configurable prediction option.
It is not a validation of the saved production ensemble or betting returns.

## Implementation

`use_previous_season` in `config/season.json` enables filling each team's
20-game statistical window with its latest completed games from the immediately
previous season. Current-season games gradually replace those records.
`include_preseason` applies to both seasons. Postseason records remain eligible.
Only results strictly before the prediction date are used. Duplicates and
unfinished games are excluded. No roster adjustment or recency reweighting is
introduced.

Streak/form and fatigue use the current season only. Opening-game fatigue uses
the existing neutral three-day rest convention. Standings and head-to-head
remain current-season inputs. CLI output and the web API expose window counts.
Live feature computation now includes the newest completed result rather than
reusing that result's pregame features. This makes five eligible games sufficient
for most rolling averages/stds, and twenty sufficient for the full window.

## Historical comparison

Run `python -X utf8 -m scripts.evaluate_rolling_history` to reproduce the offline
comparison from the local game cache. It saves
[`rolling_history_evaluation.json`](rolling_history_evaluation.json).

The evaluation covers 2022-23 through 2025-26 and includes a matchup only when
both teams are within their first twenty regular-season games. For each target
season, a standardized logistic classifier is fitted using only earlier seasons,
starting with 2015-16. Training uses games where both teams already have twenty
current-season results; the two input policies therefore coincide in training.
The same fitted model evaluates both policies. Both policies use the corrected
upcoming-game calculation and reset season form/fatigue. No target result enters
its features. Tests verify the vectorized evaluation matches live features at
opening, intermediate, and full-window cutoffs.

| Preseason included | Common games | Current-only accuracy | Carryover accuracy | Current-only Brier | Carryover Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yes | 1,163 | 62.51% | 62.94% | 0.23870 | 0.22546 |
| No | 1,100 | 63.45% | 65.36% | 0.25992 | 0.22121 |

Lower Brier scores indicate lower mean squared probability error. A paired
5,000-sample bootstrap resampling calendar dates gives 95% intervals for the
carryover-minus-current Brier difference of [-0.02453, -0.00296] with preseason
and [-0.05915, -0.02005] without it. These intervals are below zero. Probability
error improves in all four individual seasons; winner accuracy does not improve
in every season. Without preseason, carryover also covers 63 matchups skipped
by the current-only policy (1,163 versus 1,100 predictions).

## Merge decision and limits

The comparison supports merging the implementation as a reversible configuration
option: aggregate probability error improves with both preseason settings,
aggregate winner accuracy increases, and data availability improves. The saved
four-model ensemble also passes an offline compatibility smoke check with 112
finite features and valid probabilities, including a regular-season opener with
zero current-season results when preseason is excluded.

Saved ensemble artifacts have no auditable training cutoff, so their historical
accuracy cannot be treated as an honest holdout result. The measured percentages
above belong to an independently trained logistic baseline. Historical bookmaker
odds and ROI were not evaluated. The comparison does not establish production
ensemble improvement, betting profitability, or protection from roster changes.
Neither production models nor prediction history were modified by evaluation.
