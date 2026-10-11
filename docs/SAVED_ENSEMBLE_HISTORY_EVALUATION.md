# Direct comparison with the saved production ensemble

The previous evaluation trained a separate logistic baseline. It did not verify
improved accuracy for the saved four-model ensemble. The direct comparison below
is mixed; previous-season carryover remains experimental.

Run `python -X utf8 -m scripts.evaluate_saved_ensemble_history` to repeat the
comparison from local cached results. Add `--recent-only` for the sample after
the saved-model file date. Detailed per-game probabilities and model SHA-256
fingerprints are written to `docs/saved_ensemble_history_evaluation.json` (a
generated, ignored report). The script checks that models do not change during
evaluation. It does not retrain, fetch APIs, or write prediction history.

## What was compared

All four saved models, their scalers, the production feature order, random-forest
feature selection, ensemble weights, and winner threshold are used. Batched
inference is checked against `PredictionPipeline.predict_from_features`.

Three policies are evaluated using otherwise identical inputs:

- Legacy current-season statistics: features for the most recently completed
  game, omitting that game's result from its rolling statistics.
- Corrected current-season statistics: upcoming-game features using all results
  strictly before the target date, with carryover disabled.
- Corrected statistics with previous-season carryover enabled.

Head-to-head and reconstructed standings use only earlier current-season results.
Standings use regular-season results, so preseason results do not create wins or
losses. NBA API tie rankings cannot be reconstructed exactly. No historical odds
are used; the saved feature list has no odds columns. The production CLI winner
uses the aggregate base-model probability; the separately calculated stacking
output does not determine that winner, so aggregation is preserved here.

## Results from the October 11, 2026 local cache

Only common games should be used when comparing accuracy. Extra carryover
predictions are reported separately.

| Sample | Preseason inputs | Common games | Legacy correct | Corrected current-only correct | Carryover correct |
| --- | --- | ---: | ---: | ---: | ---: |
| 2026–27 October 7–10 | Included | 12 | 7 (58.3%) | 8 (66.7%) | 5 (41.7%) |
| 2025–26 opening replay | Included | 290 | 190 (65.5%) | 192 (66.2%) | 199 (68.6%) |
| 2025–26 opening replay | Excluded | 274 | 185 (67.5%) | 184 (67.2%) | 192 (70.1%) |

The recent comparison has nineteen eligible NBA matchups. One additional cached
matchup lacked a usable pair of NBA home/away records and was excluded. With
preseason inputs enabled, carryover covers seven additional games skipped by
current-only history; it gets nine of all nineteen correct (47.4%). With preseason
inputs excluded, current-only history skips all nineteen games, while carryover
gets ten of nineteen correct (52.6%). That is a coverage result, not a paired
accuracy improvement.

On the twelve comparable recent games, probability error also worsens with
carryover: Brier score increases from 0.24347 to 0.28039. On historical common
games, Brier improves from 0.21087 to 0.20750 with preseason and from 0.21065 to
0.20533 without it. Lower Brier is better.

## Limits and corrected conclusion

The saved files are dated October 6, 2026; the cached training-feature and matchup
CSV snapshots end October 5. The recent sample excludes October 6 and October 11,
and uses October 7–10 completed results. This is consistent with results after the
model snapshot, but file timestamps and local cache snapshots are not an audited
training-data manifest. The nineteen-game sample is preseason-only and far too
small to establish stable regular-season improvement.

The 2025–26 replay includes matchups where both teams are within their first twenty
regular-season games. It is diagnostic, not an independent holdout: the training
code uses later games for model validation/early stopping and calibration. Older
replays can therefore give optimistic results even when inputs exclude the
target game's result.

We have not established that enabling carryover improves the deployed ensemble's
accuracy. The older replay is encouraging, but the small recent sample goes the
other way. The original merge should have been described as experimental. Current
configuration is unchanged; enabling the option should not be represented as a
verified accuracy or profitability improvement. More games recorded after a
frozen model snapshot are needed for a meaningful comparison.
