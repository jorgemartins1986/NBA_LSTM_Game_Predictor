"""The production CLI must finish all fitting before final evaluation."""

from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd


def test_cli_fits_stacking_then_evaluates_then_saves():
    from train import train_ensemble

    frame = pd.DataFrame({'HOME_WIN': [0, 1]})
    result = SimpleNamespace(ensemble_accuracy=.6)
    with patch('src.nba_data_manager.NBADataFetcher') as fetcher, \
            patch('src.nba_predictor.NBAPredictor') as predictor, \
            patch('src.training.EnsembleTrainer') as factory:
        fetcher.return_value.fetch_games.return_value = frame
        predictor.return_value.prepare_matchup_data.return_value = frame
        coordinator = factory.return_value
        coordinator.train.return_value = result
        coordinator.train_stacking.return_value = result
        assert train_ensemble() is result
        assert [call[0] for call in coordinator.method_calls] == [
            'train', 'train_stacking', 'evaluate', 'save']
        assert factory.call_args.kwargs['config'].test_size == .1
