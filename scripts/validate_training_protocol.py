"""Run the real production training workflow into an isolated reports directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')

import pandas as pd

from scripts.evaluate_saved_ensemble_history import model_manifest
from src.paths import MATCHUPS_CSV, PREDICTION_HISTORY_FILE
from src.training import EnsembleConfig, EnsembleTrainer


class ProgressTrainer(EnsembleTrainer):
    def train_single_model(self, architecture, data, model_index):
        print(f'Fitting {architecture}; only base_train/early_stopping are supplied.', flush=True)
        return super().train_single_model(architecture, data, model_index)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='reports/production_protocol_validation')
    args = parser.parse_args()
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    before = model_manifest()
    source_hash = digest(MATCHUPS_CSV)
    history_hash = digest(PREDICTION_HISTORY_FILE) if Path(PREDICTION_HISTORY_FILE).exists() else None
    frame = pd.read_csv(MATCHUPS_CSV, dtype={'GAME_ID': str}, parse_dates=['GAME_DATE'])
    coordinator = ProgressTrainer(EnsembleConfig(verbose=False))
    result = coordinator.train(frame)
    print(json.dumps(result.partitions.summary(), indent=2), flush=True)
    coordinator.train_stacking(result)
    coordinator.evaluate(result)
    output = Path(args.output_dir).resolve()
    from src.paths import MODELS_DIR
    if output == Path(MODELS_DIR).resolve() or Path(MODELS_DIR).resolve() in output.parents:
        raise ValueError('Validation output must be outside production models/')
    coordinator.save(result, output_dir=output)
    unchanged = (before == model_manifest() and source_hash == digest(MATCHUPS_CSV)
        and history_hash == (digest(PREDICTION_HISTORY_FILE) if Path(PREDICTION_HISTORY_FILE).exists() else None))
    assert unchanged, 'Production files changed during validation'
    report = {'periods': result.partitions.summary(),
              'final_holdout_metrics': {name: item.metrics() for name, item in result.final_evaluations.items()},
              'production_files_unchanged': unchanged, 'production_models': before}
    (output / 'validation_result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report['final_holdout_metrics'], indent=2), flush=True)
    print('Production model/cache/history fingerprints unchanged.', flush=True)


if __name__ == '__main__':
    main()
