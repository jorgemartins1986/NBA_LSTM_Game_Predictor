"""
Ensemble Training Coordinator
=============================
Orchestrates training of multiple models and combines them into an ensemble.
This is the refactored version of train_ensemble_models().

Design principles:
- Composition over inheritance: uses injected trainers
- Configuration via dataclasses
- Testable at each level (unit, integration, end-to-end)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pickle

from .data_prep import DataPreparation, TrainTestData
from .trainers import (
    TrainerFactory, TrainerConfig, TrainingResult,
    XGBoostConfig, RandomForestConfig, LogisticConfig, LSTMConfig
)
from .evaluation import ModelEvaluator, EvaluationResult
from .partitions import TrainingPartitions, prepare_partitions


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training"""
    architectures: List[str] = field(default_factory=lambda: ['xgboost', 'random_forest', 'logistic', 'lstm'])
    test_size: float = 0.1  # Final holdout, never used for fitting
    validation_size: float = 0.1
    stacking_size: float = 0.05
    calibration_size: float = 0.05
    random_state: int = 42
    use_sample_weights: bool = True
    weight_decay: float = 1.2
    verbose: bool = True


@dataclass
class EnsembleResult:
    """Result of training an ensemble"""
    models: List[Any]
    scalers: List[Any]
    feature_cols: List[str]
    model_types: List[str]
    training_results: List[TrainingResult]
    ensemble_accuracy: Optional[float] = None
    individual_accuracies: Optional[List[float]] = None
    # Stacking artifacts
    meta_clf: Optional[Any] = None
    platt: Optional[Any] = None
    partitions: Optional[TrainingPartitions] = None
    final_evaluations: Dict[str, EvaluationResult] = field(default_factory=dict)
    final_model_evaluations: List[EvaluationResult] = field(default_factory=list)
    
    @property
    def n_models(self) -> int:
        return len(self.models)
    
    @property
    def best_model_index(self) -> int:
        if self.individual_accuracies:
            return int(np.argmax(self.individual_accuracies))
        return 0
    
    @property
    def average_accuracy(self) -> float:
        accs = [r.accuracy for r in self.training_results]
        return float(np.mean(accs))


class EnsembleTrainer:
    """
    Coordinates training of multiple models into an ensemble.
    
    This is the refactored version of train_ensemble_models() that:
    - Separates data preparation from training
    - Uses dependency injection for trainers
    - Is fully testable at each level
    
    Example usage:
        trainer = EnsembleTrainer()
        result = trainer.train(matchup_df)
        
    For testing with mock trainers:
        mock_factory = MockTrainerFactory()
        trainer = EnsembleTrainer(trainer_factory=mock_factory)
    """
    
    def __init__(self, 
                 config: Optional[EnsembleConfig] = None,
                 data_prep: Optional[DataPreparation] = None,
                 trainer_factory: Optional[TrainerFactory] = None,
                 evaluator: Optional[ModelEvaluator] = None,
                 print_fn: Optional[Callable] = None):
        """
        Initialize ensemble trainer with injectable dependencies.
        
        Args:
            config: Ensemble configuration
            data_prep: Data preparation instance (injectable for testing)
            trainer_factory: Factory for creating trainers (injectable for testing)
            evaluator: Model evaluator (injectable for testing)
            print_fn: Print function to use (for capturing output in tests)
        """
        self.config = config or EnsembleConfig()
        self.data_prep = data_prep or DataPreparation(
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        self.trainer_factory = trainer_factory or TrainerFactory
        self.evaluator = evaluator or ModelEvaluator()
        self.print_fn = print_fn or print
    
    def _log(self, message: str):
        """Log a message using configured print function"""
        if self.config.verbose:
            self.print_fn(message)
    
    def prepare_data(self, matchup_df: pd.DataFrame) -> Tuple[TrainTestData, List[str]]:
        """
        Prepare data for training.
        
        Args:
            matchup_df: Matchup dataframe with features
            
        Returns:
            Tuple of (TrainTestData, feature_cols)
        """
        self._log("Preparing data...")
        data, feature_cols = self.data_prep.prepare_for_training(matchup_df)
        self._log(f"✓ Prepared {data.n_train_samples} training samples, {data.n_test_samples} test samples")
        if isinstance(data.train_dates, np.ndarray) and isinstance(data.test_dates, np.ndarray):
            self._log(f"Training dates: {pd.Timestamp(data.train_dates[0]).date()} to "
                      f"{pd.Timestamp(data.train_dates[-1]).date()}")
            self._log(f"Validation dates: {pd.Timestamp(data.test_dates[0]).date()} to "
                      f"{pd.Timestamp(data.test_dates[-1]).date()} (no shared dates)")
        return data, feature_cols
    
    def compute_sample_weights(self, n_samples: int) -> Optional[np.ndarray]:
        """Compute sample weights if enabled in config"""
        if not self.config.use_sample_weights:
            return None
        return DataPreparation.compute_sample_weights(n_samples, self.config.weight_decay)
    
    def train_single_model(self, 
                          architecture: str,
                          data: TrainTestData,
                          model_index: int) -> TrainingResult:
        """
        Train a single model of the specified architecture.
        
        Args:
            architecture: Model architecture ('xgboost', 'random_forest', etc.)
            data: Prepared train/test data
            model_index: Index of this model in the ensemble
            
        Returns:
            TrainingResult with trained model and metrics
        """
        self._log(f"\n{'='*60}")
        self._log(f"Training model {model_index + 1}: {architecture.upper()}")
        self._log(f"{'='*60}")
        
        # Create trainer with model-specific random state
        config = self._get_trainer_config(architecture, model_index)
        trainer = self.trainer_factory.create(architecture, config)
        
        # Scale data (each model gets its own scaler)
        scaled_data = self.data_prep.scale_features(data)
        
        # Compute sample weights
        sample_weights = self.compute_sample_weights(len(data.y_train))
        
        # Train
        result = trainer.train(scaled_data, sample_weights)
        
        self._log(f"✓ {architecture} accuracy: {result.accuracy*100:.2f}%")
        
        return result
    
    def _get_trainer_config(self, architecture: str, model_index: int) -> TrainerConfig:
        """Get appropriate config for trainer with unique random state"""
        random_state = self.config.random_state + model_index
        verbose = self.config.verbose
        
        if architecture == 'xgboost':
            return XGBoostConfig(random_state=random_state, verbose=verbose)
        elif architecture == 'random_forest':
            return RandomForestConfig(random_state=random_state, verbose=verbose)
        elif architecture == 'logistic':
            return LogisticConfig(random_state=random_state, verbose=verbose)
        elif architecture in ('lstm', 'deep'):
            return LSTMConfig(architecture=architecture, random_state=random_state, verbose=verbose)
        else:
            return TrainerConfig(random_state=random_state, verbose=verbose)
    
    def train(self, matchup_df: pd.DataFrame) -> EnsembleResult:
        """
        Train the complete ensemble.
        
        Args:
            matchup_df: Matchup dataframe with features and labels
            
        Returns:
            EnsembleResult with all trained models
        """
        self._log("="*70)
        self._log(f"TRAINING ENSEMBLE OF {len(self.config.architectures)} MODELS")
        self._log("="*70)
        
        # Freeze all five periods before fitting any model. The final period is
        # retained for evaluation only and never passed to a trainer/callback.
        partitions = prepare_partitions(matchup_df, self.data_prep,
            self.config.validation_size, self.config.stacking_size, self.config.calibration_size,
            test_size=self.config.test_size)
        data, feature_cols = partitions.base_model_data(), partitions.feature_cols
        for name, period in partitions.summary().items():
            self._log(f"{name}: {period['games']} games, {period['start']} to {period['end']}")
        
        # Train each model
        training_results: List[TrainingResult] = []
        models = []
        scalers = []
        model_types = []
        
        for i, arch in enumerate(self.config.architectures):
            result = self.train_single_model(arch, data, i)
            
            training_results.append(result)
            models.append(result.model)
            scalers.append(result.scaler)
            model_types.append(result.model_type)
        
        # Create ensemble result
        individual_accs = [r.accuracy for r in training_results]
        
        self._log(f"\n{'='*70}")
        self._log("ENSEMBLE TRAINING COMPLETE")
        self._log(f"{'='*70}")
        self._log("Early-stopping validation accuracies (not final holdout):")
        for i, (acc, arch) in enumerate(zip(individual_accs, self.config.architectures)):
            self._log(f"  Model {i+1} ({arch}): {acc*100:.2f}%")
        self._log(f"\nEarly-stopping validation mean: {np.mean(individual_accs)*100:.2f}%")
        
        return EnsembleResult(
            models=models,
            scalers=scalers,
            feature_cols=feature_cols,
            model_types=model_types,
            training_results=training_results,
            individual_accuracies=individual_accs,
            partitions=partitions
        )
    
    @staticmethod
    def _require_partitions(result: EnsembleResult) -> TrainingPartitions:
        if not isinstance(result.partitions, TrainingPartitions):
            raise ValueError("Call train() to create five chronological periods before stacking/evaluation")
        if result.feature_cols != result.partitions.feature_cols:
            raise ValueError("Feature schema differs from the frozen training periods")
        if not result.models or not (len(result.models) == len(result.scalers) == len(result.model_types)):
            raise ValueError("Models, scalers, and model types must have matching nonzero lengths")
        return result.partitions

    def _period_predictions(self, result: EnsembleResult, matrix: np.ndarray) -> np.ndarray:
        columns = []
        for model, scaler, kind in zip(result.models, result.scalers, result.model_types):
            trainer = self.trainer_factory.create(kind)
            columns.append(trainer.predict_proba(model, scaler.transform(matrix)))
        raw = np.column_stack(columns)
        if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
            raise ValueError("Invalid model probabilities")
        return raw

    @staticmethod
    def _meta_features(raw: np.ndarray) -> np.ndarray:
        return np.hstack([raw, np.abs(raw - .5)])

    def train_stacking(self, result: EnsembleResult,
                       matchup_df: Optional[pd.DataFrame] = None) -> EnsembleResult:
        """Fit stacking and calibration on their own frozen chronological periods.

        matchup_df is accepted for old callers, but the train() snapshot is used.
        This method never reads or scores the final-test period.
        """
        from sklearn.linear_model import LogisticRegression

        periods = self._require_partitions(result)
        if result.final_evaluations:
            raise ValueError("Cannot refit stacking after final holdout evaluation")
        self._log("TRAINING STACKING META-MODEL ON ITS SEPARATE PERIOD")
        raw = self._period_predictions(result, periods.stacker_fit.X)
        meta_clf = LogisticRegression(C=1., class_weight='balanced', max_iter=1000,
                                      random_state=self.config.random_state)
        meta_clf.fit(self._meta_features(raw), periods.stacker_fit.y)
        raw_calibration = self._period_predictions(result, periods.calibrator_fit.X)
        meta_probs = meta_clf.predict_proba(self._meta_features(raw_calibration))[:, 1]
        platt = LogisticRegression(C=1., max_iter=1000, random_state=self.config.random_state)
        platt.fit(meta_probs.reshape(-1, 1), periods.calibrator_fit.y)
        result.meta_clf, result.platt = meta_clf, platt
        self._log("Stacker and Platt calibrator fitted on separate periods; final holdout untouched.")
        return result

    def evaluate(self, result: EnsembleResult,
                 matchup_df: Optional[pd.DataFrame] = None) -> EvaluationResult:
        """Score already fitted methods on the frozen final holdout, without fitting.

        matchup_df is retained for call compatibility; new data cannot replace
        the final period chosen before train(). No method is auto-selected here.
        """
        periods = self._require_partitions(result)
        if result.meta_clf is None or result.platt is None:
            raise ValueError("Fit stacking/calibration before inspecting the final holdout")
        self._log("EVALUATING ALL METHODS ON THE UNTOUCHED FINAL HOLDOUT")
        final = periods.final_test
        raw = self._period_predictions(result, final.X)
        result.final_model_evaluations = [
            self.evaluator.evaluate_predictions(final.y, (p > .5).astype(int), p)
            for p in raw.T]
        averaged = self.evaluator.evaluate_ensemble(list(raw.T), final.y, result.model_types)
        stacked = result.meta_clf.predict_proba(self._meta_features(raw))[:, 1]
        calibrated = result.platt.predict_proba(stacked.reshape(-1, 1))[:, 1]
        result.final_evaluations = {
            'average': averaged,
            'stacking': self.evaluator.evaluate_predictions(final.y, (stacked > .5).astype(int), stacked),
            'calibrated_stacking': self.evaluator.evaluate_predictions(
                final.y, (calibrated > .5).astype(int), calibrated)}
        result.ensemble_accuracy = averaged.accuracy
        self._log(f"Final holdout: {len(final.y)} games, "
                  f"{pd.Timestamp(final.dates[0]).date()} to {pd.Timestamp(final.dates[-1]).date()}")
        self._log(f"{'Model / method':<32} {'Correct / games':>16} {'Accuracy':>10} "
                  f"{'Brier':>10} {'Log loss':>10}")
        self._log('-' * 82)
        rows = [(f'Model {i} ({kind})', evaluated) for i, (kind, evaluated) in
                enumerate(zip(result.model_types, result.final_model_evaluations), 1)]
        rows.extend((('Averaging ensemble', averaged),
                     ('Stacking ensemble', result.final_evaluations['stacking']),
                     ('Calibrated stacking', result.final_evaluations['calibrated_stacking'])))
        for name, evaluated in rows:
            scores = evaluated.metrics()
            correct = int(np.count_nonzero(evaluated.y_pred == evaluated.y_true))
            counts = f"{correct} / {scores['games']}"
            self._log(f"{name:<32} {counts:>16} {scores['accuracy']:>10.2%} "
                      f"{scores['brier']:>10.5f} {scores['log_loss']:>10.5f}")
        mean_accuracy = float(np.mean([item.accuracy for item in result.final_model_evaluations]))
        self._log(f"Final holdout mean of individual accuracies: {mean_accuracy:.2%} "
                  "(arithmetic mean, not an ensemble prediction)")
        self._log(f"Averaging ensemble versus individual mean: "
                  f"{100 * (averaged.accuracy - mean_accuracy):+.2f} percentage points")
        self._log("Accuracy: higher is better. Brier and log loss: lower is better.")
        self._log("Prediction method remains averaging; holdout scores do not auto-select a method.")
        return averaged

    def save(self, result: EnsembleResult, output_dir=None):
        """Save an evaluated model set and its chronological training manifest.

        output_dir supports isolated validation runs without replacing production.
        Existing external weights must be explicitly revalidated, never inherited.
        """
        import hashlib
        import json
        from dataclasses import asdict
        from datetime import datetime, timezone
        from importlib.metadata import version
        from pathlib import Path
        from ..paths import MODELS_DIR

        periods = self._require_partitions(result)
        if set(result.final_evaluations) != {'average', 'stacking', 'calibrated_stacking'}:
            raise ValueError("Evaluate all fitted methods on the final holdout before saving")
        directory = Path(output_dir or MODELS_DIR)
        if (directory / 'ensemble_weights.pkl').exists():
            raise ValueError("External ensemble_weights.pkl must be removed or revalidated before saving a new equal-weight model set")
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = []

        def save_pickle(name, value):
            path = directory / name
            with path.open('wb') as stream:
                pickle.dump(value, stream)
            artifacts.append(path)

        for i, (model, kind) in enumerate(zip(result.models, result.model_types), 1):
            if kind == 'logistic':
                save_pickle(f'nba_ensemble_logistic_{i}.pkl', model)
            elif kind == 'random_forest':
                save_pickle(f'nba_ensemble_rf_{i}.pkl', model)
            elif kind == 'xgboost':
                path = directory / f'nba_ensemble_xgboost_{i}.json'
                try:
                    model.get_booster().save_model(str(path))
                except Exception:
                    model.save_model(str(path))
                artifacts.append(path)
            else:
                path = directory / f'nba_ensemble_model_{i}.keras'
                model.save(str(path))
                artifacts.append(path)
        save_pickle('ensemble_scalers.pkl', result.scalers)
        save_pickle('ensemble_features.pkl', result.feature_cols)
        save_pickle('ensemble_model_types.pkl', result.model_types)
        save_pickle('ensemble_stack_meta_lr.pkl', result.meta_clf)
        save_pickle('ensemble_stack_platt.pkl', result.platt)
        manifest = {
            'protocol': 'chronological-five-period-v1',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'periods': periods.summary(), 'input_sha256': periods.input_sha256,
            'feature_cols': result.feature_cols, 'model_types': result.model_types,
            'config': asdict(self.config),
            'packages': {name: version(name) for name in
                         ('numpy', 'pandas', 'scikit-learn', 'tensorflow', 'xgboost')},
            'prediction_method': 'average', 'threshold': .5,
            'method_selected_using_final_test': False,
            'early_stopping_accuracies': [float(item.accuracy) for item in result.training_results],
            'final_holdout_metrics': {name: value.metrics() for name, value in result.final_evaluations.items()},
            'final_holdout_individual_metrics': [
                {'model_index': i, 'model_type': kind, **evaluated.metrics(),
                 'correct': int(np.count_nonzero(evaluated.y_pred == evaluated.y_true))}
                for i, (kind, evaluated) in
                enumerate(zip(result.model_types, result.final_model_evaluations), 1)],
            'final_holdout_mean_individual_accuracy': float(np.mean(
                [item.accuracy for item in result.final_model_evaluations])),
            'artifact_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifacts}}
        (directory / 'ensemble_training_manifest.json').write_text(
            json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        self._log(f"Ensemble and five-period training manifest saved to {directory}")


def train_ensemble_models(matchup_df: pd.DataFrame,
                         n_models: int = 4,
                         architectures: Optional[List[str]] = None) -> Tuple[List, List, List, List]:
    """
    Backward-compatible wrapper for ensemble training.
    
    This function maintains the original interface while using the new
    refactored implementation internally.
    
    Args:
        matchup_df: Matchup dataframe
        n_models: Number of models (ignored if architectures specified)
        architectures: List of architecture names
        
    Returns:
        Tuple of (models, scalers, feature_cols, model_types)
    """
    if architectures is None:
        architectures = ['xgboost', 'random_forest', 'logistic', 'lstm'][:n_models]
    
    config = EnsembleConfig(architectures=architectures)
    trainer = EnsembleTrainer(config)
    result = trainer.train(matchup_df)
    
    return result.models, result.scalers, result.feature_cols, result.model_types
