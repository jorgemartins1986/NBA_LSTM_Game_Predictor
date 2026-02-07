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

from .data_prep import DataPreparation, TrainTestData, ScaledData
from .trainers import (
    ModelTrainer, TrainerFactory, TrainerConfig, TrainingResult,
    XGBoostConfig, RandomForestConfig, LogisticConfig, LSTMConfig
)
from .evaluation import ModelEvaluator, EvaluationResult


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training"""
    architectures: List[str] = field(default_factory=lambda: ['xgboost', 'random_forest', 'logistic', 'lstm'])
    test_size: float = 0.2
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
        
        # Prepare data once
        data, feature_cols = self.prepare_data(matchup_df)
        
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
        self._log(f"Individual model accuracies:")
        for i, (acc, arch) in enumerate(zip(individual_accs, self.config.architectures)):
            self._log(f"  Model {i+1} ({arch}): {acc*100:.2f}%")
        self._log(f"\nAverage: {np.mean(individual_accs)*100:.2f}%")
        
        return EnsembleResult(
            models=models,
            scalers=scalers,
            feature_cols=feature_cols,
            model_types=model_types,
            training_results=training_results,
            individual_accuracies=individual_accs
        )
    
    def evaluate(self, result: EnsembleResult, matchup_df: pd.DataFrame) -> EvaluationResult:
        """
        Evaluate the ensemble on test data.
        
        Args:
            result: EnsembleResult from training
            matchup_df: Matchup dataframe for evaluation
            
        Returns:
            EvaluationResult with ensemble metrics
        """
        self._log("\n" + "="*70)
        self._log("EVALUATING ENSEMBLE")
        self._log("="*70)
        
        # Get test data
        data, _ = self.prepare_data(matchup_df)
        
        # Get predictions from each model
        all_preds = []
        model_names = []
        
        for model, scaler, model_type in zip(result.models, result.scalers, result.model_types):
            X_test_scaled = scaler.transform(data.X_test)
            
            trainer = self.trainer_factory.create(model_type)
            
            # predict_proba handles feature selection internally
            preds = trainer.predict_proba(model, X_test_scaled)
            all_preds.append(preds)
            model_names.append(model_type)
        
        # Evaluate ensemble
        eval_result = self.evaluator.evaluate_ensemble(
            all_preds, data.y_test, model_names
        )
        
        result.ensemble_accuracy = eval_result.accuracy
        
        self._log(f"\n🎯 ENSEMBLE ACCURACY: {eval_result.accuracy*100:.2f}%")
        self._log(eval_result.classification_report)
        
        return eval_result
    
    def train_stacking(self, result: EnsembleResult, matchup_df: pd.DataFrame) -> EnsembleResult:
        """
        Train stacking meta-model and Platt calibrator on test set predictions.
        
        The stacking approach:
        1. Get each base model's probability predictions on test set
        2. Build meta-features: [raw_probs, confidences]
        3. Train LogisticRegression as meta-classifier
        4. Train Platt calibrator on meta-classifier output
        
        Args:
            result: EnsembleResult with trained base models
            matchup_df: Matchup dataframe (same as used for training)
            
        Returns:
            Updated EnsembleResult with meta_clf and platt
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        
        self._log("\n" + "="*70)
        self._log("TRAINING STACKING META-MODEL")
        self._log("="*70)
        
        # Get test data
        data, _ = self.prepare_data(matchup_df)
        
        # Collect predictions from each base model on test set
        all_test_preds = []
        
        for model, scaler, model_type in zip(result.models, result.scalers, result.model_types):
            X_test_scaled = scaler.transform(data.X_test)
            
            trainer = self.trainer_factory.create(model_type)
            preds = trainer.predict_proba(model, X_test_scaled)
            all_test_preds.append(preds)
        
        # Build meta-features: [raw_probs, confidences]
        raw_probs = np.column_stack(all_test_preds)  # Shape: (n_test, n_models)
        confidences = np.abs(raw_probs - 0.5)  # Distance from 50%
        
        meta_X = np.hstack([raw_probs, confidences])  # Shape: (n_test, 2*n_models)
        meta_y = data.y_test
        
        self._log(f"✓ Built meta-features: {meta_X.shape}")
        
        # Train meta-classifier (logistic regression)
        meta_clf = LogisticRegression(
            C=1.0, 
            class_weight='balanced',
            max_iter=1000,
            random_state=self.config.random_state
        )
        meta_clf.fit(meta_X, meta_y)
        
        # Get meta-classifier predictions for Platt calibration
        meta_probs = meta_clf.predict_proba(meta_X)[:, 1]
        
        # Train Platt calibrator (another logistic regression on meta-probs)
        platt = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.config.random_state
        )
        platt.fit(meta_probs.reshape(-1, 1), meta_y)
        
        # Evaluate stacking
        stacked_preds = platt.predict_proba(meta_probs.reshape(-1, 1))[:, 1]
        stacked_accuracy = np.mean((stacked_preds >= 0.5) == meta_y)
        
        self._log(f"✓ Trained meta-classifier")
        self._log(f"✓ Trained Platt calibrator")
        self._log(f"🎯 STACKED ACCURACY: {stacked_accuracy*100:.2f}%")
        
        # Update result with stacking artifacts
        result.meta_clf = meta_clf
        result.platt = platt
        
        return result
    
    def save(self, result: EnsembleResult):
        """
        Save the trained ensemble to disk.
        
        Args:
            result: EnsembleResult from training
        """
        from ..paths import (
            get_model_path, ENSEMBLE_SCALERS_FILE, 
            ENSEMBLE_FEATURES_FILE, ENSEMBLE_TYPES_FILE,
            ENSEMBLE_META_LR_FILE, ENSEMBLE_PLATT_FILE
        )
        
        self._log("\n💾 Saving ensemble...")
        
        for i, (model, model_type) in enumerate(zip(result.models, result.model_types)):
            if model_type == 'logistic':
                with open(get_model_path(f'nba_ensemble_logistic_{i+1}.pkl'), 'wb') as f:
                    pickle.dump(model, f)
                self._log(f"✓ Saved Logistic Regression model {i+1}")
            elif model_type == 'xgboost':
                try:
                    booster = model.get_booster()
                    booster.save_model(get_model_path(f'nba_ensemble_xgboost_{i+1}.json'))
                    self._log(f"✓ Saved XGBoost Booster model {i+1}")
                except Exception:
                    try:
                        model.save_model(get_model_path(f'nba_ensemble_xgboost_{i+1}.json'))
                        self._log(f"✓ Saved XGBoost model {i+1}")
                    except Exception as e:
                        self._log(f"❌ Failed to save XGBoost model {i+1}: {e}")
            elif model_type == 'random_forest':
                with open(get_model_path(f'nba_ensemble_rf_{i+1}.pkl'), 'wb') as f:
                    pickle.dump(model, f)
                self._log(f"✓ Saved Random Forest model {i+1}")
            else:  # keras
                model.save(get_model_path(f'nba_ensemble_model_{i+1}.keras'))
                self._log(f"✓ Saved Keras model {i+1}")
        
        with open(ENSEMBLE_SCALERS_FILE, 'wb') as f:
            pickle.dump(result.scalers, f)
        
        with open(ENSEMBLE_FEATURES_FILE, 'wb') as f:
            pickle.dump(result.feature_cols, f)
        
        with open(ENSEMBLE_TYPES_FILE, 'wb') as f:
            pickle.dump(result.model_types, f)
        
        # Save stacking artifacts if trained
        if result.meta_clf is not None:
            with open(ENSEMBLE_META_LR_FILE, 'wb') as f:
                pickle.dump(result.meta_clf, f)
            self._log("✓ Saved stacking meta-classifier")
        
        if result.platt is not None:
            with open(ENSEMBLE_PLATT_FILE, 'wb') as f:
                pickle.dump(result.platt, f)
            self._log("✓ Saved Platt calibrator")
        
        self._log("✓ Ensemble saved successfully")


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
