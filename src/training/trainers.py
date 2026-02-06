"""
Model Trainers
==============
Individual trainer classes for each model architecture.
Each trainer follows the same interface for consistency and testability.

Design principles:
- Single responsibility: each trainer handles one model type
- Dependency injection: sklearn/xgboost/keras models can be mocked
- Configuration via dataclasses: easy to test with different configs
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Protocol, Union
import numpy as np
from sklearn.preprocessing import StandardScaler

from .data_prep import ScaledData


@dataclass
class TrainerConfig:
    """Base configuration for all trainers"""
    random_state: int = 42
    verbose: bool = True


@dataclass
class XGBoostConfig(TrainerConfig):
    """XGBoost hyperparameters (optimized via grid search)"""
    n_estimators: int = 1253
    max_depth: int = 2
    learning_rate: float = 0.015
    subsample: float = 0.776
    colsample_bytree: float = 0.986
    gamma: float = 0.206
    reg_alpha: float = 0.531
    reg_lambda: float = 1.396
    min_child_weight: int = 1
    early_stopping_rounds: int = 50


@dataclass
class RandomForestConfig(TrainerConfig):
    """Random Forest hyperparameters"""
    n_estimators: int = 500
    max_depth: Optional[int] = None
    min_samples_split: int = 5
    min_samples_leaf: int = 5
    max_features: str = 'sqrt'
    n_top_features: int = 40  # For feature selection


@dataclass
class LogisticConfig(TrainerConfig):
    """Logistic Regression hyperparameters"""
    C: float = 0.5
    penalty: str = 'l2'
    solver: str = 'lbfgs'
    max_iter: int = 1000
    class_weight: Optional[str] = 'balanced'


@dataclass
class LSTMConfig(TrainerConfig):
    """LSTM/Keras model hyperparameters"""
    architecture: str = 'lstm'  # 'lstm' or 'deep'
    epochs: int = 200
    batch_size: int = 64
    early_stopping_patience: int = 25
    reduce_lr_patience: int = 10
    min_lr: float = 1e-6


@dataclass
class TrainingResult:
    """Result of training a single model"""
    model: Any
    scaler: StandardScaler
    accuracy: float
    y_pred: np.ndarray
    y_pred_prob: np.ndarray
    y_test: np.ndarray
    model_type: str
    classification_report: str = ""
    feature_indices: Optional[np.ndarray] = None  # For RF feature selection


class ModelTrainer(ABC):
    """
    Abstract base class for all model trainers.
    
    Defines the interface that all trainers must implement.
    This allows for consistent testing and easy swapping of implementations.
    """
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return string identifier for this model type"""
        pass
    
    @abstractmethod
    def train(self, data: ScaledData, 
              sample_weights: Optional[np.ndarray] = None) -> TrainingResult:
        """
        Train the model on scaled data.
        
        Args:
            data: ScaledData with scaled train/test features and labels
            sample_weights: Optional sample weights for training
            
        Returns:
            TrainingResult with trained model and metrics
        """
        pass
    
    @abstractmethod
    def predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions from trained model.
        
        Args:
            model: Trained model
            X: Features to predict on (already scaled)
            
        Returns:
            Array of probabilities for positive class (home win)
        """
        pass


class XGBoostTrainer(ModelTrainer):
    """Trainer for XGBoost classifier"""
    
    def __init__(self, config: Optional[XGBoostConfig] = None):
        self.config = config or XGBoostConfig()
    
    @property
    def model_type(self) -> str:
        return 'xgboost'
    
    def create_model(self) -> Any:
        """Create and return an XGBoost classifier with configured params"""
        import xgboost as xgb
        
        return xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            gamma=self.config.gamma,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            min_child_weight=self.config.min_child_weight,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=self.config.random_state,
            early_stopping_rounds=self.config.early_stopping_rounds,
            n_jobs=-1
        )
    
    def train(self, data: ScaledData, 
              sample_weights: Optional[np.ndarray] = None) -> TrainingResult:
        """Train XGBoost classifier"""
        from sklearn.metrics import accuracy_score, classification_report
        
        model = self.create_model()
        
        # Train with early stopping
        model.fit(
            data.X_train_scaled, data.y_train,
            sample_weight=sample_weights,
            eval_set=[(data.X_test_scaled, data.y_test)],
            verbose=100 if self.config.verbose else 0
        )
        
        # Evaluate
        y_pred_prob = model.predict_proba(data.X_test_scaled)[:, 1]
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(data.y_test, y_pred)
        report = classification_report(data.y_test, y_pred,
                                       target_names=['Away Win', 'Home Win'])
        
        return TrainingResult(
            model=model,
            scaler=data.scaler,
            accuracy=accuracy,
            y_pred=y_pred,
            y_pred_prob=y_pred_prob,
            y_test=data.y_test,
            model_type=self.model_type,
            classification_report=report
        )
    
    def predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from XGBoost"""
        return model.predict_proba(X)[:, 1]


class RandomForestTrainer(ModelTrainer):
    """Trainer for Random Forest with feature selection"""
    
    def __init__(self, config: Optional[RandomForestConfig] = None):
        self.config = config or RandomForestConfig()
    
    @property
    def model_type(self) -> str:
        return 'random_forest'
    
    def select_features(self, X_train: np.ndarray, y_train: np.ndarray,
                       sample_weights: Optional[np.ndarray] = None) -> Tuple[Any, np.ndarray]:
        """
        Train preliminary RF to select top features.
        
        Returns:
            Tuple of (preliminary model, top feature indices)
        """
        from sklearn.ensemble import RandomForestClassifier
        
        prelim_rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10, 
            random_state=self.config.random_state, 
            n_jobs=-1
        )
        prelim_rf.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Get top feature indices
        importances = prelim_rf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:self.config.n_top_features]
        
        return prelim_rf, top_indices
    
    def create_model(self) -> Any:
        """Create Random Forest classifier"""
        from sklearn.ensemble import RandomForestClassifier
        
        return RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            bootstrap=True,
            oob_score=True,
            class_weight=None,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=0
        )
    
    def train(self, data: ScaledData, 
              sample_weights: Optional[np.ndarray] = None) -> TrainingResult:
        """Train Random Forest with feature selection"""
        from sklearn.metrics import accuracy_score, classification_report
        
        # Step 1: Feature selection
        _, top_indices = self.select_features(
            data.X_train_scaled, data.y_train, sample_weights
        )
        
        # Select top features
        X_train_top = data.X_train_scaled[:, top_indices]
        X_test_top = data.X_test_scaled[:, top_indices]
        
        # Step 2: Train final model
        model = self.create_model()
        model.fit(X_train_top, data.y_train, sample_weight=sample_weights)
        
        # Store feature indices for prediction time
        model._top_feature_indices = top_indices
        
        # Evaluate
        y_pred_prob = model.predict_proba(X_test_top)[:, 1]
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(data.y_test, y_pred)
        report = classification_report(data.y_test, y_pred,
                                       target_names=['Away Win', 'Home Win'])
        
        return TrainingResult(
            model=model,
            scaler=data.scaler,
            accuracy=accuracy,
            y_pred=y_pred,
            y_pred_prob=y_pred_prob,
            y_test=data.y_test,
            model_type=self.model_type,
            classification_report=report,
            feature_indices=top_indices
        )
    
    def predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from Random Forest"""
        # Apply feature selection if applicable
        if hasattr(model, '_top_feature_indices'):
            X = X[:, model._top_feature_indices]
        return model.predict_proba(X)[:, 1]


class LogisticTrainer(ModelTrainer):
    """Trainer for Logistic Regression"""
    
    def __init__(self, config: Optional[LogisticConfig] = None):
        self.config = config or LogisticConfig()
    
    @property
    def model_type(self) -> str:
        return 'logistic'
    
    def create_model(self) -> Any:
        """Create Logistic Regression classifier"""
        from sklearn.linear_model import LogisticRegression
        
        return LogisticRegression(
            C=self.config.C,
            penalty=self.config.penalty,
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state
        )
    
    def train(self, data: ScaledData, 
              sample_weights: Optional[np.ndarray] = None) -> TrainingResult:
        """Train Logistic Regression classifier"""
        from sklearn.metrics import accuracy_score, classification_report
        
        model = self.create_model()
        model.fit(data.X_train_scaled, data.y_train)  # LR doesn't use sample weights in fit
        
        # Evaluate
        y_pred_prob = model.predict_proba(data.X_test_scaled)[:, 1]
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(data.y_test, y_pred)
        report = classification_report(data.y_test, y_pred,
                                       target_names=['Away Win', 'Home Win'])
        
        return TrainingResult(
            model=model,
            scaler=data.scaler,
            accuracy=accuracy,
            y_pred=y_pred,
            y_pred_prob=y_pred_prob,
            y_test=data.y_test,
            model_type=self.model_type,
            classification_report=report
        )
    
    def predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from Logistic Regression"""
        return model.predict_proba(X)[:, 1]


class LSTMTrainer(ModelTrainer):
    """Trainer for LSTM/Keras neural networks"""
    
    def __init__(self, config: Optional[LSTMConfig] = None, 
                 model_builder: Any = None):
        """
        Args:
            config: LSTM configuration
            model_builder: NBAPredictor instance for building models (injectable for testing)
        """
        self.config = config or LSTMConfig()
        self._model_builder = model_builder
    
    @property
    def model_type(self) -> str:
        return 'lstm'
    
    @property
    def model_builder(self):
        """Lazy load model builder to avoid import at module level"""
        if self._model_builder is None:
            from ..nba_predictor import NBAPredictor
            self._model_builder = NBAPredictor()
        return self._model_builder
    
    def create_model(self, input_shape: Tuple[int, ...]) -> Any:
        """Create LSTM or deep MLP model"""
        return self.model_builder.build_lstm_model(
            input_shape, 
            architecture=self.config.architecture
        )
    
    def create_callbacks(self) -> list:
        """Create Keras callbacks for training"""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=self.config.early_stopping_patience,
            restore_best_weights=True,
            mode='max',
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config.reduce_lr_patience,
            min_lr=self.config.min_lr,
            verbose=0
        )
        
        return [early_stop, reduce_lr]
    
    def train(self, data: ScaledData, 
              sample_weights: Optional[np.ndarray] = None) -> TrainingResult:
        """Train LSTM/Keras model"""
        from sklearn.metrics import accuracy_score, classification_report
        
        # Create model
        model = self.create_model((data.X_train_scaled.shape[1],))
        
        # Train
        callbacks = self.create_callbacks()
        model.fit(
            data.X_train_scaled, data.y_train,
            sample_weight=sample_weights,
            validation_data=(data.X_test_scaled, data.y_test),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1 if self.config.verbose else 0
        )
        
        # Evaluate
        y_pred_prob = model.predict(data.X_test_scaled, verbose=0).reshape(-1)
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(data.y_test, y_pred)
        report = classification_report(data.y_test, y_pred,
                                       target_names=['Away Win', 'Home Win'])
        
        return TrainingResult(
            model=model,
            scaler=data.scaler,
            accuracy=accuracy,
            y_pred=y_pred,
            y_pred_prob=y_pred_prob,
            y_test=data.y_test,
            model_type=self.model_type,
            classification_report=report
        )
    
    def predict_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get probability predictions from Keras model"""
        return model.predict(X, verbose=0).reshape(-1)


class TrainerFactory:
    """
    Factory for creating model trainers.
    
    Allows easy swapping of implementations for testing.
    """
    
    _trainers = {
        'xgboost': XGBoostTrainer,
        'random_forest': RandomForestTrainer,
        'logistic': LogisticTrainer,
        'lstm': LSTMTrainer,
        'deep': LSTMTrainer,  # Deep MLP uses same trainer with different config
    }
    
    @classmethod
    def create(cls, architecture: str, config: Optional[TrainerConfig] = None,
               model_builder: Any = None) -> ModelTrainer:
        """
        Create a trainer for the specified architecture.
        
        Args:
            architecture: Model architecture name
            config: Optional configuration (type depends on architecture)
            model_builder: Optional model builder for LSTM (for testing)
            
        Returns:
            ModelTrainer instance
        """
        if architecture not in cls._trainers:
            raise ValueError(f"Unknown architecture: {architecture}. "
                           f"Available: {list(cls._trainers.keys())}")
        
        trainer_cls = cls._trainers[architecture]
        
        # Handle LSTM/deep special case with architecture in config
        if architecture in ('lstm', 'deep'):
            if config is None:
                config = LSTMConfig(architecture=architecture)
            elif isinstance(config, LSTMConfig):
                config.architecture = architecture
            return trainer_cls(config, model_builder)
        
        return trainer_cls(config)
    
    @classmethod
    def register(cls, name: str, trainer_cls: type):
        """Register a new trainer type"""
        cls._trainers[name] = trainer_cls
    
    @classmethod
    def available_architectures(cls) -> List[str]:
        """Get list of available architecture names"""
        return list(cls._trainers.keys())
