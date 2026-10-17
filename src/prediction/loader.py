"""
Model Loader
============
Handles loading trained ensemble models from disk.
Separated from prediction logic for testability.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import pickle
import os


@dataclass  
class LoadedEnsemble:
    """Container for loaded ensemble artifacts"""
    models: List[Any]
    scalers: List[Any]
    feature_cols: List[str]
    model_types: List[str]
    meta_clf: Optional[Any] = None  # Stacking meta-classifier
    platt: Optional[Any] = None  # Platt calibrator
    ensemble_weights: Optional[List[float]] = None
    ensemble_threshold: Optional[float] = None
    
    @property
    def n_models(self) -> int:
        return len(self.models)
    
    def get_model(self, index: int) -> Tuple[Any, Any, str]:
        """Get model, scaler, and type by index"""
        return self.models[index], self.scalers[index], self.model_types[index]


class ModelLoader:
    """
    Loads ensemble models from disk.
    
    Uses dependency injection for file operations to enable testing
    without actual file I/O.
    """
    
    def __init__(self,
                 path_resolver: Optional[Callable[[str], str]] = None,
                 pickle_loader: Optional[Callable] = None):
        """
        Initialize model loader.
        
        Args:
            path_resolver: Function to resolve model paths (injectable for testing)
            pickle_loader: Function to load pickle files (injectable for testing)
        """
        self._path_resolver = path_resolver
        self._pickle_loader = pickle_loader or pickle.load
    
    @property
    def path_resolver(self):
        """Lazy load path resolver"""
        if self._path_resolver is None:
            from ..paths import get_model_path
            self._path_resolver = get_model_path
        return self._path_resolver
    
    def _load_pickle(self, path: str) -> Any:
        """Load a pickle file"""
        with open(path, 'rb') as f:
            return self._pickle_loader(f)
    
    def _load_keras_model(self, path: str) -> Any:
        """Load a Keras model with custom objects"""
        # Apply before TensorFlow's first import; preserve explicit user settings.
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
        # This also sets up the narrow compatibility-warning filters before TF.
        from ..nba_predictor import SumPooling1D
        import tensorflow as tf
        from tensorflow import keras
        
        custom_objects = {
            'SumPooling1D': SumPooling1D,
            'reduce_sum_axis1': lambda x: tf.reduce_sum(x, axis=1)
        }
        
        # Enable unsafe deserialization for Lambda layers
        keras.config.enable_unsafe_deserialization()
        
        return keras.models.load_model(
            path,
            safe_mode=False,
            custom_objects=custom_objects
        )
    
    def _load_xgboost_model(self, path: str) -> Any:
        """Load an XGBoost model"""
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(path)
        return model
    
    def load_model_types(self) -> List[str]:
        """Load the list of model types"""
        from ..paths import ENSEMBLE_TYPES_FILE
        return self._load_pickle(ENSEMBLE_TYPES_FILE)
    
    def load_scalers(self) -> List[Any]:
        """Load scalers"""
        from ..paths import ENSEMBLE_SCALERS_FILE
        return self._load_pickle(ENSEMBLE_SCALERS_FILE)
    
    def load_feature_cols(self) -> List[str]:
        """Load feature column names"""
        from ..paths import ENSEMBLE_FEATURES_FILE
        return self._load_pickle(ENSEMBLE_FEATURES_FILE)
    
    def load_ensemble(self, verbose: bool = True) -> LoadedEnsemble:
        """
        Load the complete ensemble.
        
        Args:
            verbose: Whether to print loading progress
            
        Returns:
            LoadedEnsemble with all models and artifacts
        """
        from ..paths import (
            ENSEMBLE_META_LR_FILE, ENSEMBLE_PLATT_FILE, ENSEMBLE_WEIGHTS_FILE
        )
        
        if verbose:
            print("📥 Loading ensemble models...")
        
        # Load model types first
        model_types = self.load_model_types()
        
        # Load each model
        models = []
        loaded_types = []
        loaded_indices = []
        
        for i, model_type in enumerate(model_types):
            try:
                if model_type == 'xgboost':
                    path = self.path_resolver(f'nba_ensemble_xgboost_{i+1}.json')
                    model = self._load_xgboost_model(path)
                    if verbose:
                        print(f"✓ Loaded XGBoost model {i+1}")
                        
                elif model_type == 'random_forest':
                    path = self.path_resolver(f'nba_ensemble_rf_{i+1}.pkl')
                    model = self._load_pickle(path)
                    if verbose:
                        print(f"✓ Loaded Random Forest model {i+1}")
                        
                elif model_type == 'logistic':
                    path = self.path_resolver(f'nba_ensemble_logistic_{i+1}.pkl')
                    model = self._load_pickle(path)
                    if verbose:
                        print(f"✓ Loaded Logistic Regression model {i+1}")
                        
                else:  # keras
                    path = self.path_resolver(f'nba_ensemble_model_{i+1}.keras')
                    model = self._load_keras_model(path)
                    if verbose:
                        print(f"✓ Loaded Keras model {i+1}")
                
                models.append(model)
                loaded_types.append(model_type)
                loaded_indices.append(i)
                
            except Exception as e:
                if verbose:
                    print(f"⚠ Skipping model {i+1}: {type(e).__name__}")
                continue
        
        # Load scalers (only for loaded models)
        all_scalers = self.load_scalers()
        scalers = [all_scalers[i] for i in loaded_indices]
        
        # Load feature columns
        feature_cols = self.load_feature_cols()
        
        if verbose:
            print(f"✓ Loaded {len(models)} models")
            print(f"✓ Loaded {len(feature_cols)} features")
        
        # Load optional stacking artifacts
        meta_clf = None
        platt = None
        try:
            meta_clf = self._load_pickle(ENSEMBLE_META_LR_FILE)
            platt = self._load_pickle(ENSEMBLE_PLATT_FILE)
            if verbose:
                print('✓ Loaded stacking meta-model')
        except Exception:
            pass
        
        # Load optional ensemble weights
        ensemble_weights = None
        ensemble_threshold = None
        try:
            w = self._load_pickle(ENSEMBLE_WEIGHTS_FILE)
            ensemble_weights = w.get('weights')
            ensemble_threshold = w.get('threshold')
            if verbose:
                print('✓ Loaded ensemble weights')
        except Exception:
            pass
        
        return LoadedEnsemble(
            models=models,
            scalers=scalers,
            feature_cols=feature_cols,
            model_types=loaded_types,
            meta_clf=meta_clf,
            platt=platt,
            ensemble_weights=ensemble_weights,
            ensemble_threshold=ensemble_threshold
        )
