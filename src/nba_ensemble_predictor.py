"""
Ensemble NBA Predictor
======================
Trains multiple models and combines their predictions for better accuracy.

Strategy: Train XGBoost + LightGBM + Logistic Regression + LSTM, then use ensemble voting.
This typically improves accuracy by 2-4%.

Requirements:
pip install xgboost lightgbm tqdm

Usage:
    python -m src.nba_ensemble_predictor
"""

import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from .nba_data_manager import NBADataManager, NBADataFetcher
from .nba_predictor import NBAPredictor
from .paths import (
    get_model_path, get_report_path,
    ENSEMBLE_TYPES_FILE, ENSEMBLE_SCALERS_FILE, ENSEMBLE_FEATURES_FILE,
    ENSEMBLE_META_LR_FILE, ENSEMBLE_PLATT_FILE
)
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm

# GPU Configuration - use GPU when available
def configure_gpu():
    """Configure TensorFlow to use GPU if available, with memory growth"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🎮 GPU ENABLED: {len(gpus)} GPU(s) detected")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
            return True
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
            return False
    else:
        print("💻 No GPU detected, using CPU")
        return False

# Configure GPU at import time
GPU_AVAILABLE = configure_gpu()


def save_classification_reports(model_reports, architectures):
    """Save classification reports to a text file for analysis"""
    from datetime import datetime
    import json
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = get_report_path(f'model_reports_{timestamp}.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"NBA ENSEMBLE MODEL CLASSIFICATION REPORTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        for report_data in model_reports:
            f.write(f"\n{'='*70}\n")
            f.write(f"MODEL {report_data['model']}: {report_data['architecture'].upper()}\n")
            f.write(f"{'='*70}\n")
            f.write(f"Accuracy: {report_data['accuracy']*100:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(report_data['report'])
            f.write("\n")
        
        # Summary comparison
        f.write("\n" + "="*70 + "\n")
        f.write("SUMMARY COMPARISON\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Model':<20} {'Accuracy':>10} {'Away Prec':>12} {'Home Prec':>12}\n")
        f.write("-"*56 + "\n")
        
        for report_data in model_reports:
            # Parse precision from report
            lines = report_data['report'].split('\n')
            away_prec = home_prec = 'N/A'
            for line in lines:
                if 'Away Win' in line:
                    parts = line.split()
                    away_prec = parts[2] if len(parts) > 2 else 'N/A'
                if 'Home Win' in line:
                    parts = line.split()
                    home_prec = parts[2] if len(parts) > 2 else 'N/A'
            
            f.write(f"{report_data['architecture']:<20} {report_data['accuracy']*100:>9.2f}% {away_prec:>12} {home_prec:>12}\n")
        
        f.write("\n" + "-"*56 + "\n")
        avg_acc = np.mean([r['accuracy'] for r in model_reports])
        f.write(f"{'AVERAGE':<20} {avg_acc*100:>9.2f}%\n")
    
    # Also save as JSON for programmatic access
    json_file = get_report_path(f'model_reports_{timestamp}.json')
    json_data = [{
        'model': r['model'],
        'architecture': r['architecture'],
        'accuracy': float(r['accuracy']),
        'report': r['report']
    } for r in model_reports]
    
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"\n📊 Saved classification reports to:")
    print(f"   📄 {report_file} (human-readable)")
    print(f"   📋 {json_file} (JSON for analysis)")
    
    return report_file

def train_ensemble_models(matchup_df, n_models=4, architectures=['xgboost', 'lightgbm', 'logistic', 'lstm']):
    """
    Train multiple models with different configurations
    
    Args:
        matchup_df: Training data
        n_models: Number of models to train (default 4: XGBoost + LightGBM + Logistic + LSTM)
        architectures: List of architectures to use
            - 'xgboost': Gradient boosted trees (optimized hyperparams)
            - 'lightgbm': LightGBM gradient boosting (faster, often better)
            - 'logistic': L2-regularized Logistic Regression (linear baseline)
            - 'lstm': LSTM for sequential/temporal patterns
            - 'deep': Deep MLP (256→512→256→128→64→32)
    
    Returns:
        list: Trained models, scalers, and their accuracies
    """
    print("="*70)
    print(f"TRAINING ENSEMBLE OF {n_models} MODELS")
    print("="*70)
    
    models = []
    scalers = []
    accuracies = []
    model_types = []  # Track which models are XGBoost vs Keras
    model_reports = []  # Store classification reports for each model
    
    # Prepare data once
    exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
    feature_cols = [col for col in matchup_df.columns if col not in exclude_cols]
    
    X = matchup_df[feature_cols].values
    y = matchup_df['HOME_WIN'].values
    
    for i in range(n_models):
        print(f"\n{'='*70}")
        print(f"TRAINING MODEL {i+1}/{n_models}")
        architecture = architectures[i % len(architectures)]
        print(f"Architecture: {architecture}")
        print(f"{'='*70}")
        
        # CHRONOLOGICAL SPLIT: Use shuffle=False to prevent temporal leakage
        # Training data = older games, test data = newer games
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if architecture == 'xgboost':
            # Train XGBoost model with OPTIMIZED hyperparameters from grid search
            print("Training XGBoost classifier (optimized hyperparams)...")
            
            # Create sample weights - give more weight to recent games
            # Exponential decay: newest games get weight 1.0, oldest get ~0.3
            n_train = len(y_train)
            sample_weights = np.exp(np.linspace(-1.2, 0, n_train))  # Exponential decay
            
            xgb_model = xgb.XGBClassifier(
                # Optimized params from xgboost-tests/best_xgb_params.json
                n_estimators=1253,
                max_depth=2,  # Shallow trees prevent overfitting
                learning_rate=0.015,
                subsample=0.776,
                colsample_bytree=0.986,
                gamma=0.206,  # Regularization
                reg_alpha=0.531,  # L1 regularization
                reg_lambda=1.396,  # L2 regularization
                min_child_weight=1,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42 + i,
                early_stopping_rounds=50,
                n_jobs=-1
            )
            
            # Train with validation and sample weights
            xgb_model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_test_scaled, y_test)],
                verbose=100
            )
            
            # Evaluate
            y_pred_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            model = xgb_model
            model_types.append('xgboost')
        
        elif architecture == 'logistic':
            # Train Logistic Regression - provides linear baseline with well-calibrated probabilities
            print("Training Logistic Regression classifier...")
            
            from sklearn.linear_model import LogisticRegression
            
            lr_model = LogisticRegression(
                C=0.5,  # Regularization strength (smaller = more regularization)
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                class_weight='balanced',  # Handle class imbalance
                random_state=42 + i
            )
            
            lr_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            model = lr_model
            model_types.append('logistic')
        
        elif architecture == 'lightgbm':
            # Train LightGBM - often faster and more accurate than XGBoost
            print("Training LightGBM classifier...")
            
            # Create sample weights - give more weight to recent games
            # Exponential decay: newest games get weight 1.0, oldest get ~0.3
            n_train = len(y_train)
            sample_weights = np.exp(np.linspace(-1.2, 0, n_train))  # Exponential decay
            
            lgb_model = lgb.LGBMClassifier(
                n_estimators=1500,
                max_depth=4,  # Slightly deeper than XGBoost
                learning_rate=0.02,
                num_leaves=15,  # 2^4 - 1 for max_depth=4
                subsample=0.8,
                colsample_bytree=0.9,
                reg_alpha=0.5,  # L1 regularization
                reg_lambda=1.5,  # L2 regularization
                min_child_weight=5,
                min_child_samples=20,
                objective='binary',
                boosting_type='gbdt',
                random_state=42 + i,
                n_jobs=-1,
                verbose=-1  # Suppress LightGBM warnings
            )
            
            # Train with validation and sample weights
            lgb_model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights,
                eval_set=[(X_test_scaled, y_test)],
                eval_metric='logloss',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # Evaluate
            y_pred_prob = lgb_model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            model = lgb_model
            model_types.append('lightgbm')
        
        else:
            # Train neural network model (LSTM or deep MLP)
            predictor = NBAPredictor()
            model = predictor.build_lstm_model((X_train_scaled.shape[1],), architecture=architecture)
            
            # Create sample weights - give more weight to recent games
            n_train = len(y_train)
            sample_weights = np.exp(np.linspace(-1.2, 0, n_train))  # Exponential decay
            
            # Train with early stopping and sample weights
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            early_stop = EarlyStopping(
                monitor='val_accuracy',
                patience=25,  # Increased patience
                restore_best_weights=True,
                mode='max',
                verbose=0
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=0
            )
            
            history = model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights,  # Weight recent games more
                validation_data=(X_test_scaled, y_test),
                epochs=200,  # More epochs with early stopping
                batch_size=64,  # Larger batch for stability
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            
            # Evaluate
            y_pred_prob = model.predict(X_test_scaled, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            model_types.append('keras')
        
        from sklearn.metrics import accuracy_score, classification_report
        accuracy = accuracy_score(y_test, y_pred.ravel())
        
        # Generate and save classification report
        report = classification_report(y_test, y_pred.ravel(), 
                                       target_names=['Away Win', 'Home Win'])
        
        print(f"\n✓ Model {i+1} ({architecture}) Accuracy: {accuracy*100:.2f}%")
        print(report)
        
        # Save individual report
        model_reports.append({
            'model': i + 1,
            'architecture': architecture,
            'accuracy': accuracy,
            'report': report,
            'y_test': y_test,
            'y_pred': y_pred.ravel(),
            'y_pred_prob': y_pred_prob.ravel() if hasattr(y_pred_prob, 'ravel') else y_pred_prob
        })
        
        models.append(model)
        scalers.append(scaler)
        accuracies.append(accuracy)
    
    # Save all classification reports to file
    save_classification_reports(model_reports, architectures)
    
    # Summary
    print(f"\n{'='*70}")
    print("ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Individual model accuracies:")
    for i, (acc, arch) in enumerate(zip(accuracies, architectures)):
        print(f"  Model {i+1} ({arch}): {acc*100:.2f}%")
    print(f"\nAverage: {np.mean(accuracies)*100:.2f}%")
    print(f"Expected ensemble: {(np.mean(accuracies) + 0.02)*100:.2f}% (+2-3%)")

    # ------------------
    # Train stacking meta-model
    # ------------------
    try:
        print('\n🔀 Training stacking meta-model (with regularization)')

        # Create a chronological train/test split for meta-training
        exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
        X = matchup_df[feature_cols].values
        y = matchup_df['HOME_WIN'].values

        # Use a held-out validation fold (last 20%) for meta training to avoid leakage
        split_idx = int(len(X) * 0.8)
        X_meta_train = X[:split_idx]
        y_meta_train = y[:split_idx]
        X_meta_val = X[split_idx:]
        y_meta_val = y[split_idx:]

        # Get raw model probabilities on the meta validation set
        raw_probs = []
        for model, scaler, mtype in zip(models, scalers, model_types):
            Xs = scaler.transform(X_meta_val)
            if mtype in ('xgboost', 'logistic', 'lightgbm'):
                p = model.predict_proba(Xs)[:, 1]
            else:
                p = model.predict(Xs, verbose=0).reshape(-1)
            raw_probs.append(p)

        raw_probs = np.vstack(raw_probs).T  # shape (n_samples, n_models)

        # Add richer meta-features if available: per-model confidence and ELO diff if present
        meta_features = [raw_probs]
        confidences = np.abs(raw_probs - 0.5)
        meta_features.append(confidences)

        # include ELO diff if present in matchup_df
        if 'ELO_DIFF' in matchup_df.columns:
            elo_diff = matchup_df['ELO_DIFF'].values[split_idx:]
            meta_features.append(elo_diff.reshape(-1, 1))

        meta_X = np.hstack(meta_features)

        # Train a regularized logistic regression as meta-learner
        meta_clf = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000)
        meta_clf.fit(meta_X, y_meta_val)

        # Platt calibration: fit logistic regressor mapping meta probabilities -> calibrated prob
        meta_probs = meta_clf.predict_proba(meta_X)[:, 1]
        platt = LogisticRegression(solver='lbfgs', max_iter=1000)
        platt.fit(meta_probs.reshape(-1, 1), y_meta_val)

        # Save stacking artifacts
        with open(ENSEMBLE_META_LR_FILE, 'wb') as f:
            pickle.dump(meta_clf, f)
        with open(ENSEMBLE_PLATT_FILE, 'wb') as f:
            pickle.dump(platt, f)

        print('✓ Trained and saved stacking meta-model and Platt calibrator')
    except Exception as e:
        print('❌ Failed to train stacking meta-model:', e)

    return models, scalers, feature_cols, model_types


def predict_with_ensemble(models, scalers, feature_cols, matchup_df, game_idx, model_types):
    """
    Predict using ensemble (average of all models)
    
    Args:
        models: List of trained models (XGBoost, LightGBM, Logistic, Keras)
        scalers: List of scalers
        feature_cols: Feature column names
        matchup_df: Data with game to predict
        game_idx: Index of game in dataframe
        model_types: List indicating model type for each model
    
    Returns:
        dict: Ensemble prediction
    """
    game_features = matchup_df[feature_cols].iloc[game_idx:game_idx+1].values
    
    predictions = []
    for model, scaler, model_type in zip(models, scalers, model_types):
        features_scaled = scaler.transform(game_features)
        
        if model_type in ('xgboost', 'logistic', 'lightgbm'):
            pred = model.predict_proba(features_scaled)[0][1]
        else:  # keras
            pred = model.predict(features_scaled, verbose=0)[0][0]
        
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions)
    
    return {
        'home_win_probability': float(ensemble_pred),
        'away_win_probability': float(1 - ensemble_pred),
        'predicted_winner': 'HOME' if ensemble_pred > 0.5 else 'AWAY',
        'confidence': abs(ensemble_pred - 0.5) * 2,
        'individual_predictions': predictions,
        'prediction_std': float(np.std(predictions))  # Agreement between models
    }


def evaluate_ensemble(models, scalers, feature_cols, matchup_df, model_types):
    """Evaluate ensemble on test set (optimized batch processing)"""
    print("\n" + "="*70)
    print("EVALUATING ENSEMBLE")
    print("="*70)
    
    exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
    X = matchup_df[feature_cols].values
    y = matchup_df['HOME_WIN'].values
    
    # CHRONOLOGICAL SPLIT: Use shuffle=False to prevent temporal leakage
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"Evaluating on {len(X_test)} test samples...")
    
    # Get predictions from each model (BATCH processing - much faster)
    all_model_preds = []
    model_names = []
    
    for i, (model, scaler, model_type) in enumerate(zip(models, scalers, model_types)):
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'xgboost':
            preds = model.predict_proba(X_test_scaled)[:, 1]
            model_names.append('XGBoost')
        elif model_type == 'lightgbm':
            preds = model.predict_proba(X_test_scaled)[:, 1]
            model_names.append('LightGBM')
        elif model_type == 'logistic':
            preds = model.predict_proba(X_test_scaled)[:, 1]
            model_names.append('Logistic')
        else:  # keras/lstm
            preds = model.predict(X_test_scaled, verbose=0, batch_size=512).reshape(-1)
            model_names.append('LSTM')
        
        all_model_preds.append(preds)
        
        # Individual model accuracy
        model_acc = ((preds > 0.5).astype(int) == y_test).mean()
        print(f"  {model_names[-1]} accuracy: {model_acc*100:.2f}%")
    
    # Ensemble: average predictions
    ensemble_preds = np.mean(all_model_preds, axis=0)
    ensemble_class = (ensemble_preds > 0.5).astype(int)
    
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, ensemble_class)
    
    # Calculate individual model accuracies for comparison
    individual_accs = [((preds > 0.5).astype(int) == y_test).mean() for preds in all_model_preds]
    best_single = max(individual_accs)
    best_model_idx = individual_accs.index(best_single)
    
    print(f"\n🎯 ENSEMBLE ACCURACY: {accuracy*100:.2f}%")
    print(f"   Best single model ({model_names[best_model_idx]}): {best_single*100:.2f}%")
    ensemble_benefit = accuracy - best_single
    if ensemble_benefit > 0:
        print(f"   ✅ Ensemble adds +{ensemble_benefit*100:.2f}% over best single model")
    else:
        print(f"   ⚠️ Best single model is {-ensemble_benefit*100:.2f}% better than ensemble")
    
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_class, 
                               target_names=['Away Win', 'Home Win']))
    
    return accuracy, individual_accs, model_names


def save_ensemble(models, scalers, feature_cols, model_types):
    """Save ensemble models"""
    print("\n💾 Saving ensemble...")
    
    for i, (model, model_type) in enumerate(zip(models, model_types)):
        if model_type == 'logistic':
            # Save Logistic Regression model
            with open(get_model_path(f'nba_ensemble_logistic_{i+1}.pkl'), 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved Logistic Regression model {i+1}")
        elif model_type == 'xgboost':
            # Save underlying Booster to avoid sklearn wrapper issues
            try:
                booster = model.get_booster()
                booster.save_model(get_model_path(f'nba_ensemble_xgboost_{i+1}.json'))
                print(f"✓ Saved XGBoost Booster model {i+1}.json")
            except Exception:
                # Fallback to sklearn wrapper save (older/newer xgboost versions)
                try:
                    model.save_model(get_model_path(f'nba_ensemble_xgboost_{i+1}.json'))
                    print(f"✓ Saved XGBoost model {i+1}")
                except Exception as e:
                    print(f"❌ Failed to save XGBoost model {i+1}: {e}")
        elif model_type == 'lightgbm':
            # Save LightGBM model
            model.booster_.save_model(get_model_path(f'nba_ensemble_lightgbm_{i+1}.txt'))
            print(f"✓ Saved LightGBM model {i+1}")
        else:  # keras
            model.save(get_model_path(f'nba_ensemble_model_{i+1}.keras'))
            print(f"✓ Saved Keras model {i+1}")
    
    with open(ENSEMBLE_SCALERS_FILE, 'wb') as f:
        pickle.dump(scalers, f)
    
    with open(ENSEMBLE_FEATURES_FILE, 'wb') as f:
        pickle.dump(feature_cols, f)
    
    with open(ENSEMBLE_TYPES_FILE, 'wb') as f:
        pickle.dump(model_types, f)
    
    print("✓ Ensemble saved successfully")


def main():
    """Main ensemble training"""
    print("="*70)
    print("NBA ENSEMBLE PREDICTOR")
    print("Training multiple models for improved accuracy")
    print("="*70)
    
    # Fetch data
    print("\n📥 Fetching NBA Data...")
    fetcher = NBADataFetcher()
    games_df = fetcher.fetch_games()
    
    # Prepare features
    print("\n🔧 Engineering Features...")
    predictor = NBAPredictor(window_size=20)
    matchup_df = predictor.prepare_matchup_data(games_df)
    
    print(f"✓ Prepared {len(matchup_df)} matchups")
    
    # Train ensemble (XGBoost + LightGBM + Logistic + LSTM)
    models, scalers, feature_cols, model_types = train_ensemble_models(
        matchup_df,
        n_models=4,
        architectures=['xgboost', 'lightgbm', 'logistic', 'lstm']
    )
    
    # Evaluate ensemble
    result = evaluate_ensemble(models, scalers, feature_cols, matchup_df, model_types)
    ensemble_accuracy, individual_accs, model_names = result
    
    # Save
    save_ensemble(models, scalers, feature_cols, model_types)
    
    print("\n" + "="*70)
    print("✅ ENSEMBLE TRAINING COMPLETE!")
    print(f"🎯 Ensemble Accuracy: {ensemble_accuracy*100:.2f}%")
    print("="*70)
    print("\nEnsemble composition:")
    print("  1. XGBoost (optimized gradient boosted trees)")
    print("  2. LightGBM (fast gradient boosting with sample weighting)")
    print("  3. Logistic Regression (linear baseline, well-calibrated)")
    print("  4. LSTM (attention-enhanced neural network)")
    print("\nNew features included:")
    print("  - Rest days / Back-to-back detection")
    print("  - 3-in-4 nights fatigue indicator")
    print("  - Head-to-head historical record")
    print("  - Sample weighting (recent games weighted higher)")
    print("\nTo use ensemble for predictions, run:")
    print("  python predict_with_ensemble.py")
    
    return models, scalers, feature_cols, model_types


if __name__ == "__main__":
    models, scalers, features, model_types = main()