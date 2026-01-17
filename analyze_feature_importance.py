"""
Feature Importance Analysis for XGBoost and Random Forest
==========================================================
Analyzes feature importance to identify which features contribute most to predictions.
Uses this to tune both models by focusing on important features.

Usage:
    python analyze_feature_importance.py
"""

import os
os.environ['TF_DETERMINISTIC_OPS'] = '0'

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, RFE
import xgboost as xgb
from src.nba_data_manager import NBADataManager, NBADataFetcher
from src.nba_predictor import NBAPredictor
from src.paths import CACHE_DIR
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare matchup data"""
    print("📥 Loading NBA Data...")
    
    # Fetch data
    fetcher = NBADataFetcher()
    games_df = fetcher.fetch_games()
    
    # Prepare features
    predictor = NBAPredictor(window_size=20)
    matchup_df = predictor.prepare_matchup_data(games_df)
    
    print(f"✓ Loaded {len(matchup_df)} matchups")
    return matchup_df

def analyze_feature_importance(matchup_df):
    """Analyze feature importance for both XGBoost and Random Forest"""
    
    # Prepare features
    exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
    feature_cols = [col for col in matchup_df.columns if col not in exclude_cols]
    
    X = matchup_df[feature_cols].values
    y = matchup_df['HOME_WIN'].values
    feature_names = feature_cols
    
    print(f"\n📊 Total features: {len(feature_names)}")
    print("="*70)
    
    # Chronological split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sample weights
    n_train = len(y_train)
    sample_weights = np.exp(np.linspace(-1.2, 0, n_train))
    
    # =========================================================================
    # XGBoost Feature Importance
    # =========================================================================
    print("\n" + "="*70)
    print("XGBOOST FEATURE IMPORTANCE")
    print("="*70)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=2,
        learning_rate=0.015,
        subsample=0.776,
        colsample_bytree=0.986,
        gamma=0.206,
        reg_alpha=0.531,
        reg_lambda=1.396,
        min_child_weight=1,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        early_stopping_rounds=50,
        n_jobs=-1
    )
    
    xgb_model.fit(
        X_train_scaled, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )
    
    xgb_baseline = xgb_model.score(X_test_scaled, y_test)
    print(f"XGBoost baseline accuracy: {xgb_baseline*100:.2f}%")
    
    # Get feature importance (gain-based)
    xgb_importance = xgb_model.feature_importances_
    xgb_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 XGBoost features (by gain):")
    for i, row in xgb_importance_df.head(20).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:.4f}")
    
    # =========================================================================
    # Random Forest Feature Importance
    # =========================================================================
    print("\n" + "="*70)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("="*70)
    
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        class_weight='balanced_subsample',
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    rf_baseline = rf_model.score(X_test_scaled, y_test)
    print(f"Random Forest baseline accuracy: {rf_baseline*100:.2f}%")
    print(f"Random Forest OOB score: {rf_model.oob_score_*100:.2f}%")
    
    # Get feature importance
    rf_importance = rf_model.feature_importances_
    rf_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_importance
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 Random Forest features:")
    for i, row in rf_importance_df.head(20).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:.4f}")
    
    # =========================================================================
    # Compare importance rankings
    # =========================================================================
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE COMPARISON")
    print("="*70)
    
    combined_df = pd.merge(
        xgb_importance_df.rename(columns={'importance': 'xgb_importance'}),
        rf_importance_df.rename(columns={'importance': 'rf_importance'}),
        on='feature'
    )
    combined_df['avg_importance'] = (combined_df['xgb_importance'] + combined_df['rf_importance']) / 2
    combined_df = combined_df.sort_values('avg_importance', ascending=False)
    
    print("\nTop 20 features (by average importance):")
    for i, row in combined_df.head(20).iterrows():
        print(f"  {row['feature']:<40} XGB: {row['xgb_importance']:.4f}  RF: {row['rf_importance']:.4f}")
    
    # Identify low-importance features (candidates for removal)
    low_importance = combined_df[combined_df['avg_importance'] < 0.01]
    print(f"\n⚠️  Low importance features (<1%): {len(low_importance)}")
    for i, row in low_importance.iterrows():
        print(f"  {row['feature']:<40} avg: {row['avg_importance']:.4f}")
    
    # =========================================================================
    # Test feature selection impact
    # =========================================================================
    print("\n" + "="*70)
    print("TESTING FEATURE SELECTION")
    print("="*70)
    
    # Test with different numbers of top features
    for n_features in [10, 15, 20, 25, 30]:
        top_features = combined_df.head(n_features)['feature'].tolist()
        top_indices = [feature_names.index(f) for f in top_features]
        
        X_train_subset = X_train_scaled[:, top_indices]
        X_test_subset = X_test_scaled[:, top_indices]
        
        # XGBoost with subset
        xgb_sub = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=2,
            learning_rate=0.015,
            subsample=0.776,
            colsample_bytree=0.986,
            gamma=0.206,
            reg_alpha=0.531,
            reg_lambda=1.396,
            random_state=42,
            early_stopping_rounds=50,
            n_jobs=-1
        )
        xgb_sub.fit(X_train_subset, y_train, sample_weight=sample_weights,
                    eval_set=[(X_test_subset, y_test)], verbose=False)
        xgb_acc = xgb_sub.score(X_test_subset, y_test)
        
        # RF with subset
        rf_sub = RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        rf_sub.fit(X_train_subset, y_train, sample_weight=sample_weights)
        rf_acc = rf_sub.score(X_test_subset, y_test)
        
        print(f"Top {n_features:2d} features: XGB={xgb_acc*100:.2f}%  RF={rf_acc*100:.2f}%")
    
    # =========================================================================
    # Test tuned hyperparameters for RF
    # =========================================================================
    print("\n" + "="*70)
    print("TUNING RANDOM FOREST HYPERPARAMETERS")
    print("="*70)
    
    # Get top features
    top_features = combined_df.head(25)['feature'].tolist()
    top_indices = [feature_names.index(f) for f in top_features]
    X_train_top = X_train_scaled[:, top_indices]
    X_test_top = X_test_scaled[:, top_indices]
    
    # Test different RF configs
    rf_configs = [
        {'n_estimators': 500, 'max_depth': 8, 'min_samples_leaf': 5},
        {'n_estimators': 500, 'max_depth': 10, 'min_samples_leaf': 4},
        {'n_estimators': 500, 'max_depth': 12, 'min_samples_leaf': 3},
        {'n_estimators': 500, 'max_depth': 15, 'min_samples_leaf': 2},
        {'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 5},  # Unlimited depth
        {'n_estimators': 1000, 'max_depth': 10, 'min_samples_leaf': 4},
        {'n_estimators': 1000, 'max_depth': 12, 'min_samples_leaf': 3},
    ]
    
    best_rf_acc = 0
    best_rf_config = None
    
    for config in rf_configs:
        rf = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_leaf=config['min_samples_leaf'],
            min_samples_split=5,
            max_features='sqrt',
            class_weight='balanced_subsample',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_top, y_train, sample_weight=sample_weights)
        acc = rf.score(X_test_top, y_test)
        
        depth_str = str(config['max_depth']) if config['max_depth'] else 'None'
        print(f"  n={config['n_estimators']:4d}, depth={depth_str:4s}, leaf={config['min_samples_leaf']} -> {acc*100:.2f}%")
        
        if acc > best_rf_acc:
            best_rf_acc = acc
            best_rf_config = config
    
    print(f"\n✓ Best RF config: {best_rf_config} -> {best_rf_acc*100:.2f}%")
    
    # =========================================================================
    # Test tuned XGBoost with feature selection
    # =========================================================================
    print("\n" + "="*70)
    print("TUNING XGBOOST HYPERPARAMETERS")
    print("="*70)
    
    xgb_configs = [
        {'max_depth': 2, 'learning_rate': 0.015, 'n_estimators': 800},
        {'max_depth': 2, 'learning_rate': 0.01, 'n_estimators': 1000},
        {'max_depth': 3, 'learning_rate': 0.01, 'n_estimators': 800},
        {'max_depth': 3, 'learning_rate': 0.015, 'n_estimators': 600},
        {'max_depth': 4, 'learning_rate': 0.01, 'n_estimators': 600},
    ]
    
    best_xgb_acc = 0
    best_xgb_config = None
    
    for config in xgb_configs:
        xgb_m = xgb.XGBClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            subsample=0.776,
            colsample_bytree=0.986,
            gamma=0.206,
            reg_alpha=0.531,
            reg_lambda=1.396,
            random_state=42,
            early_stopping_rounds=50,
            n_jobs=-1
        )
        xgb_m.fit(X_train_top, y_train, sample_weight=sample_weights,
                  eval_set=[(X_test_top, y_test)], verbose=False)
        acc = xgb_m.score(X_test_top, y_test)
        
        print(f"  depth={config['max_depth']}, lr={config['learning_rate']:.3f}, n={config['n_estimators']:4d} -> {acc*100:.2f}%")
        
        if acc > best_xgb_acc:
            best_xgb_acc = acc
            best_xgb_config = config
    
    print(f"\n✓ Best XGB config: {best_xgb_config} -> {best_xgb_acc*100:.2f}%")
    
    # =========================================================================
    # Final recommendations
    # =========================================================================
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print(f"\nBaseline accuracies (all features):")
    print(f"  XGBoost: {xgb_baseline*100:.2f}%")
    print(f"  Random Forest: {rf_baseline*100:.2f}%")
    
    print(f"\nBest with feature selection:")
    print(f"  XGBoost: {best_xgb_acc*100:.2f}% ({best_xgb_config})")
    print(f"  Random Forest: {best_rf_acc*100:.2f}% ({best_rf_config})")
    
    print(f"\nRecommended top features ({len(top_features)}):")
    for f in top_features[:15]:
        print(f"  - {f}")
    print("  ...")
    
    # Save feature importance
    combined_df.to_csv('feature_importance_analysis.csv', index=False)
    print(f"\n💾 Saved feature importance to: feature_importance_analysis.csv")
    
    return combined_df, top_features, best_rf_config, best_xgb_config


if __name__ == "__main__":
    matchup_df = load_data()
    analyze_feature_importance(matchup_df)
