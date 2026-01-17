"""
Random Forest Hyperparameter Tuning
====================================
Comprehensive search for optimal RF hyperparameters.
"""

import os
os.environ['TF_DETERMINISTIC_OPS'] = '0'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from src.nba_data_manager import NBADataFetcher
from src.nba_predictor import NBAPredictor
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare matchup data"""
    print("📥 Loading NBA Data...")
    fetcher = NBADataFetcher()
    games_df = fetcher.fetch_games()
    predictor = NBAPredictor(window_size=20)
    matchup_df = predictor.prepare_matchup_data(games_df)
    print(f"✓ Loaded {len(matchup_df)} matchups")
    return matchup_df

def tune_rf(matchup_df):
    """Comprehensive RF hyperparameter search"""
    
    # Prepare features
    exclude_cols = ['HOME_WIN', 'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ID', 'AWAY_TEAM_ID']
    feature_cols = [col for col in matchup_df.columns if col not in exclude_cols]
    
    X = matchup_df[feature_cols].values
    y = matchup_df['HOME_WIN'].values
    
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
    
    # First: Get feature importance for feature selection
    print("\n📊 Computing feature importance for selection...")
    prelim_rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    prelim_rf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    feature_importance = prelim_rf.feature_importances_
    
    # =========================================================================
    # Test different feature counts
    # =========================================================================
    print("\n" + "="*70)
    print("TESTING FEATURE SELECTION COUNTS")
    print("="*70)
    
    for n_features in [15, 20, 25, 30, 40, 50, 96]:  # 96 = all features
        top_indices = np.argsort(feature_importance)[::-1][:n_features]
        X_train_sub = X_train_scaled[:, top_indices]
        X_test_sub = X_test_scaled[:, top_indices]
        
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=5,
            bootstrap=True, oob_score=True,
            random_state=42, n_jobs=-1
        )
        rf.fit(X_train_sub, y_train, sample_weight=sample_weights)
        acc = rf.score(X_test_sub, y_test)
        oob = rf.oob_score_ if hasattr(rf, 'oob_score_') else 0
        print(f"  Top {n_features:3d} features: {acc*100:.2f}%  (OOB: {oob*100:.2f}%)")
    
    # Use best feature count from above (typically 30-50)
    best_n_features = 40
    top_indices = np.argsort(feature_importance)[::-1][:best_n_features]
    X_train_top = X_train_scaled[:, top_indices]
    X_test_top = X_test_scaled[:, top_indices]
    
    print(f"\n✓ Using top {best_n_features} features for hyperparameter search")
    
    # =========================================================================
    # Grid search over RF hyperparameters
    # =========================================================================
    print("\n" + "="*70)
    print("HYPERPARAMETER GRID SEARCH")
    print("="*70)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [300, 500, 800],
        'max_depth': [8, 10, 12, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 3, 5, 8],
        'max_features': ['sqrt', 'log2', 0.3],
        'class_weight': ['balanced', 'balanced_subsample', None],
    }
    
    # Simplified grid for faster search
    param_grid_fast = {
        'n_estimators': [500, 800],
        'max_depth': [10, 15, None],
        'min_samples_leaf': [3, 5, 8],
        'max_features': ['sqrt', 0.3],
        'class_weight': ['balanced_subsample', None],
    }
    
    results = []
    
    total_combos = 1
    for v in param_grid_fast.values():
        total_combos *= len(v)
    
    print(f"Testing {total_combos} hyperparameter combinations...\n")
    
    combo_idx = 0
    for n_est in param_grid_fast['n_estimators']:
        for max_d in param_grid_fast['max_depth']:
            for min_leaf in param_grid_fast['min_samples_leaf']:
                for max_feat in param_grid_fast['max_features']:
                    for cw in param_grid_fast['class_weight']:
                        combo_idx += 1
                        
                        rf = RandomForestClassifier(
                            n_estimators=n_est,
                            max_depth=max_d,
                            min_samples_split=5,
                            min_samples_leaf=min_leaf,
                            max_features=max_feat,
                            class_weight=cw,
                            bootstrap=True,
                            oob_score=True,
                            random_state=42,
                            n_jobs=-1
                        )
                        
                        rf.fit(X_train_top, y_train, sample_weight=sample_weights)
                        test_acc = rf.score(X_test_top, y_test)
                        oob_acc = rf.oob_score_
                        
                        results.append({
                            'n_estimators': n_est,
                            'max_depth': max_d,
                            'min_samples_leaf': min_leaf,
                            'max_features': max_feat,
                            'class_weight': cw,
                            'test_acc': test_acc,
                            'oob_acc': oob_acc
                        })
                        
                        if combo_idx % 10 == 0:
                            print(f"  Progress: {combo_idx}/{total_combos} ({combo_idx*100/total_combos:.0f}%)")
    
    # Sort by test accuracy
    results_df = pd.DataFrame(results).sort_values('test_acc', ascending=False)
    
    print("\n" + "="*70)
    print("TOP 15 CONFIGURATIONS")
    print("="*70)
    print(f"\n{'Rank':<5} {'n_est':<6} {'depth':<6} {'leaf':<5} {'features':<10} {'class_wt':<18} {'Test':<8} {'OOB':<8}")
    print("-"*70)
    
    for i, row in results_df.head(15).iterrows():
        depth_str = str(row['max_depth']) if row['max_depth'] else 'None'
        cw_str = str(row['class_weight']) if row['class_weight'] else 'None'
        print(f"{results_df.index.get_loc(i)+1:<5} {row['n_estimators']:<6} {depth_str:<6} {row['min_samples_leaf']:<5} {row['max_features']:<10} {cw_str:<18} {row['test_acc']*100:.2f}%   {row['oob_acc']*100:.2f}%")
    
    # Best config
    best = results_df.iloc[0]
    print("\n" + "="*70)
    print("BEST CONFIGURATION")
    print("="*70)
    print(f"""
    n_estimators: {best['n_estimators']}
    max_depth: {best['max_depth']}
    min_samples_leaf: {best['min_samples_leaf']}
    max_features: {best['max_features']}
    class_weight: {best['class_weight']}
    
    Test Accuracy: {best['test_acc']*100:.2f}%
    OOB Accuracy: {best['oob_acc']*100:.2f}%
    """)
    
    # =========================================================================
    # Test with/without sample weights
    # =========================================================================
    print("\n" + "="*70)
    print("TESTING SAMPLE WEIGHTS IMPACT")
    print("="*70)
    
    # Convert max_depth from pandas (nan -> None)
    best_max_depth = None if pd.isna(best['max_depth']) else int(best['max_depth'])
    
    # Without sample weights
    rf_no_weights = RandomForestClassifier(
        n_estimators=int(best['n_estimators']),
        max_depth=best_max_depth,
        min_samples_leaf=int(best['min_samples_leaf']),
        max_features=best['max_features'],
        class_weight=best['class_weight'] if pd.notna(best['class_weight']) else None,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf_no_weights.fit(X_train_top, y_train)  # No sample_weight
    acc_no_weights = rf_no_weights.score(X_test_top, y_test)
    
    # With sample weights (already computed above)
    rf_with_weights = RandomForestClassifier(
        n_estimators=int(best['n_estimators']),
        max_depth=best_max_depth,
        min_samples_leaf=int(best['min_samples_leaf']),
        max_features=best['max_features'],
        class_weight=best['class_weight'] if pd.notna(best['class_weight']) else None,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf_with_weights.fit(X_train_top, y_train, sample_weight=sample_weights)
    acc_with_weights = rf_with_weights.score(X_test_top, y_test)
    
    print(f"  Without sample weights: {acc_no_weights*100:.2f}%")
    print(f"  With sample weights:    {acc_with_weights*100:.2f}%")
    
    # =========================================================================
    # Test criterion (gini vs entropy)
    # =========================================================================
    print("\n" + "="*70)
    print("TESTING SPLIT CRITERION")
    print("="*70)
    
    for criterion in ['gini', 'entropy', 'log_loss']:
        rf = RandomForestClassifier(
            n_estimators=int(best['n_estimators']),
            max_depth=best_max_depth,
            min_samples_leaf=int(best['min_samples_leaf']),
            max_features=best['max_features'],
            class_weight=best['class_weight'] if pd.notna(best['class_weight']) else None,
            criterion=criterion,
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train_top, y_train, sample_weight=sample_weights)
        acc = rf.score(X_test_top, y_test)
        print(f"  {criterion:<10}: {acc*100:.2f}%")
    
    # Save results
    results_df.to_csv('rf_tuning_results.csv', index=False)
    print(f"\n💾 Saved all results to: rf_tuning_results.csv")
    
    return best


if __name__ == "__main__":
    matchup_df = load_data()
    best_config = tune_rf(matchup_df)
