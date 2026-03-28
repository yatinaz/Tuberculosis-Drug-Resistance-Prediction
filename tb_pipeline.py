"""
TB Drug Resistance Prediction Pipeline
Consolidated script for training and evaluating models across 10 anti-TB drugs.
IIT Kanpur Computational Genomics Project
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, accuracy_score, f1_score, recall_score, precision_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
N_SPLITS = 5
OUTPUT_DIR = 'results'

# Drug information
DRUGS = ['RIF', 'INH', 'PZA', 'EMB', 'STR', 'CIP', 'CAP', 'AMK', 'MOXI', 'OFLX', 'KAN']

# Drug class mapping for clinical context
DRUG_INFO = {
    'RIF': {'class': '1st-line', 'gene': 'rpoB', 'full_name': 'Rifampicin'},
    'INH': {'class': '1st-line', 'gene': 'katG/inhA', 'full_name': 'Isoniazid'},
    'PZA': {'class': '1st-line', 'gene': 'pncA', 'full_name': 'Pyrazinamide'},
    'EMB': {'class': '1st-line', 'gene': 'embB', 'full_name': 'Ethambutol'},
    'STR': {'class': '2nd-line', 'gene': 'rpsL', 'full_name': 'Streptomycin'},
    'CIP': {'class': 'Fluoroquinolone', 'gene': 'gyrA', 'full_name': 'Ciprofloxacin'},
    'CAP': {'class': 'Injectable', 'gene': 'rrs', 'full_name': 'Capreomycin'},
    'AMK': {'class': 'Injectable', 'gene': 'rrs/eis', 'full_name': 'Amikacin'},
    'MOXI': {'class': 'Fluoroquinolone', 'gene': 'gyrA', 'full_name': 'Moxifloxacin'},
    'OFLX': {'class': 'Fluoroquinolone', 'gene': 'gyrA', 'full_name': 'Ofloxacin'},
    'KAN': {'class': 'Injectable', 'gene': 'rrs/eis', 'full_name': 'Kanamycin'},
}


def load_data():
    """Load training data."""
    print("Loading data...")
    df_X = pd.read_csv('X_trainData_1.csv')
    df_y = pd.read_csv('Y_trainData_1.csv')
    print(f"X shape: {df_X.shape}")
    print(f"y shape: {df_y.shape}")
    return df_X, df_y


def get_class_distribution(df_y, drug):
    """Get class distribution for a drug."""
    counts = df_y[drug].value_counts().sort_index()
    distribution = {}
    for label in [-1, 0, 1]:
        distribution[str(label)] = int(counts.get(label, 0))
    return distribution


def prepare_data(df_X, df_y, drug):
    """Prepare data for a specific drug."""
    # Concatenate features and labels
    res = pd.concat([df_X, df_y[[drug]]], axis=1)

    # Filter out rows where label is -1 (not tested)
    res = res[res[drug] != -1]

    # Separate features and labels
    y = res[drug].values
    X = res.drop(columns=[drug]).values

    # Get feature names
    feature_names = res.drop(columns=[drug]).columns.tolist()

    # Calculate class imbalance ratio
    n_susceptible = np.sum(y == 0)
    n_resistant = np.sum(y == 1)
    imbalance_ratio = n_susceptible / n_resistant if n_resistant > 0 else 1.0

    return X, y, feature_names, n_susceptible, n_resistant, imbalance_ratio


# Per-drug tuned hyperparameters for high-variance drugs (MOXI, KAN, OFLX).
# Derived from 81-config XGB grid + 9-config RF grid with 5-fold CV.
# All other drugs use defaults — their performance is already clinical-grade.
DRUG_XGB_PARAMS = {
    'MOXI': {'n_estimators': 500, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8},
    'KAN':  {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8},
    'OFLX': {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 1.0},
}
DRUG_RF_PARAMS = {
    'MOXI': {'n_estimators': 500,  'max_depth': None},
    'KAN':  {'n_estimators': 1000, 'max_depth': 20},
    'OFLX': {'n_estimators': 1000, 'max_depth': None},
}


def create_models(imbalance_ratio, drug=None):
    """Create models with appropriate class weighting.

    Uses per-drug tuned hyperparameters for MOXI, KAN, OFLX;
    falls back to defaults for all other drugs.
    """
    xgb_params = DRUG_XGB_PARAMS.get(drug, {'n_estimators': 200, 'max_depth': 6,
                                             'learning_rate': 0.1, 'subsample': 0.8})
    rf_params = DRUG_RF_PARAMS.get(drug, {'n_estimators': 500, 'max_depth': 20})

    models = {
        'XGBoost': xgb.XGBClassifier(
            **xgb_params,
            random_state=RANDOM_STATE,
            scale_pos_weight=imbalance_ratio
        ),
        'RandomForest': RandomForestClassifier(
            **rf_params,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'LogisticRegression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            solver='saga',
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }

    # Create soft voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', models['XGBoost']),
            ('rf', models['RandomForest']),
            ('lr', models['LogisticRegression'])
        ],
        voting='soft'
    )
    models['Ensemble'] = ensemble

    return models


def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate comprehensive metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Basic metrics
    auroc = roc_auc_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # Confusion matrix for sensitivity/specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for class 1
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for class 0

    return {
        'auroc': auroc,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def find_threshold_for_sensitivity(y_true, y_pred_proba, target_sensitivity=0.90):
    """Find threshold that achieves target sensitivity."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # Find threshold closest to target sensitivity
    idx = np.argmin(np.abs(tpr - target_sensitivity))
    threshold = thresholds[idx]

    # Calculate specificity at this threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return threshold, tpr[idx], specificity


def cross_validate_model(model, X, y, cv):
    """Perform cross-validation and return predictions."""
    y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    return y_pred_proba


def get_feature_importance(model, feature_names):
    """Extract feature importance from trained model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None

    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return importance_df


def train_and_evaluate_drug(df_X, df_y, drug, cv):
    """Train and evaluate all models for a single drug."""
    print(f"\n{'='*60}")
    print(f"Processing {drug} ({DRUG_INFO[drug]['full_name']})")
    print(f"{'='*60}")

    # Prepare data
    X, y, feature_names, n_susceptible, n_resistant, imbalance_ratio = prepare_data(df_X, df_y, drug)

    print(f"Samples: {len(y)}")
    print(f"Susceptible (0): {n_susceptible}")
    print(f"Resistant (1): {n_resistant}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    if len(y) < 50:
        print(f"WARNING: Very few samples for {drug}. Results may be unreliable.")

    # Create models (uses per-drug tuned params for MOXI, KAN, OFLX)
    models = create_models(imbalance_ratio, drug=drug)

    # Store results
    drug_results = {
        'drug': drug,
        'n_samples': len(y),
        'n_susceptible': n_susceptible,
        'n_resistant': n_resistant,
        'imbalance_ratio': imbalance_ratio,
        'models': {}
    }

    # Evaluate each model
    for model_name, model in models.items():
        print(f"\n  Training {model_name}...")

        # Cross-validation
        fold_metrics = []
        fold_aurocs = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model
            model.fit(X_train, y_train)

            # Predict
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            # Calculate metrics
            metrics = calculate_metrics(y_val, y_pred_proba)
            fold_metrics.append(metrics)
            fold_aurocs.append(metrics['auroc'])

        # Aggregate metrics across folds
        aggregated = {}
        for key in ['auroc', 'accuracy', 'f1_macro', 'sensitivity', 'specificity']:
            values = [m[key] for m in fold_metrics]
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }

        # Find threshold for 90% sensitivity
        # Train on full data for threshold search
        model.fit(X, y)
        y_pred_proba_full = model.predict_proba(X)[:, 1]
        threshold, sens_at_thresh, spec_at_thresh = find_threshold_for_sensitivity(
            y, y_pred_proba_full, target_sensitivity=0.90
        )

        aggregated['threshold_90_sens'] = float(threshold)
        aggregated['specificity_at_90_sens'] = float(spec_at_thresh)

        # Feature importance (for XGBoost)
        feature_importance = None
        if model_name == 'XGBoost':
            importance_df = get_feature_importance(model, feature_names)
            if importance_df is not None:
                feature_importance = importance_df.head(20).to_dict('records')
                drug_results['top_features'] = feature_importance

        drug_results['models'][model_name] = aggregated

        print(f"    AUROC: {aggregated['auroc']['mean']:.4f} ± {aggregated['auroc']['std']:.4f}")
        print(f"    Sensitivity: {aggregated['sensitivity']['mean']:.4f}")
        print(f"    Specificity: {aggregated['specificity']['mean']:.4f}")

    return drug_results


def main():
    """Main execution function."""
    print("="*60)
    print("TB Drug Resistance Prediction Pipeline")
    print("IIT Kanpur Computational Genomics")
    print("="*60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df_X, df_y = load_data()

    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Store all results
    all_results = {}
    class_distributions = {}

    # Process each drug
    for drug in DRUGS:
        # Get class distribution
        class_distributions[drug] = get_class_distribution(df_y, drug)

        # Train and evaluate
        drug_results = train_and_evaluate_drug(df_X, df_y, drug, cv)
        all_results[drug] = drug_results

    # Save results
    output_file = os.path.join(OUTPUT_DIR, 'model_results.json')

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    serializable_results = convert_to_serializable({
        'drug_results': all_results,
        'class_distributions': class_distributions
    })

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")

    # Print summary
    print("\nSUMMARY: Best AUROC per drug")
    print("-" * 60)
    for drug in DRUGS:
        best_model = max(all_results[drug]['models'].items(), key=lambda x: x[1]['auroc']['mean'])
        print(f"{drug:5s}: {best_model[0]:20s} AUROC = {best_model[1]['auroc']['mean']:.4f}")


if __name__ == '__main__':
    main()
