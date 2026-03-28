"""
Generate publication-quality visualizations for TB Drug Resistance Prediction
IIT Kanpur Computational Genomics Project
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")
sns.set_context("paper", font_scale=1.2)

# Directories
FIGURES_DIR = Path('figures')
RESULTS_DIR = Path('results')

# Ensure directories exist
FIGURES_DIR.mkdir(exist_ok=True)

# Drug information
DRUGS = ['RIF', 'INH', 'PZA', 'EMB', 'STR', 'CIP', 'CAP', 'AMK', 'MOXI', 'OFLX', 'KAN']
DRUG_NAMES = {
    'RIF': 'Rifampicin',
    'INH': 'Isoniazid',
    'PZA': 'Pyrazinamide',
    'EMB': 'Ethambutol',
    'STR': 'Streptomycin',
    'CIP': 'Ciprofloxacin',
    'CAP': 'Capreomycin',
    'AMK': 'Amikacin',
    'MOXI': 'Moxifloxacin',
    'OFLX': 'Ofloxacin',
    'KAN': 'Kanamycin'
}


def load_results():
    """Load model results from JSON."""
    with open(RESULTS_DIR / 'model_results.json', 'r') as f:
        data = json.load(f)
    return data['drug_results'], data['class_distributions']


def figure_1_class_imbalance(drug_results, class_distributions):
    """Figure 1: Class Imbalance Overview"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data
    drugs = []
    susceptible = []
    resistant = []
    not_tested = []
    ratios = []

    for drug in DRUGS:
        dist = class_distributions[drug]
        n_sus = dist.get('0', 0)
        n_res = dist.get('1', 0)
        n_not = dist.get('-1', 0)
        ratio = n_sus / n_res if n_res > 0 else 0

        drugs.append(drug)
        susceptible.append(n_sus)
        resistant.append(n_res)
        not_tested.append(n_not)
        ratios.append(ratio)

    # Sort by total labeled samples
    total_labeled = [susceptible[i] + resistant[i] for i in range(len(drugs))]
    sorted_indices = np.argsort(total_labeled)[::-1]

    drugs = [drugs[i] for i in sorted_indices]
    susceptible = [susceptible[i] for i in sorted_indices]
    resistant = [resistant[i] for i in sorted_indices]
    not_tested = [not_tested[i] for i in sorted_indices]
    ratios = [ratios[i] for i in sorted_indices]

    x = np.arange(len(drugs))
    width = 0.35

    # Create bars
    bars1 = ax.barh(x - width/2, susceptible, width, label='Susceptible (0)', color='#2E86AB')
    bars2 = ax.barh(x + width/2, resistant, width, label='Resistant (1)', color='#A23B72')

    # Add imbalance ratio annotations
    for i, (drug, ratio) in enumerate(zip(drugs, ratios)):
        total = susceptible[i] + resistant[i]
        ax.text(max(susceptible[i], resistant[i]) + 50, i, f'ratio: {ratio:.2f}',
                va='center', fontsize=9, color='gray')

    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Drug', fontsize=12)
    ax.set_title('Class Distribution Across 10 Anti-TB Drugs', fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels([f"{drug}\n({DRUG_NAMES[drug]})" for drug in drugs])
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(max(susceptible), max(resistant)) + 300)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '01_class_imbalance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 01_class_imbalance.png")


def figure_2_auroc_heatmap(drug_results):
    """Figure 2: AUROC Heatmap"""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Prepare data
    models = ['XGBoost', 'RandomForest', 'LogisticRegression', 'Ensemble']
    auroc_data = []

    for model in models:
        row = []
        for drug in DRUGS:
            auroc = drug_results[drug]['models'][model]['auroc']['mean']
            row.append(auroc)
        auroc_data.append(row)

    auroc_data = np.array(auroc_data)

    # Create heatmap
    cmap = sns.diverging_palette(10, 133, s=85, l=55, n=9, center="light", as_cmap=True)
    im = ax.imshow(auroc_data, cmap=cmap, aspect='auto', vmin=0.7, vmax=1.0)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(DRUGS)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(DRUGS)
    ax.set_yticklabels(models)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    for i in range(len(models)):
        for j in range(len(DRUGS)):
            text = ax.text(j, i, f'{auroc_data[i, j]:.3f}',
                          ha="center", va="center", color="black" if auroc_data[i, j] > 0.85 else "white",
                          fontsize=9)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("AUROC", rotation=-90, va="bottom")

    ax.set_title("5-Fold Cross-Validated AUROC by Drug and Model", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Drug", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '02_auroc_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 02_auroc_heatmap.png")


def figure_3_roc_curves(drug_results):
    """Figure 3: ROC Curves for Yatin's Drugs (OFLX, AMK, INH)"""
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    drugs_target = ['OFLX', 'AMK', 'INH']
    drug_context = {
        'OFLX': 'Ofloxacin (Fluoroquinolone, 2nd-line)',
        'AMK': 'Amikacin (Injectable, 2nd-line)',
        'INH': 'Isoniazid (1st-line, katG mutations)'
    }

    # Load data
    df_X = pd.read_csv('X_trainData_1.csv')
    df_y = pd.read_csv('Y_trainData_1.csv')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for idx, drug in enumerate(drugs_target):
        ax = axes[idx]

        # Prepare data
        res = pd.concat([df_X, df_y[[drug]]], axis=1)
        res = res[res[drug] != -1]
        y = res[drug].values
        X = res.drop(columns=[drug]).values

        # Get imbalance ratio
        n_sus = np.sum(y == 0)
        n_res = np.sum(y == 1)
        imbalance_ratio = n_sus / n_res if n_res > 0 else 1.0

        # Models
        models = {
            'XGBoost': xgb.XGBClassifier(
                learning_rate=0.1, max_depth=6, n_estimators=200,
                subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
                random_state=42, scale_pos_weight=imbalance_ratio
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=500, max_depth=20, class_weight='balanced',
                random_state=42, n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0, class_weight='balanced', solver='saga',
                max_iter=1000, random_state=42
            )
        }

        colors = {'XGBoost': '#E63946', 'RandomForest': '#2A9D8F', 'LogisticRegression': '#457B9D', 'Ensemble': '#F4A261'}

        for name, model in models.items():
            # Cross-validated predictions
            y_pred_proba = np.zeros(len(y))
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model.fit(X_train, y_train)
                y_pred_proba[val_idx] = model.predict_proba(X_val)[:, 1]

            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            auc = drug_results[drug]['models'][name]['auroc']['mean']
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', color=colors[name], linewidth=2)

        # Plot Ensemble
        ensemble = VotingClassifier(
            estimators=[('xgb', models['XGBoost']), ('rf', models['RandomForest']), ('lr', models['LogisticRegression'])],
            voting='soft'
        )
        y_pred_proba = np.zeros(len(y))
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            ensemble.fit(X_train, y_train)
            y_pred_proba[val_idx] = ensemble.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        auc = drug_results[drug]['models']['Ensemble']['auroc']['mean']
        ax.plot(fpr, tpr, label=f'Ensemble (AUC={auc:.3f})', color=colors['Ensemble'], linewidth=2.5, linestyle='--')

        # Mark operating point at sensitivity=0.90
        ax.axvline(x=0.10, color='gray', linestyle=':', alpha=0.7, label='Sens=0.90')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate', fontsize=10)
        ax.set_ylabel('True Positive Rate', fontsize=10)
        ax.set_title(drug_context[drug], fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlim([-0.02, 1.0])
        ax.set_ylim([0.0, 1.02])

    plt.suptitle('ROC Curves — Yatin\'s Drug Contributions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '03_roc_curves_yatin.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 03_roc_curves_yatin.png")


def figure_4_sensitivity_specificity(drug_results):
    """Figure 4: Sensitivity vs Specificity Tradeoff"""
    fig, ax = plt.subplots(figsize=(12, 8))

    models = ['XGBoost', 'RandomForest', 'LogisticRegression', 'Ensemble']
    colors = {'XGBoost': '#E63946', 'RandomForest': '#2A9D8F', 'LogisticRegression': '#457B9D', 'Ensemble': '#F4A261'}
    markers = {'RIF': 'o', 'INH': 's', 'PZA': '^', 'EMB': 'D', 'STR': 'v',
               'CIP': 'p', 'CAP': '*', 'AMK': 'h', 'MOXI': '+', 'OFLX': 'x', 'KAN': 'X'}

    for drug in DRUGS:
        for model in models:
            sens = drug_results[drug]['models'][model]['sensitivity']['mean']
            spec = drug_results[drug]['models'][model]['specificity']['mean']
            ax.scatter(spec, sens, c=colors[model], marker=markers[drug], s=100, alpha=0.8,
                      edgecolors='black', linewidth=0.5)

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')

    # Shade clinical sweet spot
    ax.axhspan(0.85, 1.0, xmin=0.7, xmax=1.0, alpha=0.2, color='green', label='Clinical Sweet Spot (Sens>0.85, Spec>0.70)')

    # Create custom legends
    from matplotlib.lines import Line2D
    model_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[m], markersize=10, label=m)
                    for m in models]
    drug_legend = [Line2D([0], [0], marker=markers[d], color='w', markerfacecolor='gray', markersize=8, label=d)
                   for d in DRUGS]

    legend1 = ax.legend(handles=model_legend, loc='lower left', title='Models', framealpha=0.9)
    ax.add_artist(legend1)
    ax.legend(handles=drug_legend, loc='upper right', title='Drugs', framealpha=0.9, ncol=2)

    ax.set_xlabel('Specificity', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.set_title('Sensitivity-Specificity Tradeoff Across All Drug-Model Combinations', fontsize=14, fontweight='bold')
    ax.set_xlim([0.3, 1.0])
    ax.set_ylim([0.7, 1.0])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '04_sensitivity_specificity.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 04_sensitivity_specificity.png")


def figure_5_feature_importance(drug_results):
    """Figure 5: Feature Importance for OFLX, AMK, INH"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    drugs_target = ['OFLX', 'AMK', 'INH']

    for idx, drug in enumerate(drugs_target):
        ax = axes[idx]

        if 'top_features' in drug_results[drug]:
            features = drug_results[drug]['top_features'][:15]
            feature_names = [f['feature'][:30] for f in features]  # Truncate long names
            importances = [f['importance'] for f in features]

            # Create horizontal bar chart with color gradient
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importances)))[::-1]
            bars = ax.barh(range(len(importances)), importances, color=colors)

            ax.set_yticks(range(len(importances)))
            ax.set_yticklabels(feature_names, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance', fontsize=10)
            ax.set_title(f'Top 15 Genomic Predictors — {drug}', fontsize=11, fontweight='bold')

            # Add note
            ax.text(0.02, -0.12, 'Features are binary SNP/indel variants',
                   transform=ax.transAxes, fontsize=8, style='italic', color='gray')
        else:
            ax.text(0.5, 0.5, 'No feature importance data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{drug} - Feature Importance', fontsize=11, fontweight='bold')

    plt.suptitle('XGBoost Feature Importance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '05_feature_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 05_feature_importance.png")


def figure_6_model_comparison(drug_results):
    """Figure 6: Model Comparison Summary"""
    fig, ax = plt.subplots(figsize=(14, 7))

    models = ['XGBoost', 'RandomForest', 'LogisticRegression', 'Ensemble']
    colors = {'XGBoost': '#E63946', 'RandomForest': '#2A9D8F', 'LogisticRegression': '#457B9D', 'Ensemble': '#F4A261'}

    x = np.arange(len(DRUGS))
    width = 0.2

    for i, model in enumerate(models):
        aurocs = [drug_results[drug]['models'][model]['auroc']['mean'] for drug in DRUGS]
        stds = [drug_results[drug]['models'][model]['auroc']['std'] for drug in DRUGS]
        offset = (i - 1.5) * width
        ax.bar(x + offset, aurocs, width, yerr=stds, label=model, color=colors[model],
               capsize=3, alpha=0.85, edgecolor='black', linewidth=0.5)

    # Add clinical utility threshold
    ax.axhline(y=0.85, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Clinical threshold (0.85)')

    ax.set_xlabel('Drug', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Model Performance Comparison Across Anti-TB Drugs (5-Fold CV)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(DRUGS, rotation=45, ha='right')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_ylim([0.85, 1.0])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '06_model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 06_model_comparison.png")


def figure_7_confusion_matrices():
    """Figure 7: Confusion Matrix Grid"""
    from sklearn.model_selection import StratifiedKFold
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    import xgboost as xgb

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))

    drugs_target = ['OFLX', 'AMK', 'INH']
    models = ['XGBoost', 'RandomForest', 'LogisticRegression', 'Ensemble']

    # Load data
    df_X = pd.read_csv('X_trainData_1.csv')
    df_y = pd.read_csv('Y_trainData_1.csv')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for i, model_name in enumerate(models):
        for j, drug in enumerate(drugs_target):
            ax = axes[i, j]

            # Prepare data
            res = pd.concat([df_X, df_y[[drug]]], axis=1)
            res = res[res[drug] != -1]
            y = res[drug].values
            X = res.drop(columns=[drug]).values

            # Get imbalance ratio
            n_sus = np.sum(y == 0)
            n_res = np.sum(y == 1)
            imbalance_ratio = n_sus / n_res if n_res > 0 else 1.0

            # Get model
            if model_name == 'XGBoost':
                model = xgb.XGBClassifier(
                    learning_rate=0.1, max_depth=6, n_estimators=200,
                    subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
                    random_state=42, scale_pos_weight=imbalance_ratio
                )
            elif model_name == 'RandomForest':
                model = RandomForestClassifier(
                    n_estimators=500, max_depth=20, class_weight='balanced',
                    random_state=42, n_jobs=-1
                )
            elif model_name == 'LogisticRegression':
                model = LogisticRegression(
                    C=1.0, class_weight='balanced', solver='saga',
                    max_iter=1000, random_state=42
                )
            else:  # Ensemble
                model = VotingClassifier(
                    estimators=[
                        ('xgb', xgb.XGBClassifier(
                            learning_rate=0.1, max_depth=6, n_estimators=200,
                            subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
                            random_state=42, scale_pos_weight=imbalance_ratio
                        )),
                        ('rf', RandomForestClassifier(
                            n_estimators=500, max_depth=20, class_weight='balanced',
                            random_state=42, n_jobs=-1
                        )),
                        ('lr', LogisticRegression(
                            C=1.0, class_weight='balanced', solver='saga',
                            max_iter=1000, random_state=42
                        ))
                    ],
                    voting='soft'
                )

            # Cross-validated predictions
            y_pred_all = np.zeros(len(y))
            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model.fit(X_train, y_train)
                y_pred_all[val_idx] = model.predict(X_val)

            # Compute normalized confusion matrix
            cm = confusion_matrix(y, y_pred_all)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Plot
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
            ax.set_title(f'{model_name}\n{drug}', fontsize=10, fontweight='bold')

            # Add text annotations
            thresh = cm_normalized.max() / 2.
            for k in range(cm.shape[0]):
                for l in range(cm.shape[1]):
                    ax.text(l, k, format(cm_normalized[k, l], '.2f'),
                           ha="center", va="center", color="white" if cm_normalized[k, l] > thresh else "black",
                           fontsize=12)

            ax.set_ylabel('True Label', fontsize=9)
            ax.set_xlabel('Predicted Label', fontsize=9)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Susceptible', 'Resistant'], fontsize=8)
            ax.set_yticklabels(['Susceptible', 'Resistant'], fontsize=8)

            # Add sensitivity and specificity annotations
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            ax.text(0.5, -0.25, f'TP Rate: {sens:.2f} | TN Rate: {spec:.2f}',
                   transform=ax.transAxes, ha='center', fontsize=8, style='italic')

    plt.suptitle('Normalised Confusion Matrices — OFLX, AMK, INH', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '07_confusion_matrices.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: 07_confusion_matrices.png")


def main():
    """Main function to generate all visualizations."""
    print("="*60)
    print("Generating Publication-Quality Visualizations")
    print("="*60)

    # Load results
    print("\nLoading results...")
    drug_results, class_distributions = load_results()

    # Generate figures
    print("\nGenerating figures...")
    figure_1_class_imbalance(drug_results, class_distributions)
    figure_2_auroc_heatmap(drug_results)
    figure_3_roc_curves(drug_results)
    figure_4_sensitivity_specificity(drug_results)
    figure_5_feature_importance(drug_results)
    figure_6_model_comparison(drug_results)
    figure_7_confusion_matrices()

    print("\n" + "="*60)
    print("All figures saved to 'figures/' directory")
    print("="*60)


if __name__ == '__main__':
    main()
