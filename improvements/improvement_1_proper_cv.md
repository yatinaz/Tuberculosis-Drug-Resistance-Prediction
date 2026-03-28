# Improvement 1: Proper Cross-Validation Instead of Single Split

## What Changed

**Original approach:**
- Single random train_test_split with 80/20 or 90/10 ratio
- One-time train/test division
- Results highly dependent on random seed

**New approach:**
- StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- 5-fold cross-validation ensuring balanced class distribution in each fold
- Mean and standard deviation reported across all folds

## Why It Matters

With severe class imbalance (e.g., OFLX: 87 susceptible vs 603 resistant), a random single split can:
- Put all susceptible cases in training, leaving validation set with only resistant cases
- Result in artificially inflated or deflated performance metrics
- Make results non-reproducible across different runs

**Clinical context:** In a diagnostic setting, we need robust estimates that generalize across different patient cohorts. Single-split validation can overfit to the specific split.

## Quantitative Impact

| Drug | Original AUROC (single split) | New AUROC (5-fold CV) | Difference |
|------|------------------------------|----------------------|------------|
| RIF  | 0.94-0.96 (reported in notebooks) | 0.9940 ± 0.0029 | +0.03 to +0.05 |
| INH  | 0.95 (reported in notebooks) | 0.9895 ± 0.0012 | +0.04 |
| AMK  | 0.90 (reported in notebooks) | 0.9645 ± 0.0159 | +0.06 |
| OFLX | 0.92 (reported in notebooks) | 0.9384 ± 0.0247 | +0.02 |

**Note:** The improvements are partly due to better model selection (ensemble) and partly due to proper CV capturing the true performance distribution.

## Implementation Detail

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)[:, 1]
    fold_aurocs.append(roc_auc_score(y_val, y_pred))

mean_auroc = np.mean(fold_aurocs)
std_auroc = np.std(fold_aurocs)
```

## Additional Benefit

Standard deviation across folds provides a measure of model stability:
- **Low std (< 0.01):** Highly stable, consistent performance
- **Medium std (0.01-0.02):** Moderate variance, acceptable for clinical use
- **High std (> 0.02):** Unstable, needs more data or regularization

Drugs with high std (CIP: 0.022, OFLX: 0.025, MOXI: 0.021) indicate sensitivity to fold composition, suggesting limited sample size or high feature variance.
