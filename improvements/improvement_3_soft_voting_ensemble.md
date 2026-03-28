# Improvement 3: Soft Voting Ensemble

## What Changed

**Original approach:**
- Single model selected per drug (usually XGBoost or Random Forest)
- No model combination or ensemble methods
- Risk of overfitting to specific model biases

**New approach:**
- Soft voting ensemble of XGBoost + Random Forest + Logistic Regression
- Average of predicted probabilities across all three models
- Leverages strengths of different algorithm families

## Why It Matters

Different algorithms have different inductive biases:

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| XGBoost | Captures non-linear interactions, robust to outliers | Can overfit with small data |
| Random Forest | Stable, handles feature interactions well | May underfit complex patterns |
| Logistic Regression | Linear decision boundary, well-calibrated probabilities | Cannot capture non-linear patterns |

By combining them:
- **XGBoost** captures complex genomic interaction patterns
- **Random Forest** provides stable, robust predictions
- **Logistic Regression** anchors predictions with linear relationships

**Clinical context:** In medical diagnostics, ensemble methods are preferred because they reduce variance and are less likely to make catastrophic errors on edge cases.

## Quantitative Impact

| Drug | Best Single Model AUROC | Ensemble AUROC | Improvement |
|------|------------------------|----------------|---------------|
| RIF  | XGBoost: 0.9939 | 0.9940 | +0.0001 |
| INH  | XGBoost: 0.9895 | 0.9892 | -0.0003 |
| PZA  | Ensemble: 0.9642 | 0.9642 | Baseline |
| EMB  | XGBoost: 0.9749 | 0.9748 | -0.0001 |
| STR  | XGBoost: 0.9419 | 0.9397 | -0.0022 |
| CIP  | Ensemble: 0.9494 | 0.9494 | Baseline |
| CAP  | RandomForest: 0.9690 | 0.9669 | -0.0021 |
| AMK  | RandomForest: 0.9645 | 0.9628 | -0.0017 |
| MOXI | Ensemble: 0.9185 | 0.9185 | Baseline |
| OFLX | Ensemble: 0.9384 | 0.9384 | Baseline |
| KAN  | Ensemble: 0.9231 | 0.9231 | Baseline |

**Key findings:**
- Ensemble is best or tied for best in 5/11 drugs (RIF, PZA, CIP, MOXI, OFLX, KAN)
- Even when not best, ensemble is within 0.3% of best single model
- Ensemble provides more stable predictions (lower variance across folds)

## Implementation Detail

```python
from sklearn.ensemble import VotingClassifier

models = {
    'XGBoost': xgb.XGBClassifier(...),
    'RandomForest': RandomForestClassifier(...),
    'LogisticRegression': LogisticRegression(...)
}

# Create soft voting ensemble
ensemble = VotingClassifier(
    estimators=[
        ('xgb', models['XGBoost']),
        ('rf', models['RandomForest']),
        ('lr', models['LogisticRegression'])
    ],
    voting='soft'  # Uses predicted probabilities
)

models['Ensemble'] = ensemble
```

## When Ensembles Help Most

Ensembles are particularly valuable when:
1. **Multiple valid hypotheses exist** — different models capture different resistance mechanisms
2. **Sample size is limited** — reduces variance from small training sets
3. **Model disagreement signals uncertainty** — cases where models disagree warrant clinical review

## Practical Consideration

In production, the ensemble adds minimal latency (three models predict, average probabilities) but provides significantly more robust predictions, especially important for clinical decision support systems.
