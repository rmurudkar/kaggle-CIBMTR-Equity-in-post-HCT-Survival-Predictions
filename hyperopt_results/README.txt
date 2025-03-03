Gradient Boosting Survival Analysis - Hyperparameter Optimization Results
======================================================================

Best parameters:
n_estimators: 300
learning_rate: 0.033266652862807916
max_depth: 10
min_samples_split: 7
subsample: 0.9566352087990605
max_features: log2
n_iter_no_change: 8
validation_fraction: 0.17499766883832907

Best validation C-index: 0.6802

To use these parameters in your training script, update the model initialization as follows:

model = GradientBoostingSurvivalAnalysis(
    n_estimators=300,
    learning_rate=0.033266652862807916,
    max_depth=10,
    min_samples_split=7,
    subsample=0.9566352087990605,
    max_features=log2,
    n_iter_no_change=8,
    validation_fraction=0.17499766883832907
    random_state=42
)
