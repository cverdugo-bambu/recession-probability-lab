# Probability Learning Path (Applied to This Project)

## Module 1: Binary events and Bernoulli thinking
- Question: recession in next 6 months, yes or no.
- Your model predicts a probability between 0 and 1.
- Practice: compare predicted probability to actual outcomes in `artifacts/predictions_test.csv`.

## Module 2: Base rates and class imbalance
- Recessions are rare, so naive accuracy can be misleading.
- Track `positive_rate` in `metrics.json`.
- Practice: test different thresholds in the dashboard and see precision/recall tradeoffs.

## Module 3: Proper scoring rules
- Brier score: mean squared error on probabilities.
- Log loss: heavier penalty for confident wrong predictions.
- Practice: optimize model settings for lower Brier score, not only higher AUC.

## Module 4: Calibration
- If model says 30%, events should happen around 30% of the time in similar bins.
- Practice: use `artifacts/calibration.csv` to plot reliability curves.

## Module 5: Conditional probability with features
- Learn how each indicator shifts recession odds.
- In logistic regression, coefficients move log-odds.
- Practice: inspect `feature_importance.csv` and interpret sign and magnitude.

## Module 6: Forecast horizon sensitivity
- 3-month horizon and 12-month horizon are different probability tasks.
- Practice: rerun pipeline with different `HORIZON_MONTHS` and compare metrics.

## Suggested next experiments
1. Add lagged versions of each feature.
2. Compare logistic regression with gradient boosting.
3. Run walk-forward backtests instead of one static train/test split.
4. Evaluate model stability by decade.
