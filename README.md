# Recession Probability Lab

A Python-first research project to estimate near-term U.S. recession risk using public data and learn probability in a practical setting.

## What this project does
- Pulls historical macro indicators from FRED plus market indicators from Yahoo Finance and Fama-French factors.
- Builds a Florida Stage 1 stress index (0-100) from Florida labor, housing, and claims data.
- Builds an early-warning target: recession in the next N months.
- Trains multiple probabilistic models (logistic regression, histogram gradient boosting, Bayesian dynamic model with uncertainty bands, NGBoost, and ensemble).
- Runs a Markov switching regime detector (calm/watch/stress) on the model risk signal.
- Stores an algorithm catalog in `metrics.json` with plain-language descriptions of what each algorithm does.
- Runs walk-forward backtests to reduce one-split bias.
- Generates a recession episode review to inspect misses (including the 2008 period) and compare with the current regime (2026).
- Produces visuals in a Streamlit dashboard.

## Project structure
- `src/recession_project/data.py`: data ingestion + feature engineering.
- `src/recession_project/model.py`: model training + evaluation.
- `src/recession_project/pipeline.py`: end-to-end run and artifact saving.
- `app/dashboard.py`: interactive visuals.
- `docs/probability_path.md`: probability study guide tied to the code.
- `docs/non_finance_guide.md`: plain-language guide for non-finance learners.
- `docs/us_florida_stage_plan.md`: roadmap for expanding from U.S. model to Florida pilot.

## Quickstart
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Build dataset and train model:

```bash
python run_pipeline.py
```

4. Launch dashboard:

```bash
python -m streamlit run app/dashboard.py
```

## Outputs
After running the pipeline, artifacts are saved in `artifacts/`:
- `metrics.json`
- `model_metrics.csv`
- `predictions_test.csv`
- `predictions_train.csv`
- `feature_importance.csv`
- `backtest_walkforward.csv`
- `episode_review.csv`
- `period_comparison.csv`
- `probability_history.csv`
- `florida_stage1_index.csv`
- `florida_stage1_latest.json`
- `markov_regimes.csv`
- `markov_summary.json`
- `model.pkl`
- `dataset_snapshot.csv`
- `latest_nowcast.json`

## Notes
- This is a learning and research project, not investment or policy advice.
- You can adjust forecast horizon and test window in `src/recession_project/config.py`.
- Yahoo Finance can rate-limit requests. The pipeline still runs using FRED + Fama-French data when that happens.
