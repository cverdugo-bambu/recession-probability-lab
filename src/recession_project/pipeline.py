from __future__ import annotations

import json
from datetime import datetime

import joblib
import pandas as pd

from .analysis import build_episode_review, build_period_comparison
from .config import (
    ARTIFACT_DIR,
    DEFAULT_ALERT_THRESHOLD,
    HORIZON_MONTHS,
    MIN_DATE,
    MIN_TRAIN_MONTHS_WALKFORWARD,
    TEST_MONTHS,
)
from .data import build_inference_features, load_and_prepare
from .florida import build_florida_stage1, fetch_florida_data, save_florida_stage1
from .model import (
    fit_and_evaluate,
    fit_final_models,
    get_algorithm_catalog,
    predict_selected_probability,
    predict_selected_probability_interval,
    walk_forward_backtest,
)
from .regimes import run_markov_switching, save_markov_outputs


def run_pipeline(
    start_date: str = MIN_DATE,
    horizon_months: int = HORIZON_MONTHS,
    test_months: int = TEST_MONTHS,
    min_train_months_walkforward: int = MIN_TRAIN_MONTHS_WALKFORWARD,
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
) -> dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    florida_stage1_latest: dict[str, object]
    try:
        florida_raw = fetch_florida_data(start_date=start_date)
        florida_stage1 = build_florida_stage1(florida_raw)
        save_florida_stage1(florida_stage1, ARTIFACT_DIR)
        florida_stage1_latest = florida_stage1.latest
    except Exception as exc:  # noqa: BLE001
        florida_stage1_latest = {"status": "unavailable", "error": str(exc)}

    bundle = load_and_prepare(start_date=start_date, horizon_months=horizon_months)
    dataset = bundle.features

    result = fit_and_evaluate(dataset, test_months=test_months)
    selected_model = result.selected_model_name

    metrics_by_model: dict[str, dict[str, float]] = {}
    for row in result.model_metrics.to_dict(orient="records"):
        model_name = str(row.pop("model"))
        metrics_by_model[model_name] = {k: float(v) for k, v in row.items()}

    selected_metrics = metrics_by_model[selected_model]
    metrics = {
        "selected_model": selected_model,
        "selected_test_metrics": selected_metrics,
        "all_model_test_metrics": metrics_by_model,
        "algorithm_catalog": get_algorithm_catalog(),
        "horizon_months": horizon_months,
        "test_months": test_months,
        "walkforward_min_train_months": min_train_months_walkforward,
        "alert_threshold": alert_threshold,
        "florida_stage1_latest": florida_stage1_latest,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
    }

    result.train_predictions.to_csv(ARTIFACT_DIR / "predictions_train.csv", index_label="date")
    result.test_predictions.to_csv(ARTIFACT_DIR / "predictions_test.csv", index_label="date")
    result.feature_importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    result.calibration.to_csv(ARTIFACT_DIR / "calibration.csv", index=False)
    result.model_metrics.to_csv(ARTIFACT_DIR / "model_metrics.csv", index=False)
    dataset.to_csv(ARTIFACT_DIR / "dataset_snapshot.csv", index_label="date")

    final_models, feature_cols = fit_final_models(dataset, selected_model_name=selected_model)
    inference_features = build_inference_features(bundle.raw)
    probability_history = pd.DataFrame(index=inference_features.index)
    probability_history.index.name = "date"
    probability_history["y_prob"] = predict_selected_probability(
        fitted_models=final_models,
        x=inference_features[feature_cols],
        selected_model_name=selected_model,
    )
    probability_interval = predict_selected_probability_interval(
        fitted_models=final_models,
        x=inference_features[feature_cols],
        selected_model_name=selected_model,
    )
    if probability_interval is not None:
        probability_history["y_prob_lower"] = probability_interval[0]
        probability_history["y_prob_upper"] = probability_interval[1]
    probability_history["current_recession"] = inference_features["current_recession"].astype(int)
    probability_history.to_csv(ARTIFACT_DIR / "probability_history.csv", index_label="date")

    latest_prob = float(probability_history.iloc[-1]["y_prob"])
    latest_date = pd.to_datetime(bundle.raw.dropna(how="all").index.max()).date().isoformat()
    latest_nowcast = {
        "date": latest_date,
        "recession_probability": latest_prob,
        "selected_model": selected_model,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
    }
    if "y_prob_lower" in probability_history.columns and "y_prob_upper" in probability_history.columns:
        latest_nowcast["recession_probability_lower"] = float(probability_history.iloc[-1]["y_prob_lower"])
        latest_nowcast["recession_probability_upper"] = float(probability_history.iloc[-1]["y_prob_upper"])

    markov_df, markov_summary = run_markov_switching(probability_history["y_prob"], num_regimes=3)
    save_markov_outputs(markov_df, markov_summary, ARTIFACT_DIR)
    metrics["markov_switching_latest"] = markov_summary

    backtest_df = walk_forward_backtest(
        dataset,
        selected_model_name=selected_model,
        min_train_months=min_train_months_walkforward,
    )
    backtest_df.to_csv(ARTIFACT_DIR / "backtest_walkforward.csv", index=False)

    episode_review = build_episode_review(backtest_df, alert_threshold=alert_threshold)
    period_comparison = build_period_comparison(
        probability_history_df=probability_history.reset_index(),
        alert_threshold=alert_threshold,
    )
    episode_review.to_csv(ARTIFACT_DIR / "episode_review.csv", index=False)
    period_comparison.to_csv(ARTIFACT_DIR / "period_comparison.csv", index=False)

    model_package = {
        "selected_model": selected_model,
        "feature_cols": feature_cols,
        "models": final_models,
    }
    joblib.dump(model_package, ARTIFACT_DIR / "model.pkl")

    with (ARTIFACT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (ARTIFACT_DIR / "latest_nowcast.json").open("w", encoding="utf-8") as f:
        json.dump(latest_nowcast, f, indent=2)

    return metrics
