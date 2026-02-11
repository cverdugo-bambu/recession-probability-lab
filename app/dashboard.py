from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recession_project.config import DEFAULT_ALERT_THRESHOLD, FEATURE_DESCRIPTIONS, FEATURE_LABELS

ARTIFACT_DIR = ROOT / "artifacts"

MODEL_LABELS = {
    "logistic": "Logistic Regression (simple + interpretable)",
    "hist_gb": "Gradient Boosted Trees (captures nonlinear patterns)",
    "bayesian_dynamic": "Bayesian Dynamic Model (with uncertainty bands)",
    "ngboost": "NGBoost (distribution-aware boosting)",
    "ensemble": "Ensemble (average of logistic + boosted trees)",
    "markov_switching": "Markov Switching (regime change detector)",
}


def _feature_label(raw: str) -> str:
    return FEATURE_LABELS.get(raw, raw.replace("_", " ").title())


def _model_label(raw: str) -> str:
    return MODEL_LABELS.get(raw, raw)


def _load_artifacts() -> tuple[
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict | None,
    dict | None,
    dict | None,
]:
    paths = {
        "metrics": ARTIFACT_DIR / "metrics.json",
        "test": ARTIFACT_DIR / "predictions_test.csv",
        "train": ARTIFACT_DIR / "predictions_train.csv",
        "importance": ARTIFACT_DIR / "feature_importance.csv",
        "snapshot": ARTIFACT_DIR / "dataset_snapshot.csv",
        "model_metrics": ARTIFACT_DIR / "model_metrics.csv",
        "prob_history": ARTIFACT_DIR / "probability_history.csv",
        "backtest": ARTIFACT_DIR / "backtest_walkforward.csv",
        "episode_review": ARTIFACT_DIR / "episode_review.csv",
        "period_compare": ARTIFACT_DIR / "period_comparison.csv",
        "florida_index": ARTIFACT_DIR / "florida_stage1_index.csv",
        "florida_latest": ARTIFACT_DIR / "florida_stage1_latest.json",
        "markov_regimes": ARTIFACT_DIR / "markov_regimes.csv",
        "markov_summary": ARTIFACT_DIR / "markov_summary.json",
        "nowcast": ARTIFACT_DIR / "latest_nowcast.json",
    }

    required = [paths["metrics"], paths["test"], paths["train"], paths["importance"], paths["snapshot"]]
    if not all(p.exists() for p in required):
        missing = [str(p) for p in required if not p.exists()]
        raise FileNotFoundError(
            "Missing artifacts. Run `python run_pipeline.py` first. Missing: " + ", ".join(missing)
        )

    with paths["metrics"].open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    test_df = pd.read_csv(paths["test"], parse_dates=["date"])
    train_df = pd.read_csv(paths["train"], parse_dates=["date"])
    importance_df = pd.read_csv(paths["importance"])
    snapshot_df = pd.read_csv(paths["snapshot"], parse_dates=["date"])
    model_metrics_df = pd.read_csv(paths["model_metrics"]) if paths["model_metrics"].exists() else pd.DataFrame()
    prob_history_df = pd.read_csv(paths["prob_history"], parse_dates=["date"]) if paths["prob_history"].exists() else pd.DataFrame()
    backtest_df = pd.read_csv(paths["backtest"], parse_dates=["date"]) if paths["backtest"].exists() else pd.DataFrame()
    episode_review_df = pd.read_csv(paths["episode_review"]) if paths["episode_review"].exists() else pd.DataFrame()
    period_compare_df = pd.read_csv(paths["period_compare"]) if paths["period_compare"].exists() else pd.DataFrame()
    florida_index_df = pd.read_csv(paths["florida_index"], parse_dates=["date"]) if paths["florida_index"].exists() else pd.DataFrame()
    markov_regimes_df = pd.read_csv(paths["markov_regimes"], parse_dates=["date"]) if paths["markov_regimes"].exists() else pd.DataFrame()

    nowcast = None
    if paths["nowcast"].exists():
        with paths["nowcast"].open("r", encoding="utf-8") as f:
            nowcast = json.load(f)
    florida_latest = None
    if paths["florida_latest"].exists():
        with paths["florida_latest"].open("r", encoding="utf-8") as f:
            florida_latest = json.load(f)
    markov_summary = None
    if paths["markov_summary"].exists():
        with paths["markov_summary"].open("r", encoding="utf-8") as f:
            markov_summary = json.load(f)

    return (
        metrics,
        train_df,
        test_df,
        importance_df,
        snapshot_df,
        model_metrics_df,
        prob_history_df,
        backtest_df,
        episode_review_df,
        period_compare_df,
        florida_index_df,
        markov_regimes_df,
        markov_summary,
        florida_latest,
        nowcast,
    )


def _probability_chart(df: pd.DataFrame, prob_col: str, title: str, threshold: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[prob_col],
            mode="lines",
            name="Predicted recession probability",
            line={"width": 2},
        )
    )

    if "current_recession" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["current_recession"] * 0.15,
                name="Actual recession periods (scaled bars)",
                opacity=0.22,
            )
        )

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Alert threshold {threshold:.0%}",
            annotation_position="top left",
        )

    fig.update_layout(
        yaxis_title="Probability",
        xaxis_title="Date",
        title=title,
        bargap=0,
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def _threshold_breakdown(df: pd.DataFrame, prob_col: str, threshold: float) -> dict[str, float]:
    y_true = df["y_true"].to_numpy(dtype=int)
    y_pred = (df[prob_col].to_numpy() >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _friendly_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["indicator"] = out["feature"].map(_feature_label)
    out["what_this_sign_means"] = "Not directional for this model"
    logistic_mask = out["model"] == "logistic"
    out.loc[logistic_mask & (out["importance"] > 0), "what_this_sign_means"] = (
        "If this indicator rises, model risk usually rises"
    )
    out.loc[logistic_mask & (out["importance"] < 0), "what_this_sign_means"] = (
        "If this indicator rises, model risk usually falls"
    )

    return out[
        [
            "model",
            "indicator",
            "feature",
            "importance_type",
            "importance",
            "abs_importance",
            "what_this_sign_means",
        ]
    ]


def _current_direction_summary(prob_history_df: pd.DataFrame, threshold: float) -> tuple[str, str]:
    if prob_history_df.empty:
        return "Unknown", "No probability history available yet."

    recent = prob_history_df.sort_values("date").tail(6).copy()
    latest = float(recent["y_prob"].iloc[-1])
    change_6m = float(recent["y_prob"].iloc[-1] - recent["y_prob"].iloc[0]) if len(recent) >= 2 else 0.0

    if latest >= threshold:
        status = "Warning"
        note = (
            f"Current probability is {latest:.1%}, above your {threshold:.0%} threshold. "
            f"6-month change: {change_6m:+.1%}."
        )
    elif latest >= threshold * 0.5 or change_6m > 0.05:
        status = "Watch"
        note = (
            f"Current probability is {latest:.1%}, below threshold but elevated or rising. "
            f"6-month change: {change_6m:+.1%}."
        )
    else:
        status = "Calm"
        note = (
            f"Current probability is {latest:.1%}, well below threshold and not rising sharply. "
            f"6-month change: {change_6m:+.1%}."
        )
    return status, note


def _episode_story(episode_review_df: pd.DataFrame, threshold: float) -> tuple[str, str]:
    if episode_review_df.empty:
        return "No episode review available yet.", "Run the pipeline to generate walk-forward episode diagnostics."

    row_2008 = episode_review_df[
        episode_review_df["episode_start"].astype(str).str.startswith(("2007", "2008"))
    ]
    misses = episode_review_df[episode_review_df["missed_early_warning"] == True]  # noqa: E712

    if row_2008.empty:
        part_2008 = "2008 row not found in this backtest window."
    else:
        row = row_2008.iloc[0]
        if pd.isna(row["lead_months"]):
            part_2008 = (
                f"For the recession starting {row['episode_start']}, the model did not cross "
                f"the {threshold:.0%} threshold before recession began."
            )
        else:
            part_2008 = (
                f"For the recession starting {row['episode_start']}, the model crossed "
                f"the {threshold:.0%} threshold about {int(row['lead_months'])} months earlier."
            )

    if misses.empty:
        misses_part = "In this backtest window, every recession got an early warning signal."
    else:
        miss_dates = ", ".join(misses["episode_start"].astype(str).tolist())
        misses_part = (
            f"Missed early warning happened before recession starts at: {miss_dates}. "
            "In those periods, pre-recession probabilities stayed below your selected threshold."
        )

    return part_2008, misses_part


def _build_monthly_explanations(
    prob_history_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    if prob_history_df.empty or snapshot_df.empty:
        return pd.DataFrame()

    logistic = importance_df[importance_df["model"] == "logistic"]
    coef_map = logistic.set_index("feature")["importance"].to_dict()

    merged = prob_history_df.merge(snapshot_df, on="date", how="left", suffixes=("", "_snapshot"))
    feature_cols = [
        c
        for c in snapshot_df.columns
        if c not in {"date", "target_recession_next_horizon", "current_recession"}
    ]
    merged = merged.sort_values("date").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for i in range(1, len(merged)):
        curr = merged.iloc[i]
        prev = merged.iloc[i - 1]
        prob_delta = float(curr["y_prob"] - prev["y_prob"])

        driver_rows: list[tuple[float, str, float, str]] = []
        for feature in feature_cols:
            if feature not in merged.columns:
                continue
            curr_val = curr[feature]
            prev_val = prev[feature]
            if pd.isna(curr_val) or pd.isna(prev_val):
                continue

            delta = float(curr_val - prev_val)
            if abs(delta) < 1e-10:
                continue

            coef = float(coef_map.get(feature, 0.0))
            score = abs(delta * (coef if coef != 0 else 1.0))

            if coef > 0 and delta > 0:
                effect_text = "pushed risk up"
            elif coef > 0 and delta < 0:
                effect_text = "pulled risk down"
            elif coef < 0 and delta > 0:
                effect_text = "pulled risk down"
            elif coef < 0 and delta < 0:
                effect_text = "pushed risk up"
            else:
                effect_text = "had uncertain direction"

            driver_rows.append((score, feature, delta, effect_text))

        top_drivers = sorted(driver_rows, key=lambda x: x[0], reverse=True)[:top_n]
        driver_texts = [
            f"{_feature_label(f)} moved {d:+.2f} and {e}" for _, f, d, e in top_drivers
        ]

        if prob_delta > 0:
            headline = f"Probability rose by {prob_delta:+.1%} versus previous month."
        elif prob_delta < 0:
            headline = f"Probability fell by {prob_delta:+.1%} versus previous month."
        else:
            headline = "Probability was flat versus previous month."

        rows.append(
            {
                "date": pd.to_datetime(curr["date"]),
                "y_prob": float(curr["y_prob"]),
                "prob_delta": prob_delta,
                "headline": headline,
                "top_drivers": " | ".join(driver_texts) if driver_texts else "No major feature movement found.",
            }
        )

    return pd.DataFrame(rows)


def _model_goal_winners(model_metrics_df: pd.DataFrame) -> dict[str, str]:
    if model_metrics_df.empty:
        return {}

    df = model_metrics_df.copy()
    winners = {
        "Best Calibration (lowest Brier)": str(df.loc[df["brier_score"].idxmin(), "model"]),
        "Best Ranking (highest AUC)": str(df.loc[df["roc_auc"].idxmax(), "model"]),
        "Best Event Capture (highest Recall proxy: avg precision)": str(
            df.loc[df["average_precision"].idxmax(), "model"]
        ),
        "Lowest Confidence Penalty (lowest log loss)": str(df.loc[df["log_loss"].idxmin(), "model"]),
    }
    return winners


def _friendly_model_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model"] = out["model"].map(_model_label)
    out = out.rename(
        columns={
            "model": "Model",
            "roc_auc": "AUC (higher is better)",
            "average_precision": "Average Precision (higher is better)",
            "brier_score": "Brier Score (lower is better)",
            "log_loss": "Log Loss (lower is better)",
            "positive_rate": "Actual Recession Share",
            "predicted_positive_rate": "Warning Share",
        }
    )
    return out


def _friendly_algorithm_catalog(catalog: list[dict[str, str]]) -> pd.DataFrame:
    if not catalog:
        return pd.DataFrame()
    out = pd.DataFrame(catalog)
    out["status"] = out["status"].map({"active": "Used now", "planned": "Planned next"}).fillna(out["status"])
    out = out.rename(
        columns={
            "id": "Algorithm ID",
            "name": "Algorithm Name",
            "status": "Status",
            "what_it_does": "What It Does",
        }
    )
    cols = ["Algorithm Name", "Algorithm ID", "Status", "What It Does"]
    return out[cols]


def _florida_zone_text(zone: str) -> str:
    mapping = {
        "Calm": "Low stress",
        "Watch": "Medium stress",
        "Stress": "High stress",
        "Unknown": "Unknown stress",
    }
    return mapping.get(zone, zone)


def _florida_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["florida_stress_index"],
            mode="lines",
            name="Florida Stress Index",
            line={"width": 2},
        )
    )
    fig.add_hrect(y0=0, y1=33, fillcolor="green", opacity=0.08, line_width=0, annotation_text="Calm")
    fig.add_hrect(y0=33, y1=66, fillcolor="orange", opacity=0.08, line_width=0, annotation_text="Watch")
    fig.add_hrect(y0=66, y1=100, fillcolor="red", opacity=0.08, line_width=0, annotation_text="Stress")
    fig.update_layout(
        title="Florida Stage 1 Stress Index (0-100)",
        yaxis_title="Stress Score",
        xaxis_title="Date",
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def _markov_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    prob_cols = [c for c in df.columns if c.startswith("regime_prob_")]
    for col in prob_cols:
        label = col.replace("regime_prob_", "").title()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[col],
                mode="lines",
                stackgroup="one",
                name=f"{label} probability",
            )
        )
    fig.update_layout(
        title="Markov Switching Regime Probabilities",
        yaxis_title="Probability",
        xaxis_title="Date",
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def _bayesian_band_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["bayesian_dynamic_prob_upper"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            name="Upper",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["bayesian_dynamic_prob_lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.20)",
            line={"width": 0},
            name="Uncertainty band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["bayesian_dynamic_prob"],
            mode="lines",
            line={"width": 2},
            name="Bayesian dynamic probability",
        )
    )
    fig.update_layout(
        title="Bayesian Dynamic Model: Probability with Uncertainty Band",
        yaxis_title="Probability",
        xaxis_title="Date",
    )
    fig.update_yaxes(range=[0, 1])
    return fig


def main() -> None:
    st.set_page_config(page_title="Recession Probability Lab", layout="wide")
    st.title("Recession Probability Lab")
    st.caption("Plain-language recession monitor with optional technical detail")

    try:
        (
            metrics,
            train_df,
            test_df,
            importance_df,
            snapshot_df,
            model_metrics_df,
            prob_history_df,
            backtest_df,
            episode_review_df,
            period_compare_df,
            florida_index_df,
            markov_regimes_df,
            markov_summary,
            florida_latest,
            nowcast,
        ) = _load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    selected_model = str(metrics.get("selected_model", "unknown"))
    selected_test_metrics = metrics.get("selected_test_metrics", {})
    alert_threshold = float(metrics.get("alert_threshold", DEFAULT_ALERT_THRESHOLD))
    algorithm_catalog = metrics.get("algorithm_catalog", [])

    if nowcast:
        latest_prob = float(nowcast["recession_probability"])
        latest_date = str(nowcast["date"])
        latest_prob_lower = (
            float(nowcast["recession_probability_lower"])
            if "recession_probability_lower" in nowcast
            else None
        )
        latest_prob_upper = (
            float(nowcast["recession_probability_upper"])
            if "recession_probability_upper" in nowcast
            else None
        )
    elif not prob_history_df.empty:
        latest_prob = float(prob_history_df.iloc[-1]["y_prob"])
        latest_date = str(pd.to_datetime(prob_history_df.iloc[-1]["date"]).date())
        latest_prob_lower = (
            float(prob_history_df.iloc[-1]["y_prob_lower"])
            if "y_prob_lower" in prob_history_df.columns and not pd.isna(prob_history_df.iloc[-1]["y_prob_lower"])
            else None
        )
        latest_prob_upper = (
            float(prob_history_df.iloc[-1]["y_prob_upper"])
            if "y_prob_upper" in prob_history_df.columns and not pd.isna(prob_history_df.iloc[-1]["y_prob_upper"])
            else None
        )
    else:
        latest_prob = float(test_df.iloc[-1]["y_prob"])
        latest_date = str(pd.to_datetime(test_df.iloc[-1]["date"]).date())
        latest_prob_lower = None
        latest_prob_upper = None

    status, status_note = _current_direction_summary(prob_history_df, threshold=alert_threshold)
    status_map = {"Calm": "OK for now", "Watch": "Watch closely", "Warning": "Higher risk"}
    status_display = status_map.get(status, status)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current risk", status_display)
    c2.metric("Latest nowcast", f"{latest_prob:.1%}")
    c3.metric("Nowcast date", latest_date)
    c4.metric("Model in use", _model_label(selected_model))
    c5.metric("Alert threshold", f"{alert_threshold:.0%}")
    if latest_prob_lower is not None and latest_prob_upper is not None:
        st.caption(f"Uncertainty band: {latest_prob_lower:.1%} to {latest_prob_upper:.1%}")
    st.info(status_note)

    st.subheader("Florida Stage 1 Pilot")
    if florida_index_df.empty:
        st.info("Florida Stage 1 artifacts not found yet. Run `python run_pipeline.py`.")
    else:
        florida_latest_row = florida_latest or {}
        fl_score = float(florida_latest_row.get("florida_stress_index", florida_index_df.iloc[-1]["florida_stress_index"]))
        fl_zone = str(florida_latest_row.get("florida_stress_zone", florida_index_df.iloc[-1]["florida_stress_zone"]))
        fl_change = float(
            florida_latest_row.get(
                "florida_stress_mom_change",
                florida_index_df.iloc[-1].get("florida_stress_mom_change", 0.0),
            )
        )
        fl_date = str(florida_latest_row.get("date", pd.to_datetime(florida_index_df.iloc[-1]["date"]).date()))
        fl_drivers = str(
            florida_latest_row.get(
                "florida_top_drivers",
                florida_index_df.iloc[-1].get("florida_top_drivers", "No driver text available."),
            )
        )

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Florida stress level", _florida_zone_text(fl_zone))
        f2.metric("Florida stress score", f"{fl_score:.1f}/100")
        f3.metric("Monthly change", f"{fl_change:+.1f}")
        f4.metric("Florida data date", fl_date)

        st.plotly_chart(_florida_chart(florida_index_df), use_container_width=True)
        st.caption(f"Latest Florida stress drivers: {fl_drivers}")

        with st.expander("How Florida Stage 1 works (simple version)"):
            st.markdown(
                "- We combine labor, housing, and claims indicators for Florida into one score from 0 to 100.\n"
                "- `0-33`: calm, `34-66`: watch, `67-100`: stress.\n"
                "- The score is relative to Florida's own history, so it helps detect worsening direction over time."
            )

    st.subheader("Markov Switching Regime Detector")
    if markov_regimes_df.empty:
        st.info("Markov switching outputs not found yet. Run `python run_pipeline.py`.")
    else:
        markov_latest = markov_summary or {}
        latest_regime = str(markov_latest.get("latest_regime", markov_regimes_df.iloc[-1]["regime_label"])).title()
        regime_probs = markov_latest.get("regime_probabilities", {})

        r1, r2, r3 = st.columns(3)
        r1.metric("Latest regime", latest_regime)
        if isinstance(regime_probs, dict) and regime_probs:
            calm_prob = float(regime_probs.get("calm", np.nan))
            watch_prob = float(regime_probs.get("watch", np.nan))
            stress_prob = float(regime_probs.get("stress", np.nan))
            r2.metric("Stress probability", "n/a" if np.isnan(stress_prob) else f"{stress_prob:.1%}")
            r3.metric("Calm probability", "n/a" if np.isnan(calm_prob) else f"{calm_prob:.1%}")
            st.caption(
                f"Regime probabilities now -> Calm: {calm_prob:.1%} | Watch: {watch_prob:.1%} | Stress: {stress_prob:.1%}"
            )
        else:
            r2.metric("Stress probability", "n/a")
            r3.metric("Calm probability", "n/a")

        st.plotly_chart(_markov_chart(markov_regimes_df), use_container_width=True)
        with st.expander("How Markov switching works (simple version)"):
            st.markdown(
                "- It looks for hidden economic states in the probability signal over time.\n"
                "- States are labeled `calm`, `watch`, and `stress` by how high the signal tends to be.\n"
                "- This helps detect regime changes, not just level changes."
            )

    with st.expander("How to read this dashboard"):
        st.markdown(
            "- `Latest nowcast`: probability of recession within the next forecast window.\n"
            "- `Alert threshold`: your warning line. Crossing it means the model says risk is high.\n"
            "- `Current risk`: plain-language status based on level and recent direction.\n"
            "- Bars on charts show actual recession periods from historical NBER data.\n"
            "- This is a learning tool; treat outputs as signals, not certainty."
        )

    prob_columns = [c for c in test_df.columns if c.endswith("_prob")]
    default_prob = "y_prob" if "y_prob" in prob_columns else prob_columns[0]
    chosen_prob_col = st.selectbox(
        "Backtest probability series",
        options=prob_columns,
        index=prob_columns.index(default_prob),
        format_func=lambda x: _model_label(x.replace("_prob", "")),
    )

    st.plotly_chart(
        _probability_chart(
            test_df,
            prob_col=chosen_prob_col,
            title="Historical Backtest (periods where outcomes are known)",
            threshold=alert_threshold,
        ),
        use_container_width=True,
    )

    st.subheader("Bayesian Uncertainty Bands")
    bayes_cols = {"bayesian_dynamic_prob", "bayesian_dynamic_prob_lower", "bayesian_dynamic_prob_upper"}
    if bayes_cols.issubset(set(test_df.columns)):
        st.plotly_chart(_bayesian_band_chart(test_df), use_container_width=True)
        st.caption(
            "Center line is Bayesian Dynamic risk estimate. Shaded band is uncertainty range "
            "(wider band means lower confidence)."
        )
    else:
        st.info("Bayesian uncertainty columns not found. Re-run `python run_pipeline.py`.")

    st.subheader("Threshold Slider")
    threshold = st.slider(
        "When should we call it a warning?",
        min_value=0.05,
        max_value=0.95,
        value=alert_threshold,
        step=0.05,
    )
    breakdown = _threshold_breakdown(test_df, prob_col=chosen_prob_col, threshold=threshold)

    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{breakdown['precision']:.2f}")
    m2.metric("Recall", f"{breakdown['recall']:.2f}")
    m3.metric("F1", f"{breakdown['f1']:.2f}")

    st.caption(
        f"At {threshold:.0%} threshold: true warnings={breakdown['tp']}, false warnings={breakdown['fp']}, "
        f"missed warnings={breakdown['fn']}, calm correct={breakdown['tn']}."
    )

    with st.expander("What Precision / Recall / F1 mean in plain words"):
        st.markdown(
            "- `Precision`: of the months we warned, how many were truly followed by recession.\n"
            "- `Recall`: of all months that really had upcoming recession, how many we caught.\n"
            "- `F1`: one score that balances precision and recall.\n"
            "- High precision + low recall means careful but misses many.\n"
            "- High recall + low precision means catches more, but with more false alarms."
        )

    st.subheader("2008 vs Current Period")
    story_2008, story_misses = _episode_story(episode_review_df, threshold=threshold)
    st.write(story_2008)
    st.write(story_misses)

    left, right = st.columns(2)
    if not backtest_df.empty:
        backtest_2008 = backtest_df[
            (backtest_df["date"] >= pd.Timestamp("2005-01-01"))
            & (backtest_df["date"] <= pd.Timestamp("2011-12-31"))
        ]
        left.plotly_chart(
            _probability_chart(
                backtest_2008,
                prob_col="y_prob",
                title="2008 Era Walk-forward Probability",
                threshold=threshold,
            ),
            use_container_width=True,
        )

    if not prob_history_df.empty:
        recent = prob_history_df[prob_history_df["date"] >= pd.Timestamp("2023-01-01")]
        right.plotly_chart(
            _probability_chart(
                recent,
                prob_col="y_prob",
                title="Current Era Probability (2023-2026)",
                threshold=threshold,
            ),
            use_container_width=True,
        )

    if not period_compare_df.empty:
        st.markdown("**Period Comparison (simple summary)**")
        st.dataframe(period_compare_df, use_container_width=True)

    st.subheader("Monthly Explanations")
    explanations_df = _build_monthly_explanations(prob_history_df, snapshot_df, importance_df)
    if explanations_df.empty:
        st.info("No monthly explanation data available.")
    else:
        recent_explanations = explanations_df.sort_values("date").tail(24).copy()
        selected_date = st.selectbox(
            "Pick a month",
            options=list(recent_explanations["date"]),
            index=len(recent_explanations) - 1,
            format_func=lambda d: pd.to_datetime(d).date().isoformat(),
        )
        row = recent_explanations[recent_explanations["date"] == pd.Timestamp(selected_date)].iloc[0]
        st.write(row["headline"])
        st.write(row["top_drivers"])
        st.caption("Driver text is heuristic: useful for learning, not strict causality.")

    st.subheader("Feature Importance")
    if importance_df.empty:
        st.info("No feature importance data found.")
    else:
        model_options = sorted(importance_df["model"].unique().tolist())
        chosen_model = st.selectbox(
            "Importance model",
            options=model_options,
            index=0,
            format_func=_model_label,
        )
        display_importance = (
            importance_df[importance_df["model"] == chosen_model]
            .sort_values("abs_importance", ascending=False)
            .head(20)
        )
        st.dataframe(_friendly_feature_df(display_importance), use_container_width=True)

    with st.expander("What positive vs negative importance means"):
        st.markdown(
            "- In logistic regression, a `positive` coefficient means higher values usually push risk up.\n"
            "- In logistic regression, a `negative` coefficient means higher values usually push risk down.\n"
            "- In boosted trees, we use permutation importance instead; it shows usefulness, not direction."
        )

    st.subheader("Indicator Explorer")
    candidate_features = [
        c
        for c in snapshot_df.columns
        if c not in {"date", "target_recession_next_horizon", "current_recession"}
    ]
    label_to_feature = {_feature_label(feature): feature for feature in candidate_features}
    selected_label = st.selectbox("Choose an indicator", options=sorted(label_to_feature.keys()), index=0)
    selected_feature = label_to_feature[selected_label]

    indicator_fig = go.Figure()
    indicator_fig.add_trace(
        go.Scatter(x=snapshot_df["date"], y=snapshot_df[selected_feature], mode="lines", name=selected_label)
    )
    indicator_fig.update_layout(title=f"{selected_label} over time", xaxis_title="Date", yaxis_title=selected_label)
    st.plotly_chart(indicator_fig, use_container_width=True)
    st.caption(FEATURE_DESCRIPTIONS.get(selected_feature, "No description available."))

    with st.expander("What are Fama-French factors?"):
        st.markdown(
            "- `Market excess return`: stock market return above risk-free rate.\n"
            "- `SMB (size factor)`: performance of smaller firms vs larger firms.\n"
            "- `HML (value factor)`: performance of value stocks vs growth stocks.\n"
            "- They are classic finance learning factors and give extra signal beyond FRED."
        )

    st.subheader("Model Comparison")
    if model_metrics_df.empty:
        st.info("Model comparison file not found. Re-run pipeline to generate it.")
    else:
        winners = _model_goal_winners(model_metrics_df)
        if winners:
            st.markdown("**Best Model by Goal**")
            for goal, model_name in winners.items():
                st.write(f"- {goal}: {_model_label(model_name)}")

        st.dataframe(_friendly_model_metrics(model_metrics_df), use_container_width=True)

        with st.expander("How to read AUC, Brier, Log Loss, and other table columns"):
            st.markdown(
                "- `AUC`: how well the model ranks risky months above safer months. "
                "0.50 is random, 1.00 is perfect.\n"
                "- `Average Precision`: useful for rare events like recessions; higher means better detection quality.\n"
                "- `Brier Score`: average squared error of probabilities. Lower means probabilities are more accurate.\n"
                "- `Log Loss`: penalizes confident wrong predictions more heavily. Lower is better.\n"
                "- `Actual Recession Share`: how often recession was truly present in this test set.\n"
                "- `Warning Share`: how often model issued warnings at default 50% cutoff."
            )

    with st.expander("Advanced algorithms status (friendly guide)"):
        st.markdown(
            "- `Bayesian Dynamic Model` is active now: it gives uncertainty bands over time.\n"
            "- `Markov Switching Model` is active now: it detects calm/watch/stress regime shifts.\n"
            "- `NGBoost` is active now: it models predictive distributions for richer uncertainty handling.\n"
            "- `Temporal Fusion Transformer` is still future/optional: powerful but more complex."
        )

    st.subheader("Algorithms Used And Planned")
    algo_df = _friendly_algorithm_catalog(algorithm_catalog)
    if algo_df.empty:
        st.info("Algorithm catalog not found in metrics yet. Re-run `python run_pipeline.py`.")
    else:
        st.dataframe(algo_df, use_container_width=True)

    st.subheader("Training Diagnostics")
    st.caption(
        f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,} | "
        f"Forecast horizon: {metrics['horizon_months']} months | "
        f"Walk-forward min train window: {metrics.get('walkforward_min_train_months', 'n/a')}"
    )


if __name__ == "__main__":
    main()
