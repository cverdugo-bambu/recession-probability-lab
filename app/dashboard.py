from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

MODEL_FRIENDLY_NAMES = {
    "logistic": "Logistic Regression",
    "hist_gb": "Gradient Boosted Trees",
    "bayesian_dynamic": "Bayesian Dynamic",
    "ngboost": "NGBoost",
    "ensemble": "Ensemble Average",
    "markov_switching": "Markov Switching",
}

MODEL_ANALOGIES = {
    "logistic": {
        "one_liner": "Finds the simplest straight-line pattern between indicators and recessions.",
        "analogy": (
            "Think of logistic regression like a weighted checklist. Each economic indicator "
            "gets a score based on how strongly it has historically been linked to recessions. "
            "If the total weighted score crosses a line, it says 'watch out.' It's transparent — "
            "you can see exactly which indicators are pushing risk up or down."
        ),
        "best_at": "Telling you which indicators matter most and in which direction.",
    },
    "hist_gb": {
        "one_liner": "Learns complex if-then rules from the data to spot nonlinear warning patterns.",
        "analogy": (
            "Imagine a decision tree: 'If unemployment is rising AND the yield curve is inverted "
            "AND credit spreads are widening, then risk is high.' Gradient boosting builds hundreds "
            "of these trees, each one correcting the mistakes of the previous ones. It can catch "
            "subtle combinations that simpler models miss."
        ),
        "best_at": "Detecting complex interactions between multiple indicators at once.",
    },
    "bayesian_dynamic": {
        "one_liner": "A probability model that also tells you how uncertain it is.",
        "analogy": (
            "Most models give you a single number. This one says 'I think the probability is 15%, "
            "but it could reasonably be anywhere from 8% to 25%.' It tracks how confident it is, "
            "which is especially valuable when the economic picture is murky."
        ),
        "best_at": "Quantifying uncertainty — showing when the model is confident vs. unsure.",
    },
    "ngboost": {
        "one_liner": "Boosted trees that predict the full range of possible outcomes, not just a point estimate.",
        "analogy": (
            "Traditional models say 'the probability is X%.' NGBoost says 'here is the full "
            "distribution of where the probability could be.' It combines the pattern-finding "
            "power of boosted trees with a richer picture of uncertainty."
        ),
        "best_at": "Providing distribution-aware forecasts with natural uncertainty estimates.",
    },
    "ensemble": {
        "one_liner": "Averages multiple models together for a more stable, balanced forecast.",
        "analogy": (
            "Like asking several doctors for their opinion and taking the consensus. Each model "
            "has different strengths and blind spots. By averaging them, extreme predictions get "
            "tempered and the overall signal tends to be more reliable."
        ),
        "best_at": "Reducing the chance that one model's quirks lead you astray.",
    },
}

# Stress component columns in the Florida index CSV
FLORIDA_STRESS_COMPONENTS = {
    "stress_unemployment_level": "Unemployment Level",
    "stress_unemployment_change": "Unemployment Change",
    "stress_payroll_growth": "Payroll Growth",
    "stress_manufacturing_growth": "Manufacturing Growth",
    "stress_building_permits": "Building Permits",
    "stress_initial_claims": "Initial Claims",
}

# NBER recession date ranges for overlaying on charts
NBER_RECESSIONS = [
    ("1960-04-01", "1961-02-01"),
    ("1969-12-01", "1970-11-01"),
    ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]


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


# ---------------------------------------------------------------------------
# Existing helper functions (kept from original)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NEW helper functions
# ---------------------------------------------------------------------------

def _generate_situation_summary(
    latest_prob: float,
    status: str,
    change_6m: float,
    markov_summary: dict | None,
    florida_latest: dict | None,
) -> str:
    """Auto-generate plain-language summary from latest data."""
    direction = "rising" if change_6m > 0.01 else ("falling" if change_6m < -0.01 else "roughly flat")

    regime_text = "unknown"
    if markov_summary:
        regime_text = str(markov_summary.get("latest_regime", "unknown"))

    fl_text = ""
    if florida_latest:
        fl_zone = florida_latest.get("florida_stress_zone", "Unknown")
        fl_score = florida_latest.get("florida_stress_index", 0)
        fl_text = f" Florida's economic stress index is at {fl_score:.0f}/100, in the **{fl_zone}** zone."

    summary = (
        f"The US recession probability stands at **{latest_prob:.1%}**, which is {direction} "
        f"over the past 6 months (change: {change_6m:+.1%}). "
        f"The overall risk status is **{status}**. "
        f"The Markov regime model classifies the current economic state as **{regime_text}**, "
        f"suggesting the economy is in a {regime_text}-level risk environment."
        f"{fl_text}"
    )
    return summary


def _generate_key_findings(
    latest_prob: float,
    change_6m: float,
    threshold: float,
    markov_summary: dict | None,
    florida_latest: dict | None,
    episode_review_df: pd.DataFrame,
) -> list[str]:
    """Auto-generate key findings bullets from latest data."""
    findings = []

    # Probability direction
    if change_6m > 0.02:
        findings.append(f"Recession probability is **rising** ({change_6m:+.1%} over 6 months) — bears watching.")
    elif change_6m < -0.02:
        findings.append(f"Recession probability is **falling** ({change_6m:+.1%} over 6 months) — a positive signal.")
    else:
        findings.append(f"Recession probability is **roughly flat** ({change_6m:+.1%} over 6 months).")

    # Threshold comparison
    if latest_prob >= threshold:
        findings.append(f"We are **above** the {threshold:.0%} alert threshold at {latest_prob:.1%}.")
    else:
        gap = threshold - latest_prob
        findings.append(f"We are **below** the {threshold:.0%} alert threshold by {gap:.1%} ({latest_prob:.1%} current).")

    # Markov regime vs pre-2008
    if markov_summary:
        regime = markov_summary.get("latest_regime", "unknown")
        stress_prob = markov_summary.get("regime_probabilities", {}).get("stress", 0)
        if regime == "stress":
            findings.append("The regime model shows **stress** — this level was last seen during major downturns.")
        elif regime == "watch":
            findings.append(f"The regime model shows **watch** with {stress_prob:.0%} chance of stress — elevated but not crisis-level.")
        else:
            findings.append(f"The regime model shows **{regime}** — consistent with normal economic conditions.")

    # Florida
    if florida_latest:
        fl_zone = florida_latest.get("florida_stress_zone", "Unknown")
        fl_change = florida_latest.get("florida_stress_mom_change", 0)
        trend = "rising" if fl_change > 1 else ("falling" if fl_change < -1 else "stable")
        findings.append(f"Florida stress is in the **{fl_zone}** zone and {trend} (monthly change: {fl_change:+.1f} points).")

    return findings


def _generate_policy_implications(
    snapshot_df: pd.DataFrame,
    florida_latest: dict | None,
) -> list[tuple[str, str]]:
    """Map stressed indicators to policy area recommendations."""
    implications: list[tuple[str, str]] = []
    if snapshot_df.empty:
        return implications

    latest = snapshot_df.sort_values("date").iloc[-1]

    # Unemployment rising
    if "unemployment_3m_delta" in latest.index and not pd.isna(latest["unemployment_3m_delta"]):
        if latest["unemployment_3m_delta"] > 0.2:
            implications.append((
                "Labor market support",
                f"Unemployment has risen {latest['unemployment_3m_delta']:+.1f}pp over 3 months. "
                "Consider: expanded job training programs, extended unemployment insurance, workforce development initiatives."
            ))

    # Credit spread widening
    if "credit_spread" in latest.index and not pd.isna(latest["credit_spread"]):
        if latest["credit_spread"] > 2.5:
            implications.append((
                "Financial stability",
                f"Corporate credit spreads are at {latest['credit_spread']:.2f}%, indicating elevated financial stress. "
                "Consider: monitoring bank lending conditions, reviewing stress test results, ensuring liquidity backstops."
            ))

    # Yield curve inverted
    if "yield_spread" in latest.index and not pd.isna(latest["yield_spread"]):
        if latest["yield_spread"] < 0:
            implications.append((
                "Monetary policy watch",
                f"The yield curve is inverted at {latest['yield_spread']:.2f}%. "
                "Consider: rate path assessment, forward guidance review, communication strategy."
            ))
        elif latest["yield_spread"] < 0.5:
            implications.append((
                "Monetary policy watch",
                f"The yield curve is nearly flat at {latest['yield_spread']:.2f}%. "
                "Monitor for potential inversion — historically a leading recession indicator."
            ))

    # Florida stress high
    if florida_latest:
        fl_zone = florida_latest.get("florida_stress_zone", "Calm")
        if fl_zone in ("Watch", "Stress"):
            implications.append((
                "State-level: Florida",
                f"Florida is in the {fl_zone} zone. "
                "Consider: housing market support, construction sector monitoring, tourism industry assessment."
            ))

    # Industrial production falling
    if "industrial_prod_yoy" in latest.index and not pd.isna(latest["industrial_prod_yoy"]):
        if latest["industrial_prod_yoy"] < -1:
            implications.append((
                "Manufacturing sector",
                f"Industrial production is down {latest['industrial_prod_yoy']:.1f}% year-over-year. "
                "Consider: trade policy review, supply chain support, manufacturing investment incentives."
            ))

    # VIX elevated
    if "vix_level" in latest.index and not pd.isna(latest["vix_level"]):
        if latest["vix_level"] > 25:
            implications.append((
                "Market confidence",
                f"The VIX (market fear index) is elevated at {latest['vix_level']:.1f}. "
                "Consider: clear economic communication, stability measures, reducing policy uncertainty."
            ))

    if not implications:
        implications.append((
            "No immediate red flags",
            "Current indicators are within normal ranges. Continue monitoring and maintain readiness."
        ))

    return implications


def _generate_conclusions(
    status: str,
    latest_prob: float,
    change_6m: float,
    markov_summary: dict | None,
    florida_latest: dict | None,
) -> str:
    """Generate 2-3 sentence conclusion."""
    regime = markov_summary.get("latest_regime", "unknown") if markov_summary else "unknown"
    fl_zone = florida_latest.get("florida_stress_zone", "N/A") if florida_latest else "N/A"

    if status == "Warning":
        conclusion = (
            f"Overall risk posture: **Elevated**. The probability is above threshold and the Markov regime "
            f"is in {regime} mode. Close monitoring is warranted across all indicators, with particular "
            f"attention to employment and credit conditions next month."
        )
    elif status == "Watch":
        conclusion = (
            f"Overall risk posture: **Watchful**. While probability is below threshold ({latest_prob:.1%}), "
            f"the {regime} regime and recent trends suggest this is not a time for complacency. "
            f"Key indicators to watch next month: unemployment trajectory, credit spreads, and yield curve shape."
        )
    else:
        conclusion = (
            f"Overall risk posture: **Low**. Recession probability is well contained at {latest_prob:.1%} "
            f"and the regime model reads {regime}. "
        )
        if fl_zone in ("Watch", "Stress"):
            conclusion += f"However, Florida's {fl_zone} zone status deserves continued attention."
        else:
            conclusion += "Continue routine monitoring; no immediate policy pivots appear necessary."

    return conclusion


def _indicator_comparison_chart(
    snapshot_df: pd.DataFrame,
    indicators: list[str],
    period1_range: tuple[str, str],
    period2_range: tuple[str, str],
    period1_label: str = "2008 Crisis",
    period2_label: str = "Current",
) -> go.Figure:
    """Small multiples comparing indicator values across two periods."""
    p1 = snapshot_df[(snapshot_df["date"] >= period1_range[0]) & (snapshot_df["date"] <= period1_range[1])].copy()
    p2 = snapshot_df[(snapshot_df["date"] >= period2_range[0]) & (snapshot_df["date"] <= period2_range[1])].copy()

    available = [ind for ind in indicators if ind in snapshot_df.columns]
    n = len(available)
    if n == 0:
        fig = go.Figure()
        fig.add_annotation(text="No indicators available for comparison", showarrow=False)
        return fig

    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[_feature_label(ind) for ind in available],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for idx, ind in enumerate(available):
        r = idx // cols + 1
        c = idx % cols + 1
        if not p1.empty and ind in p1.columns:
            fig.add_trace(
                go.Scatter(x=p1["date"], y=p1[ind], mode="lines", name=period1_label,
                           line={"color": "red", "dash": "dash"}, legendgroup="p1",
                           showlegend=(idx == 0)),
                row=r, col=c,
            )
        if not p2.empty and ind in p2.columns:
            fig.add_trace(
                go.Scatter(x=p2["date"], y=p2[ind], mode="lines", name=period2_label,
                           line={"color": "blue"}, legendgroup="p2",
                           showlegend=(idx == 0)),
                row=r, col=c,
            )

    fig.update_layout(height=300 * rows, title_text="Indicator Comparison: Then vs Now")
    return fig


def _indicator_comparison_narrative(
    snapshot_df: pd.DataFrame,
    indicators: list[str],
    period1_range: tuple[str, str],
    period2_range: tuple[str, str],
) -> str:
    """Auto-generate a narrative comparing indicators across two periods."""
    p1 = snapshot_df[(snapshot_df["date"] >= period1_range[0]) & (snapshot_df["date"] <= period1_range[1])].copy()
    p2 = snapshot_df[(snapshot_df["date"] >= period2_range[0]) & (snapshot_df["date"] <= period2_range[1])].copy()

    lines = []
    for ind in indicators:
        if ind not in snapshot_df.columns:
            continue
        label = _feature_label(ind)
        p1_vals = p1[ind].dropna()
        p2_vals = p2[ind].dropna()
        if p1_vals.empty or p2_vals.empty:
            continue

        p1_latest = float(p1_vals.iloc[-1])
        p1_mean = float(p1_vals.mean())
        p2_latest = float(p2_vals.iloc[-1])
        p2_mean = float(p2_vals.mean())

        lines.append(
            f"- **{label}**: In the 2008 crisis, it averaged {p1_mean:.2f} (peaked around {p1_latest:.2f} at period end). "
            f"Currently, it averages {p2_mean:.2f} (latest: {p2_latest:.2f})."
        )

    if not lines:
        return "Insufficient data for narrative comparison."
    return "\n".join(lines)


def _recession_aligned_chart(
    snapshot_df: pd.DataFrame,
    indicator: str,
    episode_review_df: pd.DataFrame,
    prob_history_df: pd.DataFrame,
    months_before: int = 24,
    months_after: int = 12,
) -> go.Figure:
    """Overlay indicator paths aligned to recession start dates."""
    fig = go.Figure()
    if indicator not in snapshot_df.columns or episode_review_df.empty:
        fig.add_annotation(text="Data not available", showarrow=False)
        return fig

    label = _feature_label(indicator)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    for idx, row in episode_review_df.iterrows():
        start = pd.to_datetime(row["episode_start"])
        window_start = start - pd.DateOffset(months=months_before)
        window_end = start + pd.DateOffset(months=months_after)
        window = snapshot_df[(snapshot_df["date"] >= window_start) & (snapshot_df["date"] <= window_end)].copy()
        if window.empty:
            continue

        # Align to months relative to recession start
        window["months_relative"] = (((window["date"] - start).dt.days) / 30.44).round().astype(int)
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=window["months_relative"], y=window[indicator],
                mode="lines", name=str(start.date()),
                line={"color": color, "width": 1.5},
                opacity=0.7,
            )
        )

    # Add current period aligned to "now" as T-0
    if not prob_history_df.empty:
        current_start = prob_history_df["date"].max()
        window_start = current_start - pd.DateOffset(months=months_before)
        window = snapshot_df[(snapshot_df["date"] >= window_start) & (snapshot_df["date"] <= current_start)].copy()
        if not window.empty:
            window["months_relative"] = (((window["date"] - current_start).dt.days) / 30.44).round().astype(int)
            fig.add_trace(
                go.Scatter(
                    x=window["months_relative"], y=window[indicator],
                    mode="lines", name="Current",
                    line={"color": "black", "width": 3},
                )
            )

    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="T-0 (recession start / now)")
    fig.update_layout(
        title=f"{label}: Path Aligned to Recession Start",
        xaxis_title="Months Relative to Recession Start",
        yaxis_title=label,
    )
    return fig


def _florida_component_bar_chart(florida_index_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of Florida stress components (z-scores) with color coding."""
    if florida_index_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No Florida data", showarrow=False)
        return fig

    latest = florida_index_df.sort_values("date").iloc[-1]
    prev = florida_index_df.sort_values("date").iloc[-2] if len(florida_index_df) >= 2 else None

    components = []
    values = []
    changes = []
    colors = []

    for col, label in FLORIDA_STRESS_COMPONENTS.items():
        if col in latest.index and not pd.isna(latest[col]):
            val = float(latest[col])
            components.append(label)
            values.append(val)
            # Red = adding stress (positive z-score), green = reducing stress (negative z-score)
            colors.append("#d62728" if val > 0 else "#2ca02c")
            if prev is not None and col in prev.index and not pd.isna(prev[col]):
                changes.append(float(latest[col] - prev[col]))
            else:
                changes.append(0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=components,
            x=values,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
        )
    )
    fig.add_vline(x=0, line_color="gray", line_width=1)
    fig.update_layout(
        title="Florida Stress Components (z-scores)",
        xaxis_title="Z-Score (positive = adding stress)",
        yaxis_title="",
        height=350,
    )
    return fig, changes, components


def _florida_vs_national_chart(
    florida_index_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
) -> go.Figure:
    """Side-by-side Florida vs US indicator comparison."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Unemployment Rate", "Payroll Growth (YoY)"],
        horizontal_spacing=0.1,
    )

    # Unemployment: FL vs US
    if "fl_unemployment_rate" in florida_index_df.columns:
        fig.add_trace(
            go.Scatter(x=florida_index_df["date"], y=florida_index_df["fl_unemployment_rate"],
                       mode="lines", name="Florida", line={"color": "#ff7f0e"}),
            row=1, col=1,
        )
    if "unemployment" in snapshot_df.columns:
        fig.add_trace(
            go.Scatter(x=snapshot_df["date"], y=snapshot_df["unemployment"],
                       mode="lines", name="US National", line={"color": "#1f77b4"}),
            row=1, col=1,
        )

    # Payroll growth: FL vs US
    if "fl_total_payrolls_yoy" in florida_index_df.columns:
        fig.add_trace(
            go.Scatter(x=florida_index_df["date"], y=florida_index_df["fl_total_payrolls_yoy"],
                       mode="lines", name="Florida", line={"color": "#ff7f0e"}, showlegend=False),
            row=1, col=2,
        )
    if "payrolls_yoy" in snapshot_df.columns:
        fig.add_trace(
            go.Scatter(x=snapshot_df["date"], y=snapshot_df["payrolls_yoy"],
                       mode="lines", name="US National", line={"color": "#1f77b4"}, showlegend=False),
            row=1, col=2,
        )

    fig.update_layout(title="Florida vs National: Key Labor Indicators", height=400)
    fig.update_yaxes(title_text="%", row=1, col=1)
    fig.update_yaxes(title_text="% YoY", row=1, col=2)
    return fig


def _model_agreement_chart(all_model_metrics: dict) -> go.Figure:
    """Horizontal bar of all models' current probabilities (from metrics)."""
    # We use test metrics predicted positive rate as a proxy since we don't have
    # per-model live probabilities — but the real comparison uses model_metrics
    fig = go.Figure()
    models = []
    aucs = []
    for model_id, met in all_model_metrics.items():
        models.append(MODEL_FRIENDLY_NAMES.get(model_id, model_id))
        aucs.append(met.get("roc_auc", 0))

    fig.add_trace(
        go.Bar(
            y=models,
            x=aucs,
            orientation="h",
            marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"][:len(models)],
            text=[f"{v:.3f}" for v in aucs],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Model Performance Comparison (AUC)",
        xaxis_title="AUC Score",
        yaxis_title="",
        height=300,
    )
    return fig


def _model_agreement_panel(
    prob_history_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[go.Figure, str]:
    """Show all models' latest probabilities in a bar chart + agreement text."""
    prob_cols = [c for c in test_df.columns if c.endswith("_prob")]

    if not prob_cols:
        fig = go.Figure()
        fig.add_annotation(text="No model probabilities available", showarrow=False)
        return fig, "No data available."

    latest = test_df.sort_values("date").iloc[-1]
    model_names = []
    model_probs = []
    for col in prob_cols:
        model_id = col.replace("_prob", "")
        if model_id == "y":
            model_id = "selected"
        model_names.append(MODEL_FRIENDLY_NAMES.get(model_id, model_id))
        model_probs.append(float(latest[col]))

    colors = []
    for p in model_probs:
        if p >= 0.4:
            colors.append("#d62728")
        elif p >= 0.2:
            colors.append("#ff7f0e")
        else:
            colors.append("#2ca02c")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=model_names,
            x=model_probs,
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1%}" for p in model_probs],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="All Models: Latest Test Probabilities",
        xaxis_title="Recession Probability",
        yaxis_title="",
        height=max(250, 50 * len(model_names)),
    )
    fig.update_xaxes(range=[0, 1])

    # Agreement text
    high = sum(1 for p in model_probs if p >= 0.4)
    low = sum(1 for p in model_probs if p < 0.2)
    mid = len(model_probs) - high - low
    spread = max(model_probs) - min(model_probs)

    if spread < 0.15:
        agreement_text = f"**{len(model_probs)} out of {len(model_probs)} models broadly agree** — the signal is consistent."
    elif spread > 0.4:
        agreement_text = (
            f"**Models disagree significantly** (spread: {spread:.0%}). "
            "This often means the economic signal is mixed — proceed with caution."
        )
    else:
        agreement_text = (
            f"{high} model(s) show high risk, {low} show low risk, {mid} are in between. "
            f"Spread: {spread:.0%}."
        )

    return fig, agreement_text


def _driver_bar_chart(
    importance_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    top_n: int = 10,
) -> go.Figure:
    """Top features as colored horizontal bars with descriptions."""
    logistic = importance_df[importance_df["model"] == "logistic"].copy()
    if logistic.empty:
        # Fall back to any model
        logistic = importance_df.copy()
    if logistic.empty:
        fig = go.Figure()
        fig.add_annotation(text="No importance data", showarrow=False)
        return fig

    top = logistic.nlargest(top_n, "abs_importance")
    coef_map = logistic.set_index("feature")["importance"].to_dict()

    # Determine direction based on latest snapshot change
    latest = snapshot_df.sort_values("date").iloc[-1] if not snapshot_df.empty else None
    prev = snapshot_df.sort_values("date").iloc[-2] if len(snapshot_df) >= 2 else None

    labels = []
    values = []
    colors = []

    for _, row in top.iterrows():
        feat = row["feature"]
        labels.append(_feature_label(feat))
        val = float(row["abs_importance"])
        coef = float(row["importance"])

        # Determine if this feature is currently pushing risk up or down
        pushing_up = True  # default
        if latest is not None and prev is not None and feat in latest.index and feat in prev.index:
            curr_val = latest[feat]
            prev_val = prev[feat]
            if not pd.isna(curr_val) and not pd.isna(prev_val):
                delta = float(curr_val - prev_val)
                if coef > 0:
                    pushing_up = delta > 0
                elif coef < 0:
                    pushing_up = delta < 0
                else:
                    pushing_up = delta > 0

        values.append(val)
        colors.append("#d62728" if pushing_up else "#2ca02c")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels[::-1],
            x=values[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.4f}" for v in values[::-1]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="What's Driving the Forecast (Top Features)",
        xaxis_title="Importance",
        yaxis_title="",
        height=max(350, 40 * top_n),
    )
    return fig


def _enhanced_episode_table(episode_review_df: pd.DataFrame, period_compare_df: pd.DataFrame) -> pd.DataFrame:
    """Episode review with current period row appended."""
    if episode_review_df.empty:
        return pd.DataFrame()

    df = episode_review_df.copy()
    df = df.rename(columns={
        "episode_start": "Period Start",
        "episode_end": "Period End",
        "max_prob_pre_12m": "Max Prob (12m Before)",
        "mean_prob_pre_12m": "Mean Prob (12m Before)",
        "first_signal_date": "First Signal Date",
        "lead_months": "Lead Time (months)",
        "peak_prob_during_episode": "Peak Prob During",
        "missed_early_warning": "Missed Warning?",
    })

    # Add current period row from period_compare_df
    if not period_compare_df.empty:
        current_row = period_compare_df[period_compare_df["period"].str.contains("Current", case=False, na=False)]
        if not current_row.empty:
            cr = current_row.iloc[0]
            new_row = {
                "Period Start": cr.get("start", "2023"),
                "Period End": cr.get("end", "Present"),
                "Max Prob (12m Before)": np.nan,
                "Mean Prob (12m Before)": np.nan,
                "First Signal Date": np.nan,
                "Lead Time (months)": np.nan,
                "Peak Prob During": cr.get("max_prob", np.nan),
                "Missed Warning?": "N/A (ongoing)",
            }
            new_df = pd.DataFrame([new_row])
            for col in df.columns:
                if col in new_df.columns:
                    new_df[col] = new_df[col].astype(df[col].dtype, errors="ignore")
            df = pd.concat([df, new_df], ignore_index=True)

    return df


def _florida_recession_chart(florida_index_df: pd.DataFrame) -> go.Figure:
    """Florida stress index with recession bars overlaid."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=florida_index_df["date"],
            y=florida_index_df["florida_stress_index"],
            mode="lines",
            name="Florida Stress Index",
            line={"width": 2, "color": "#ff7f0e"},
        )
    )

    # Add recession shading
    for start, end in NBER_RECESSIONS:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="gray", opacity=0.2, line_width=0,
        )

    fig.add_hrect(y0=0, y1=33, fillcolor="green", opacity=0.05, line_width=0)
    fig.add_hrect(y0=33, y1=66, fillcolor="orange", opacity=0.05, line_width=0)
    fig.add_hrect(y0=66, y1=100, fillcolor="red", opacity=0.05, line_width=0)

    fig.update_layout(
        title="Florida Stress Index with Recession Periods (gray bars)",
        yaxis_title="Stress Score",
        xaxis_title="Date",
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def _florida_vulnerabilities(florida_index_df: pd.DataFrame) -> tuple[list[str], list[str], str]:
    """Check each component to find vulnerabilities and strengths."""
    if florida_index_df.empty:
        return [], [], "No data available."

    latest = florida_index_df.sort_values("date").iloc[-1]
    vulnerabilities = []
    strengths = []

    for col, label in FLORIDA_STRESS_COMPONENTS.items():
        if col in latest.index and not pd.isna(latest[col]):
            val = float(latest[col])
            if val > 0.5:
                vulnerabilities.append(f"{label} (z-score: {val:+.2f})")
            elif val < -0.3:
                strengths.append(f"{label} (z-score: {val:+.2f})")

    if vulnerabilities:
        text = f"**Current Florida vulnerabilities**: {', '.join(vulnerabilities)}. "
    else:
        text = "**No major Florida vulnerabilities** detected in the latest data. "

    if strengths:
        text += f"**Current strengths**: {', '.join(strengths)}."
    else:
        text += "No indicators showing strong positive contribution."

    return vulnerabilities, strengths, text


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Recession Probability Lab", layout="wide")
    st.title("Recession Probability Lab")
    st.caption("Insight-driven recession monitor — US national and Florida state analysis")

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
    all_model_test_metrics = metrics.get("all_model_test_metrics", {})

    # Compute latest probability and date
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

    # Compute 6-month change for reuse
    if not prob_history_df.empty:
        recent_6 = prob_history_df.sort_values("date").tail(6)
        change_6m = float(recent_6["y_prob"].iloc[-1] - recent_6["y_prob"].iloc[0]) if len(recent_6) >= 2 else 0.0
    else:
        change_6m = 0.0

    # -----------------------------------------------------------------------
    # TABS
    # -----------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Executive Summary",
        "Recessions: Then vs Now",
        "Florida Deep Dive",
        "How the Models Work",
        "Technical Details",
    ])

    # ===================================================================
    # TAB 1: EXECUTIVE SUMMARY
    # ===================================================================
    with tab1:
        # Status Banner
        status_colors = {"Calm": "#2ca02c", "Watch": "#ff7f0e", "Warning": "#d62728"}
        banner_color = status_colors.get(status, "#999999")
        status_map = {"Calm": "OK for now", "Watch": "Watch closely", "Warning": "Higher risk"}
        status_display = status_map.get(status, status)

        st.markdown(
            f'<div style="background-color: {banner_color}; color: white; padding: 16px 24px; '
            f'border-radius: 8px; margin-bottom: 16px;">'
            f'<h2 style="margin:0; color: white;">Risk Level: {status_display.upper()}</h2>'
            f'<p style="margin:4px 0 0 0; font-size: 1.1em; color: white;">'
            f'Recession probability: <strong>{latest_prob:.1%}</strong> as of {latest_date}'
            f'{"  |  Uncertainty: " + f"{latest_prob_lower:.1%} to {latest_prob_upper:.1%}" if latest_prob_lower is not None and latest_prob_upper is not None else ""}'
            f'</p></div>',
            unsafe_allow_html=True,
        )

        # What's Happening Right Now
        st.subheader("What's Happening Right Now")
        summary = _generate_situation_summary(latest_prob, status, change_6m, markov_summary, florida_latest)
        st.markdown(summary)

        # Key Findings
        st.subheader("Key Findings")
        findings = _generate_key_findings(latest_prob, change_6m, alert_threshold, markov_summary, florida_latest, episode_review_df)
        for finding in findings:
            st.markdown(f"- {finding}")

        # "Are We Like 2008?" quick comparison
        st.subheader("Are We Like 2008?")
        if not period_compare_df.empty:
            crisis_row = period_compare_df[period_compare_df["period"].str.contains("2008", na=False)]
            current_row = period_compare_df[period_compare_df["period"].str.contains("Current", case=False, na=False)]

            if not crisis_row.empty and not current_row.empty:
                cr = crisis_row.iloc[0]
                cu = current_row.iloc[0]
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Current Probability", f"{cu['latest_prob']:.1%}", help="Latest recession probability")
                col_a.caption(f"2008 peak: {cr['max_prob']:.1%}")
                col_b.metric("Months Above Threshold", int(cu["months_above_threshold"]))
                col_b.caption(f"2008 era: {int(cr['months_above_threshold'])} months")

                regime_now = markov_summary.get("latest_regime", "N/A").title() if markov_summary else "N/A"
                col_c.metric("Current Regime", regime_now)
                col_c.caption("2008 regime: Stress")

                if latest_prob < cr["max_prob"] * 0.3:
                    st.success("Current conditions are significantly milder than the 2008 crisis period.")
                elif latest_prob < cr["max_prob"] * 0.6:
                    st.warning("Some similarities to early 2008 conditions, but not at crisis levels yet.")
                else:
                    st.error("Conditions are approaching 2008 crisis-era levels — heightened vigilance warranted.")
            else:
                st.info("Period comparison data not available for 2008 vs current.")
        else:
            st.info("Run the pipeline to generate period comparison data.")

        # Policy Implications
        st.subheader("Policy Implications & Action Areas")
        st.caption("Areas where policymakers could focus attention based on current indicator readings")
        implications = _generate_policy_implications(snapshot_df, florida_latest)
        for area, detail in implications:
            with st.expander(f"**{area}**"):
                st.write(detail)

        # Conclusions
        st.subheader("Conclusions")
        conclusions = _generate_conclusions(status, latest_prob, change_6m, markov_summary, florida_latest)
        st.markdown(conclusions)

    # ===================================================================
    # TAB 2: RECESSIONS: THEN VS NOW
    # ===================================================================
    with tab2:
        # Recession Timeline Chart
        st.subheader("Recession Timeline: Full Probability History")
        if not backtest_df.empty:
            st.plotly_chart(
                _probability_chart(
                    backtest_df,
                    prob_col="y_prob",
                    title="Walk-Forward Backtest Probability with Recession Bars",
                    threshold=alert_threshold,
                ),
                use_container_width=True,
            )
        elif not prob_history_df.empty:
            st.plotly_chart(
                _probability_chart(
                    prob_history_df,
                    prob_col="y_prob",
                    title="Probability History with Recession Bars",
                    threshold=alert_threshold,
                ),
                use_container_width=True,
            )

        # Episode Scorecard Table
        st.subheader("Episode Scorecard: How the Model Performed in Past Recessions")
        enhanced_episodes = _enhanced_episode_table(episode_review_df, period_compare_df)
        if not enhanced_episodes.empty:
            st.dataframe(enhanced_episodes, use_container_width=True)
            st.caption(
                "Each row is a historical recession. 'Lead Time' shows how many months before the recession "
                "the model crossed the alert threshold. 'Missed Warning' means the model didn't cross the "
                "threshold before the recession started."
            )
        else:
            st.info("Episode review data not available. Run the pipeline.")

        # Deep Dive: 2008 vs Now
        st.subheader("Deep Dive: 2008 vs Now")
        story_2008, story_misses = _episode_story(episode_review_df, threshold=alert_threshold)
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
                    threshold=alert_threshold,
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
                    threshold=alert_threshold,
                ),
                use_container_width=True,
            )

        # Side-by-side key indicator comparison
        st.subheader("Key Indicators: 2008 vs Now")
        comparison_indicators = ["unemployment", "yield_spread", "credit_spread", "vix_level", "payrolls_yoy", "industrial_prod_yoy"]
        available_indicators = [i for i in comparison_indicators if i in snapshot_df.columns]

        if available_indicators:
            fig = _indicator_comparison_chart(
                snapshot_df,
                indicators=available_indicators,
                period1_range=("2006-01-01", "2010-12-31"),
                period2_range=("2023-01-01", "2026-12-31"),
                period1_label="2008 Crisis Era",
                period2_label="Current (2023-2026)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Narrative
            narrative = _indicator_comparison_narrative(
                snapshot_df,
                indicators=available_indicators,
                period1_range=("2006-01-01", "2010-12-31"),
                period2_range=("2023-01-01", "2026-12-31"),
            )
            with st.expander("Detailed indicator comparison narrative"):
                st.markdown(narrative)

        # Multi-Recession Indicator Comparison
        st.subheader("Multi-Recession Indicator Comparison")
        st.caption("Select an indicator to see how its path compares across recessions, aligned to T-0 (recession start).")

        candidate_features = [
            c for c in snapshot_df.columns
            if c not in {"date", "target_recession_next_horizon", "current_recession"}
        ]
        label_to_feature = {_feature_label(f): f for f in candidate_features}
        default_labels = sorted(label_to_feature.keys())
        selected_indicator_label = st.selectbox(
            "Choose an indicator to align across recessions",
            options=default_labels,
            index=default_labels.index("Unemployment Rate") if "Unemployment Rate" in default_labels else 0,
            key="tab2_recession_indicator",
        )
        selected_indicator = label_to_feature[selected_indicator_label]

        aligned_fig = _recession_aligned_chart(
            snapshot_df, selected_indicator, episode_review_df, prob_history_df,
        )
        st.plotly_chart(aligned_fig, use_container_width=True)
        st.caption(
            "Each colored line shows the indicator's path for a past recession, aligned so T-0 is the recession start. "
            "The black line shows the current trajectory aligned to today."
        )

        # Period Comparison Table
        if not period_compare_df.empty:
            st.subheader("Period Comparison Summary")
            st.dataframe(period_compare_df, use_container_width=True)

    # ===================================================================
    # TAB 3: FLORIDA DEEP DIVE
    # ===================================================================
    with tab3:
        if florida_index_df.empty:
            st.info("Florida Stage 1 artifacts not found yet. Run `python run_pipeline.py`.")
        else:
            # Florida Status Banner
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

            fl_banner_color = {"Calm": "#2ca02c", "Watch": "#ff7f0e", "Stress": "#d62728"}.get(fl_zone, "#999")
            st.markdown(
                f'<div style="background-color: {fl_banner_color}; color: white; padding: 14px 20px; '
                f'border-radius: 8px; margin-bottom: 16px;">'
                f'<h2 style="margin:0; color: white;">Florida: {_florida_zone_text(fl_zone)}</h2>'
                f'<p style="margin:4px 0 0 0; color: white;">'
                f'Stress score: <strong>{fl_score:.1f}/100</strong> | '
                f'Monthly change: <strong>{fl_change:+.1f}</strong> | '
                f'Data date: {fl_date}</p></div>',
                unsafe_allow_html=True,
            )

            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Florida stress level", _florida_zone_text(fl_zone))
            f2.metric("Florida stress score", f"{fl_score:.1f}/100")
            f3.metric("Monthly change", f"{fl_change:+.1f}")
            f4.metric("Florida data date", fl_date)

            # Florida Stress Index Chart
            st.subheader("Florida Stress Index Over Time")
            st.plotly_chart(_florida_chart(florida_index_df), use_container_width=True)

            # Florida Stress Drivers — Component Bar Chart
            st.subheader("Florida Stress Drivers")
            st.caption(f"Latest drivers: {fl_drivers}")
            result = _florida_component_bar_chart(florida_index_df)
            if isinstance(result, tuple):
                comp_fig, changes, comp_names = result
                st.plotly_chart(comp_fig, use_container_width=True)

                # Month-over-month change arrows
                if changes and comp_names:
                    change_text = " | ".join(
                        f"{name}: {'up' if ch > 0.01 else ('down' if ch < -0.01 else 'flat')} ({ch:+.2f})"
                        for name, ch in zip(comp_names, changes)
                        if abs(ch) > 0.001
                    )
                    if change_text:
                        st.caption(f"Month-over-month changes: {change_text}")

            with st.expander("How Florida Stage 1 works"):
                st.markdown(
                    "- We combine labor, housing, and claims indicators for Florida into one score from 0 to 100.\n"
                    "- `0-33`: Calm, `34-66`: Watch, `67-100`: Stress.\n"
                    "- Each component is a z-score: positive values add stress, negative values reduce it.\n"
                    "- The score is relative to Florida's own history."
                )

            # Florida vs National Comparison
            st.subheader("Florida vs National Comparison")
            fl_nat_fig = _florida_vs_national_chart(florida_index_df, snapshot_df)
            st.plotly_chart(fl_nat_fig, use_container_width=True)

            # Florida vs national table
            if not snapshot_df.empty:
                latest_national = snapshot_df.sort_values("date").iloc[-1]
                latest_fl = florida_index_df.sort_values("date").iloc[-1]
                comparison_data = []
                fl_nat_pairs = [
                    ("fl_unemployment_rate", "unemployment", "Unemployment Rate (%)"),
                    ("fl_total_payrolls_yoy", "payrolls_yoy", "Payroll Growth (YoY %)"),
                ]
                for fl_col, nat_col, label in fl_nat_pairs:
                    fl_val = float(latest_fl[fl_col]) if fl_col in latest_fl.index and not pd.isna(latest_fl[fl_col]) else None
                    nat_val = float(latest_national[nat_col]) if nat_col in latest_national.index and not pd.isna(latest_national[nat_col]) else None
                    comparison_data.append({
                        "Indicator": label,
                        "Florida": f"{fl_val:.2f}" if fl_val is not None else "N/A",
                        "US National": f"{nat_val:.2f}" if nat_val is not None else "N/A",
                    })
                if comparison_data:
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

            # Florida During Past Recessions
            st.subheader("Florida During Past Recessions")
            st.plotly_chart(_florida_recession_chart(florida_index_df), use_container_width=True)
            st.caption("Gray bars show NBER recession periods. Notice how Florida stress typically rises before or during recessions.")

            # Florida stress at start of each historical recession
            recession_fl_data = []
            for start, end in NBER_RECESSIONS:
                start_dt = pd.to_datetime(start)
                close = florida_index_df.loc[(florida_index_df["date"] - start_dt).abs().argsort()[:1]]
                if not close.empty:
                    row = close.iloc[0]
                    recession_fl_data.append({
                        "Recession Start": start,
                        "Florida Stress Index": f"{row['florida_stress_index']:.1f}" if not pd.isna(row["florida_stress_index"]) else "N/A",
                        "Florida Zone": row.get("florida_stress_zone", "N/A"),
                    })
            if recession_fl_data:
                st.dataframe(pd.DataFrame(recession_fl_data), use_container_width=True, hide_index=True)

            st.markdown(
                "**How Florida typically behaves around recessions:** Florida's economy is sensitive to housing, "
                "tourism, and construction. Historically, Florida stress tends to rise ahead of or during national "
                "recessions, often driven by building permits declining and unemployment rising. During the 2008 "
                "crisis, Florida was hit especially hard due to its housing bubble."
            )

            # Vulnerabilities & Strengths
            st.subheader("Florida Vulnerabilities & Strengths")
            vulns, strengths, vuln_text = _florida_vulnerabilities(florida_index_df)
            st.markdown(vuln_text)

            if fl_zone in ("Watch", "Stress"):
                st.markdown(
                    "**Florida-specific recommendations:** Monitor housing permits and construction employment "
                    "closely. Consider: state-level support for tourism sector, construction workforce programs, "
                    "and housing affordability measures."
                )

    # ===================================================================
    # TAB 4: HOW THE MODELS WORK
    # ===================================================================
    with tab4:
        # Plain language intro
        st.subheader("What We're Doing")
        st.markdown(
            "We feed economic indicators — things like unemployment, interest rates, credit conditions, "
            "and stock market signals — into mathematical models that have learned from **60+ years of history**. "
            "Each model looks at the data differently: some find simple linear patterns, others detect complex "
            "interactions, and others track how confident they are in their predictions.\n\n"
            "The result is a **recession probability**: a number between 0% and 100% that represents the model's "
            "estimate of whether a recession will begin within the next 6 months. No model is perfect, but by "
            "using several approaches and comparing their answers, we get a more reliable picture than any single "
            "method alone."
        )

        # Model Cards
        st.subheader("Model Cards")
        for model_id, info in MODEL_ANALOGIES.items():
            model_met = all_model_test_metrics.get(model_id, {})
            friendly_name = MODEL_FRIENDLY_NAMES.get(model_id, model_id)
            with st.expander(f"**{friendly_name}**: {info['one_liner']}"):
                st.markdown(f"**How it works:** {info['analogy']}")
                st.markdown(f"**What it's best at:** {info['best_at']}")
                if model_met:
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("AUC (ranking ability)", f"{model_met.get('roc_auc', 0):.3f}")
                    mc2.metric("Brier Score (accuracy)", f"{model_met.get('brier_score', 0):.4f}")
                    mc3.metric("Log Loss (confidence)", f"{model_met.get('log_loss', 0):.3f}")
                    st.caption("AUC: higher is better (1.0 = perfect). Brier & Log Loss: lower is better.")

        # Markov Switching (separate card since it's different)
        if markov_summary:
            with st.expander("**Markov Switching**: Detects regime shifts in the economy"):
                st.markdown(
                    "**How it works:** Instead of predicting a probability directly, this model looks for "
                    "'hidden states' in the economy. It detects whether we're in a Calm, Watch, or Stress "
                    "regime based on how the probability signal behaves over time. Think of it like a traffic "
                    "light for the economy — it tells you which mode we're in, not just the speed."
                )
                st.markdown("**What it's best at:** Detecting when the economic environment fundamentally changes (regime shifts).")
                regime = markov_summary.get("latest_regime", "N/A").title()
                probs = markov_summary.get("regime_probabilities", {})
                rm1, rm2, rm3 = st.columns(3)
                rm1.metric("Current Regime", regime)
                rm2.metric("Stress Probability", f"{probs.get('stress', 0):.1%}")
                rm3.metric("Calm Probability", f"{probs.get('calm', 0):.1%}")

        # Model Agreement Panel
        st.subheader("Model Agreement Panel")
        agreement_fig, agreement_text = _model_agreement_panel(prob_history_df, test_df)
        st.plotly_chart(agreement_fig, use_container_width=True)
        st.markdown(agreement_text)

        # What's Driving the Forecast
        st.subheader("What's Driving the Forecast")
        st.caption("These are the economic signals that matter most to the model right now. "
                   "Red = currently pushing risk up. Green = currently pushing risk down.")
        driver_fig = _driver_bar_chart(importance_df, snapshot_df, top_n=10)
        st.plotly_chart(driver_fig, use_container_width=True)

        # Feature description table
        with st.expander("What each indicator means"):
            top_features = (
                importance_df[importance_df["model"] == "logistic"]
                .nlargest(10, "abs_importance")["feature"]
                .tolist()
            )
            if not top_features:
                top_features = importance_df.nlargest(10, "abs_importance")["feature"].tolist()
            for feat in top_features:
                desc = FEATURE_DESCRIPTIONS.get(feat, "No description available.")
                st.markdown(f"- **{_feature_label(feat)}**: {desc}")

        # Interactive Threshold Explorer
        st.subheader("Interactive Threshold Explorer")
        st.markdown(
            "The 'threshold' is the probability level at which we declare a warning. "
            "A lower threshold catches more recessions but also gives more false alarms. "
            "A higher threshold is more selective but risks missing early signals."
        )

        prob_columns = [c for c in test_df.columns if c.endswith("_prob")]
        default_prob = "y_prob" if "y_prob" in prob_columns else prob_columns[0]

        threshold = st.slider(
            "When should we call it a warning?",
            min_value=0.05,
            max_value=0.95,
            value=alert_threshold,
            step=0.05,
            key="tab4_threshold",
        )
        breakdown = _threshold_breakdown(test_df, prob_col=default_prob, threshold=threshold)

        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{breakdown['precision']:.2f}")
        m2.metric("Recall", f"{breakdown['recall']:.2f}")
        m3.metric("F1", f"{breakdown['f1']:.2f}")

        st.markdown(
            f"If we set the warning line at **{threshold:.0%}**, here's what would have happened historically: "
            f"we would have correctly warned {breakdown['tp']} times, given {breakdown['fp']} false alarms, "
            f"missed {breakdown['fn']} recessions, and correctly stayed quiet {breakdown['tn']} times."
        )

        # Confidence & Uncertainty
        st.subheader("Confidence & Uncertainty")
        bayes_cols = {"bayesian_dynamic_prob", "bayesian_dynamic_prob_lower", "bayesian_dynamic_prob_upper"}
        if bayes_cols.issubset(set(test_df.columns)):
            st.plotly_chart(_bayesian_band_chart(test_df), use_container_width=True, key="tab4_bayesian_bands")
            st.markdown(
                "The shaded area shows **how confident the model is**. A wider band means the model is "
                "less sure about its estimate. This is especially useful during transitional periods when "
                "the economic picture is unclear — if the band is wide, treat the point estimate with more caution."
            )
        else:
            st.info("Bayesian uncertainty columns not found. Re-run the pipeline.")

    # ===================================================================
    # TAB 5: TECHNICAL DETAILS
    # ===================================================================
    with tab5:
        st.caption("Full technical output for deeper analysis.")

        # Backtest Probability Series Selector
        st.subheader("Backtest Probability Series")
        prob_columns = [c for c in test_df.columns if c.endswith("_prob")]
        default_prob = "y_prob" if "y_prob" in prob_columns else prob_columns[0]
        chosen_prob_col = st.selectbox(
            "Backtest probability series",
            options=prob_columns,
            index=prob_columns.index(default_prob),
            format_func=lambda x: _model_label(x.replace("_prob", "")),
            key="tab5_prob_series",
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

        # Bayesian Uncertainty Bands
        st.subheader("Bayesian Uncertainty Bands")
        bayes_cols = {"bayesian_dynamic_prob", "bayesian_dynamic_prob_lower", "bayesian_dynamic_prob_upper"}
        if bayes_cols.issubset(set(test_df.columns)):
            st.plotly_chart(_bayesian_band_chart(test_df), use_container_width=True, key="tab5_bayesian_bands")
            st.caption(
                "Center line is Bayesian Dynamic risk estimate. Shaded band is uncertainty range "
                "(wider band means lower confidence)."
            )
        else:
            st.info("Bayesian uncertainty columns not found. Re-run `python run_pipeline.py`.")

        # Full Feature Importance Table
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
                key="tab5_importance_model",
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

        # Indicator Explorer
        st.subheader("Indicator Explorer")
        candidate_features = [
            c
            for c in snapshot_df.columns
            if c not in {"date", "target_recession_next_horizon", "current_recession"}
        ]
        label_to_feature_t5 = {_feature_label(feature): feature for feature in candidate_features}
        selected_label = st.selectbox(
            "Choose an indicator", options=sorted(label_to_feature_t5.keys()), index=0,
            key="tab5_indicator",
        )
        selected_feature = label_to_feature_t5[selected_label]

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

        # Full Model Comparison Table
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

        # Algorithm Catalog
        st.subheader("Algorithms Used And Planned")
        algo_df = _friendly_algorithm_catalog(algorithm_catalog)
        if algo_df.empty:
            st.info("Algorithm catalog not found in metrics yet. Re-run `python run_pipeline.py`.")
        else:
            st.dataframe(algo_df, use_container_width=True)

        with st.expander("Advanced algorithms status"):
            st.markdown(
                "- `Bayesian Dynamic Model` is active: gives uncertainty bands over time.\n"
                "- `Markov Switching Model` is active: detects calm/watch/stress regime shifts.\n"
                "- `NGBoost` is active: models predictive distributions for richer uncertainty handling.\n"
                "- `Temporal Fusion Transformer` is future/optional: powerful but more complex."
            )

        # Markov Switching
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

        # Training Diagnostics
        st.subheader("Training Diagnostics")
        st.caption(
            f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,} | "
            f"Forecast horizon: {metrics['horizon_months']} months | "
            f"Walk-forward min train window: {metrics.get('walkforward_min_train_months', 'n/a')}"
        )

        # Threshold Breakdown (detailed)
        st.subheader("Calibration & Threshold Breakdown")
        threshold_t5 = st.slider(
            "Threshold for detailed breakdown",
            min_value=0.05,
            max_value=0.95,
            value=alert_threshold,
            step=0.05,
            key="tab5_threshold",
        )
        breakdown_t5 = _threshold_breakdown(test_df, prob_col=chosen_prob_col, threshold=threshold_t5)

        m1t5, m2t5, m3t5 = st.columns(3)
        m1t5.metric("Precision", f"{breakdown_t5['precision']:.2f}")
        m2t5.metric("Recall", f"{breakdown_t5['recall']:.2f}")
        m3t5.metric("F1", f"{breakdown_t5['f1']:.2f}")
        st.caption(
            f"At {threshold_t5:.0%} threshold: true warnings={breakdown_t5['tp']}, false warnings={breakdown_t5['fp']}, "
            f"missed warnings={breakdown_t5['fn']}, calm correct={breakdown_t5['tn']}."
        )

        with st.expander("What Precision / Recall / F1 mean"):
            st.markdown(
                "- `Precision`: of the months we warned, how many were truly followed by recession.\n"
                "- `Recall`: of all months that really had upcoming recession, how many we caught.\n"
                "- `F1`: one score that balances precision and recall.\n"
                "- High precision + low recall means careful but misses many.\n"
                "- High recall + low precision means catches more, but with more false alarms."
            )

        # Monthly Explanations
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
                key="tab5_monthly_explanation",
            )
            row = recent_explanations[recent_explanations["date"] == pd.Timestamp(selected_date)].iloc[0]
            st.write(row["headline"])
            st.write(row["top_drivers"])
            st.caption("Driver text is heuristic: useful for learning, not strict causality.")


if __name__ == "__main__":
    main()
