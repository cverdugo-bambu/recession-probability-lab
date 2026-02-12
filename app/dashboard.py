from __future__ import annotations

import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st
from scipy.special import expit as sigmoid
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recession_project.config import DEFAULT_ALERT_THRESHOLD, FEATURE_DESCRIPTIONS, FEATURE_LABELS

ARTIFACT_DIR = ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Modern Plotly theme — white background, clean grid, Inter-style font
# ---------------------------------------------------------------------------
_MODERN_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=13, color="#1a1a2e"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        title=dict(font=dict(size=17, color="#1a1a2e"), x=0.0, xanchor="left"),
        colorway=[
            "#1f77b4", "#ff6b35", "#2ecc71", "#e74c3c",
            "#9b59b6", "#1abc9c", "#f39c12", "#3498db",
            "#e67e22", "#2c3e50",
        ],
        xaxis=dict(
            showgrid=True,
            gridcolor="#eaedf2",
            gridwidth=1,
            zeroline=False,
            linecolor="#d0d5dd",
            linewidth=1,
            title=dict(font=dict(size=12, color="#475467")),
            tickfont=dict(size=11, color="#475467"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#eaedf2",
            gridwidth=1,
            zeroline=False,
            linecolor="#d0d5dd",
            linewidth=1,
            title=dict(font=dict(size=12, color="#475467")),
            tickfont=dict(size=11, color="#475467"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#eaedf2",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=48, r=24, t=56, b=40),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#d0d5dd",
            font=dict(size=12, color="#1a1a2e"),
        ),
        hovermode="x unified",
    ),
)
pio.templates["modern_white"] = _MODERN_TEMPLATE
pio.templates.default = "modern_white"

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

# ---------------------------------------------------------------------------
# Action Center constants
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}

SEVERITY_COLORS = {
    "Critical": "#d62728",
    "High": "#ff7f0e",
    "Medium": "#f0c929",
    "Low": "#2ca02c",
}

# Each entry: thresholds (from worst to least bad), area, review cadence,
# direction ("higher" or "lower" is worse), danger_zone value for early warning
INDICATOR_THRESHOLDS: dict[str, dict] = {
    "unemployment_3m_delta": {
        "thresholds": [
            (0.5, "Critical", "Unemployment rising rapidly (+{val:.2f}pp in 3 months). Activate emergency labor market programs: expand UI benefits, accelerate workforce retraining, coordinate with state agencies."),
            (0.3, "High", "Unemployment rising significantly (+{val:.2f}pp in 3 months). Prepare labor market support: review UI eligibility, fund job placement services, monitor mass layoff filings."),
            (0.1, "Medium", "Unemployment ticking up (+{val:.2f}pp in 3 months). Monitor weekly claims data and regional employment trends for acceleration."),
        ],
        "area": "Labor",
        "review": "weekly",
        "direction": "higher",
        "danger_zone": 0.5,
    },
    "credit_spread": {
        "thresholds": [
            (3.5, "Critical", "Credit spreads at {val:.2f}% — severe financial stress. Review bank lending conditions, ensure liquidity backstops are operational, monitor corporate default risk."),
            (2.5, "High", "Credit spreads elevated at {val:.2f}%. Monitor corporate bond markets, stress-test bank portfolios, review counterparty exposures."),
            (2.0, "Medium", "Credit spreads at {val:.2f}% — above normal. Track weekly issuance and secondary market liquidity."),
        ],
        "area": "Financial",
        "review": "daily",
        "direction": "higher",
        "danger_zone": 3.5,
    },
    "yield_spread": {
        "thresholds": [
            (-0.5, "Critical", "Yield curve deeply inverted at {val:.2f}%. Historically precedes recessions by 6-18 months. Assess monetary policy stance and forward rate expectations."),
            (0.0, "High", "Yield curve inverted at {val:.2f}%. Historically a strong recession signal. Review rate path, forward guidance, duration positioning."),
            (0.5, "Medium", "Yield curve flattening at {val:.2f}%. Monitor for potential inversion — a classic leading indicator."),
        ],
        "area": "Monetary",
        "review": "weekly",
        "direction": "lower",
        "danger_zone": -0.5,
    },
    "vix_level": {
        "thresholds": [
            (35, "Critical", "VIX at {val:.1f} — extreme market fear. Expect elevated volatility, potential margin calls. Ensure risk limits are tight and hedging is in place."),
            (25, "High", "VIX at {val:.1f} — elevated fear. Review portfolio hedges, reduce policy uncertainty in communications, monitor options market."),
            (20, "Medium", "VIX at {val:.1f} — above average. Watch for catalysts that could spike volatility further."),
        ],
        "area": "Market",
        "review": "daily",
        "direction": "higher",
        "danger_zone": 35,
    },
    "nfci": {
        "thresholds": [
            (0.5, "Critical", "Financial conditions severely tight (NFCI: {val:.2f}). Credit markets stressed — monitor bank lending, commercial paper, and money markets."),
            (0.0, "High", "Financial conditions tightening (NFCI: {val:.2f}). Review credit availability, interbank rates, and financial sector stability."),
            (-0.3, "Medium", "Financial conditions firming (NFCI: {val:.2f}). Track weekly for acceleration of tightening."),
        ],
        "area": "Financial",
        "review": "weekly",
        "direction": "higher",
        "danger_zone": 0.5,
    },
    "payrolls_yoy": {
        "thresholds": [
            (-1.0, "Critical", "Payrolls contracting {val:.1f}% YoY — active job losses. Coordinate federal/state employment support, consider stimulus measures."),
            (0.0, "High", "Payroll growth stalled at {val:.1f}% YoY. Economy near zero job creation — prepare contingency plans."),
            (0.5, "Medium", "Payroll growth slowing to {val:.1f}% YoY. Monitor hiring surveys and job openings for further weakening."),
        ],
        "area": "Labor",
        "review": "biweekly",
        "direction": "lower",
        "danger_zone": -1.0,
    },
    "industrial_prod_yoy": {
        "thresholds": [
            (-3.0, "Critical", "Industrial production down {val:.1f}% YoY — severe contraction. Assess supply chain disruptions, manufacturing sector support, trade policy impacts."),
            (-1.0, "High", "Industrial production falling {val:.1f}% YoY. Manufacturing sector weakening — review sector-specific support and trade conditions."),
            (0.0, "Medium", "Industrial production flat/declining at {val:.1f}% YoY. Monitor factory orders and capacity utilization."),
        ],
        "area": "Labor",
        "review": "monthly",
        "direction": "lower",
        "danger_zone": -3.0,
    },
    "unemployment": {
        "thresholds": [
            (7.0, "Critical", "Unemployment at {val:.1f}% — crisis level. Full employment support measures warranted."),
            (5.5, "High", "Unemployment at {val:.1f}% — elevated. Expand job training and placement programs."),
            (4.5, "Medium", "Unemployment at {val:.1f}% — rising above natural rate. Watch for continued deterioration."),
        ],
        "area": "Labor",
        "review": "biweekly",
        "direction": "higher",
        "danger_zone": 7.0,
    },
    "inflation_yoy": {
        "thresholds": [
            (6.0, "Critical", "Inflation at {val:.1f}% YoY — price stability risk. Assess monetary tightening impact, cost-of-living support, supply-side measures."),
            (4.0, "High", "Inflation at {val:.1f}% YoY — above target. Monitor Fed response, wage-price dynamics, and inflation expectations."),
            (3.0, "Medium", "Inflation at {val:.1f}% YoY — above 2% target. Track core vs. headline divergence and persistence."),
        ],
        "area": "Monetary",
        "review": "monthly",
        "direction": "higher",
        "danger_zone": 6.0,
    },
    "fedfunds": {
        "thresholds": [
            (5.5, "High", "Fed funds rate at {val:.2f}% — restrictive territory. Monitor for over-tightening risk, credit contraction, and housing impact."),
            (4.0, "Medium", "Fed funds rate at {val:.2f}% — elevated. Track forward guidance and rate-sensitive sectors."),
        ],
        "area": "Monetary",
        "review": "biweekly",
        "direction": "higher",
        "danger_zone": 5.5,
    },
}

FLORIDA_ACTION_THRESHOLDS = [
    {
        "zone": "Stress",
        "severity": "High",
        "action": "Florida in Stress zone (score: {score:.0f}/100). Activate state-level contingency: housing market support, construction monitoring, tourism sector assessment. Coordinate with state agencies.",
        "area": "State",
        "review": "weekly",
    },
    {
        "zone": "Watch",
        "severity": "Medium",
        "action": "Florida in Watch zone (score: {score:.0f}/100). Increase monitoring of state labor indicators, building permits, and initial claims. Prepare support plans.",
        "area": "State",
        "review": "biweekly",
    },
]

PROBABILITY_ACTION_THRESHOLDS = [
    {
        "threshold": 0.40,
        "severity": "Critical",
        "action": "Recession probability at {prob:.1%} — above alert threshold. Activate full monitoring protocol: daily indicator reviews, cross-model validation, scenario planning.",
        "area": "Overall",
        "review": "daily",
    },
    {
        "threshold": 0.25,
        "severity": "High",
        "action": "Recession probability at {prob:.1%} — elevated. Increase monitoring frequency, review portfolio hedges, prepare contingency communications.",
        "area": "Overall",
        "review": "weekly",
    },
    {
        "threshold": 0.15,
        "severity": "Medium",
        "action": "Recession probability at {prob:.1%} — above baseline. Maintain heightened awareness and weekly check-ins.",
        "area": "Overall",
        "review": "biweekly",
    },
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
# Labor Market Deep Dive — chart helpers
# ---------------------------------------------------------------------------

def _load_labor_deep_dive() -> pd.DataFrame | None:
    """Load labor deep dive CSV artifact. Returns None if not available."""
    path = ARTIFACT_DIR / "labor_deep_dive.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df if not df.empty else None


def _labor_insight_card(title: str, body: str, accent_color: str = "#1f77b4") -> str:
    """Return styled HTML card matching existing dashboard card pattern."""
    return (
        f'<div style="border-left: 5px solid {accent_color}; padding: 10px 16px; '
        f'margin-bottom: 12px; background-color: #f8f9fa; border-radius: 0 6px 6px 0;">'
        f'<strong style="font-size:1.05em;">{title}</strong><br/>'
        f'<span style="font-size:0.95em; color:#333;">{body}</span></div>'
    )


def _add_recession_shading(fig: go.Figure) -> None:
    """Add NBER recession shading rectangles to a plotly figure."""
    for start, end in NBER_RECESSIONS:
        fig.add_vrect(x0=start, x1=end, fillcolor="gray", opacity=0.18, line_width=0)


def _headline_vs_reality_chart(df: pd.DataFrame) -> go.Figure | None:
    """U-3 vs U-6 unemployment overlay with spread subplot."""
    if "u6_rate" not in df.columns or "u3_rate" not in df.columns:
        return None
    merged = df[["date", "u6_rate", "u3_rate"]].dropna()
    if merged.empty:
        return None

    merged["spread"] = merged["u6_rate"] - merged["u3_rate"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Official (U-3) vs Broad (U-6) Unemployment", "U-6 minus U-3 Spread"),
    )
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["u3_rate"], name="U-3 (Official)",
                             line=dict(width=2, color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["u6_rate"], name="U-6 (Broad)",
                             line=dict(width=2, color="#e74c3c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["spread"], name="Spread",
                             line=dict(width=2, color="#9b59b6"),
                             fill="tozeroy", fillcolor="rgba(155, 89, 182, 0.1)"), row=2, col=1)
    _add_recession_shading(fig)
    fig.update_yaxes(title_text="Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (pp)", row=2, col=1)
    fig.update_xaxes(range=[merged["date"].min(), merged["date"].max()])
    fig.update_layout(height=520, legend=dict(orientation="h", y=1.06))
    return fig


def _sector_employment_chart(df: pd.DataFrame) -> go.Figure | None:
    """YoY % change for 5 employment sectors."""
    sector_cols = {
        "info_sector_emp": ("Information", "#e74c3c"),
        "prof_business_emp": ("Prof. & Business Svc.", "#ff6b35"),
        "computer_systems_emp": ("Computer Systems Design", "#9b59b6"),
        "healthcare_emp": ("Education & Health", "#2ecc71"),
        "government_emp": ("Government", "#3498db"),
    }
    available = {k: v for k, v in sector_cols.items() if k in df.columns}
    if not available:
        return None

    fig = go.Figure()
    for col, (label, color) in available.items():
        s = df.set_index("date")[col].dropna()
        yoy = s.pct_change(12) * 100
        fig.add_trace(go.Scatter(x=yoy.index, y=yoy.values, name=label,
                                 line=dict(width=2, color=color)))
    _add_recession_shading(fig)
    fig.add_hline(y=0, line_dash="dot", line_color="#999", line_width=1)
    fig.update_layout(
        title="Year-over-Year Employment Growth by Sector",
        yaxis_title="YoY Change (%)",
        height=440,
        xaxis_range=[df["date"].min(), df["date"].max()],
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def _tech_focus_indexed_chart(df: pd.DataFrame, base_date: str) -> go.Figure | None:
    """Indexed (base=100) employment for tech-adjacent sectors from a chosen base date."""
    cols = {
        "info_sector_emp": ("Information", "#e74c3c"),
        "computer_systems_emp": ("Computer Systems Design", "#9b59b6"),
        "prof_business_emp": ("Prof. & Business Svc.", "#ff6b35"),
    }
    available = {k: v for k, v in cols.items() if k in df.columns}
    if not available:
        return None

    base_ts = pd.Timestamp(base_date)
    fig = go.Figure()
    for col, (label, color) in available.items():
        s = df.set_index("date")[col].dropna()
        if base_ts not in s.index:
            # Find nearest date
            idx = s.index.get_indexer([base_ts], method="nearest")[0]
            base_val = s.iloc[idx]
        else:
            base_val = s.loc[base_ts]
        if base_val == 0 or pd.isna(base_val):
            continue
        indexed = (s / base_val) * 100
        fig.add_trace(go.Scatter(x=indexed.index, y=indexed.values, name=label,
                                 line=dict(width=2.5, color=color)))
    _add_recession_shading(fig)
    fig.add_hline(y=100, line_dash="dot", line_color="#999", line_width=1,
                  annotation_text="Base = 100", annotation_position="bottom right",
                  annotation_font_size=10, annotation_font_color="#999")
    fig.update_layout(
        title=f"Tech-Adjacent Employment (Indexed, {base_date[:7]} = 100)",
        yaxis_title="Index (base = 100)",
        height=440,
        xaxis_range=[base_ts, df["date"].max()],
    )
    return fig


def _jolts_chart(df: pd.DataFrame) -> tuple[go.Figure | None, go.Figure | None]:
    """JOLTS openings/hires/quits lines + openings-to-hires ratio."""
    jolts_cols = {
        "jolts_openings": ("Job Openings", "#1f77b4"),
        "jolts_hires": ("Hires", "#2ecc71"),
        "jolts_quits": ("Quits Rate", "#ff6b35"),
    }
    available = {k: v for k, v in jolts_cols.items() if k in df.columns}
    if not available:
        return None, None

    # Main JOLTS chart
    fig1 = go.Figure()
    for col, (label, color) in available.items():
        s = df.set_index("date")[col].dropna()
        fig1.add_trace(go.Scatter(x=s.index, y=s.values, name=label,
                                  line=dict(width=2, color=color)))
    _add_recession_shading(fig1)
    fig1.update_layout(
        title="JOLTS: Job Openings, Hires & Quits",
        yaxis_title="Thousands / Rate",
        height=400,
        xaxis_range=[df["date"].min(), df["date"].max()],
        legend=dict(orientation="h", y=-0.15),
    )

    # Ratio chart
    fig2 = None
    if "jolts_openings" in df.columns and "jolts_hires" in df.columns:
        ratio_df = df[["date", "jolts_openings", "jolts_hires"]].dropna()
        if not ratio_df.empty:
            ratio_df = ratio_df.copy()
            ratio_df["ratio"] = ratio_df["jolts_openings"] / ratio_df["jolts_hires"]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=ratio_df["date"], y=ratio_df["ratio"], name="Openings / Hires",
                line=dict(width=2.5, color="#1f77b4"),
                fill="tozeroy", fillcolor="rgba(31, 119, 180, 0.08)",
            ))
            fig2.add_hline(y=1.0, line_dash="dot", line_color="#e74c3c", line_width=1.5,
                           annotation_text="1:1 ratio", annotation_position="bottom right",
                           annotation_font_size=10, annotation_font_color="#e74c3c")
            _add_recession_shading(fig2)
            fig2.update_layout(
                title="Openings-to-Hires Ratio (higher = jobs posted but not filled)",
                yaxis_title="Ratio",
                height=340,
                xaxis_range=[df["date"].min(), df["date"].max()],
            )
    return fig1, fig2


def _hidden_slack_chart(df: pd.DataFrame) -> go.Figure | None:
    """Avg unemployment duration + involuntary part-time subplots."""
    has_dur = "unemp_duration_mean" in df.columns
    has_pt = "involuntary_part_time" in df.columns
    if not has_dur and not has_pt:
        return None

    rows = (1 if has_dur else 0) + (1 if has_pt else 0)
    titles = []
    if has_dur:
        titles.append("Average Weeks Unemployed")
    if has_pt:
        titles.append("Involuntary Part-Time Workers (thousands)")

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        subplot_titles=titles)
    row_idx = 1
    if has_dur:
        s = df.set_index("date")["unemp_duration_mean"].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Avg Weeks Unemployed",
                                 line=dict(width=2, color="#e74c3c"),
                                 fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.08)"),
                      row=row_idx, col=1)
        fig.update_yaxes(title_text="Weeks", row=row_idx, col=1)
        row_idx += 1
    if has_pt:
        s = df.set_index("date")["involuntary_part_time"].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Involuntary Part-Time",
                                 line=dict(width=2, color="#ff6b35"),
                                 fill="tozeroy", fillcolor="rgba(255, 107, 53, 0.08)"),
                      row=row_idx, col=1)
        fig.update_yaxes(title_text="Thousands", row=row_idx, col=1)

    _add_recession_shading(fig)
    fig.update_xaxes(range=[df["date"].min(), df["date"].max()])
    fig.update_layout(height=420 if rows == 2 else 300, legend=dict(orientation="h", y=1.06))
    return fig


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
            line={"width": 2.5, "color": "#1f77b4"},
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.08)",
        )
    )

    if "current_recession" in df.columns:
        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["current_recession"] * 0.15,
                name="Actual recession periods",
                marker_color="rgba(214, 39, 40, 0.18)",
                marker_line_width=0,
            )
        )

    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dot",
            line_color="#ff6b35",
            line_width=1.5,
            annotation_text=f"Alert {threshold:.0%}",
            annotation_position="top left",
            annotation_font_size=11,
            annotation_font_color="#ff6b35",
        )

    fig.update_layout(
        yaxis_title="Probability",
        xaxis_title="",
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
            line={"width": 2.5, "color": "#ff6b35"},
            fill="tozeroy",
            fillcolor="rgba(255, 107, 53, 0.06)",
        )
    )
    fig.add_hrect(y0=0, y1=33, fillcolor="#2ecc71", opacity=0.06, line_width=0, annotation_text="Calm")
    fig.add_hrect(y0=33, y1=66, fillcolor="#f39c12", opacity=0.06, line_width=0, annotation_text="Watch")
    fig.add_hrect(y0=66, y1=100, fillcolor="#e74c3c", opacity=0.06, line_width=0, annotation_text="Stress")
    fig.update_layout(
        title="Florida Stage 1 Stress Index (0-100)",
        yaxis_title="Stress Score",
        xaxis_title="",
    )
    fig.update_yaxes(range=[0, 100])
    return fig


def _markov_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    regime_colors = {"calm": "#2ecc71", "watch": "#f39c12", "stress": "#e74c3c"}
    prob_cols = [c for c in df.columns if c.startswith("regime_prob_")]
    for col in prob_cols:
        regime_id = col.replace("regime_prob_", "")
        label = regime_id.title()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[col],
                mode="lines",
                stackgroup="one",
                name=f"{label} probability",
                line={"width": 0.5, "color": regime_colors.get(regime_id, "#999")},
                fillcolor=regime_colors.get(regime_id, "#999"),
            )
        )
    fig.update_layout(
        title="Markov Switching Regime Probabilities",
        yaxis_title="Probability",
        xaxis_title="",
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
            fillcolor="rgba(31, 119, 180, 0.12)",
            line={"width": 0},
            name="Uncertainty band",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["bayesian_dynamic_prob"],
            mode="lines",
            line={"width": 2.5, "color": "#1f77b4"},
            name="Bayesian dynamic probability",
        )
    )
    fig.update_layout(
        title="Bayesian Dynamic Model: Probability with Uncertainty Band",
        yaxis_title="Probability",
        xaxis_title="",
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
# Action Center helper functions
# ---------------------------------------------------------------------------


def _generate_action_items(
    snapshot_df: pd.DataFrame,
    latest_prob: float,
    florida_latest: dict | None,
) -> list[dict]:
    """Check all indicators against thresholds, return severity-sorted action list."""
    items: list[dict] = []
    if snapshot_df.empty:
        return items

    latest = snapshot_df.sort_values("date").iloc[-1]

    # Per-indicator thresholds
    for indicator, cfg in INDICATOR_THRESHOLDS.items():
        if indicator not in latest.index or pd.isna(latest[indicator]):
            continue
        val = float(latest[indicator])
        direction = cfg["direction"]

        for level_val, severity, template in cfg["thresholds"]:
            triggered = (val > level_val) if direction == "higher" else (val < level_val)
            if triggered:
                items.append({
                    "severity": severity,
                    "area": cfg["area"],
                    "indicator": indicator,
                    "action": template.format(val=val),
                    "data_value": val,
                    "data_label": f"{_feature_label(indicator)}: {val:.2f}",
                    "review": cfg["review"],
                })
                break  # Only the worst matching threshold

    # Florida stress zone
    if florida_latest:
        fl_zone = florida_latest.get("florida_stress_zone", "Calm")
        fl_score = float(florida_latest.get("florida_stress_index", 0))
        for flt in FLORIDA_ACTION_THRESHOLDS:
            if fl_zone == flt["zone"]:
                items.append({
                    "severity": flt["severity"],
                    "area": flt["area"],
                    "indicator": "florida_stress",
                    "action": flt["action"].format(score=fl_score),
                    "data_value": fl_score,
                    "data_label": f"Florida Stress Index: {fl_score:.0f}/100 ({fl_zone})",
                    "review": flt["review"],
                })
                break

    # Recession probability level
    for pt in PROBABILITY_ACTION_THRESHOLDS:
        if latest_prob >= pt["threshold"]:
            items.append({
                "severity": pt["severity"],
                "area": pt["area"],
                "indicator": "recession_probability",
                "action": pt["action"].format(prob=latest_prob),
                "data_value": latest_prob,
                "data_label": f"Recession Probability: {latest_prob:.1%}",
                "review": pt["review"],
            })
            break

    # Sort by severity
    items.sort(key=lambda x: SEVERITY_ORDER.get(x["severity"], 99))
    return items


def _generate_what_changed(
    snapshot_df: pd.DataFrame,
    prob_history_df: pd.DataFrame,
) -> dict:
    """Compute month-over-month deltas, biggest movers, threshold crossings."""
    result: dict = {
        "prob_current": None,
        "prob_previous": None,
        "prob_delta": None,
        "movers_up": [],
        "movers_down": [],
        "threshold_crossings": [],
        "all_changes": [],
    }

    if prob_history_df.empty or len(prob_history_df) < 2:
        return result

    sorted_prob = prob_history_df.sort_values("date")
    result["prob_current"] = float(sorted_prob.iloc[-1]["y_prob"])
    result["prob_previous"] = float(sorted_prob.iloc[-2]["y_prob"])
    result["prob_delta"] = result["prob_current"] - result["prob_previous"]

    if snapshot_df.empty or len(snapshot_df) < 2:
        return result

    sorted_snap = snapshot_df.sort_values("date")
    latest = sorted_snap.iloc[-1]
    previous = sorted_snap.iloc[-2]

    feature_cols = [
        c for c in snapshot_df.columns
        if c not in {"date", "target_recession_next_horizon", "current_recession"}
    ]

    changes = []
    for col in feature_cols:
        if col not in latest.index or pd.isna(latest[col]) or pd.isna(previous[col]):
            continue
        curr_val = float(latest[col])
        prev_val = float(previous[col])
        delta = curr_val - prev_val
        if abs(delta) < 1e-10:
            continue
        changes.append({
            "indicator": col,
            "label": _feature_label(col),
            "current": curr_val,
            "previous": prev_val,
            "delta": delta,
            "pct_change": (delta / abs(prev_val) * 100) if abs(prev_val) > 1e-10 else 0.0,
        })

    changes.sort(key=lambda x: abs(x["delta"]), reverse=True)
    result["all_changes"] = changes

    # Top 5 movers up and down
    movers_up = sorted([c for c in changes if c["delta"] > 0], key=lambda x: x["delta"], reverse=True)[:5]
    movers_down = sorted([c for c in changes if c["delta"] < 0], key=lambda x: x["delta"])[:5]
    result["movers_up"] = movers_up
    result["movers_down"] = movers_down

    # Threshold crossings: indicators that crossed a threshold boundary this month
    for indicator, cfg in INDICATOR_THRESHOLDS.items():
        if indicator not in latest.index or pd.isna(latest[indicator]) or pd.isna(previous[indicator]):
            continue
        curr_val = float(latest[indicator])
        prev_val = float(previous[indicator])
        direction = cfg["direction"]

        for level_val, severity, _ in cfg["thresholds"]:
            if direction == "higher":
                crossed_now = curr_val > level_val
                crossed_before = prev_val > level_val
            else:
                crossed_now = curr_val < level_val
                crossed_before = prev_val < level_val

            if crossed_now and not crossed_before:
                result["threshold_crossings"].append({
                    "indicator": indicator,
                    "label": _feature_label(indicator),
                    "severity": severity,
                    "direction": "worsened into" if crossed_now else "improved out of",
                    "threshold_value": level_val,
                    "current": curr_val,
                    "previous": prev_val,
                })
                break
            elif crossed_before and not crossed_now:
                result["threshold_crossings"].append({
                    "indicator": indicator,
                    "label": _feature_label(indicator),
                    "severity": severity,
                    "direction": "improved out of",
                    "threshold_value": level_val,
                    "current": curr_val,
                    "previous": prev_val,
                })
                break

    return result


def _generate_indicator_heatmap_data(
    snapshot_df: pd.DataFrame,
) -> list[dict]:
    """Distance-to-danger, direction, historical percentile per indicator."""
    if snapshot_df.empty or len(snapshot_df) < 2:
        return []

    sorted_snap = snapshot_df.sort_values("date")
    latest = sorted_snap.iloc[-1]
    previous = sorted_snap.iloc[-2]

    rows: list[dict] = []
    for indicator, cfg in INDICATOR_THRESHOLDS.items():
        if indicator not in latest.index or pd.isna(latest[indicator]):
            continue
        val = float(latest[indicator])
        prev_val = float(previous[indicator]) if not pd.isna(previous[indicator]) else val
        danger = cfg["danger_zone"]
        direction = cfg["direction"]

        # Distance to danger zone (0% = at danger, 100% = maximally safe)
        if direction == "higher":
            # Higher is worse: danger when val > danger_zone
            distance_abs = danger - val
            distance_pct = max(0.0, min(100.0, (distance_abs / abs(danger) * 100) if danger != 0 else 100.0))
            crossed = val >= danger
        else:
            # Lower is worse: danger when val < danger_zone
            distance_abs = val - danger
            distance_pct = max(0.0, min(100.0, (distance_abs / abs(danger) * 100) if danger != 0 else 100.0))
            crossed = val <= danger

        # Direction: improving or worsening
        delta = val - prev_val
        if direction == "higher":
            improving = delta < 0
        else:
            improving = delta > 0

        # Historical percentile
        col_data = sorted_snap[indicator].dropna()
        if len(col_data) > 0:
            percentile = float((col_data < val).sum() / len(col_data) * 100)
        else:
            percentile = 50.0

        # Color zone
        if crossed:
            color = "red"
            zone_label = "Crossed"
        elif distance_pct < 15:
            color = "orange"
            zone_label = "Near"
        elif distance_pct < 40:
            color = "gold"
            zone_label = "Approaching"
        else:
            color = "green"
            zone_label = "Safe"

        rows.append({
            "indicator": indicator,
            "label": _feature_label(indicator),
            "current_value": val,
            "danger_zone": danger,
            "distance_pct": distance_pct,
            "crossed": crossed,
            "direction_text": "Improving" if improving else "Worsening",
            "direction_symbol": "\u2193" if improving else "\u2191" if not improving and abs(delta) > 1e-10 else "\u2194",
            "percentile": percentile,
            "color": color,
            "zone_label": zone_label,
            "area": cfg["area"],
        })

    # Sort by distance (closest to danger first)
    rows.sort(key=lambda x: x["distance_pct"])
    return rows


def _compute_whatif_probability(
    current_prob: float,
    importance_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    scenario: dict[str, float],
) -> tuple[float, list[str]]:
    """Delta-logit scenario estimation using logistic coefficients.

    new_prob = sigmoid(logit(current_prob) + sum(coef * (z_new - z_current)))
    """
    explanations: list[str] = []

    if current_prob <= 0.001:
        current_prob = 0.001
    if current_prob >= 0.999:
        current_prob = 0.999

    current_logit = float(logit(current_prob))

    logistic_imp = importance_df[importance_df["model"] == "logistic"]
    if logistic_imp.empty:
        return current_prob, ["No logistic model coefficients available."]
    coef_map = logistic_imp.set_index("feature")["importance"].to_dict()

    feature_cols = [
        c for c in snapshot_df.columns
        if c not in {"date", "target_recession_next_horizon", "current_recession"}
    ]

    # Compute mean and std from full training dataset for standardization
    stats = {}
    for col in feature_cols:
        col_data = snapshot_df[col].dropna()
        if len(col_data) > 1:
            stats[col] = {"mean": float(col_data.mean()), "std": float(col_data.std())}

    sorted_snap = snapshot_df.sort_values("date")
    latest = sorted_snap.iloc[-1]

    logit_delta = 0.0
    for feature, new_val in scenario.items():
        if feature not in coef_map or feature not in stats:
            continue
        coef = coef_map[feature]
        mean = stats[feature]["mean"]
        std = stats[feature]["std"]
        if std < 1e-10:
            continue

        current_val = float(latest[feature]) if feature in latest.index and not pd.isna(latest[feature]) else mean
        z_current = (current_val - mean) / std
        z_new = (new_val - mean) / std
        delta_z = z_new - z_current
        contribution = coef * delta_z
        logit_delta += contribution
        explanations.append(
            f"{_feature_label(feature)}: {current_val:.2f} -> {new_val:.2f} "
            f"(coef={coef:.4f}, delta-z={delta_z:+.2f}, logit contribution={contribution:+.4f})"
        )

    new_logit = current_logit + logit_delta
    new_prob = float(sigmoid(new_logit))
    return new_prob, explanations


def _generate_briefing_text(
    latest_prob: float,
    latest_date: str,
    status: str,
    change_6m: float,
    action_items: list[dict],
    what_changed: dict,
    snapshot_df: pd.DataFrame,
    markov_summary: dict | None,
    florida_latest: dict | None,
    alert_threshold: float,
) -> str:
    """Assemble all data into downloadable Markdown briefing."""
    now_str = datetime.date.today().isoformat()
    lines: list[str] = []

    lines.append(f"# Recession Risk Briefing — {now_str}")
    lines.append("")
    lines.append(f"**Generated:** {now_str}")
    lines.append(f"**Data as of:** {latest_date}")
    lines.append(f"**Risk Level:** {status}")
    lines.append(f"**Recession Probability:** {latest_prob:.1%}")
    lines.append(f"**6-Month Change:** {change_6m:+.1%}")
    lines.append(f"**Alert Threshold:** {alert_threshold:.0%}")
    lines.append("")

    # Regime and Florida
    if markov_summary:
        regime = markov_summary.get("latest_regime", "unknown")
        lines.append(f"**Markov Regime:** {regime}")
    if florida_latest:
        fl_zone = florida_latest.get("florida_stress_zone", "Unknown")
        fl_score = florida_latest.get("florida_stress_index", 0)
        lines.append(f"**Florida Stress:** {fl_zone} ({fl_score:.0f}/100)")
    lines.append("")

    # Action Items
    lines.append("## Priority Action Items")
    lines.append("")
    if action_items:
        for item in action_items:
            lines.append(f"- **[{item['severity']}]** ({item['area']}) {item['action']} *(Review: {item['review']})*")
    else:
        lines.append("- No action items triggered — all indicators within normal ranges.")
    lines.append("")

    # What Changed
    lines.append("## What Changed This Month")
    lines.append("")
    if what_changed.get("prob_delta") is not None:
        arrow = "\u2191" if what_changed["prob_delta"] > 0 else "\u2193" if what_changed["prob_delta"] < 0 else "\u2194"
        lines.append(f"- Probability: {what_changed['prob_previous']:.1%} -> {what_changed['prob_current']:.1%} ({what_changed['prob_delta']:+.1%}) {arrow}")
    if what_changed.get("movers_up"):
        lines.append("")
        lines.append("**Top movers up:**")
        for m in what_changed["movers_up"]:
            lines.append(f"- {m['label']}: {m['previous']:.2f} -> {m['current']:.2f} ({m['delta']:+.2f})")
    if what_changed.get("movers_down"):
        lines.append("")
        lines.append("**Top movers down:**")
        for m in what_changed["movers_down"]:
            lines.append(f"- {m['label']}: {m['previous']:.2f} -> {m['current']:.2f} ({m['delta']:+.2f})")
    if what_changed.get("threshold_crossings"):
        lines.append("")
        lines.append("**Threshold crossings:**")
        for tc in what_changed["threshold_crossings"]:
            lines.append(f"- {tc['label']} {tc['direction']} {tc['severity']} zone ({tc['previous']:.2f} -> {tc['current']:.2f})")
    lines.append("")

    # Indicator Snapshot
    lines.append("## Current Indicator Snapshot")
    lines.append("")
    if not snapshot_df.empty:
        latest = snapshot_df.sort_values("date").iloc[-1]
        feature_cols = [
            c for c in snapshot_df.columns
            if c not in {"date", "target_recession_next_horizon", "current_recession"}
        ]
        lines.append("| Indicator | Value |")
        lines.append("|---|---|")
        for col in feature_cols:
            if col in latest.index and not pd.isna(latest[col]):
                lines.append(f"| {_feature_label(col)} | {float(latest[col]):.4f} |")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by Recession Probability Lab — Action Center*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="Recession Probability Lab", layout="wide")

    # Modern CSS overrides
    st.markdown(
        """
        <style>
        /* Clean white background throughout */
        .stApp { background-color: #ffffff; }
        section[data-testid="stSidebar"] { background-color: #f5f7fa; }

        /* Tighter, modern tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            border-bottom: 2px solid #eaedf2;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            font-weight: 500;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #f5f7fa;
            border-bottom: 2px solid #1f77b4;
        }

        /* Metric cards */
        [data-testid="stMetric"] {
            background-color: #f5f7fa;
            padding: 14px 18px;
            border-radius: 10px;
            border: 1px solid #eaedf2;
        }
        [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #475467; }
        [data-testid="stMetricValue"] { font-weight: 600; }

        /* Expander styling */
        .streamlit-expanderHeader { font-weight: 500; }

        /* Dataframe styling */
        .stDataFrame { border-radius: 8px; border: 1px solid #eaedf2; }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
    tab1, tab2, tab3, tab7, tab4, tab5, tab6 = st.tabs([
        "Executive Summary",
        "Recessions: Then vs Now",
        "Florida Deep Dive",
        "Labor Market Deep Dive",
        "How the Models Work",
        "Technical Details",
        "Action Center",
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
    # TAB 7: LABOR MARKET DEEP DIVE
    # ===================================================================
    with tab7:
        st.caption(
            "Headline employment numbers can mask painful realities in specific sectors. "
            "This tab exposes the gap between official stats and the lived experience of job seekers, "
            "especially in tech and data fields."
        )

        labor_df = _load_labor_deep_dive()

        if labor_df is None:
            st.warning(
                "Labor market data not available. Run `python run_pipeline.py` to generate it."
            )
        else:
            # -----------------------------------------------------------
            # Section 1: The Headline vs Reality Gap
            # -----------------------------------------------------------
            st.subheader("The Headline vs Reality Gap")
            st.markdown(
                "The official unemployment rate (U-3) only counts people *actively looking* for work. "
                "The broader U-6 rate includes discouraged workers and people stuck in part-time jobs "
                "who want full-time work. The gap between them reveals hidden labor market pain."
            )

            # Metrics row
            if "u6_rate" in labor_df.columns:
                latest = labor_df.dropna(subset=["u6_rate"]).iloc[-1] if not labor_df.dropna(subset=["u6_rate"]).empty else None
                if latest is not None:
                    u3_latest = latest.get("u3_rate") if "u3_rate" in labor_df.columns else None
                    if pd.notna(u3_latest):
                        u3_latest = float(u3_latest)
                    else:
                        u3_latest = None
                    mc1, mc2, mc3 = st.columns(3)
                    if u3_latest is not None:
                        mc1.metric("U-3 (Official)", f"{u3_latest:.1f}%")
                    mc2.metric("U-6 (Broad)", f"{latest['u6_rate']:.1f}%")
                    if u3_latest is not None:
                        gap = latest["u6_rate"] - u3_latest
                        mc3.metric("Gap (U-6 minus U-3)", f"{gap:.1f} pp")
                        if gap > 4.0:
                            st.markdown(_labor_insight_card(
                                "Significant hidden slack",
                                f"The {gap:.1f} pp gap between U-6 and U-3 suggests a meaningful share of workers "
                                "are underemployed or have given up searching. The headline number understates real pain.",
                                "#e74c3c",
                            ), unsafe_allow_html=True)
                        else:
                            st.markdown(_labor_insight_card(
                                "Gap within normal range",
                                f"The {gap:.1f} pp spread is relatively contained, suggesting most labor market "
                                "slack is captured by the headline rate.",
                                "#2ecc71",
                            ), unsafe_allow_html=True)

            fig_headline = _headline_vs_reality_chart(labor_df)
            if fig_headline:
                st.plotly_chart(fig_headline, use_container_width=True)

            st.markdown("---")

            # -----------------------------------------------------------
            # Section 2: Where Are Jobs Actually Going?
            # -----------------------------------------------------------
            st.subheader("Where Are Jobs Actually Going?")
            st.markdown(
                "Not all job growth is created equal. Healthcare and government have driven most "
                "recent gains, while the Information sector (which includes tech) and Professional "
                "Services have stagnated or declined. This chart shows year-over-year employment "
                "growth by sector."
            )

            fig_sector = _sector_employment_chart(labor_df)
            if fig_sector:
                st.plotly_chart(fig_sector, use_container_width=True)

                # Summary card with latest values
                sector_cols = {
                    "info_sector_emp": "Information",
                    "prof_business_emp": "Prof. & Business Svc.",
                    "healthcare_emp": "Education & Health",
                    "government_emp": "Government",
                }
                parts = []
                for col, label in sector_cols.items():
                    if col in labor_df.columns:
                        s = labor_df.set_index("date")[col].dropna()
                        if len(s) >= 13:
                            yoy = ((s.iloc[-1] / s.iloc[-13]) - 1) * 100
                            direction = "+" if yoy > 0 else ""
                            parts.append(f"<b>{label}:</b> {direction}{yoy:.1f}%")
                if parts:
                    st.markdown(_labor_insight_card(
                        "Latest YoY Growth Rates",
                        " &nbsp;|&nbsp; ".join(parts),
                        "#ff6b35",
                    ), unsafe_allow_html=True)

            st.markdown("---")

            # -----------------------------------------------------------
            # Section 3: Tech & Data Job Market Health
            # -----------------------------------------------------------
            st.subheader("Tech & Data Job Market Health")
            st.markdown(
                "Indexing employment to a base date reveals how much tech-adjacent sectors have "
                "grown (or shrunk) relative to a starting point. Pick a base date to compare."
            )

            base_options = {
                "Pre-COVID (Jan 2020)": "2020-01-01",
                "Post-COVID Peak (Jan 2022)": "2022-01-01",
                "Five Years Ago (Jan 2021)": "2021-01-01",
                "Start of Data (Jan 2000)": "2000-01-01",
            }
            base_choice = st.radio(
                "Index base date",
                list(base_options.keys()),
                horizontal=True,
                key="labor_base_date",
            )
            base_date = base_options[base_choice]

            fig_tech = _tech_focus_indexed_chart(labor_df, base_date)
            if fig_tech:
                st.plotly_chart(fig_tech, use_container_width=True)

                # Peak-to-current change card
                tech_cols = {
                    "info_sector_emp": "Information",
                    "computer_systems_emp": "Computer Systems Design",
                }
                peak_parts = []
                for col, label in tech_cols.items():
                    if col in labor_df.columns:
                        s = labor_df.set_index("date")[col].dropna()
                        if not s.empty:
                            peak_val = s.max()
                            current_val = s.iloc[-1]
                            pct_from_peak = ((current_val / peak_val) - 1) * 100
                            peak_parts.append(f"<b>{label}:</b> {pct_from_peak:+.1f}% from peak")
                if peak_parts:
                    st.markdown(_labor_insight_card(
                        "Distance from Peak Employment",
                        " &nbsp;|&nbsp; ".join(peak_parts),
                        "#9b59b6",
                    ), unsafe_allow_html=True)

            with st.expander("Why does this matter for tech workers?"):
                st.markdown(
                    "Official payroll numbers aggregate across all sectors. When total payrolls grow by 200k, "
                    "it sounds great — but if 180k of those are in healthcare and government while Information "
                    "sector jobs are flat or declining, the experience for tech/data professionals is very "
                    "different from the headline.\n\n"
                    "**Computer Systems Design** (NAICS 5415) is the closest FRED series to 'tech jobs.' "
                    "It covers software development, IT consulting, and systems integration. When this series "
                    "stagnates while total payrolls grow, it means the labor market is healthy in aggregate "
                    "but structurally challenging for tech workers."
                )

            st.markdown("---")

            # -----------------------------------------------------------
            # Section 4: JOLTS — Are Companies Actually Hiring?
            # -----------------------------------------------------------
            st.subheader("JOLTS: Are Companies Actually Hiring?")
            st.markdown(
                "The Job Openings and Labor Turnover Survey (JOLTS) tells us whether employers are "
                "posting jobs, actually filling them, and whether workers feel confident enough to quit. "
                "A high openings-to-hires ratio means lots of postings but few actual hires — "
                "'ghost jobs' or very selective hiring."
            )

            fig_jolts, fig_ratio = _jolts_chart(labor_df)
            if fig_jolts:
                st.plotly_chart(fig_jolts, use_container_width=True)
            if fig_ratio:
                st.plotly_chart(fig_ratio, use_container_width=True)

            # Hiring reality check
            if "jolts_openings" in labor_df.columns and "jolts_hires" in labor_df.columns:
                recent = labor_df[["date", "jolts_openings", "jolts_hires"]].dropna()
                if not recent.empty:
                    r = recent.iloc[-1]
                    ratio_val = r["jolts_openings"] / r["jolts_hires"] if r["jolts_hires"] > 0 else 0
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Job Openings", f"{r['jolts_openings']:,.0f}k")
                    mc2.metric("Actual Hires", f"{r['jolts_hires']:,.0f}k")
                    mc3.metric("Openings/Hires Ratio", f"{ratio_val:.2f}")
                    if ratio_val > 1.5:
                        st.markdown(_labor_insight_card(
                            "Hiring Disconnect",
                            f"With a {ratio_val:.2f}x openings-to-hires ratio, employers are posting far more jobs "
                            "than they're filling. This can reflect ghost postings, unrealistic requirements, "
                            "or very selective hiring — all of which make the job search feel harder than "
                            "headline openings suggest.",
                            "#e74c3c",
                        ), unsafe_allow_html=True)

            # Quits rate insight
            if "jolts_quits" in labor_df.columns:
                quits = labor_df.set_index("date")["jolts_quits"].dropna()
                if len(quits) >= 2:
                    latest_qr = quits.iloc[-1]
                    pre_covid_avg = quits.loc[:"2020-01"].mean() if len(quits.loc[:"2020-01"]) > 12 else None
                    if pre_covid_avg is not None:
                        st.markdown(_labor_insight_card(
                            "Worker Confidence (Quits Rate)",
                            f"Current quits rate: <b>{latest_qr:.1f}%</b> vs pre-COVID average: <b>{pre_covid_avg:.1f}%</b>. "
                            + ("Workers are quitting less than normal — a sign of caution." if latest_qr < pre_covid_avg
                               else "Workers feel confident enough to quit at above-normal rates."),
                            "#3498db",
                        ), unsafe_allow_html=True)

            st.markdown("---")

            # -----------------------------------------------------------
            # Section 5: The Hidden Slack
            # -----------------------------------------------------------
            st.subheader("The Hidden Slack")
            st.markdown(
                "Two indicators that never make headlines but reveal real pain: how long the average "
                "unemployed person stays jobless, and how many people are stuck in part-time work "
                "when they want full-time hours."
            )

            fig_slack = _hidden_slack_chart(labor_df)
            if fig_slack:
                st.plotly_chart(fig_slack, use_container_width=True)

            # Metrics row
            slack_mc1, slack_mc2 = st.columns(2)
            if "unemp_duration_mean" in labor_df.columns:
                dur = labor_df.set_index("date")["unemp_duration_mean"].dropna()
                if not dur.empty:
                    slack_mc1.metric("Avg Weeks Unemployed", f"{dur.iloc[-1]:.1f}")
            if "involuntary_part_time" in labor_df.columns:
                ipt = labor_df.set_index("date")["involuntary_part_time"].dropna()
                if not ipt.empty:
                    slack_mc1_val = ipt.iloc[-1]
                    slack_mc2.metric("Involuntary Part-Time", f"{slack_mc1_val:,.0f}k")

            if "unemp_duration_mean" in labor_df.columns:
                dur = labor_df.set_index("date")["unemp_duration_mean"].dropna()
                if not dur.empty and dur.iloc[-1] > 20:
                    st.markdown(_labor_insight_card(
                        "Extended Job Searches",
                        f"At {dur.iloc[-1]:.1f} weeks on average, job searches are taking longer than "
                        "the historical norm (~15-20 weeks outside recessions). This disproportionately "
                        "affects specialized workers who can't easily switch sectors.",
                        "#e74c3c",
                    ), unsafe_allow_html=True)
                elif not dur.empty:
                    st.markdown(_labor_insight_card(
                        "Search Duration Normal",
                        f"At {dur.iloc[-1]:.1f} weeks, average unemployment duration is within normal range.",
                        "#2ecc71",
                    ), unsafe_allow_html=True)

            st.markdown("---")

            # -----------------------------------------------------------
            # Section 6: What This Means
            # -----------------------------------------------------------
            st.subheader("What This Means")
            st.markdown(
                "The five sections above tell a story that headline numbers hide. "
                "Here are the conclusions — not just data points — and what they mean "
                "for job seekers, employers, and policymakers."
            )

            # Compute narrative-driving figures from the data
            _narr_ratio = None
            if "jolts_openings" in labor_df.columns and "jolts_hires" in labor_df.columns:
                _jr = labor_df[["jolts_openings", "jolts_hires"]].dropna()
                if not _jr.empty:
                    _narr_ratio = _jr.iloc[-1]["jolts_openings"] / _jr.iloc[-1]["jolts_hires"]

            _narr_info_yoy = None
            if "info_sector_emp" in labor_df.columns:
                _is = labor_df.set_index("date")["info_sector_emp"].dropna()
                if len(_is) >= 13:
                    _narr_info_yoy = ((_is.iloc[-1] / _is.iloc[-13]) - 1) * 100

            _narr_health_yoy = None
            if "healthcare_emp" in labor_df.columns:
                _hs = labor_df.set_index("date")["healthcare_emp"].dropna()
                if len(_hs) >= 13:
                    _narr_health_yoy = ((_hs.iloc[-1] / _hs.iloc[-13]) - 1) * 100

            _narr_gap = None
            if "u6_rate" in labor_df.columns and "u3_rate" in labor_df.columns:
                _ug = labor_df[["u6_rate", "u3_rate"]].dropna()
                if not _ug.empty:
                    _narr_gap = _ug.iloc[-1]["u6_rate"] - _ug.iloc[-1]["u3_rate"]

            # --- Conclusion 1: The Matching Problem ---
            st.markdown("#### 1. The Hiring System Is Broken — Not the Workers")
            ratio_text = f" Currently at **{_narr_ratio:.2f}x**," if _narr_ratio else ""
            st.markdown(
                f"The openings-to-hires ratio is the single most important chart on this page.{ratio_text} "
                "it means employers are posting far more jobs than they actually fill. A significant "
                "share of these are **ghost jobs** — postings kept open for pipeline building, compliance "
                "requirements, or investor optics with no immediate intent to hire.\n\n"
                "For job seekers, this means: if you're applying to 200 jobs and hearing back from 10, "
                "the problem is likely not your resume. The *effective* number of real positions is "
                "probably **50-65% of what's posted**. Focus on companies showing actual headcount growth "
                "in recent quarters rather than spray-applying to every listing."
            )

            # --- Conclusion 2: Where the jobs are vs where people are ---
            st.markdown("#### 2. Jobs Exist — But the Economy Is Trading Down")
            if _narr_info_yoy is not None and _narr_health_yoy is not None:
                st.markdown(
                    f"Information sector employment is at **{_narr_info_yoy:+.1f}%** year-over-year while "
                    f"Education & Health is at **{_narr_health_yoy:+.1f}%**. "
                    "Yes, people *can* reskill into healthcare or government — but telling a software engineer "
                    "earning $140k to become a home health aide at $38k isn't a 'skills gap' solution. "
                    "It's **value destruction**.\n\n"
                    "The economy is replacing high-productivity, high-wage jobs with lower-productivity ones. "
                    "Aggregate payroll numbers look healthy, but the *composition* of job growth matters enormously "
                    "for long-term economic output and individual livelihoods."
                )
            else:
                st.markdown(
                    "The sector charts show healthcare and government driving most job growth while "
                    "tech-adjacent sectors stagnate. Aggregate payroll numbers mask a composition problem: "
                    "the economy is adding jobs, but not the kind that leverage the skills of displaced "
                    "knowledge workers."
                )

            # --- Conclusion 3: The Information Asymmetry ---
            st.markdown("#### 3. The Real Crisis Is Information, Not Skills")
            gap_text = f" (the current **{_narr_gap:.1f} pp** U-6/U-3 gap confirms this)" if _narr_gap else ""
            st.markdown(
                "There is a massive **information asymmetry** in the labor market. Employers know their "
                "budget, timeline, competing candidates, and whether a role is real. Workers know almost "
                f"nothing. Hidden slack{gap_text} "
                "shows that the headline unemployment rate understates real pain — discouraged workers "
                "and involuntary part-timers don't show up in U-3.\n\n"
                "Meanwhile, the staffing industry — which should reduce friction — often makes it worse. "
                "Their business model profits from opacity: a 20-35% markup on placements means efficiency "
                "is bad for margins. Multiple agencies frequently post the same role, inflating apparent "
                "demand while fragmenting the worker's ability to negotiate."
            )

            # --- Conclusion 4: What would actually help ---
            st.markdown("#### 4. Transparency Would Be the Cheapest, Highest-Impact Fix")
            st.markdown(
                "The data on this page points to one clear policy conclusion: we don't need more "
                "retraining programs — we need **transparency mandates**.\n\n"
                "Concrete changes that the data supports:\n"
                "- **Require salary ranges on all postings** — states doing this (Colorado, NYC, Washington) "
                "are already reducing the apply-interview-discover-lowball cycle\n"
                "- **Mandate fill-rate disclosure** — if a company posts 500 roles and fills 30, that should "
                "be public. It would instantly expose ghost postings\n"
                "- **Publish JOLTS data by sector and company size** — aggregate openings numbers are misleading. "
                "If 60% of 'openings' come from companies with <5% hire rates, the entire narrative changes\n"
                "- **Require staffing agencies to disclose the client, markup, and whether the role has a "
                "confirmed start date** — this alone would eliminate the worst practices"
            )

            # --- Conclusion 5: For job seekers right now ---
            st.markdown("#### 5. Practical Advice for Job Seekers Today")
            st.markdown(
                "While waiting for systemic change, here's what the data says you should do *now*:\n\n"
                "1. **Don't take JOLTS at face value.** Many postings aren't real. Prioritize companies "
                "with confirmed headcount growth (check LinkedIn employee trends, earnings calls, or "
                "recent funding rounds)\n"
                "2. **Track sector trends, not headlines.** A 200k payroll beat means nothing if your "
                "sector is flat. Use this dashboard to know where *your* sector stands\n"
                "3. **Negotiate from data.** If the openings-to-hires ratio is high, employers are "
                "struggling to fill roles even if it doesn't feel that way. You have more leverage than "
                "the rejection rate suggests\n"
                "4. **Consider adjacent sectors growing in your skill range** — not a full career pivot, "
                "but lateral moves where your existing expertise transfers at comparable compensation\n"
                "5. **Time your search.** The quits rate signals worker confidence. When it's low, "
                "fewer people are leaving voluntarily — meaning fewer backfill openings. The JOLTS charts "
                "above help you read the cycle"
            )

            st.markdown(
                '<div style="border-left: 5px solid #1a1a2e; padding: 10px 16px; margin-top: 16px; '
                'background-color: #f8f9fa; border-radius: 0 6px 6px 0;">'
                '<strong>Bottom line:</strong> The U.S. labor market is functioning well <em>in aggregate</em> '
                'but failing <em>in allocation</em>. The matching mechanism between employers and workers is '
                'broken — not because workers lack skills, but because the system lacks transparency. '
                'The data to fix this already exists. It just needs to be connected and made public.'
                '</div>',
                unsafe_allow_html=True,
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

    # ===================================================================
    # TAB 6: ACTION CENTER
    # ===================================================================
    with tab6:
        st.caption("Prioritized actions, change tracking, early warnings, scenario exploration, and exportable briefings.")

        # --- Section 1: Priority Action Items ---
        st.subheader("Priority Action Items")
        action_items = _generate_action_items(snapshot_df, latest_prob, florida_latest)

        if action_items:
            # Summary bar
            counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
            for item in action_items:
                counts[item["severity"]] = counts.get(item["severity"], 0) + 1
            badge_parts = []
            for sev in ["Critical", "High", "Medium", "Low"]:
                if counts[sev] > 0:
                    color = SEVERITY_COLORS[sev]
                    badge_parts.append(
                        f'<span style="background-color:{color}; color:white; padding:4px 12px; '
                        f'border-radius:12px; font-weight:bold; margin-right:8px;">'
                        f'{sev}: {counts[sev]}</span>'
                    )
            st.markdown(" ".join(badge_parts), unsafe_allow_html=True)
            st.markdown("")

            # Action item cards
            for item in action_items:
                sev_color = SEVERITY_COLORS[item["severity"]]
                st.markdown(
                    f'<div style="border-left: 5px solid {sev_color}; padding: 10px 16px; '
                    f'margin-bottom: 8px; background-color: #f8f9fa; border-radius: 0 6px 6px 0;">'
                    f'<span style="background-color:{sev_color}; color:white; padding:2px 8px; '
                    f'border-radius:8px; font-size:0.85em; font-weight:bold;">{item["severity"]}</span> '
                    f'<span style="background-color:#e0e0e0; padding:2px 8px; border-radius:8px; '
                    f'font-size:0.85em; margin-left:4px;">{item["area"]}</span> '
                    f'<span style="background-color:#d0e8ff; padding:2px 8px; border-radius:8px; '
                    f'font-size:0.85em; margin-left:4px;">Review: {item["review"]}</span>'
                    f'<br/><span style="font-size:1.0em; margin-top:6px; display:inline-block;">'
                    f'{item["action"]}</span>'
                    f'<br/><span style="font-size:0.85em; color:#666;">Data: {item["data_label"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("No action items triggered — all indicators are within normal ranges.")

        st.markdown("---")

        # --- Section 2: What Changed This Month ---
        st.subheader("What Changed This Month")
        what_changed = _generate_what_changed(snapshot_df, prob_history_df)

        if what_changed["prob_delta"] is not None:
            delta = what_changed["prob_delta"]
            arrow = "\u2191" if delta > 0 else "\u2193" if delta < 0 else "\u2194\ufe0f"
            delta_color = "#d62728" if delta > 0 else "#2ca02c" if delta < 0 else "#666"
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("Previous Probability", f"{what_changed['prob_previous']:.1%}")
            wc2.metric("Current Probability", f"{what_changed['prob_current']:.1%}")
            wc3.markdown(
                f'<div style="text-align:center; padding-top:14px;">'
                f'<span style="font-size:2.5em; color:{delta_color};">{arrow}</span><br/>'
                f'<span style="font-size:1.3em; color:{delta_color}; font-weight:bold;">{delta:+.2%}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Top movers
            col_up, col_down = st.columns(2)
            with col_up:
                st.markdown("**Top Movers \u2191 (Increased)**")
                if what_changed["movers_up"]:
                    for m in what_changed["movers_up"]:
                        st.markdown(f"- **{m['label']}**: {m['previous']:.2f} \u2192 {m['current']:.2f} ({m['delta']:+.2f})")
                else:
                    st.caption("No significant increases this month.")
            with col_down:
                st.markdown("**Top Movers \u2193 (Decreased)**")
                if what_changed["movers_down"]:
                    for m in what_changed["movers_down"]:
                        st.markdown(f"- **{m['label']}**: {m['previous']:.2f} \u2192 {m['current']:.2f} ({m['delta']:+.2f})")
                else:
                    st.caption("No significant decreases this month.")

            # Threshold crossings
            if what_changed["threshold_crossings"]:
                st.markdown("**Threshold Crossings This Month**")
                for tc in what_changed["threshold_crossings"]:
                    sev_color = SEVERITY_COLORS.get(tc["severity"], "#666")
                    st.markdown(
                        f'- <span style="color:{sev_color}; font-weight:bold;">[{tc["severity"]}]</span> '
                        f'**{tc["label"]}** {tc["direction"]} zone '
                        f'({tc["previous"]:.2f} \u2192 {tc["current"]:.2f})',
                        unsafe_allow_html=True,
                    )

            # Expandable full table
            with st.expander("Full indicator change table"):
                if what_changed["all_changes"]:
                    change_df = pd.DataFrame(what_changed["all_changes"])[
                        ["label", "previous", "current", "delta", "pct_change"]
                    ].rename(columns={
                        "label": "Indicator",
                        "previous": "Previous",
                        "current": "Current",
                        "delta": "Change",
                        "pct_change": "% Change",
                    })
                    st.dataframe(change_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No changes to display.")
        else:
            st.info("Insufficient data to compute month-over-month changes. Need at least 2 months of history.")

        st.markdown("---")

        # --- Section 3: Early Warning Dashboard ---
        st.subheader("Early Warning Dashboard")
        st.caption("Distance of each indicator to its danger zone. Sorted by proximity — closest to danger first.")

        heatmap_data = _generate_indicator_heatmap_data(snapshot_df)
        if heatmap_data:
            labels = [r["label"] for r in heatmap_data]
            distances = [r["distance_pct"] for r in heatmap_data]
            bar_colors = [
                {"red": "#d62728", "orange": "#ff7f0e", "gold": "#f0c929", "green": "#2ca02c"}[r["color"]]
                for r in heatmap_data
            ]
            hover_texts = [
                f"Current: {r['current_value']:.2f}<br>"
                f"Danger zone: {r['danger_zone']:.2f}<br>"
                f"Direction: {r['direction_text']} {r['direction_symbol']}<br>"
                f"Hist. percentile: {r['percentile']:.0f}%<br>"
                f"Area: {r['area']}"
                for r in heatmap_data
            ]
            annotation_texts = [
                f"{r['current_value']:.2f} {r['direction_symbol']} | {r['zone_label']} | P{r['percentile']:.0f}"
                for r in heatmap_data
            ]

            fig_ew = go.Figure()
            fig_ew.add_trace(
                go.Bar(
                    y=labels[::-1],
                    x=distances[::-1],
                    orientation="h",
                    marker_color=bar_colors[::-1],
                    text=annotation_texts[::-1],
                    textposition="outside",
                    hovertext=hover_texts[::-1],
                    hoverinfo="text",
                )
            )
            fig_ew.update_layout(
                title="Distance to Danger Zone (%)",
                xaxis_title="Distance to Danger Zone (%) — lower = closer to danger",
                yaxis_title="",
                height=max(400, 45 * len(heatmap_data)),
                xaxis={"range": [0, max(120, max(distances) + 20)]},
            )
            st.plotly_chart(fig_ew, use_container_width=True)

            # Legend
            st.markdown(
                '<span style="color:#d62728;">\u25a0</span> Crossed danger zone &nbsp;&nbsp;'
                '<span style="color:#ff7f0e;">\u25a0</span> Near (<15%) &nbsp;&nbsp;'
                '<span style="color:#f0c929;">\u25a0</span> Approaching (15-40%) &nbsp;&nbsp;'
                '<span style="color:#2ca02c;">\u25a0</span> Safe (>40%)',
                unsafe_allow_html=True,
            )
        else:
            st.info("Insufficient data to generate early warning chart.")

        st.markdown("---")

        # --- Section 4: What-If Scenario Explorer ---
        st.subheader("What-If Scenario Explorer")
        st.caption("Adjust key indicators to see how recession probability would change. Uses logistic regression coefficients with delta-logit estimation.")

        sorted_snap_latest = snapshot_df.sort_values("date").iloc[-1] if not snapshot_df.empty else None

        whatif_cols = st.columns(4)
        scenario_inputs: dict[str, float] = {}

        # Unemployment slider
        if sorted_snap_latest is not None and "unemployment" in sorted_snap_latest.index and not pd.isna(sorted_snap_latest["unemployment"]):
            current_unemp = float(sorted_snap_latest["unemployment"])
            with whatif_cols[0]:
                scenario_inputs["unemployment"] = st.slider(
                    "Unemployment Rate (%)",
                    min_value=2.0, max_value=15.0,
                    value=current_unemp,
                    step=0.1,
                    key="whatif_unemployment",
                )

        # Credit Spread slider
        if sorted_snap_latest is not None and "credit_spread" in sorted_snap_latest.index and not pd.isna(sorted_snap_latest["credit_spread"]):
            current_cs = float(sorted_snap_latest["credit_spread"])
            with whatif_cols[1]:
                scenario_inputs["credit_spread"] = st.slider(
                    "Credit Spread (%)",
                    min_value=0.5, max_value=8.0,
                    value=current_cs,
                    step=0.1,
                    key="whatif_credit_spread",
                )

        # Yield Spread slider
        if sorted_snap_latest is not None and "yield_spread" in sorted_snap_latest.index and not pd.isna(sorted_snap_latest["yield_spread"]):
            current_ys = float(sorted_snap_latest["yield_spread"])
            with whatif_cols[2]:
                scenario_inputs["yield_spread"] = st.slider(
                    "Yield Spread (%)",
                    min_value=-3.0, max_value=4.0,
                    value=current_ys,
                    step=0.1,
                    key="whatif_yield_spread",
                )

        # VIX slider
        if sorted_snap_latest is not None and "vix_level" in sorted_snap_latest.index and not pd.isna(sorted_snap_latest["vix_level"]):
            current_vix = float(sorted_snap_latest["vix_level"])
            with whatif_cols[3]:
                scenario_inputs["vix_level"] = st.slider(
                    "VIX Level",
                    min_value=8.0, max_value=80.0,
                    value=current_vix,
                    step=0.5,
                    key="whatif_vix",
                )

        if scenario_inputs:
            scenario_prob, scenario_explanations = _compute_whatif_probability(
                latest_prob, importance_df, snapshot_df, scenario_inputs,
            )
            prob_change = scenario_prob - latest_prob

            sp1, sp2, sp3 = st.columns(3)
            sp1.metric("Current Probability", f"{latest_prob:.1%}")
            sp2.metric("Scenario Probability", f"{scenario_prob:.1%}")
            change_color = "#d62728" if prob_change > 0 else "#2ca02c" if prob_change < 0 else "#666"
            sp3.markdown(
                f'<div style="text-align:center; padding-top:10px;">'
                f'<span style="font-size:1.6em; color:{change_color}; font-weight:bold;">{prob_change:+.2%}</span><br/>'
                f'<span style="font-size:0.9em; color:#666;">Change</span></div>',
                unsafe_allow_html=True,
            )

            # Narrative
            direction = "increase" if prob_change > 0 else "decrease" if prob_change < 0 else "remain unchanged"
            st.markdown(
                f"Under this scenario, recession probability would **{direction}** "
                f"from {latest_prob:.1%} to {scenario_prob:.1%} ({prob_change:+.2%})."
            )

            with st.expander("Calculation details"):
                for exp in scenario_explanations:
                    st.markdown(f"- {exp}")
                st.caption(
                    "Approximation using logistic regression coefficients with delta-logit approach. "
                    "Actual model (ensemble/boosted trees) may produce different results. "
                    "Standardization uses training data mean/std."
                )
        else:
            st.info("Indicator data not available for scenario sliders.")

        st.markdown("---")

        # --- Section 5: Export Briefing ---
        st.subheader("Export Briefing")
        st.caption("Download a Markdown briefing summarizing the current risk assessment, action items, and changes.")

        briefing_text = _generate_briefing_text(
            latest_prob=latest_prob,
            latest_date=latest_date,
            status=status,
            change_6m=change_6m,
            action_items=action_items,
            what_changed=what_changed,
            snapshot_df=snapshot_df,
            markov_summary=markov_summary,
            florida_latest=florida_latest,
            alert_threshold=alert_threshold,
        )

        with st.expander("Preview briefing"):
            st.markdown(briefing_text)

        file_date = datetime.date.today().strftime("%Y-%m-%d")
        st.download_button(
            label="Download Briefing (.md)",
            data=briefing_text,
            file_name=f"recession_briefing_{file_date}.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
