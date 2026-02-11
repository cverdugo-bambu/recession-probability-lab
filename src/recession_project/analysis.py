from __future__ import annotations

import numpy as np
import pandas as pd


def _month_diff(later: pd.Timestamp, earlier: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _recession_episodes(df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    episodes: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    in_recession = False
    start = None

    for row in df.itertuples():
        is_recession = int(row.current_recession) == 1
        date = pd.Timestamp(row.date)
        if is_recession and not in_recession:
            start = date
            in_recession = True
        if not is_recession and in_recession and start is not None:
            episodes.append((start, date))
            in_recession = False
            start = None

    if in_recession and start is not None:
        episodes.append((start, pd.Timestamp(df["date"].max())))

    return episodes


def build_episode_review(backtest_df: pd.DataFrame, alert_threshold: float) -> pd.DataFrame:
    if backtest_df.empty:
        return pd.DataFrame()

    df = backtest_df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    rows: list[dict[str, object]] = []

    for start, end in _recession_episodes(df):
        pre_window = df[(df["date"] >= start - pd.DateOffset(months=12)) & (df["date"] < start)]
        in_window = df[(df["date"] >= start) & (df["date"] <= end)]

        signaled = pre_window[pre_window["y_prob"] >= alert_threshold]
        first_signal_date = pd.NaT if signaled.empty else pd.Timestamp(signaled.iloc[0]["date"])
        lead_months = np.nan if pd.isna(first_signal_date) else _month_diff(start, first_signal_date)

        rows.append(
            {
                "episode_start": start.date().isoformat(),
                "episode_end": end.date().isoformat(),
                "max_prob_pre_12m": float(pre_window["y_prob"].max()) if not pre_window.empty else np.nan,
                "mean_prob_pre_12m": float(pre_window["y_prob"].mean()) if not pre_window.empty else np.nan,
                "first_signal_date": None if pd.isna(first_signal_date) else first_signal_date.date().isoformat(),
                "lead_months": None if np.isnan(lead_months) else int(lead_months),
                "peak_prob_during_episode": float(in_window["y_prob"].max()) if not in_window.empty else np.nan,
                "missed_early_warning": bool(signaled.empty),
            }
        )

    return pd.DataFrame(rows)


def build_period_comparison(
    probability_history_df: pd.DataFrame,
    alert_threshold: float,
) -> pd.DataFrame:
    df = probability_history_df.copy()
    if "date" not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    windows = {
        "2008 Crisis Build-Up (2006-2010)": ("2006-01-01", "2010-12-31"),
        "Current Regime (2023-2026)": ("2023-01-01", None),
    }

    rows: list[dict[str, object]] = []
    for label, (start, end) in windows.items():
        subset = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            subset = subset[subset["date"] <= pd.Timestamp(end)]

        if subset.empty:
            continue

        rows.append(
            {
                "period": label,
                "start": pd.Timestamp(subset["date"].min()).date().isoformat(),
                "end": pd.Timestamp(subset["date"].max()).date().isoformat(),
                "latest_prob": float(subset["y_prob"].iloc[-1]),
                "mean_prob": float(subset["y_prob"].mean()),
                "max_prob": float(subset["y_prob"].max()),
                "months_above_threshold": int((subset["y_prob"] >= alert_threshold).sum()),
                "observations": int(len(subset)),
            }
        )

    return pd.DataFrame(rows)
