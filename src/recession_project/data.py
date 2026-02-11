from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from pandas_datareader.data import DataReader
import yfinance as yf

from .config import SERIES_CODES, YAHOO_TICKERS


@dataclass
class DataBundle:
    raw: pd.DataFrame
    features: pd.DataFrame


def fetch_fred_data(start_date: str) -> pd.DataFrame:
    """Download configured series from FRED and align into one table."""
    parts: list[pd.DataFrame] = []
    for name, code in SERIES_CODES.items():
        series = DataReader(code, "fred", start=start_date)
        series.columns = [name]
        parts.append(series)

    raw = pd.concat(parts, axis=1).sort_index()
    return raw


def fetch_yahoo_data(start_date: str) -> pd.DataFrame:
    """Download selected daily market series from Yahoo Finance."""
    parts: list[pd.Series] = []
    tickers = list(YAHOO_TICKERS.values())

    try:
        hist = yf.download(
            tickers=tickers,
            start=start_date,
            auto_adjust=True,
            progress=False,
            interval="1d",
            threads=False,
            group_by="ticker",
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Yahoo batch download failed: {exc}", stacklevel=2)
        return pd.DataFrame()

    if hist.empty:
        return pd.DataFrame()

    for name, ticker in YAHOO_TICKERS.items():
        close = None
        if isinstance(hist.columns, pd.MultiIndex):
            if ticker in hist.columns.get_level_values(0):
                ticker_frame = hist[ticker]
                if "Close" in ticker_frame.columns:
                    close = ticker_frame["Close"]
                elif len(ticker_frame.columns) > 0:
                    close = ticker_frame.iloc[:, 0]
            elif ticker in hist.columns.get_level_values(1) and "Close" in hist.columns.get_level_values(0):
                close = hist["Close"][ticker]
        else:
            close_col = "Close" if "Close" in hist.columns else hist.columns[0]
            close = hist[close_col]

        if close is None or close.dropna().empty:
            continue

        idx = pd.to_datetime(close.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        close = close.copy()
        close.index = idx.normalize()
        close.name = name
        parts.append(close)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, axis=1).sort_index()


def fetch_famafrench_data(start_date: str) -> pd.DataFrame:
    """Download monthly Fama-French factors (outside FRED)."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The argument 'date_parser' is deprecated.*",
                category=FutureWarning,
            )
            ff_dict = DataReader("F-F_Research_Data_Factors", "famafrench", start=start_date)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Fama-French download failed: {exc}", stacklevel=2)
        return pd.DataFrame()

    if 0 not in ff_dict:
        return pd.DataFrame()

    ff = ff_dict[0].copy()
    ff.index = ff.index.to_timestamp(how="end").normalize()
    ff = ff.rename(
        columns={
            "Mkt-RF": "ff_mkt_excess",
            "SMB": "ff_size_factor",
            "HML": "ff_value_factor",
        }
    )
    wanted_cols = [c for c in ["ff_mkt_excess", "ff_size_factor", "ff_value_factor"] if c in ff.columns]
    return ff[wanted_cols]


def _forward_recession_label(usrec: pd.Series, horizon_months: int) -> pd.Series:
    """1 if any recession starts/continues in the next horizon window; NaN near tail."""
    target = pd.Series(index=usrec.index, dtype=float)
    for i in range(len(usrec)):
        window = usrec.iloc[i + 1 : i + 1 + horizon_months]
        if len(window) < horizon_months:
            target.iloc[i] = np.nan
        else:
            target.iloc[i] = 1.0 if window.max() >= 1 else 0.0
    return target


def _monthly_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Align mixed-frequency series into month-end snapshots."""
    # Mixed-frequency series are sparse on a daily index. Forward fill first,
    # then sample month-end snapshots to keep full monthly history.
    return raw.sort_index().ffill().resample("ME").last().ffill()


def _engineer_features(monthly: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=monthly.index)
    features["yield_spread"] = monthly["yield_spread"]
    features["yield_spread_3m_delta"] = monthly["yield_spread"].diff(3)
    features["unemployment"] = monthly["unemployment"]
    features["unemployment_3m_delta"] = monthly["unemployment"].diff(3)
    features["fedfunds"] = monthly["fedfunds"]
    features["inflation_yoy"] = monthly["cpi"].pct_change(12) * 100.0
    features["industrial_prod_yoy"] = monthly["industrial_prod"].pct_change(12) * 100.0
    features["payrolls_yoy"] = monthly["payrolls"].pct_change(12) * 100.0
    features["vix_level"] = monthly["vix"]
    features["credit_spread"] = monthly["credit_spread"]
    features["nfci"] = monthly["nfci"]

    if "sp500_price" in monthly.columns:
        features["sp500_6m_return"] = monthly["sp500_price"].pct_change(6) * 100.0
    if "oil_price" in monthly.columns:
        features["oil_6m_return"] = monthly["oil_price"].pct_change(6) * 100.0
    if "gold_price" in monthly.columns:
        features["gold_6m_return"] = monthly["gold_price"].pct_change(6) * 100.0
    if "dollar_index" in monthly.columns:
        features["dollar_3m_return"] = monthly["dollar_index"].pct_change(3) * 100.0
    if "hyg_price" in monthly.columns and "ief_price" in monthly.columns:
        hyg_3m = monthly["hyg_price"].pct_change(3)
        ief_3m = monthly["ief_price"].pct_change(3)
        features["hyg_ief_3m_relative_return"] = (hyg_3m - ief_3m) * 100.0
    if "ff_mkt_excess" in monthly.columns:
        features["ff_mkt_excess"] = monthly["ff_mkt_excess"]
    if "ff_size_factor" in monthly.columns:
        features["ff_size_factor"] = monthly["ff_size_factor"]
    if "ff_value_factor" in monthly.columns:
        features["ff_value_factor"] = monthly["ff_value_factor"]

    features["current_recession"] = monthly["usrec"]
    return features


def _enough_feature_coverage(features: pd.DataFrame, min_non_null: int = 5) -> pd.Series:
    feature_cols = [c for c in features.columns if c not in {"target_recession_next_horizon", "current_recession"}]
    return features[feature_cols].notna().sum(axis=1) >= min_non_null


def build_inference_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for latest nowcast inference (no forward label)."""
    monthly = _monthly_table(raw)
    features = _engineer_features(monthly)
    mask = _enough_feature_coverage(features)
    return features.loc[mask].copy()


def build_feature_table(raw: pd.DataFrame, horizon_months: int) -> pd.DataFrame:
    """Create monthly features and forward-looking recession label."""
    monthly = _monthly_table(raw)
    features = _engineer_features(monthly)
    features["target_recession_next_horizon"] = _forward_recession_label(
        monthly["usrec"], horizon_months=horizon_months
    )

    mask = features["target_recession_next_horizon"].notna() & _enough_feature_coverage(features)
    features = features.loc[mask].copy()
    return features


def load_and_prepare(start_date: str, horizon_months: int) -> DataBundle:
    fred = fetch_fred_data(start_date=start_date)
    yahoo = fetch_yahoo_data(start_date=start_date)
    famafrench = fetch_famafrench_data(start_date=start_date)
    raw = pd.concat([fred, yahoo, famafrench], axis=1).sort_index()
    features = build_feature_table(raw=raw, horizon_months=horizon_months)
    return DataBundle(raw=raw, features=features)
