from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from pandas_datareader.data import DataReader

from .config import (
    FLORIDA_SERIES_CODES,
    FLORIDA_STRESS_CALM_MAX,
    FLORIDA_STRESS_WATCH_MAX,
    FEATURE_LABELS,
)


@dataclass
class FloridaStage1Result:
    raw: pd.DataFrame
    monthly: pd.DataFrame
    index: pd.DataFrame
    latest: dict[str, object]


def _zscore(series: pd.Series) -> pd.Series:
    centered = series - series.mean(skipna=True)
    std = series.std(skipna=True)
    if std is None or np.isnan(std) or std == 0:
        return pd.Series(index=series.index, data=np.nan, dtype=float)
    return (centered / std).clip(-3, 3)


def _stress_zone(score: float) -> str:
    if np.isnan(score):
        return "Unknown"
    if score <= FLORIDA_STRESS_CALM_MAX:
        return "Calm"
    if score <= FLORIDA_STRESS_WATCH_MAX:
        return "Watch"
    return "Stress"


def fetch_florida_data(start_date: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for name, code in FLORIDA_SERIES_CODES.items():
        try:
            series = DataReader(code, "fred", start=start_date)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Florida series download failed for {code}: {exc}", stacklevel=2)
            continue
        series.columns = [name]
        parts.append(series)
    if not parts:
        raise ValueError("No Florida series were downloaded.")
    return pd.concat(parts, axis=1).sort_index()


def _monthly_florida_table(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.resample("ME").last().index)
    for col in raw.columns:
        if col == "fl_initial_claims":
            out[col] = raw[col].resample("ME").mean()
        else:
            out[col] = raw[col].resample("ME").last()
    return out.ffill()


def _driver_sentence(row: pd.Series, components: list[tuple[str, str]]) -> str:
    entries: list[tuple[float, str]] = []
    for comp_col, label in components:
        val = row.get(comp_col, np.nan)
        if pd.isna(val):
            continue
        if val > 0:
            direction = "added stress"
        elif val < 0:
            direction = "reduced stress"
        else:
            direction = "was neutral"
        entries.append((abs(float(val)), f"{label}: {direction}"))

    if not entries:
        return "No driver explanation available."

    top = sorted(entries, key=lambda x: x[0], reverse=True)[:3]
    return " | ".join(text for _, text in top)


def build_florida_stage1(raw: pd.DataFrame) -> FloridaStage1Result:
    monthly = _monthly_florida_table(raw)

    features = pd.DataFrame(index=monthly.index)
    if "fl_unemployment_rate" in monthly.columns:
        features["fl_unemployment_rate"] = monthly["fl_unemployment_rate"]
        features["fl_unemployment_3m_delta"] = monthly["fl_unemployment_rate"].diff(3)
    if "fl_total_payrolls" in monthly.columns:
        features["fl_total_payrolls_yoy"] = monthly["fl_total_payrolls"].pct_change(12) * 100.0
    if "fl_manufacturing_payrolls" in monthly.columns:
        features["fl_manufacturing_payrolls_yoy"] = monthly["fl_manufacturing_payrolls"].pct_change(12) * 100.0
    if "fl_private_building_permits" in monthly.columns:
        features["fl_private_building_permits_yoy"] = monthly["fl_private_building_permits"].pct_change(12) * 100.0
    if "fl_initial_claims" in monthly.columns:
        features["fl_initial_claims_3m_pct"] = monthly["fl_initial_claims"].pct_change(3) * 100.0

    stress_components = pd.DataFrame(index=features.index)
    if "fl_unemployment_rate" in features.columns:
        stress_components["stress_unemployment_level"] = _zscore(features["fl_unemployment_rate"])
    if "fl_unemployment_3m_delta" in features.columns:
        stress_components["stress_unemployment_change"] = _zscore(features["fl_unemployment_3m_delta"])
    if "fl_total_payrolls_yoy" in features.columns:
        stress_components["stress_payroll_growth"] = _zscore(-features["fl_total_payrolls_yoy"])
    if "fl_manufacturing_payrolls_yoy" in features.columns:
        stress_components["stress_manufacturing_growth"] = _zscore(-features["fl_manufacturing_payrolls_yoy"])
    if "fl_private_building_permits_yoy" in features.columns:
        stress_components["stress_building_permits"] = _zscore(-features["fl_private_building_permits_yoy"])
    if "fl_initial_claims_3m_pct" in features.columns:
        stress_components["stress_initial_claims"] = _zscore(features["fl_initial_claims_3m_pct"])

    if stress_components.shape[1] == 0:
        raise ValueError("Florida Stage 1 could not be built: no stress components available.")

    component_count = stress_components.notna().sum(axis=1)
    composite = stress_components.mean(axis=1, skipna=True)
    composite = composite.where(component_count >= 3)
    score_0_100 = composite.rank(pct=True) * 100.0

    stage1 = features.join(stress_components)
    stage1["florida_stress_composite"] = composite
    stage1["florida_stress_index"] = score_0_100
    stage1["florida_stress_zone"] = stage1["florida_stress_index"].apply(_stress_zone)
    stage1["florida_stress_mom_change"] = stage1["florida_stress_index"].diff(1)

    driver_map = [
        ("stress_unemployment_level", FEATURE_LABELS["fl_unemployment_rate"]),
        ("stress_unemployment_change", FEATURE_LABELS["fl_unemployment_3m_delta"]),
        ("stress_payroll_growth", FEATURE_LABELS["fl_total_payrolls_yoy"]),
        ("stress_manufacturing_growth", FEATURE_LABELS["fl_manufacturing_payrolls_yoy"]),
        ("stress_building_permits", FEATURE_LABELS["fl_private_building_permits_yoy"]),
        ("stress_initial_claims", FEATURE_LABELS["fl_initial_claims_3m_pct"]),
    ]
    stage1["florida_top_drivers"] = stage1.apply(lambda row: _driver_sentence(row, driver_map), axis=1)

    stage1 = stage1.dropna(subset=["florida_stress_index"]).copy()
    if stage1.empty:
        raise ValueError("Florida Stage 1 could not be built: insufficient data after feature engineering.")
    latest_row = stage1.iloc[-1]
    latest = {
        "date": pd.to_datetime(stage1.index[-1]).date().isoformat(),
        "florida_stress_index": float(latest_row["florida_stress_index"]),
        "florida_stress_zone": str(latest_row["florida_stress_zone"]),
        "florida_stress_mom_change": float(latest_row["florida_stress_mom_change"])
        if not pd.isna(latest_row["florida_stress_mom_change"])
        else 0.0,
        "florida_top_drivers": str(latest_row["florida_top_drivers"]),
    }

    return FloridaStage1Result(raw=raw, monthly=monthly, index=stage1, latest=latest)


def save_florida_stage1(result: FloridaStage1Result, artifact_dir: Path | str) -> None:
    path = Path(artifact_dir)
    result.raw.to_csv(path / "florida_raw.csv", index_label="date")
    result.monthly.to_csv(path / "florida_monthly.csv", index_label="date")
    result.index.to_csv(path / "florida_stage1_index.csv", index_label="date")
    with (path / "florida_stage1_latest.json").open("w", encoding="utf-8") as f:
        json.dump(result.latest, f, indent=2)
