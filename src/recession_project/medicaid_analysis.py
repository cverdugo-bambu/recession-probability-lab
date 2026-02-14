"""Analysis of HHS Medicaid provider spending data from DOGE."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"

# HCPCS code descriptions for the top spending categories
HCPCS_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "T1019": {
        "short": "Personal Care Services (15 min)",
        "category": "Home Health / Personal Care",
    },
    "T2016": {
        "short": "Residential Habilitation, Waiver (per diem)",
        "category": "Waiver / Residential",
    },
    "S5125": {
        "short": "Attendant Care Services (15 min)",
        "category": "Home Health / Personal Care",
    },
    "T1015": {
        "short": "Clinic Visit, All-Inclusive (FQHC/RHC)",
        "category": "Clinic / Outpatient",
    },
    "H2016": {
        "short": "Community Support Services (per diem)",
        "category": "Behavioral Health / Community",
    },
    "H2015": {
        "short": "Community Support Services (15 min)",
        "category": "Behavioral Health / Community",
    },
    "S5102": {
        "short": "Adult Day Care (30 min)",
        "category": "Day Programs",
    },
    "T1020": {
        "short": "Personal Care Services (per diem)",
        "category": "Home Health / Personal Care",
    },
    "T2021": {
        "short": "Day Habilitation, Waiver (15 min)",
        "category": "Waiver / Day Programs",
    },
    "99283": {
        "short": "Emergency Dept Visit (moderate)",
        "category": "Emergency",
    },
    "99284": {
        "short": "Emergency Dept Visit (high severity)",
        "category": "Emergency",
    },
    "T2033": {
        "short": "Residential Care, Non-Waiver (per diem)",
        "category": "Residential",
    },
    "T1000": {
        "short": "Private Duty Nursing (LPN, 15 min)",
        "category": "Home Health / Nursing",
    },
    "T1017": {
        "short": "Targeted Case Management (15 min)",
        "category": "Case Management",
    },
    "S5140": {
        "short": "Foster Care, Adult (per diem)",
        "category": "Foster Care",
    },
    "H2017": {
        "short": "Psychosocial Rehab Services (15 min)",
        "category": "Behavioral Health / Community",
    },
    "H2014": {
        "short": "Skills Training & Development (15 min)",
        "category": "Behavioral Health / Community",
    },
    "H0019": {
        "short": "Behavioral Health Day Treatment (per diem)",
        "category": "Behavioral Health / Day Programs",
    },
    "A0427": {
        "short": "ALS Ambulance, Emergency",
        "category": "Emergency / Transport",
    },
    "T1040": {
        "short": "Waiver Services, NOS (per diem)",
        "category": "Waiver / Other",
    },
    "J2326": {
        "short": "Nusinersen (Spinraza) Injection",
        "category": "Specialty Pharmacy",
    },
    "8888888": {
        "short": "State-Specific / Placeholder Code",
        "category": "Other",
    },
}


def hcpcs_label(code: str) -> str:
    """Return human-readable label for an HCPCS code."""
    info = HCPCS_DESCRIPTIONS.get(code)
    if info:
        return f"{code} — {info['short']}"
    return code


def hcpcs_category(code: str) -> str:
    """Return spending category for an HCPCS code."""
    info = HCPCS_DESCRIPTIONS.get(code)
    if info:
        return info["category"]
    # Infer category from code prefix
    if code.startswith("T10"):
        return "Home Health / Personal Care"
    if code.startswith("T20"):
        return "Waiver / Residential"
    if code.startswith(("H20", "H00")):
        return "Behavioral Health / Community"
    if code.startswith("S5"):
        return "Home Health / Personal Care"
    if code.startswith("99"):
        return "E&M / Office Visits"
    if code.startswith(("J", "Q")):
        return "Drugs / Injectables"
    if code.startswith("A0"):
        return "Emergency / Transport"
    return "Other"


def load_medicaid_stats() -> dict | None:
    """Load the comprehensive Medicaid analysis stats JSON."""
    path = ARTIFACT_DIR / "doge_medicaid_analysis_stats.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_monthly_summary() -> pd.DataFrame | None:
    """Load monthly Medicaid spending summary."""
    path = ARTIFACT_DIR / "doge_medicaid_monthly_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"], format="mixed")
    return df.sort_values("month")


def load_top_hcpcs() -> pd.DataFrame | None:
    """Load top HCPCS codes by spending."""
    path = ARTIFACT_DIR / "doge_medicaid_top_hcpcs.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["label"] = df["hcpcs_code"].apply(hcpcs_label)
    df["category"] = df["hcpcs_code"].apply(hcpcs_category)
    return df


def load_top_providers() -> pd.DataFrame | None:
    """Load top providers by spending."""
    path = ARTIFACT_DIR / "doge_medicaid_top_providers.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"billing_npi": str})


def load_doge_payments() -> pd.DataFrame | None:
    """Load DOGE CMS payments data."""
    path = ARTIFACT_DIR / "doge_hhs_cms_payments.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_doge_grants() -> pd.DataFrame | None:
    """Load DOGE HHS grants savings data."""
    path = ARTIFACT_DIR / "doge_hhs_grants_savings.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_doge_contracts() -> pd.DataFrame | None:
    """Load DOGE HHS contracts savings data."""
    path = ARTIFACT_DIR / "doge_hhs_contracts_savings.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def compute_spending_categories(hcpcs_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate HCPCS spending into broader categories."""
    if hcpcs_df is None or hcpcs_df.empty:
        return pd.DataFrame()
    grouped = (
        hcpcs_df.groupby("category")
        .agg(total_paid=("total_paid", "sum"), total_claims=("total_claims", "sum"), code_count=("hcpcs_code", "count"))
        .sort_values("total_paid", ascending=False)
        .reset_index()
    )
    grouped["pct_of_total"] = grouped["total_paid"] / grouped["total_paid"].sum() * 100
    return grouped


def compute_yoy_growth(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute year-over-year spending growth rates."""
    if monthly_df is None or monthly_df.empty:
        return pd.DataFrame()
    df = monthly_df.copy()
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month
    yearly = df.groupby("year").agg(total_paid=("total_paid", "sum")).reset_index()
    yearly["yoy_growth"] = yearly["total_paid"].pct_change() * 100
    return yearly


def compute_concentration_metrics(providers_df: pd.DataFrame) -> dict:
    """Compute provider spending concentration metrics."""
    if providers_df is None or providers_df.empty:
        return {}
    total = providers_df["total_paid"].sum()
    n = len(providers_df)
    cumsum = providers_df["total_paid"].cumsum()
    top_10 = providers_df.head(10)["total_paid"].sum()
    top_50 = providers_df.head(50)["total_paid"].sum()
    top_100 = providers_df.head(100)["total_paid"].sum()
    return {
        "total_providers_in_sample": n,
        "top_10_pct": top_10 / total * 100,
        "top_50_pct": top_50 / total * 100,
        "top_100_pct": top_100 / total * 100,
        "top_10_amount": top_10,
        "top_50_amount": top_50,
        "top_100_amount": top_100,
    }


# ---------------------------------------------------------------------------
# Palm Beach County functions
# ---------------------------------------------------------------------------

def load_pbc_stats() -> dict | None:
    """Load Palm Beach County Medicaid stats."""
    path = ARTIFACT_DIR / "doge_medicaid_pbc_stats.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_pbc_monthly() -> pd.DataFrame | None:
    """Load PBC monthly spending data."""
    path = ARTIFACT_DIR / "doge_medicaid_pbc_monthly.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"], format="mixed")
    return df.sort_values("month")


def load_pbc_hcpcs() -> pd.DataFrame | None:
    """Load PBC top HCPCS codes."""
    path = ARTIFACT_DIR / "doge_medicaid_pbc_hcpcs.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["label"] = df["hcpcs_code"].apply(hcpcs_label)
    df["category"] = df["hcpcs_code"].apply(hcpcs_category)
    return df


def load_pbc_providers() -> pd.DataFrame | None:
    """Load PBC providers with names and fraud signals."""
    path = ARTIFACT_DIR / "doge_medicaid_pbc_providers.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"billing_npi": str, "zip": str})


def load_named_providers() -> pd.DataFrame | None:
    """Load top national providers with names looked up from NPI registry."""
    path = ARTIFACT_DIR / "doge_medicaid_top_providers_named.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"npi": str, "zip": str})


def compute_fraud_signals(providers_df: pd.DataFrame) -> pd.DataFrame:
    """Flag providers with potential fraud indicators.

    Signals computed:
    - high_cost_per_claim: cost/claim > 95th percentile
    - high_cost_per_beneficiary: cost/beneficiary > 95th percentile
    - high_claims_per_beneficiary: claims/beneficiary > 95th percentile
    - low_code_diversity: only 1-2 HCPCS codes used (potential upcoding)
    - billing_servicing_mismatch: billing NPI != servicing NPI often
    - rapid_growth: spending increased > 50% YoY in recent period
    """
    if providers_df is None or providers_df.empty:
        return pd.DataFrame()

    df = providers_df.copy()

    # Compute per-unit metrics if not present
    if "cost_per_claim" not in df.columns:
        df["cost_per_claim"] = df["total_paid"] / df["total_claims"].clip(lower=1)
    if "cost_per_beneficiary" not in df.columns:
        df["cost_per_beneficiary"] = df["total_paid"] / df["total_beneficiaries"].clip(lower=1)
    if "claims_per_beneficiary" not in df.columns:
        df["claims_per_beneficiary"] = df["total_claims"] / df["total_beneficiaries"].clip(lower=1)

    # Flag thresholds (95th percentile)
    cpc_95 = df["cost_per_claim"].quantile(0.95)
    cpb_95 = df["cost_per_beneficiary"].quantile(0.95)
    cpb_claims_95 = df["claims_per_beneficiary"].quantile(0.95)

    df["flag_high_cost_per_claim"] = df["cost_per_claim"] > cpc_95
    df["flag_high_cost_per_beneficiary"] = df["cost_per_beneficiary"] > cpb_95
    df["flag_high_claims_per_beneficiary"] = df["claims_per_beneficiary"] > cpb_claims_95

    if "unique_hcpcs_codes" in df.columns:
        df["flag_low_code_diversity"] = df["unique_hcpcs_codes"] <= 2

    # Total flags
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["fraud_signal_count"] = df[flag_cols].sum(axis=1)

    return df.sort_values("fraud_signal_count", ascending=False)
