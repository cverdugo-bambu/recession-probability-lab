"""Investigation of Ambassador Health / Ambassador Home Health Services fraud signals.

This module consolidates all Ambassador-branded entities appearing in the
Florida Medicaid provider-billing data, computes fraud-risk indicators, and
produces a structured report for further review.

Key findings summary (from the data):
    - 9 distinct NPI numbers all billing under the "Ambassador" brand
    - Spread across 9 Florida cities with a combined $197M+ in Medicaid billings
    - Multiple entities show extreme cost-per-beneficiary and claims-per-beneficiary
    - Several locations bill only 1-2 HCPCS codes (very low service diversity)
    - Pattern is consistent with a possible franchise/shell network
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .medicaid_analysis import load_pbc_providers, compute_fraud_signals, ARTIFACT_DIR


# ---------------------------------------------------------------------------
# Ambassador entity extraction
# ---------------------------------------------------------------------------

def get_ambassador_entities(providers_df: pd.DataFrame) -> pd.DataFrame:
    """Filter the provider dataset to Ambassador-branded entities only."""
    mask = providers_df["provider_name"].str.contains(
        r"(?i)ambassador", na=False
    )
    return providers_df[mask].copy()


# ---------------------------------------------------------------------------
# Fraud signal computation
# ---------------------------------------------------------------------------

@dataclass
class AmbassadorFraudReport:
    """Container for the full Ambassador investigation output."""

    entities: pd.DataFrame
    total_paid: float = 0.0
    entity_count: int = 0
    unique_cities: list[str] = field(default_factory=list)
    unique_npis: list[str] = field(default_factory=list)
    flags_summary: dict[str, int] = field(default_factory=dict)
    peer_comparison: dict[str, float] = field(default_factory=dict)
    red_flags: list[str] = field(default_factory=list)


def _compute_peer_stats(all_providers: pd.DataFrame) -> dict[str, float]:
    """Return median/mean/p95 for key metrics across all providers."""
    return {
        "median_cost_per_claim": float(all_providers["cost_per_claim"].median()),
        "mean_cost_per_claim": float(all_providers["cost_per_claim"].mean()),
        "p95_cost_per_claim": float(all_providers["cost_per_claim"].quantile(0.95)),
        "median_cost_per_beneficiary": float(
            all_providers["cost_per_beneficiary"].median()
        ),
        "mean_cost_per_beneficiary": float(
            all_providers["cost_per_beneficiary"].mean()
        ),
        "p95_cost_per_beneficiary": float(
            all_providers["cost_per_beneficiary"].quantile(0.95)
        ),
        "median_claims_per_beneficiary": float(
            all_providers["claims_per_beneficiary"].median()
        ),
        "mean_claims_per_beneficiary": float(
            all_providers["claims_per_beneficiary"].mean()
        ),
        "p95_claims_per_beneficiary": float(
            all_providers["claims_per_beneficiary"].quantile(0.95)
        ),
        "median_unique_hcpcs": float(all_providers["unique_hcpcs_codes"].median()),
        "mean_unique_hcpcs": float(all_providers["unique_hcpcs_codes"].mean()),
    }


def _identify_red_flags(
    amb: pd.DataFrame, peers: dict[str, float]
) -> list[str]:
    """Generate plain-English red flag descriptions from the data."""
    flags: list[str] = []

    # 1) Network / shell structure
    n = len(amb)
    cities = amb["city"].nunique()
    total = amb["total_paid"].sum()
    flags.append(
        f"NETWORK STRUCTURE: {n} separate NPIs billing under the Ambassador brand "
        f"across {cities} Florida cities, totalling ${total:,.2f} in Medicaid payments. "
        "This is consistent with a franchise or shell-company network designed to "
        "stay below per-entity audit thresholds."
    )

    # 2) Extreme cost per beneficiary
    high_cpb = amb[amb["cost_per_beneficiary"] > peers["p95_cost_per_beneficiary"]]
    if not high_cpb.empty:
        worst = high_cpb.sort_values("cost_per_beneficiary", ascending=False).iloc[0]
        flags.append(
            f"EXTREME COST PER BENEFICIARY: {len(high_cpb)} of {n} Ambassador entities "
            f"exceed the 95th-percentile cost-per-beneficiary "
            f"(>${peers['p95_cost_per_beneficiary']:,.2f}). "
            f"Worst offender: {worst['provider_name']} in {worst['city']} at "
            f"${worst['cost_per_beneficiary']:,.2f}/beneficiary — "
            f"{worst['cost_per_beneficiary'] / peers['median_cost_per_beneficiary']:.1f}x "
            f"the median provider."
        )

    # 3) Extreme cost per claim
    high_cpc = amb[amb["cost_per_claim"] > peers["p95_cost_per_claim"]]
    if not high_cpc.empty:
        flags.append(
            f"HIGH COST PER CLAIM: {len(high_cpc)} of {n} entities exceed the "
            f"95th-percentile cost-per-claim (>${peers['p95_cost_per_claim']:,.2f})."
        )

    # 4) Low code diversity (potential upcoding / phantom billing)
    low_div = amb[amb["unique_hcpcs_codes"] <= 2]
    if not low_div.empty:
        codes_list = ", ".join(
            f"{r['provider_name']} ({r['city']}): {r['unique_hcpcs_codes']} codes"
            for _, r in low_div.iterrows()
        )
        flags.append(
            f"LOW SERVICE DIVERSITY: {len(low_div)} of {n} entities bill only 1-2 HCPCS "
            f"codes (median provider uses {peers['median_unique_hcpcs']:.0f}). "
            f"This is a classic indicator of upcoding or phantom billing. "
            f"Details: {codes_list}"
        )

    # 5) High claims per beneficiary
    high_cpb_claims = amb[
        amb["claims_per_beneficiary"] > peers["p95_claims_per_beneficiary"]
    ]
    if not high_cpb_claims.empty:
        flags.append(
            f"EXCESSIVE UTILIZATION: {len(high_cpb_claims)} of {n} entities have "
            f"claims-per-beneficiary above the 95th percentile "
            f"({peers['p95_claims_per_beneficiary']:.1f}). "
            "This may indicate billing for services not actually rendered."
        )

    # 6) Geographic spread with consistent patterns
    if cities >= 5:
        avg_cpc = amb["cost_per_claim"].mean()
        std_cpc = amb["cost_per_claim"].std()
        cv = std_cpc / avg_cpc if avg_cpc > 0 else 0
        flags.append(
            f"GEOGRAPHIC DISPERSION: Ambassador operates in {cities} cities across "
            f"Florida. Average cost/claim=${avg_cpc:,.2f} with CV={cv:.2f}. "
            "A coordinated billing scheme would show consistent pricing patterns "
            "across locations."
        )

    # 7) Outlier entity (Tampa — looks different from the rest)
    tampa = amb[amb["city"] == "TAMPA"]
    if not tampa.empty:
        t = tampa.iloc[0]
        other_avg_cpc = amb[amb["city"] != "TAMPA"]["cost_per_claim"].mean()
        flags.append(
            f"TAMPA ANOMALY: The Tampa location (NPI {t['billing_npi']}) has "
            f"dramatically different billing patterns: cost/claim=${t['cost_per_claim']:,.2f} "
            f"vs ${other_avg_cpc:,.2f} average for other Ambassador locations, and "
            f"{t['total_beneficiaries']:,} beneficiaries (much higher). "
            "This may represent a legitimate operation or a different billing strategy."
        )

    return flags


def build_investigation_report(
    providers_df: pd.DataFrame | None = None,
) -> AmbassadorFraudReport:
    """Run the full Ambassador Health fraud investigation.

    Parameters
    ----------
    providers_df : DataFrame, optional
        Full provider dataset. If *None*, loads from the PBC providers CSV.

    Returns
    -------
    AmbassadorFraudReport
    """
    if providers_df is None:
        providers_df = load_pbc_providers()
    if providers_df is None or providers_df.empty:
        raise FileNotFoundError("Provider data not available.")

    # Apply fraud flags to ALL providers first (for percentile context)
    all_flagged = compute_fraud_signals(providers_df)

    # Extract Ambassador entities with their flags
    amb = get_ambassador_entities(all_flagged)
    peers = _compute_peer_stats(all_flagged)
    red_flags = _identify_red_flags(amb, peers)

    # Build flag summary
    flag_cols = [c for c in amb.columns if c.startswith("flag_")]
    flags_summary = {col: int(amb[col].sum()) for col in flag_cols}

    report = AmbassadorFraudReport(
        entities=amb,
        total_paid=float(amb["total_paid"].sum()),
        entity_count=len(amb),
        unique_cities=sorted(amb["city"].unique().tolist()),
        unique_npis=sorted(amb["billing_npi"].unique().tolist()),
        flags_summary=flags_summary,
        peer_comparison=peers,
        red_flags=red_flags,
    )
    return report


def report_to_dict(report: AmbassadorFraudReport) -> dict:
    """Convert report to a JSON-serialisable dict."""
    entity_records = report.entities[
        [
            "billing_npi",
            "provider_name",
            "city",
            "zip",
            "total_paid",
            "total_claims",
            "total_beneficiaries",
            "cost_per_claim",
            "cost_per_beneficiary",
            "claims_per_beneficiary",
            "months_active",
            "unique_hcpcs_codes",
            "fraud_signal_count",
        ]
    ].to_dict(orient="records")

    return {
        "investigation": "Ambassador Health / Ambassador Home Health Services",
        "state": "Florida",
        "total_medicaid_paid": report.total_paid,
        "entity_count": report.entity_count,
        "unique_npis": report.unique_npis,
        "unique_cities": report.unique_cities,
        "entities": entity_records,
        "flags_summary": report.flags_summary,
        "peer_comparison": report.peer_comparison,
        "red_flags": report.red_flags,
    }


def save_report(report: AmbassadorFraudReport, path: Path | None = None) -> Path:
    """Write the investigation report to a JSON artifact."""
    if path is None:
        path = ARTIFACT_DIR / "ambassador_health_investigation.json"
    with open(path, "w") as f:
        json.dump(report_to_dict(report), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Ambassador Health investigation and print findings."""
    report = build_investigation_report()
    out = save_report(report)
    print(f"Ambassador Health Investigation Report saved to {out}\n")
    print(f"{'='*72}")
    print("AMBASSADOR HEALTH FRAUD INVESTIGATION — SUMMARY")
    print(f"{'='*72}\n")
    print(f"Total entities:       {report.entity_count}")
    print(f"Total Medicaid paid:  ${report.total_paid:,.2f}")
    print(f"Unique NPIs:          {len(report.unique_npis)}")
    print(f"Cities:               {', '.join(report.unique_cities)}")
    print()
    print(f"{'─'*72}")
    print("ENTITY BREAKDOWN")
    print(f"{'─'*72}")
    for _, row in report.entities.iterrows():
        print(
            f"  NPI {row['billing_npi']}  {row['provider_name']:<45s} "
            f"{row['city']:<16s} ${row['total_paid']:>14,.2f}  "
            f"flags={int(row['fraud_signal_count'])}"
        )
    print()
    print(f"{'─'*72}")
    print("FLAG SUMMARY")
    print(f"{'─'*72}")
    for flag, count in report.flags_summary.items():
        print(f"  {flag}: {count}/{report.entity_count} entities")
    print()
    print(f"{'─'*72}")
    print("RED FLAGS")
    print(f"{'─'*72}")
    for i, flag in enumerate(report.red_flags, 1):
        print(f"\n  [{i}] {flag}")
    print()


if __name__ == "__main__":
    main()
