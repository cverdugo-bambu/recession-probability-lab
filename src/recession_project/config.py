from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"

MIN_DATE = "1960-01-01"
HORIZON_MONTHS = 6
TEST_MONTHS = 120
MIN_TRAIN_MONTHS_WALKFORWARD = 180
DEFAULT_ALERT_THRESHOLD = 0.40
FLORIDA_STRESS_CALM_MAX = 33.0
FLORIDA_STRESS_WATCH_MAX = 66.0

# FRED series codes.
SERIES_CODES = {
    "usrec": "USREC",          # NBER recession indicator
    "yield_spread": "T10Y2Y",  # 10Y - 2Y Treasury spread
    "unemployment": "UNRATE",  # unemployment rate
    "fedfunds": "FEDFUNDS",    # policy rate
    "cpi": "CPIAUCSL",         # CPI
    "industrial_prod": "INDPRO",
    "payrolls": "PAYEMS",
    "vix": "VIXCLS",           # equity volatility index
    "credit_spread": "BAA10Y", # Moody's BAA - 10Y Treasury spread
    "nfci": "NFCI",            # Chicago Fed National Financial Conditions Index
}

# Yahoo Finance market series (outside FRED).
YAHOO_TICKERS = {
    "sp500_price": "^GSPC",
    "oil_price": "CL=F",
    "gold_price": "GC=F",
    "dollar_index": "DX-Y.NYB",
    "hyg_price": "HYG",
    "ief_price": "IEF",
}

FLORIDA_SERIES_CODES = {
    "fl_unemployment_rate": "FLUR",
    "fl_total_payrolls": "FLNA",
    "fl_manufacturing_payrolls": "FLMFG",
    "fl_private_building_permits": "FLBPPRIVSA",
    "fl_initial_claims": "FLICLAIMS",
}

FEATURE_LABELS = {
    "yield_spread": "Treasury Curve (10Y minus 2Y)",
    "yield_spread_3m_delta": "Yield Curve Change (3-month)",
    "unemployment": "Unemployment Rate",
    "unemployment_3m_delta": "Unemployment Change (3-month)",
    "fedfunds": "Fed Policy Rate",
    "inflation_yoy": "Inflation (CPI, year-over-year)",
    "industrial_prod_yoy": "Industrial Production Growth (YoY)",
    "payrolls_yoy": "Payroll Employment Growth (YoY)",
    "vix_level": "Market Fear Index (VIX)",
    "credit_spread": "Corporate Credit Spread (BAA - 10Y)",
    "nfci": "Financial Conditions Index (Chicago Fed)",
    "sp500_6m_return": "S&P 500 Return (6-month)",
    "oil_6m_return": "Oil Return (6-month)",
    "gold_6m_return": "Gold Return (6-month)",
    "dollar_3m_return": "US Dollar Index Return (3-month)",
    "hyg_ief_3m_relative_return": "Risk Credit vs Treasuries (3-month)",
    "ff_mkt_excess": "Fama-French Market Excess Return",
    "ff_size_factor": "Fama-French Size Factor (SMB)",
    "ff_value_factor": "Fama-French Value Factor (HML)",
    "fl_unemployment_rate": "Florida Unemployment Rate",
    "fl_unemployment_3m_delta": "Florida Unemployment Change (3-month)",
    "fl_total_payrolls_yoy": "Florida Total Payroll Growth (YoY)",
    "fl_manufacturing_payrolls_yoy": "Florida Manufacturing Payroll Growth (YoY)",
    "fl_private_building_permits_yoy": "Florida Building Permits Growth (YoY)",
    "fl_initial_claims_3m_pct": "Florida Initial Claims Change (3-month)",
}

FEATURE_DESCRIPTIONS = {
    "yield_spread": "Negative values (inversion) often appear before recessions.",
    "yield_spread_3m_delta": "How quickly the curve is steepening or flattening.",
    "unemployment": "Higher unemployment usually reflects weakening labor demand.",
    "unemployment_3m_delta": "Fast increases can signal labor market stress.",
    "fedfunds": "Monetary policy stance from the Federal Reserve.",
    "inflation_yoy": "Consumer price growth relative to the same month last year.",
    "industrial_prod_yoy": "Factory/utilities/mining activity growth rate.",
    "payrolls_yoy": "Employment growth in nonfarm payrolls.",
    "vix_level": "Implied equity volatility; rises during market stress.",
    "credit_spread": "Extra yield for risky corporate debt versus Treasuries.",
    "nfci": "Broad financial stress indicator; higher means tighter conditions.",
    "sp500_6m_return": "Recent equity momentum and risk appetite.",
    "oil_6m_return": "Energy price shock proxy affecting costs and demand.",
    "gold_6m_return": "Risk-off and inflation-hedging behavior proxy.",
    "dollar_3m_return": "Rapid dollar strength can tighten global financial conditions.",
    "hyg_ief_3m_relative_return": "Credit risk appetite versus safer bond demand.",
    "ff_mkt_excess": "Broad equity risk premium from Fama-French data (non-FRED source).",
    "ff_size_factor": "Small-minus-big equity factor; risk appetite in smaller firms.",
    "ff_value_factor": "Value-minus-growth equity factor; style and macro sensitivity signal.",
    "fl_unemployment_rate": "Florida labor market weakness level.",
    "fl_unemployment_3m_delta": "How quickly Florida unemployment is changing.",
    "fl_total_payrolls_yoy": "Florida total jobs growth versus last year.",
    "fl_manufacturing_payrolls_yoy": "Florida manufacturing jobs growth versus last year.",
    "fl_private_building_permits_yoy": "Florida housing construction momentum.",
    "fl_initial_claims_3m_pct": "Short-term trend in Florida jobless claims.",
}
