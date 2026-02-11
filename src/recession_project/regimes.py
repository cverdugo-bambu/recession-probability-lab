from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def _label_map(num_regimes: int) -> list[str]:
    if num_regimes == 2:
        return ["calm", "stress"]
    if num_regimes == 3:
        return ["calm", "watch", "stress"]
    return [f"regime_{i}" for i in range(num_regimes)]


def run_markov_switching(
    series: pd.Series,
    num_regimes: int = 3,
) -> tuple[pd.DataFrame, dict[str, object]]:
    clean = series.dropna().astype(float)
    if clean.empty:
        raise ValueError("Markov switching requires non-empty series.")

    model = MarkovRegression(clean, k_regimes=num_regimes, trend="c", switching_variance=True)
    fit = model.fit(disp=False)

    smoothed = fit.smoothed_marginal_probabilities
    if not isinstance(smoothed, pd.DataFrame):
        smoothed = pd.DataFrame(smoothed, index=clean.index)

    regime_means = []
    for i in range(smoothed.shape[1]):
        w = smoothed.iloc[:, i].to_numpy()
        w_sum = float(np.sum(w))
        mean_val = float(np.dot(clean.to_numpy(), w) / w_sum) if w_sum > 0 else float(clean.mean())
        regime_means.append((i, mean_val))
    regime_order = [idx for idx, _ in sorted(regime_means, key=lambda x: x[1])]

    labels = _label_map(smoothed.shape[1])
    rename_map = {old_idx: labels[new_pos] for new_pos, old_idx in enumerate(regime_order)}

    out = pd.DataFrame(index=clean.index)
    for old_idx, label in rename_map.items():
        out[f"regime_prob_{label}"] = smoothed.iloc[:, old_idx].to_numpy()

    prob_cols = [c for c in out.columns if c.startswith("regime_prob_")]
    out["regime_label"] = out[prob_cols].idxmax(axis=1).str.replace("regime_prob_", "", regex=False)
    out["signal_value"] = clean

    latest = out.iloc[-1]
    summary = {
        "date": pd.to_datetime(out.index[-1]).date().isoformat(),
        "latest_regime": str(latest["regime_label"]),
        "latest_signal_value": float(latest["signal_value"]),
        "regime_probabilities": {c.replace("regime_prob_", ""): float(latest[c]) for c in prob_cols},
        "regime_signal_means": {
            rename_map[idx]: float(mean_val) for idx, mean_val in regime_means
        },
    }
    return out, summary


def save_markov_outputs(
    regimes_df: pd.DataFrame,
    summary: dict[str, object],
    artifact_dir: Path | str,
) -> None:
    path = Path(artifact_dir)
    regimes_df.to_csv(path / "markov_regimes.csv", index_label="date")
    with (path / "markov_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
