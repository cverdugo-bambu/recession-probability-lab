from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(
    "ignore",
    message="This Pipeline instance is not fitted yet.*",
    category=FutureWarning,
)

try:
    from ngboost import NGBClassifier

    HAS_NGBOOST = True
except Exception:  # noqa: BLE001
    NGBClassifier = None
    HAS_NGBOOST = False

TARGET_COL = "target_recession_next_horizon"
ENSEMBLE_MODEL_NAME = "ensemble"

ALGORITHM_CATALOG = [
    {
        "id": "logistic",
        "name": "Logistic Regression",
        "status": "active",
        "what_it_does": "Baseline probability model. Easy to interpret and explain.",
    },
    {
        "id": "hist_gb",
        "name": "Histogram Gradient Boosting",
        "status": "active",
        "what_it_does": "Tree-based model that captures nonlinear patterns and interactions.",
    },
    {
        "id": "bayesian_dynamic",
        "name": "Bayesian Dynamic Model",
        "status": "active",
        "what_it_does": "Bayesian linear model with time dynamics and uncertainty bands.",
    },
    {
        "id": "ngboost",
        "name": "NGBoost",
        "status": "active" if HAS_NGBOOST else "unavailable",
        "what_it_does": "Natural Gradient Boosting for full predictive distributions.",
    },
    {
        "id": ENSEMBLE_MODEL_NAME,
        "name": "Ensemble Average",
        "status": "active",
        "what_it_does": "Averages probabilities from active base models for stability.",
    },
    {
        "id": "markov_switching",
        "name": "Markov Switching Model",
        "status": "active",
        "what_it_does": "Detects regime shifts (calm/watch/stress) from probability dynamics.",
    },
]


def get_algorithm_catalog() -> list[dict[str, str]]:
    return [dict(item) for item in ALGORITHM_CATALOG]


class BayesianDynamicBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian regression classifier with uncertainty bands and simple time dynamics."""

    def __init__(
        self,
        ci_z: float = 1.64,
        use_time_feature: bool = True,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
    ) -> None:
        self.ci_z = ci_z
        self.use_time_feature = use_time_feature
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def _augment_train(self, x: np.ndarray) -> np.ndarray:
        if not self.use_time_feature:
            return x
        t = np.linspace(0.0, 1.0, len(x), endpoint=True).reshape(-1, 1)
        return np.hstack([x, t])

    def _augment_predict(self, x: np.ndarray) -> np.ndarray:
        if not self.use_time_feature:
            return x
        step = 1.0 / max(self._fit_length, 1)
        t = (1.0 + step * np.arange(1, len(x) + 1)).reshape(-1, 1)
        return np.hstack([x, t])

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "BayesianDynamicBinaryClassifier":
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        self._fit_length = len(x_arr)
        self._n_features = x_arr.shape[1]

        self._model = BayesianRidge(
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
        )
        self._model.fit(self._augment_train(x_arr), y_arr)

        coef_full = np.asarray(self._model.coef_)
        self.coef_ = coef_full[: self._n_features]
        self.intercept_ = float(self._model.intercept_)
        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def _predict_mean_std(self, x: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float)
        mean, std = self._model.predict(self._augment_predict(x_arr), return_std=True)
        return mean, std

    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        mean, _ = self._predict_mean_std(x)
        p = np.clip(mean, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p, p])

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)[:, 1]
        return (proba >= 0.5).astype(int)

    def predict_interval_proba(self, x: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, std = self._predict_mean_std(x)
        lower = np.clip(mean - self.ci_z * std, 1e-6, 1.0 - 1e-6)
        upper = np.clip(mean + self.ci_z * std, 1e-6, 1.0 - 1e-6)
        return lower, upper


@dataclass
class TrainResult:
    selected_model_name: str
    fitted_models_train: dict[str, Pipeline]
    feature_cols: list[str]
    train_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    model_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    calibration: pd.DataFrame


def train_test_split_time(df: pd.DataFrame, test_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= test_months + 24:
        raise ValueError("Not enough history for a stable train/test split.")
    train = df.iloc[:-test_months].copy()
    test = df.iloc[-test_months:].copy()
    return train, test


def _build_logistic() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _build_hist_gb() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_depth=3,
                    max_iter=400,
                    min_samples_leaf=20,
                    random_state=42,
                ),
            ),
        ]
    )


def _build_bayesian_dynamic() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", BayesianDynamicBinaryClassifier()),
        ]
    )


def _build_ngboost() -> Pipeline:
    if not HAS_NGBOOST or NGBClassifier is None:
        raise ValueError("NGBoost is unavailable in this environment.")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "clf",
                NGBClassifier(
                    n_estimators=400,
                    learning_rate=0.03,
                    random_state=42,
                    verbose=False,
                ),
            ),
        ]
    )


def _model_factories() -> dict[str, callable]:
    factories: dict[str, callable] = {
        "logistic": _build_logistic,
        "hist_gb": _build_hist_gb,
        "bayesian_dynamic": _build_bayesian_dynamic,
    }
    if HAS_NGBOOST and NGBClassifier is not None:
        factories["ngboost"] = _build_ngboost
    return factories


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    eps = 1e-8
    y_prob_safe = np.clip(y_prob, eps, 1.0 - eps)
    unique = np.unique(y_true)
    if unique.size < 2:
        roc_auc = float("nan")
        avg_precision = float("nan")
        logloss = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob_safe))
        avg_precision = float(average_precision_score(y_true, y_prob_safe))
        logloss = float(log_loss(y_true, np.vstack([1 - y_prob_safe, y_prob_safe]).T))
    return {
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "brier_score": float(brier_score_loss(y_true, y_prob_safe)),
        "log_loss": logloss,
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(y_pred.mean()),
    }


def _predict_ensemble(probabilities: dict[str, np.ndarray]) -> np.ndarray:
    base_models = [k for k in probabilities if k != ENSEMBLE_MODEL_NAME]
    return np.mean(np.column_stack([probabilities[k] for k in base_models]), axis=1)


def _predict_interval_from_pipeline(model: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    if hasattr(model, "predict_interval_proba"):
        try:
            return model.predict_interval_proba(x)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

    clf = model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "predict_interval_proba"):
        return None

    x_trans = x
    for _, step in model.steps[:-1]:
        x_trans = step.transform(x_trans)
    return clf.predict_interval_proba(x_trans)


def _prediction_frame(
    df: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    selected_model_name: str,
    intervals: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["y_true"] = df[TARGET_COL].astype(int)
    out["current_recession"] = df["current_recession"].astype(int)
    for model_name, probs in probabilities.items():
        out[f"{model_name}_prob"] = probs
    if intervals:
        for model_name, (lower, upper) in intervals.items():
            out[f"{model_name}_prob_lower"] = lower
            out[f"{model_name}_prob_upper"] = upper
    out["y_prob"] = out[f"{selected_model_name}_prob"]
    return out


def _feature_importance(
    fitted_models: dict[str, Pipeline],
    feature_cols: list[str],
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    if "logistic" in fitted_models:
        coefs = fitted_models["logistic"].named_steps["clf"].coef_[0]
        for feature, coef in zip(feature_cols, coefs):
            rows.append(
                {
                    "model": "logistic",
                    "feature": feature,
                    "importance": float(coef),
                    "abs_importance": float(abs(coef)),
                    "importance_type": "coefficient",
                }
            )

    if "bayesian_dynamic" in fitted_models:
        coefs = fitted_models["bayesian_dynamic"].named_steps["clf"].coef_
        for feature, coef in zip(feature_cols, coefs):
            rows.append(
                {
                    "model": "bayesian_dynamic",
                    "feature": feature,
                    "importance": float(coef),
                    "abs_importance": float(abs(coef)),
                    "importance_type": "coefficient",
                }
            )

    for model_name in ["hist_gb", "ngboost"]:
        if model_name not in fitted_models:
            continue
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This Pipeline instance is not fitted yet.*",
                    category=FutureWarning,
                )
                perm = permutation_importance(
                    fitted_models[model_name],
                    x_test,
                    y_test,
                    n_repeats=20,
                    random_state=42,
                    scoring="neg_brier_score",
                )
            importance_type = "permutation_brier"
        except Exception:  # noqa: BLE001
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This Pipeline instance is not fitted yet.*",
                    category=FutureWarning,
                )
                perm = permutation_importance(
                    fitted_models[model_name],
                    x_test,
                    y_test,
                    n_repeats=20,
                    random_state=42,
                    scoring="neg_mean_squared_error",
                )
            importance_type = "permutation_mse"
        for feature, imp in zip(feature_cols, perm.importances_mean):
            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "importance": float(imp),
                    "abs_importance": float(abs(imp)),
                    "importance_type": importance_type,
                }
            )

    return pd.DataFrame(rows).sort_values(["model", "abs_importance"], ascending=[True, False])


def fit_and_evaluate(df: pd.DataFrame, test_months: int) -> TrainResult:
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, "current_recession"}]

    train_df, test_df = train_test_split_time(df=df, test_months=test_months)

    x_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)
    x_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int)

    factories = _model_factories()
    fitted_models_train: dict[str, Pipeline] = {}
    train_prob: dict[str, np.ndarray] = {}
    test_prob: dict[str, np.ndarray] = {}
    train_intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    test_intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for model_name, factory in factories.items():
        fitted = factory()
        fitted.fit(x_train, y_train)
        fitted_models_train[model_name] = fitted
        train_prob[model_name] = fitted.predict_proba(x_train)[:, 1]
        test_prob[model_name] = fitted.predict_proba(x_test)[:, 1]

        train_int = _predict_interval_from_pipeline(fitted, x_train)
        if train_int is not None:
            train_intervals[model_name] = train_int
        test_int = _predict_interval_from_pipeline(fitted, x_test)
        if test_int is not None:
            test_intervals[model_name] = test_int

    train_prob[ENSEMBLE_MODEL_NAME] = _predict_ensemble(train_prob)
    test_prob[ENSEMBLE_MODEL_NAME] = _predict_ensemble(test_prob)

    metric_rows = []
    for model_name, probs in test_prob.items():
        row = _metrics(y_test.to_numpy(), probs)
        row["model"] = model_name
        metric_rows.append(row)
    metrics_df = pd.DataFrame(metric_rows).sort_values("brier_score").reset_index(drop=True)
    selected_model_name = str(metrics_df.iloc[0]["model"])

    train_pred = _prediction_frame(
        train_df,
        train_prob,
        selected_model_name=selected_model_name,
        intervals=train_intervals,
    )
    test_pred = _prediction_frame(
        test_df,
        test_prob,
        selected_model_name=selected_model_name,
        intervals=test_intervals,
    )

    importance = _feature_importance(
        fitted_models=fitted_models_train,
        feature_cols=feature_cols,
        x_test=x_test,
        y_test=y_test,
    )

    selected_test_prob = test_pred["y_prob"].to_numpy()
    frac_pos, mean_pred = calibration_curve(y_test, selected_test_prob, n_bins=10, strategy="uniform")
    calibration = pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})

    return TrainResult(
        selected_model_name=selected_model_name,
        fitted_models_train=fitted_models_train,
        feature_cols=feature_cols,
        train_predictions=train_pred,
        test_predictions=test_pred,
        model_metrics=metrics_df,
        feature_importance=importance,
        calibration=calibration,
    )


def fit_final_models(df: pd.DataFrame, selected_model_name: str) -> tuple[dict[str, Pipeline], list[str]]:
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, "current_recession"}]
    x = df[feature_cols]
    y = df[TARGET_COL].astype(int)

    factories = _model_factories()
    fitted: dict[str, Pipeline] = {}

    if selected_model_name == ENSEMBLE_MODEL_NAME:
        model_names = [k for k in factories.keys()]
    else:
        model_names = [selected_model_name]

    for model_name in model_names:
        if model_name not in factories:
            continue
        est = factories[model_name]()
        est.fit(x, y)
        fitted[model_name] = est

    return fitted, feature_cols


def predict_selected_probability(
    fitted_models: dict[str, Pipeline],
    x: pd.DataFrame,
    selected_model_name: str,
) -> np.ndarray:
    if selected_model_name == ENSEMBLE_MODEL_NAME:
        probs = [fitted_models[name].predict_proba(x)[:, 1] for name in sorted(fitted_models.keys())]
        return np.mean(np.column_stack(probs), axis=1)
    return fitted_models[selected_model_name].predict_proba(x)[:, 1]


def predict_selected_probability_interval(
    fitted_models: dict[str, Pipeline],
    x: pd.DataFrame,
    selected_model_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if selected_model_name == ENSEMBLE_MODEL_NAME:
        intervals = []
        for name in sorted(fitted_models.keys()):
            interval = _predict_interval_from_pipeline(fitted_models[name], x)
            if interval is None:
                continue
            intervals.append(interval)
        if not intervals:
            return None
        lowers = np.column_stack([i[0] for i in intervals])
        uppers = np.column_stack([i[1] for i in intervals])
        return lowers.mean(axis=1), uppers.mean(axis=1)

    model = fitted_models.get(selected_model_name)
    if model is None:
        return None
    return _predict_interval_from_pipeline(model, x)


def walk_forward_backtest(
    df: pd.DataFrame,
    selected_model_name: str,
    min_train_months: int,
) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in {TARGET_COL, "current_recession"}]
    factories = _model_factories()
    rows: list[dict[str, float | int | pd.Timestamp]] = []

    for i in range(min_train_months, len(df)):
        train_df = df.iloc[:i]
        test_df = df.iloc[i : i + 1]
        if train_df[TARGET_COL].nunique() < 2:
            continue

        x_train = train_df[feature_cols]
        y_train = train_df[TARGET_COL].astype(int)
        x_test = test_df[feature_cols]

        if selected_model_name == ENSEMBLE_MODEL_NAME:
            probs = []
            for model_name in sorted(factories.keys()):
                est = factories[model_name]()
                est.fit(x_train, y_train)
                probs.append(est.predict_proba(x_test)[:, 1][0])
            y_prob = float(np.mean(probs))
            y_low = np.nan
            y_high = np.nan
        else:
            est = factories[selected_model_name]()
            est.fit(x_train, y_train)
            y_prob = float(est.predict_proba(x_test)[:, 1][0])
            interval = _predict_interval_from_pipeline(est, x_test)
            if interval is None:
                y_low, y_high = np.nan, np.nan
            else:
                y_low, y_high = float(interval[0][0]), float(interval[1][0])

        rows.append(
            {
                "date": test_df.index[0],
                "y_true": int(test_df[TARGET_COL].iloc[0]),
                "current_recession": int(test_df["current_recession"].iloc[0]),
                "y_prob": y_prob,
                "y_prob_lower": y_low,
                "y_prob_upper": y_high,
            }
        )

    return pd.DataFrame(rows)
