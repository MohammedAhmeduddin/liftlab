# src/liftlab/causal/uplift.py
"""
Uplift models for heterogeneous treatment effect estimation.

Three estimators:
  - T-Learner  : separate outcome models per treatment arm (baseline)
  - X-Learner  : improves T-Learner via cross-fitting (better with imbalanced treatment)
  - CausalForest: non-parametric CATE estimator (the main model)

All return CATEResult with individual-level treatment effect estimates.
"""
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger
from typing import Optional

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from econml.metalearners import TLearner, XLearner
from econml.dml import CausalForestDML


@dataclass
class CATEResult:
    """Heterogeneous treatment effect estimation results."""
    method: str
    cate: np.ndarray             # Individual-level CATE estimates
    ate: float                   # Mean CATE = ATE
    ate_lower: float             # 95% CI lower
    ate_upper: float             # 95% CI upper
    feature_cols: list[str]
    shap_values: Optional[np.ndarray] = None
    shap_explainer: Optional[object] = None
    metadata: dict = field(default_factory=dict)

    def segment_effects(self, df: pd.DataFrame, col: str, n_bins: int = 4) -> pd.DataFrame:
        """
        Break down CATE by a feature column to find high-lift segments.
        This is how you answer: 'Which customers respond most to treatment?'
        """
        df = df.copy()
        df["_cate"] = self.cate
        df["_bin"] = pd.qcut(df[col], q=n_bins, labels=False, duplicates="drop")
        summary = (
            df.groupby("_bin")
            .agg(
                mean_cate=("_cate", "mean"),
                n=("_cate", "count"),
                feature_mean=(col, "mean"),
            )
            .reset_index()
        )
        summary["lift_vs_average"] = summary["mean_cate"] / self.ate
        return summary


# ---------------------------------------------------------------------------
# T-Learner
# ---------------------------------------------------------------------------

class TLearnerUplift:
    """
    T-Learner: train separate outcome models for treated and control.
    CATE(x) = E[Y|X=x, T=1] - E[Y|X=x, T=0]

    Weakness: with imbalanced treatment (85% treated here),
    the control model has limited data and may be noisy.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = TLearner(
            models=GradientBoostingRegressor(
                n_estimators=100, max_depth=4, random_state=seed
            )
        )
        self.scaler = StandardScaler()

    def fit_estimate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
    ) -> CATEResult:
        logger.info("Fitting T-Learner...")
        X = self.scaler.fit_transform(df[feature_cols])
        T = df[treatment_col].values.ravel()
        Y = df[outcome_col].values.astype(float).ravel()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(Y, T, X=X)
            cate = self.model.effect(X).ravel()

        ate = float(cate.mean())

        # Bootstrap CI on ATE
        rng = np.random.default_rng(self.seed)
        boot_ates = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(100):
                idx = rng.choice(len(X), len(X), replace=True)
                boot_ates.append(self.model.effect(X[idx]).mean())

        ci_lower = float(np.percentile(boot_ates, 2.5))
        ci_upper = float(np.percentile(boot_ates, 97.5))

        logger.info(f"T-Learner ATE={ate:.6f} | CI=[{ci_lower:.6f}, {ci_upper:.6f}]")

        return CATEResult(
            method="T-Learner",
            cate=cate,
            ate=ate,
            ate_lower=ci_lower,
            ate_upper=ci_upper,
            feature_cols=feature_cols,
            metadata={"n_obs": len(df)},
        )


# ---------------------------------------------------------------------------
# X-Learner
# ---------------------------------------------------------------------------

class XLearnerUplift:
    """
    X-Learner: improvement over T-Learner via cross-fitting.
    Better when treatment groups are imbalanced (our case: 85/15 split).

    Steps:
    1. Fit outcome models per arm (like T-Learner)
    2. Compute imputed treatment effects via cross-prediction
    3. Fit CATE models on imputed effects
    4. Combine using propensity-weighted average
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = XLearner(
            models=GradientBoostingRegressor(
                n_estimators=100, max_depth=4, random_state=seed
            ),
            propensity_model=LogisticRegression(max_iter=1000, random_state=seed),
        )
        self.scaler = StandardScaler()

    def fit_estimate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
    ) -> CATEResult:
        logger.info("Fitting X-Learner...")
        X = self.scaler.fit_transform(df[feature_cols])
        T = df[treatment_col].values.ravel()
        Y = df[outcome_col].values.astype(float).ravel()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(Y, T, X=X)
            cate = self.model.effect(X).ravel()

        ate = float(cate.mean())

        rng = np.random.default_rng(self.seed)
        boot_ates = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(100):
                idx = rng.choice(len(X), len(X), replace=True)
                boot_ates.append(self.model.effect(X[idx]).mean())

        ci_lower = float(np.percentile(boot_ates, 2.5))
        ci_upper = float(np.percentile(boot_ates, 97.5))

        logger.info(f"X-Learner ATE={ate:.6f} | CI=[{ci_lower:.6f}, {ci_upper:.6f}]")

        return CATEResult(
            method="X-Learner",
            cate=cate,
            ate=ate,
            ate_lower=ci_lower,
            ate_upper=ci_upper,
            feature_cols=feature_cols,
            metadata={"n_obs": len(df)},
        )


# ---------------------------------------------------------------------------
# Causal Forest DML (main model)
# ---------------------------------------------------------------------------

class CausalForestUplift:
    """
    CausalForestDML: the most robust CATE estimator.

    Uses Double Machine Learning (DML) to:
    1. Partial out the effect of X on Y (residualize outcome)
    2. Partial out the effect of X on T (residualize treatment)
    3. Fit a causal forest on the residuals → clean CATE estimates

    This removes the influence of confounders non-parametrically,
    making it robust to model misspecification.
    """

    def __init__(self, n_estimators: int = 100, seed: int = 42):
        self.seed = seed
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()
        self._X_scaled: Optional[np.ndarray] = None
        self.model: Optional[CausalForestDML] = None

    def _build_model(self) -> CausalForestDML:
        return CausalForestDML(
            n_estimators=self.n_estimators,
            min_samples_leaf=10,
            max_depth=5,
            random_state=self.seed,
            cv=3,
            discrete_treatment=True,
            model_y=GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=self.seed
            ),
            model_t=GradientBoostingClassifier(
                n_estimators=50, max_depth=3, random_state=self.seed
            ),
        )

    def fit_estimate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
    ) -> CATEResult:
        logger.info(f"Fitting CausalForestDML on {len(df):,} rows...")

        X = self.scaler.fit_transform(df[feature_cols])
        T = df[treatment_col].values.astype(float).ravel()
        Y = df[outcome_col].values.astype(float).ravel()

        self._X_scaled = X
        self.model = self._build_model()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(Y, T, X=X)

        # Point estimates
        cate = self.model.effect(X).ravel()
        ate = float(cate.mean())

        # Confidence intervals from the forest's built-in inference
        try:
            ate_interval = self.model.ate_interval(X, T0=0, T1=1)
            ci_lower = float(ate_interval[0])
            ci_upper = float(ate_interval[1])
        except Exception as e:
            logger.warning(f"Built-in CI failed ({e}) — falling back to bootstrap")
            rng = np.random.default_rng(self.seed)
            boot_ates = []
            for _ in range(100):
                idx = rng.choice(len(X), len(X), replace=True)
                boot_ates.append(self.model.effect(X[idx]).mean())
            ci_lower = float(np.percentile(boot_ates, 2.5))
            ci_upper = float(np.percentile(boot_ates, 97.5))

        logger.info(
            f"CausalForest ATE={ate:.6f} | CI=[{ci_lower:.6f}, {ci_upper:.6f}]"
        )

        # SHAP values — capped at 2K rows for speed
        shap_arr = None
        logger.info("Computing SHAP values on Causal Forest...")
        try:
            shap_sample = X[:2000]
            shap_values = self.model.shap_values(shap_sample)

            if isinstance(shap_values, dict):
                shap_arr = list(shap_values.values())[0]
                if isinstance(shap_arr, dict):
                    shap_arr = list(shap_arr.values())[0]
            else:
                shap_arr = shap_values

            # Unwrap SHAP Explanation object if needed
            if hasattr(shap_arr, "values"):
                shap_arr = shap_arr.values

            if isinstance(shap_arr, np.ndarray) and shap_arr.ndim == 3:
                shap_arr = shap_arr[:, :, 0]

            logger.info("SHAP values computed ✅")
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e} — skipping")
            shap_arr = None

        return CATEResult(
            method="CausalForest",
            cate=cate,
            ate=ate,
            ate_lower=ci_lower,
            ate_upper=ci_upper,
            feature_cols=feature_cols,
            shap_values=shap_arr,
            metadata={
                "n_estimators": self.n_estimators,
                "n_obs": len(df),
            },
        )

    def feature_importance(self, feature_cols: list[str]) -> pd.DataFrame:
        """SHAP-based feature importance for treatment effect heterogeneity."""
        if self._X_scaled is None or self.model is None:
            raise RuntimeError("Call fit_estimate() first.")

        try:
            shap_sample = self._X_scaled[:2000]
            shap_values = self.model.shap_values(shap_sample)

            if isinstance(shap_values, dict):
                shap_arr = list(shap_values.values())[0]
                if isinstance(shap_arr, dict):
                    shap_arr = list(shap_arr.values())[0]
            else:
                shap_arr = shap_values

            # Unwrap SHAP Explanation object if needed
            if hasattr(shap_arr, "values"):
                shap_arr = shap_arr.values

            if isinstance(shap_arr, np.ndarray) and shap_arr.ndim == 3:
                shap_arr = shap_arr[:, :, 0]

            importance = np.abs(shap_arr).mean(axis=0)

        except Exception as e:
            logger.warning(f"SHAP feature importance failed: {e} — using uniform importance")
            importance = np.ones(len(feature_cols)) / len(feature_cols)

        return (
            pd.DataFrame({"feature": feature_cols, "shap_importance": importance})
            .sort_values("shap_importance", ascending=False)
            .reset_index(drop=True)
        )
