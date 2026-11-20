"""
Average-effect estimators.

- difference_in_means: unbiased when treatment is truly randomized. On
  Criteo a small treated/control imbalance makes it drift, so it is
  reported but not used as the reference.
- AIPW: doubly robust, cross-fitted, covariate-adjusted ATE (or ATT) with
  an influence-function standard error. Consistent if either the outcome
  or the propensity model is right; the Criteo reference.
- PropensityScoreMatching: effect on the treated (ATT) for observational
  data, 1:1 nearest-neighbour matching with replacement on the logit
  propensity, Austin's caliper (0.2 SD), balance diagnostics, and an
  Abadie-Imbens style standard error that accounts for reused controls.
- DifferenceInDifferences: 2x2 DiD for panel data with a pre and a post
  period. Needs real time variation; it is not applicable to Criteo.

All return a CausalEstimate with the estimand stated explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from loguru import logger
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

Z95 = stats.norm.ppf(0.975)


@dataclass
class CausalEstimate:
    """Standardized output for all average-effect estimators."""
    method: str
    estimand: str                 # "ATE" or "ATT"
    ate: float                    # point estimate of the estimand
    ci_lower: float               # 95% CI
    ci_upper: float
    std_error: float
    n_treated: int
    n_control: int
    p_value: float
    metadata: dict = field(default_factory=dict)


def _check_binary(series: pd.Series, name: str) -> None:
    values = set(pd.unique(series.dropna()))
    if not values <= {0, 1}:
        raise ValueError(f"'{name}' must be binary 0/1, found {sorted(values)[:5]}")
    if len(values) < 2:
        raise ValueError(f"'{name}' needs both 0 and 1 values")


def _normal_ci(est: float, se: float) -> tuple[float, float, float]:
    p = 2 * (1 - stats.norm.cdf(abs(est / se))) if se > 0 else float("nan")
    return est - Z95 * se, est + Z95 * se, p


# ---------------------------------------------------------------------------
# Difference in means (randomized trials)
# ---------------------------------------------------------------------------

def difference_in_means(
    df: pd.DataFrame, treatment_col: str = "treatment", outcome_col: str = "conversion"
) -> CausalEstimate:
    """Unbiased ATE under randomization, Neyman (unpooled) standard error."""
    _check_binary(df[treatment_col], treatment_col)
    t = df[treatment_col].to_numpy() == 1
    y = df[outcome_col].to_numpy(dtype=float)
    y1, y0 = y[t], y[~t]
    ate = y1.mean() - y0.mean()
    se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
    lo, hi, p = _normal_ci(ate, se)
    return CausalEstimate(
        method="Difference in means", estimand="ATE",
        ate=float(ate), ci_lower=float(lo), ci_upper=float(hi), std_error=se,
        n_treated=int(t.sum()), n_control=int((~t).sum()), p_value=float(p),
        metadata={
            "treated_rate": float(y1.mean()),
            "control_rate": float(y0.mean()),
            "relative_lift": float(ate / y0.mean()) if y0.mean() > 0 else None,
        },
    )


# ---------------------------------------------------------------------------
# AIPW (doubly robust, cross-fitted)
# ---------------------------------------------------------------------------

def _default_outcome_model():
    return HistGradientBoostingRegressor(max_iter=100, max_depth=4, random_state=0)


def _default_propensity_model():
    """Gradient boosting, not logistic regression: selection that depends
    non-linearly on a skewed or heavily tied feature (common in Criteo) is
    missed by a linear-logit model, and matching or weighting on a wrong
    propensity leaves most of the bias in place."""
    return HistGradientBoostingClassifier(
        max_iter=300, max_depth=3, learning_rate=0.05, early_stopping=True,
        n_iter_no_change=10, random_state=0)


def cross_fit_nuisances(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    n_folds: int = 2,
    propensity: Optional[float] = None,
    outcome_model: Callable = _default_outcome_model,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold predictions of E[Y|X,T=0], E[Y|X,T=1] and P(T=1|X).

    Cross-fitting keeps each unit's nuisance predictions out of the data
    they were trained on, which is what makes the AIPW standard error valid.
    Pass ``propensity`` when it is known (randomized trial).
    """
    n = len(y)
    mu0, mu1, e = np.zeros(n), np.zeros(n), np.zeros(n)
    for train, test in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        for arm, target in ((0, mu0), (1, mu1)):
            idx = train[t[train] == arm]
            target[test] = outcome_model().fit(X[idx], y[idx]).predict(X[test])
        if propensity is None:
            model = _default_propensity_model().fit(X[train], t[train])
            e[test] = model.predict_proba(X[test])[:, 1]
        else:
            e[test] = propensity
    return mu0, mu1, np.clip(e, 0.01, 0.99)


def aipw_scores(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    propensity: Optional[float] = None,
    n_folds: int = 2,
    seed: int = 42,
    estimand: str = "ATE",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-unit doubly robust scores and the propensities used.

    ATE: the mean of the scores is the ATE, and the mean within a subgroup
    is that subgroup's effect. ATT: the mean is the effect on the treated
    (only E[Y|X,T=0] and the propensity are needed).
    """
    if estimand not in ("ATE", "ATT"):
        raise ValueError("estimand must be 'ATE' or 'ATT'")
    _check_binary(df[treatment_col], treatment_col)
    X = StandardScaler().fit_transform(df[feature_cols].to_numpy(dtype=float))
    t = df[treatment_col].to_numpy(dtype=int)
    y = df[outcome_col].to_numpy(dtype=float)
    mu0, mu1, e = cross_fit_nuisances(X, t, y, n_folds, propensity, seed=seed)
    if estimand == "ATT":
        psi = (t * (y - mu0) - (1 - t) * e / (1 - e) * (y - mu0)) / t.mean()
    else:
        psi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    return psi, e


def aipw(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str = "treatment",
    outcome_col: str = "conversion",
    propensity: Optional[float] = None,
    n_folds: int = 2,
    seed: int = 42,
    estimand: str = "ATE",
) -> CausalEstimate:
    """Augmented inverse-propensity weighted ATE (or ATT) with cross-fitting."""
    psi, e = aipw_scores(df, feature_cols, treatment_col, outcome_col, propensity,
                         n_folds, seed, estimand)
    t = df[treatment_col].to_numpy(dtype=int)
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
    lo, hi, p = _normal_ci(ate, se)
    return CausalEstimate(
        method="AIPW (doubly robust)", estimand=estimand,
        ate=ate, ci_lower=float(lo), ci_upper=float(hi), std_error=se,
        n_treated=int(t.sum()), n_control=int((1 - t).sum()), p_value=float(p),
        metadata={
            "n_folds": n_folds,
            "propensity": "known" if propensity is not None else "estimated",
            # share of units whose propensity hit the 0.01 / 0.99 clip: a
            # large value means poor overlap and a less trustworthy estimate
            "clipped_share": float(np.mean((e <= 0.01) | (e >= 0.99))),
        },
    )


# ---------------------------------------------------------------------------
# Propensity score matching (ATT)
# ---------------------------------------------------------------------------

def standardized_mean_differences(
    treated: pd.DataFrame, control: pd.DataFrame, cols: list[str]
) -> dict[str, float]:
    """|SMD| per covariate; < 0.1 is the usual threshold for good balance."""
    out = {}
    for c in cols:
        pooled = np.sqrt((treated[c].var() + control[c].var()) / 2)
        out[c] = float(abs(treated[c].mean() - control[c].mean()) / pooled) if pooled > 0 else 0.0
    return out


class PropensityScoreMatching:
    """1:1 nearest-neighbour matching with replacement on the logit propensity.

    Estimand: ATT (effect on the treated). Treated units with no control
    within the caliper are dropped and reported in ``match_rate``.
    """

    def __init__(self, caliper_sd: float = 0.2, seed: int = 42):
        self.caliper_sd = caliper_sd
        self.seed = seed

    @staticmethod
    def _logit_propensity(X: np.ndarray, t: np.ndarray) -> np.ndarray:
        model = _default_propensity_model().fit(X, t)
        p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _att(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Returns (ATT, matched treated indices, matched control indices)."""
        lp = self._logit_propensity(X, t)
        caliper = self.caliper_sd * lp.std()
        ti, ci = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
        nn = NearestNeighbors(n_neighbors=1).fit(lp[ci].reshape(-1, 1))
        dist, pos = nn.kneighbors(lp[ti].reshape(-1, 1))
        ok = dist.ravel() <= caliper
        mt, mc = ti[ok], ci[pos.ravel()[ok]]
        if len(mt) == 0:
            raise ValueError("No treated unit has a control within the caliper")
        return float((y[mt] - y[mc]).mean()), mt, mc

    def estimate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
    ) -> CausalEstimate:
        _check_binary(df[treatment_col], treatment_col)
        X = StandardScaler().fit_transform(df[feature_cols].to_numpy(dtype=float))
        t = df[treatment_col].to_numpy(dtype=int)
        y = df[outcome_col].to_numpy(dtype=float)
        att, mt, mc = self._att(X, t, y)

        # Abadie-Imbens style variance. The naive bootstrap is not valid for
        # nearest-neighbour matching with replacement (Abadie & Imbens 2008).
        # Pair differences capture outcome noise and effect heterogeneity;
        # the second term adds the covariance from a control being reused
        # K times (homoskedastic control variance).
        n1 = len(mt)
        d = y[mt] - y[mc]
        _, k = np.unique(mc, return_counts=True)
        sigma0 = float(np.var(y[np.unique(mc)], ddof=1)) if len(k) > 1 else 0.0
        var = (np.sum((d - att) ** 2) + np.sum(k * (k - 1)) * sigma0) / n1**2
        se = float(np.sqrt(var))
        lo, hi, p = _normal_ci(att, se)

        frame = df[feature_cols]
        smd_before = standardized_mean_differences(frame[t == 1], frame[t == 0], feature_cols)
        smd_after = standardized_mean_differences(frame.iloc[mt], frame.iloc[mc], feature_cols)
        n_treated = int((t == 1).sum())
        logger.info(f"PSM ATT={att:.6f} CI=[{lo:.6f}, {hi:.6f}] matched {len(mt)}/{n_treated}")
        return CausalEstimate(
            method="Propensity score matching", estimand="ATT",
            ate=att, ci_lower=float(lo), ci_upper=float(hi), std_error=se,
            n_treated=n_treated, n_control=int((t == 0).sum()), p_value=float(p),
            metadata={
                "match_rate": len(mt) / n_treated,
                "unique_controls_used": int(len(np.unique(mc))),
                "max_smd_before": max(smd_before.values()),
                "max_smd_after": max(smd_after.values()),
                "max_control_reuse": int(k.max()),
            },
        )


# ---------------------------------------------------------------------------
# Difference-in-differences (panel data only)
# ---------------------------------------------------------------------------

class DifferenceInDifferences:
    """2x2 DiD via OLS with heteroskedasticity-robust (HC3) errors.

    Estimand: ATT in the post period, under parallel trends. Two periods
    cannot test parallel trends; that needs several pre-periods. Requires
    real time variation: do not fabricate a time column for cross-sectional
    data such as Criteo.
    """

    def estimate(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        time_col: str,
        covariate_cols: Optional[list[str]] = None,
    ) -> CausalEstimate:
        _check_binary(df[treatment_col], treatment_col)
        _check_binary(df[time_col], time_col)
        cells = df.groupby([treatment_col, time_col]).size()
        if len(cells) < 4:
            raise ValueError("DiD needs treated and control units in both periods")

        d = df.copy()
        d["_post"], d["_treated"] = d[time_col], d[treatment_col]
        d["_did"] = d["_post"] * d["_treated"]
        formula = f"{outcome_col} ~ _post + _treated + _did"
        if covariate_cols:
            formula += " + " + " + ".join(covariate_cols)
        model = smf.ols(formula, data=d).fit(cov_type="HC3")
        lo, hi = model.conf_int().loc["_did"]
        pre = d[d["_post"] == 0]
        return CausalEstimate(
            method="Difference-in-differences", estimand="ATT",
            ate=float(model.params["_did"]), ci_lower=float(lo), ci_upper=float(hi),
            std_error=float(model.bse["_did"]),
            n_treated=int((d["_treated"] == 1).sum()), n_control=int((d["_treated"] == 0).sum()),
            p_value=float(model.pvalues["_did"]),
            metadata={
                "formula": formula,
                "r_squared": float(model.rsquared),
                "baseline_gap": float(pre[pre._treated == 1][outcome_col].mean()
                                      - pre[pre._treated == 0][outcome_col].mean()),
                "n_obs": int(len(d)),
            },
        )
