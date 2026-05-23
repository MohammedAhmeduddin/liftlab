# src/liftlab/causal/estimators.py
"""
PSM (Propensity Score Matching) and DiD (Difference-in-Differences) estimators.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.formula.api as smf
from typing import Optional


@dataclass
class CausalEstimate:
    """Standardized output for all causal estimators."""
    method: str
    ate: float                    # Average Treatment Effect
    ci_lower: float               # 95% CI lower bound
    ci_upper: float               # 95% CI upper bound
    std_error: float
    n_treated: int
    n_control: int
    p_value: float
    metadata: dict                # Method-specific diagnostics


# ---------------------------------------------------------------------------
# Propensity Score Matching (PSM)
# ---------------------------------------------------------------------------

class PropensityScoreMatching:
    """
    PSM estimator for observational data.

    Steps:
    1. Fit logistic regression to estimate P(treatment=1 | features)
    2. Match each treated unit to nearest untreated unit by propensity score
    3. Estimate ATE on matched sample
    4. Bootstrap for confidence intervals
    """

    def __init__(self, caliper: float = 0.05, n_bootstrap: int = 200, seed: int = 42):
        self.caliper = caliper          # Max allowed propensity score distance
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.propensity_model = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=seed
        )
        self.scaler = StandardScaler()
        self.propensity_scores_: Optional[np.ndarray] = None

    def _estimate_propensity(
        self, df: pd.DataFrame, feature_cols: list[str], treatment_col: str
    ) -> np.ndarray:
        """Fit logistic regression and return propensity scores P(T=1|X)."""
        X = self.scaler.fit_transform(df[feature_cols])
        y = df[treatment_col].values
        self.propensity_model.fit(X, y)
        scores = self.propensity_model.predict_proba(X)[:, 1]
        logger.info(
            f"Propensity scores | mean={scores.mean():.3f} | "
            f"std={scores.std():.3f} | min={scores.min():.3f} | max={scores.max():.3f}"
        )
        return scores

    def _match(
        self,
        df: pd.DataFrame,
        propensity_scores: np.ndarray,
        treatment_col: str,
    ) -> pd.DataFrame:
        """
        1:1 nearest-neighbor matching with caliper.
        Each treated unit is matched to the closest control unit
        within caliper distance. Unmatched treated units are dropped.
        """
        df = df.copy()
        df["_pscore"] = propensity_scores
        df["_idx"] = np.arange(len(df))

        treated = df[df[treatment_col] == 1].copy()
        control = df[df[treatment_col] == 0].copy()

        matched_pairs = []
        used_control_indices = set()

        for _, t_row in treated.iterrows():
            # Find nearest unused control within caliper
            available = control[~control["_idx"].isin(used_control_indices)]
            if available.empty:
                break
            dists = np.abs(available["_pscore"] - t_row["_pscore"])
            nearest_idx = dists.idxmin()
            nearest_dist = dists.min()

            if nearest_dist <= self.caliper:
                matched_pairs.append(t_row)
                matched_pairs.append(control.loc[nearest_idx])
                used_control_indices.add(control.loc[nearest_idx, "_idx"])

        matched_df = pd.DataFrame(matched_pairs).drop(columns=["_pscore", "_idx"])
        n_matched = len(matched_pairs) // 2
        logger.info(
            f"PSM matched {n_matched:,} pairs "
            f"({n_matched / len(treated) * 100:.1f}% of treated units matched)"
        )
        return matched_df

    def estimate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
    ) -> CausalEstimate:
        """Run full PSM pipeline and return CausalEstimate."""
        logger.info("Running Propensity Score Matching...")

        # Step 1: Propensity scores
        scores = self._estimate_propensity(df, feature_cols, treatment_col)
        self.propensity_scores_ = scores

        # Step 2: Match
        matched = self._match(df, scores, treatment_col)

        # Step 3: ATE on matched sample
        treated_outcomes = matched[matched[treatment_col] == 1][outcome_col].values
        control_outcomes = matched[matched[treatment_col] == 0][outcome_col].values
        ate = treated_outcomes.mean() - control_outcomes.mean()

        # Step 4: Bootstrap CI
        rng = np.random.default_rng(self.seed)
        boot_ates = []
        for _ in range(self.n_bootstrap):
            t_boot = rng.choice(treated_outcomes, size=len(treated_outcomes), replace=True)
            c_boot = rng.choice(control_outcomes, size=len(control_outcomes), replace=True)
            boot_ates.append(t_boot.mean() - c_boot.mean())

        ci_lower = np.percentile(boot_ates, 2.5)
        ci_upper = np.percentile(boot_ates, 97.5)
        std_error = np.std(boot_ates)

        # Step 5: t-test p-value
        _, p_value = stats.ttest_ind(treated_outcomes, control_outcomes)

        result = CausalEstimate(
            method="PSM",
            ate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_error=std_error,
            n_treated=len(treated_outcomes),
            n_control=len(control_outcomes),
            p_value=p_value,
            metadata={
                "caliper": self.caliper,
                "n_bootstrap": self.n_bootstrap,
                "propensity_score_mean": float(scores.mean()),
                "match_rate": len(treated_outcomes) / (df[treatment_col] == 1).sum(),
            },
        )

        logger.info(
            f"PSM ATE={ate:.6f} | 95% CI=[{ci_lower:.6f}, {ci_upper:.6f}] | p={p_value:.4f}"
        )
        return result


# ---------------------------------------------------------------------------
# Difference-in-Differences (DiD)
# ---------------------------------------------------------------------------

class DifferenceInDifferences:
    """
    DiD estimator for time-based promotion rollouts.

    Assumes:
    - You have pre-period and post-period data
    - Treatment group received promotion in post-period
    - Control group never received it
    - PARALLEL TRENDS: absent the treatment, both groups would have
      trended identically (we test this)

    Model: outcome ~ post + treated + post*treated + controls
    The coefficient on post*treated is the DiD estimate (ATT).
    """

    def estimate(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        time_col: str,
        covariate_cols: Optional[list[str]] = None,
    ) -> CausalEstimate:
        """
        Run DiD via OLS regression.

        Args:
            df: Must contain outcome, treatment group indicator,
                and time period indicator (0=pre, 1=post)
            outcome_col: Outcome variable
            treatment_col: Group indicator (1=treatment group, 0=control)
            time_col: Period indicator (1=post period, 0=pre period)
            covariate_cols: Optional covariates for precision
        """
        logger.info("Running Difference-in-Differences...")

        df = df.copy()
        df["_post"] = df[time_col]
        df["_treated"] = df[treatment_col]
        df["_did"] = df["_post"] * df["_treated"]  # Interaction term

        # Build regression formula
        formula = f"{outcome_col} ~ _post + _treated + _did"
        if covariate_cols:
            formula += " + " + " + ".join(covariate_cols)

        model = smf.ols(formula=formula, data=df).fit(cov_type="HC3")  # Robust SEs

        did_coef = model.params["_did"]
        did_se = model.bse["_did"]
        did_pvalue = model.pvalues["_did"]
        ci = model.conf_int().loc["_did"]

        # Parallel trends test — compare pre-period trends (informal check)
        pre_data = df[df["_post"] == 0]
        treated_pre_mean = pre_data[pre_data["_treated"] == 1][outcome_col].mean()
        control_pre_mean = pre_data[pre_data["_treated"] == 0][outcome_col].mean()
        pre_period_diff = treated_pre_mean - control_pre_mean

        logger.info(f"Pre-period difference: {pre_period_diff:.6f} "
                    f"(should be small for parallel trends)")
        logger.info(
            f"DiD ATE={did_coef:.6f} | 95% CI=[{ci[0]:.6f}, {ci[1]:.6f}] | "
            f"p={did_pvalue:.4f} | R²={model.rsquared:.4f}"
        )

        n_treated = int((df["_treated"] == 1).sum())
        n_control = int((df["_treated"] == 0).sum())

        return CausalEstimate(
            method="DiD",
            ate=did_coef,
            ci_lower=ci[0],
            ci_upper=ci[1],
            std_error=did_se,
            n_treated=n_treated,
            n_control=n_control,
            p_value=did_pvalue,
            metadata={
                "r_squared": model.rsquared,
                "formula": formula,
                "pre_period_diff": pre_period_diff,
                "parallel_trends_warning": abs(pre_period_diff) > 0.01,
                "n_obs": len(df),
            },
        )
