"""
Uplift (CATE) models: who responds to treatment?

- T-Learner:   separate outcome models per arm; CATE = difference.
- X-Learner:   cross-imputed effects, weighted by propensity; better than
               T when one arm is much smaller (Criteo: 85/15).
- CausalForestDML: orthogonalized (double ML) honest forest.

Each model is FIT ON A TRAINING SPLIT and judged on a HELD-OUT split with
the Qini coefficient and observed uplift in the top-ranked users
(liftlab.causal.evaluation). In-sample CATE averages are not evidence:
the model has already seen those outcomes.

ATE uncertainty is not taken from these models; use
liftlab.causal.estimators (difference in means / AIPW) for that.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.metalearners import TLearner, XLearner
from loguru import logger
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from liftlab.causal.evaluation import evaluate_scores, observed_uplift_by_group


def _hgb(seed: int, max_iter: int = 100) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=max_iter, max_depth=4, random_state=seed)


@dataclass
class CATEResult:
    """Held-out evaluation of one CATE model."""
    method: str
    mean_cate: float                       # average predicted effect on the test split
    test_cate: np.ndarray                  # predictions for the test split
    evaluation: dict                       # Qini, top-k uplift, deciles (held-out)
    n_train: int
    n_test: int
    feature_importance: Optional[dict[str, float]] = None
    metadata: dict = field(default_factory=dict)

    def observed_segments(self, test_df: pd.DataFrame, col: str, n_bins: int = 4,
                          treatment_col: str = "treatment",
                          outcome_col: str = "conversion") -> pd.DataFrame:
        """Observed (not predicted) uplift by quantile of ``col`` on held-out data."""
        bins = pd.qcut(test_df[col].rank(method="first"), n_bins, labels=False).to_numpy()
        return observed_uplift_by_group(test_df[outcome_col].to_numpy(float),
                                        test_df[treatment_col].to_numpy(int), bins)


class _BaseUplift:
    method = "base"

    def __init__(self, seed: int = 42, test_size: float = 0.5):
        self.seed = seed
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.model = None

    def _build(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def _importance(self, feature_cols: list[str]) -> Optional[dict[str, float]]:
        return None

    def split(self, df: pd.DataFrame, treatment_col: str):
        return train_test_split(df, test_size=self.test_size, random_state=self.seed,
                                stratify=df[treatment_col])

    def fit_evaluate(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        treatment_col: str = "treatment",
        outcome_col: str = "conversion",
    ) -> tuple[CATEResult, pd.DataFrame]:
        """Fit on a stratified train split, evaluate on the held-out split.
        Returns (result, test_df) so callers can inspect segments."""
        train, test = self.split(df, treatment_col)
        Xtr = self.scaler.fit_transform(train[feature_cols].to_numpy(float))
        Xte = self.scaler.transform(test[feature_cols].to_numpy(float))
        ttr = train[treatment_col].to_numpy(int)
        ytr = train[outcome_col].to_numpy(float)
        self.model = self._build()
        logger.info(f"Fitting {self.method} on {len(train):,} rows, evaluating on {len(test):,}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(ytr, ttr, X=Xtr)
            cate = np.asarray(self.model.effect(Xte)).ravel()
        evaluation = evaluate_scores(test[outcome_col].to_numpy(float),
                                     test[treatment_col].to_numpy(int), cate)
        result = CATEResult(
            method=self.method, mean_cate=float(cate.mean()), test_cate=cate,
            evaluation=evaluation, n_train=len(train), n_test=len(test),
            feature_importance=self._importance(feature_cols),
        )
        logger.info(f"{self.method}: Qini={evaluation['qini_coefficient']:.3f} "
                    f"top-25% lift={evaluation['top_25pct']['lift_vs_overall']:.2f}x")
        return result, test


class TLearnerUplift(_BaseUplift):
    method = "T-Learner"

    def _build(self):
        return TLearner(models=_hgb(self.seed))


class XLearnerUplift(_BaseUplift):
    method = "X-Learner"

    def _build(self):
        return XLearner(models=_hgb(self.seed),
                        propensity_model=LogisticRegression(max_iter=1000))


class CausalForestUplift(_BaseUplift):
    method = "Causal Forest (DML)"

    def __init__(self, n_estimators: int = 100, seed: int = 42, test_size: float = 0.5):
        super().__init__(seed, test_size)
        self.n_estimators = n_estimators

    def _build(self):
        return CausalForestDML(
            n_estimators=self.n_estimators, min_samples_leaf=50, max_depth=6,
            discrete_treatment=True, cv=2, random_state=self.seed,
            model_y=_hgb(self.seed, 50),
            model_t=HistGradientBoostingClassifier(max_iter=50, max_depth=3, random_state=self.seed),
        )

    def _importance(self, feature_cols: list[str]) -> Optional[dict[str, float]]:
        """Split-based importance of features for effect heterogeneity."""
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            return None
        imp = np.asarray(imp, dtype=float)
        if imp.sum() > 0:
            imp = imp / imp.sum()
        return dict(sorted(zip(feature_cols, imp.round(4).tolist()), key=lambda kv: -kv[1]))


ESTIMATORS = {
    "tlearner": TLearnerUplift,
    "xlearner": XLearnerUplift,
    "causalforest": CausalForestUplift,
}
