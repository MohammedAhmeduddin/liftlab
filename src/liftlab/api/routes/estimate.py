"""
POST /estimate: run a registered experiment's estimator on Criteo data.

Average-effect estimators (difference in means, AIPW, PSM) return an
estimate with a 95% CI and the estimand. CATE models (T, X, Causal Forest)
are fit on a training split and evaluated on a held-out split: the
response reports their held-out Qini, observed uplift in the top 25%, and
observed uplift by quartile of predicted effect. Their headline ATE is the
difference in means, which is unbiased because Criteo is randomized.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy.orm import Session

from liftlab.api.schemas import EstimateRequest, EstimateResponse, SegmentEffect
from liftlab.causal.estimators import PropensityScoreMatching, aipw, difference_in_means
from liftlab.causal.evaluation import observed_uplift_by_group
from liftlab.causal.uplift import ESTIMATORS as CATE_ESTIMATORS
from liftlab.data.loaders import load_criteo
from liftlab.db.models import EstimationRun, Experiment
from liftlab.db.session import get_db_dependency
from liftlab.mlflow_logger import log_causal_run

router = APIRouter(prefix="/estimate", tags=["estimation"])


def _num(x: Any) -> Optional[float]:
    """float, with NaN/inf -> None (JSON has no NaN)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _run_estimator(experiment: Experiment, df) -> dict:
    """Dispatch to the experiment's estimator and return a flat result dict."""
    name = experiment.estimator
    features = experiment.feature_cols
    t_col, y_col = experiment.treatment_col, experiment.outcome_col

    if name in ("difference_in_means", "aipw", "psm"):
        if name == "difference_in_means":
            r = difference_in_means(df, t_col, y_col)
        elif name == "aipw":
            r = aipw(df, features, t_col, y_col, propensity=float(df[t_col].mean()))
        else:
            r = PropensityScoreMatching(n_bootstrap=30).estimate(df, features, t_col, y_col)
        return {"estimand": r.estimand, "ate": r.ate, "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper, "std_error": r.std_error, "p_value": r.p_value,
                "n_treated": r.n_treated, "n_control": r.n_control}

    if name not in CATE_ESTIMATORS:
        raise ValueError(f"Unknown estimator: {name}")

    model = CATE_ESTIMATORS[name](seed=42)
    res, test = model.fit_evaluate(df, features, t_col, y_col)
    quartile = np.minimum((np.argsort(np.argsort(-res.test_cate)) * 4) // len(test), 3)
    segments = observed_uplift_by_group(test[y_col].to_numpy(float), test[t_col].to_numpy(int), quartile)
    dm = difference_in_means(df, t_col, y_col)
    return {
        "estimand": "ATE", "ate": dm.ate, "ci_lower": dm.ci_lower, "ci_upper": dm.ci_upper,
        "std_error": dm.std_error, "p_value": dm.p_value,
        "n_treated": dm.n_treated, "n_control": dm.n_control,
        "mean_cate": res.mean_cate,
        "qini_coefficient": res.evaluation["qini_coefficient"],
        "top_25pct_lift": res.evaluation["top_25pct"]["lift_vs_overall"],
        "feature_importance": res.feature_importance,
        "segment_effects": [
            {"group": int(s["group"]), "n": int(s["n"]), "uplift": _num(s["uplift"]),
             "ci_lower": _num(s["ci_lower"]), "ci_upper": _num(s["ci_upper"]),
             "lift_vs_overall": _num(s["lift_vs_overall"])}
            for s in segments.to_dict(orient="records")
        ],
    }


@router.post("/", response_model=EstimateResponse, status_code=status.HTTP_200_OK)
def estimate(payload: EstimateRequest, db: Session = Depends(get_db_dependency)):
    experiment = db.query(Experiment).filter(Experiment.id == payload.experiment_id).first()
    if not experiment:
        raise HTTPException(status_code=404, detail="Experiment not found")

    experiment.status = "running"
    db.flush()
    start = time.time()
    try:
        df = load_criteo(sample_frac=experiment.sample_frac)
        r = _run_estimator(experiment, df)
        duration = time.time() - start
        mlflow_run_id = log_causal_run(
            experiment_name=experiment.name, estimator=experiment.estimator,
            ate=r["ate"], ci_lower=_num(r["ci_lower"]), ci_upper=_num(r["ci_upper"]),
            n_treated=r["n_treated"], n_control=r["n_control"], p_value=_num(r["p_value"]),
            shap_importance=r.get("feature_importance"),
            extra_params={"sample_frac": experiment.sample_frac, "estimand": r["estimand"]},
            extra_metrics={"duration_seconds": duration,
                           **({"qini_coefficient": r["qini_coefficient"]}
                              if _num(r.get("qini_coefficient")) is not None else {})},
        )
        run = EstimationRun(
            id=uuid4(), experiment_id=experiment.id, mlflow_run_id=mlflow_run_id,
            ate=r["ate"], ci_lower=_num(r["ci_lower"]) or 0.0, ci_upper=_num(r["ci_upper"]) or 0.0,
            std_error=_num(r["std_error"]), p_value=_num(r["p_value"]),
            n_treated=r["n_treated"], n_control=r["n_control"],
            n_total=r["n_treated"] + r["n_control"],
            feature_importance=r.get("feature_importance"),
            segment_effects=r.get("segment_effects"),
            estimator=experiment.estimator, duration_seconds=duration,
            metadata_={"sample_frac": experiment.sample_frac, "estimand": r["estimand"],
                       "mean_cate": _num(r.get("mean_cate")),
                       "qini_coefficient": _num(r.get("qini_coefficient")),
                       "top_25pct_lift": _num(r.get("top_25pct_lift"))},
        )
        db.add(run)
        experiment.status = "done"
        db.flush()
        db.refresh(run)
    except Exception:
        experiment.status = "failed"
        db.flush()
        # Details stay in the server log; the client gets a generic message.
        logger.exception(f"Estimation failed for experiment '{experiment.name}'")
        raise HTTPException(status_code=500, detail="Estimation failed; see server logs.")

    lo, hi = _num(r["ci_lower"]), _num(r["ci_upper"])
    return EstimateResponse(
        run_id=run.id, experiment_id=experiment.id, mlflow_run_id=mlflow_run_id,
        estimator=experiment.estimator, estimand=r["estimand"],
        ate=r["ate"], ci_lower=lo, ci_upper=hi,
        ci_width=(hi - lo) if lo is not None and hi is not None else None,
        p_value=_num(r["p_value"]),
        n_treated=r["n_treated"], n_control=r["n_control"],
        n_total=r["n_treated"] + r["n_control"],
        mean_cate=_num(r.get("mean_cate")), qini_coefficient=_num(r.get("qini_coefficient")),
        top_25pct_lift=_num(r.get("top_25pct_lift")),
        feature_importance=r.get("feature_importance"),
        segment_effects=[SegmentEffect(**s) for s in r["segment_effects"]] if r.get("segment_effects") else None,
        duration_seconds=duration, created_at=run.created_at,
    )
