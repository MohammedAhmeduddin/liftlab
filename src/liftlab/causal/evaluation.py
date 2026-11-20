"""
Held-out evaluation of uplift (CATE) models.

Individual effects are never observed, so a CATE model can only be judged
indirectly: rank held-out users by predicted effect, then use the
randomization to measure the *observed* effect among the top-ranked users.
If the model is right, targeting its top users beats targeting at random.

- observed_uplift_by_group: difference in means (with CI) inside groups
- qini_curve / qini_coefficient: incremental outcomes as more users are
  targeted in model order, minus what random targeting would get
- uplift_at_k: observed effect in the top k% vs the whole population
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

Z95 = stats.norm.ppf(0.975)


def _diff_in_means(y: np.ndarray, t: np.ndarray) -> tuple[float, float, int, int]:
    y1, y0 = y[t == 1], y[t == 0]
    if len(y1) < 2 or len(y0) < 2:
        return float("nan"), float("nan"), len(y1), len(y0)
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return float(est), float(se), len(y1), len(y0)


def observed_uplift_by_group(
    y: np.ndarray, t: np.ndarray, groups: np.ndarray
) -> pd.DataFrame:
    """Observed effect (treated − control mean) with 95% CI in each group."""
    rows = []
    for g in np.unique(groups):
        m = groups == g
        est, se, n1, n0 = _diff_in_means(y[m], t[m])
        rows.append({"group": g, "n": int(m.sum()), "n_treated": n1, "n_control": n0,
                     "uplift": est, "ci_lower": est - Z95 * se, "ci_upper": est + Z95 * se})
    out = pd.DataFrame(rows)
    overall, _, _, _ = _diff_in_means(y, t)
    out["lift_vs_overall"] = out["uplift"] / overall if overall else np.nan
    return out


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray, n_points: int = 100):
    """Returns (fraction targeted, incremental outcomes, random baseline).

    Incremental outcomes after targeting the top-k users (Radcliffe):
        Q(k) = Y_T(k) − Y_C(k) · N_T(k) / N_C(k)
    """
    order = np.argsort(-score, kind="stable")
    y, t = y[order].astype(float), t[order].astype(int)
    cum_t, cum_c = np.cumsum(t), np.cumsum(1 - t)
    cum_yt, cum_yc = np.cumsum(y * t), np.cumsum(y * (1 - t))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = cum_yt - np.where(cum_c > 0, cum_yc * cum_t / cum_c, 0.0)
    n = len(y)
    idx = np.unique(np.linspace(0, n - 1, n_points).astype(int))
    frac = (idx + 1) / n
    random_line = frac * q[-1]
    return frac, q[idx], random_line


def qini_coefficient(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Area between the Qini curve and random targeting, normalized by the
    total incremental outcomes. 0 = no better than random; > 0 = better."""
    frac, q, rand = qini_curve(y, t, score, n_points=200)
    total = q[-1]
    if total == 0:
        return float("nan")
    return float(np.trapezoid(q - rand, frac) / abs(total))


def uplift_at_k(y: np.ndarray, t: np.ndarray, score: np.ndarray, k: float) -> dict:
    """Observed effect among the top ``k`` fraction by score, vs everyone."""
    n_top = max(int(round(k * len(y))), 2)
    top = np.argsort(-score, kind="stable")[:n_top]
    est, se, n1, n0 = _diff_in_means(y[top], t[top])
    overall, overall_se, _, _ = _diff_in_means(y, t)
    return {
        "k": k, "n": n_top, "uplift": est,
        "ci_lower": est - Z95 * se, "ci_upper": est + Z95 * se,
        "overall_uplift": overall,
        "lift_vs_overall": est / overall if overall else float("nan"),
    }


def evaluate_scores(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> dict:
    """Standard report for one model's held-out scores."""
    deciles = pd.qcut(pd.Series(score).rank(method="first"), 10, labels=False).to_numpy()
    by_decile = observed_uplift_by_group(y, t, 9 - deciles)  # group 0 = highest predicted
    return {
        "qini_coefficient": qini_coefficient(y, t, score),
        "top_10pct": uplift_at_k(y, t, score, 0.10),
        "top_25pct": uplift_at_k(y, t, score, 0.25),
        "deciles": by_decile.to_dict(orient="records"),
    }
