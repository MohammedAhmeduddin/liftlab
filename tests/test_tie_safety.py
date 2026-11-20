"""Regression tests for a bug found on the real Criteo file.

Criteo is stored ordered by treatment (the first quarter is 100% treated)
and 52% of users share one f8 value. Ranking with ties broken by row order
then sends treated and control users to different groups, inventing
"segments" and NaN effects where there is nothing. These tests rebuild
that situation with NO real effect and check that nothing is found.
"""

import numpy as np
import pandas as pd

from liftlab.causal.evaluation import (
    evaluate_scores,
    observed_uplift_by_group,
    qini_coefficient,
    rank_order,
    score_groups,
    uplift_at_k,
    value_bins,
)


def criteo_like(n=40_000, seed=0, effect=0.0):
    """Treatment-sorted rows, a feature with 50% ties, a uniform effect."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.binomial(1, 0.85, n))[::-1]          # all treated first, like the file
    f = np.where(rng.random(n) < 0.5, 3.97, rng.normal(3.5, 0.2, n))
    y = rng.binomial(1, 0.05 + effect * t)                # same effect for everyone
    return pd.DataFrame({"treatment": t, "f8": f, "y": y})


def test_rank_based_quartiles_on_sorted_ties_are_broken():
    """Documents the original bug: a quartile with no control users."""
    df = criteo_like()
    rank_q = (np.ceil(df.f8.rank(method="first") / len(df) * 4) - 1).astype(int)
    shares = df.groupby(rank_q).treatment.mean()
    assert shares.max() == 1.0          # one quartile is all treated -> NaN effect


def test_value_bins_keep_ties_together_and_stay_balanced():
    df = criteo_like()
    bins, edges = value_bins(df.f8.to_numpy(), 4)
    tied_bins = set(bins[df.f8.to_numpy() == 3.97])
    assert len(tied_bins) == 1                       # every tied user in one bin
    shares = df.groupby(bins).treatment.mean()
    assert shares.between(0.80, 0.90).all()          # randomization preserved in each bin
    seg = observed_uplift_by_group(df.y.to_numpy(float), df.treatment.to_numpy(), bins)
    assert seg.uplift.notna().all()                  # no group without controls
    assert edges == sorted(edges)


def test_tied_scores_do_not_follow_row_order():
    df = criteo_like()
    flat = np.zeros(len(df))                         # every user tied
    top = rank_order(flat)[: len(df) // 4]
    assert 0.80 < df.treatment.to_numpy()[top].mean() < 0.90
    groups = score_groups(flat, 4)
    assert pd.Series(df.treatment.to_numpy()).groupby(groups).mean().between(0.80, 0.90).all()


def test_tied_scores_give_random_targeting_not_file_order():
    """Uniform true effect, every score tied: targeting is random, so the Qini
    coefficient is ~0 and the top 25% shows the overall effect. Breaking ties
    by row order would put only treated users at the top (curve far off)."""
    df = criteo_like(n=200_000, effect=0.02, seed=1)
    y, t = df.y.to_numpy(float), df.treatment.to_numpy()
    flat = np.zeros(len(df))
    assert abs(qini_coefficient(y, t, flat)) < 0.1
    top = uplift_at_k(y, t, flat, 0.25)
    assert top["ci_lower"] < top["overall_uplift"] < top["ci_upper"]
    rep = evaluate_scores(y, t, flat)
    assert all(np.isfinite(d["uplift"]) for d in rep["deciles"])


def test_random_tie_breaking_is_reproducible():
    s = np.array([1.0, 1.0, 0.5, 0.5, 0.5])
    assert (rank_order(s, seed=3) == rank_order(s, seed=3)).all()
    assert set(rank_order(s)[:2]) == {0, 1}          # higher score always first
