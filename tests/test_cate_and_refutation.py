"""CATE models (held-out evaluation), evaluation metrics, refutation, DAG."""

import numpy as np
import pandas as pd
import pytest

from liftlab.causal.estimators import aipw, difference_in_means
from liftlab.causal.evaluation import (
    evaluate_scores,
    observed_uplift_by_group,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
)
from liftlab.causal.graph import backdoor_set, build_causal_model, build_criteo_causal_graph
from liftlab.causal.refutation import placebo_treatment, random_common_cause, subset_stability
from liftlab.causal.simulate import FEATURES, simulate_rct
from liftlab.causal.uplift import ESTIMATORS, CausalForestUplift


@pytest.fixture(scope="module")
def strong():
    """Strong, rare-free heterogeneity so tests are not flaky."""
    return simulate_rct(16_000, base_rate=0.2, base_effect=0.03, responder_multiplier=6, seed=21)


# ── evaluation metrics ────────────────────────────────────────────────────────

class TestEvaluation:
    def test_perfect_score_beats_random(self, strong):
        y, t = strong.conversion.to_numpy(float), strong.treatment.to_numpy(int)
        oracle = qini_coefficient(y, t, strong._true_cate.to_numpy())
        noise = qini_coefficient(y, t, np.random.default_rng(0).random(len(y)))
        assert oracle > 0.1
        assert abs(noise) < oracle

    def test_qini_curve_endpoints(self, strong):
        y, t = strong.conversion.to_numpy(float), strong.treatment.to_numpy(int)
        frac, q, rand = qini_curve(y, t, strong._true_cate.to_numpy())
        assert frac[-1] == pytest.approx(1.0)
        assert q[-1] == pytest.approx(rand[-1])

    def test_uplift_at_k_and_groups(self, strong):
        y, t = strong.conversion.to_numpy(float), strong.treatment.to_numpy(int)
        top = uplift_at_k(y, t, strong._true_cate.to_numpy(), 0.25)
        assert top["lift_vs_overall"] > 1.5
        groups = observed_uplift_by_group(y, t, (strong.f8 < strong.f8.quantile(0.25)).astype(int).to_numpy())
        assert groups.set_index("group").loc[1, "uplift"] > groups.set_index("group").loc[0, "uplift"]

    def test_evaluate_scores_report(self, strong):
        y, t = strong.conversion.to_numpy(float), strong.treatment.to_numpy(int)
        rep = evaluate_scores(y, t, strong._true_cate.to_numpy())
        assert len(rep["deciles"]) == 10
        assert {"qini_coefficient", "top_10pct", "top_25pct"} <= set(rep)

    def test_zero_total_uplift_gives_nan(self):
        y = np.zeros(10)
        t = np.array([0, 1] * 5)
        assert np.isnan(qini_coefficient(y, t, np.arange(10.0)))


# ── CATE models ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_cate_models_evaluated_on_held_out_split(strong, name):
    model = ESTIMATORS[name](n_estimators=40) if name == "causalforest" else ESTIMATORS[name]()
    res, test = model.fit_evaluate(strong, FEATURES)
    assert res.n_train + res.n_test == len(strong)
    assert len(res.test_cate) == len(test) == res.n_test
    assert set(test.index).isdisjoint(set(strong.index.difference(test.index)))
    # strong signal: every model should rank responders above random on held-out data
    assert res.evaluation["qini_coefficient"] > 0
    assert np.corrcoef(res.test_cate, test._true_cate)[0, 1] > 0.2


def test_causal_forest_finds_the_driver(strong):
    res, _ = CausalForestUplift(n_estimators=40).fit_evaluate(strong, FEATURES)
    assert next(iter(res.feature_importance)) == "f8"
    assert sum(res.feature_importance.values()) == pytest.approx(1.0, abs=0.01)


def test_observed_segments(strong):
    res, test = CausalForestUplift(n_estimators=12).fit_evaluate(strong, FEATURES)
    seg = res.observed_segments(test, "f8")
    assert len(seg) == 4
    assert seg.loc[0, "uplift"] > seg.loc[3, "uplift"]  # lowest f8 quartile responds most


# ── refutation ────────────────────────────────────────────────────────────────

class TestRefutation:
    def test_placebo_detects_real_effect(self, strong):
        r = placebo_treatment(lambda d: difference_in_means(d).ate, strong, n_permutations=30)
        assert r["passed"] and abs(r["placebo_mean"]) < abs(r["actual"]) / 5

    def test_placebo_fails_without_effect(self):
        df = simulate_rct(8_000, base_effect=0.0, responder_multiplier=1, seed=3)
        r = placebo_treatment(lambda d: difference_in_means(d).ate, df, n_permutations=30)
        assert not r["passed"]

    def test_random_common_cause(self, strong):
        fn = lambda d: aipw(d, FEATURES + (["_random_cause"] if "_random_cause" in d else []),  # noqa: E731
                            propensity=0.85).ate
        assert random_common_cause(fn, strong)["passed"]

    def test_subset_stability(self, strong):
        r = subset_stability(lambda d: difference_in_means(d).ate, strong, n_subsets=5)
        assert r["passed"] and r["share_same_sign"] == 1.0


# ── causal graph ──────────────────────────────────────────────────────────────

class TestGraph:
    def test_randomized_design_needs_no_adjustment(self, strong):
        assert backdoor_set(build_causal_model(strong.head(500))) == []

    def test_observational_design_adjusts_for_all_features(self, strong):
        assert backdoor_set(build_causal_model(strong.head(500), randomized=False)) == sorted(FEATURES)

    def test_no_edges_into_treatment_when_randomized(self):
        gml = build_criteo_causal_graph(randomized=True)
        assert 'target "treatment"' not in gml
        assert 'target "treatment"' in build_criteo_causal_graph(randomized=False)


def test_simulator_shapes():
    df = simulate_rct(1_000, seed=1)
    assert set(FEATURES) <= set(df.columns)
    assert df.treatment.between(0, 1).all() and 0.8 < df.treatment.mean() < 0.9
    assert isinstance(df, pd.DataFrame)
