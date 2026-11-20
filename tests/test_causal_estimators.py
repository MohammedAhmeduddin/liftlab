"""Average-effect estimators recover KNOWN effects on simulated data."""

import numpy as np
import pandas as pd
import pytest

from liftlab.causal.estimators import (
    DifferenceInDifferences,
    PropensityScoreMatching,
    aipw,
    difference_in_means,
    standardized_mean_differences,
)
from liftlab.causal.simulate import FEATURES, make_confounded, simulate_panel, simulate_rct


@pytest.fixture(scope="module")
def rct():
    return simulate_rct(30_000, base_effect=0.02, seed=11)


@pytest.fixture(scope="module")
def truth(rct):
    return float(rct["_true_cate"].mean())


def covers(est, value):
    return est.ci_lower <= value <= est.ci_upper


class TestDifferenceInMeans:
    def test_recovers_truth_with_valid_ci(self, rct, truth):
        est = difference_in_means(rct)
        assert est.estimand == "ATE"
        assert covers(est, truth)
        assert est.p_value < 0.001

    def test_neyman_standard_error(self):
        df = pd.DataFrame({"treatment": [1, 1, 1, 0, 0, 0], "conversion": [1, 0, 1, 0, 0, 1]})
        est = difference_in_means(df)
        se = np.sqrt(np.var([1, 0, 1], ddof=1) / 3 + np.var([0, 0, 1], ddof=1) / 3)
        assert est.ate == pytest.approx(1 / 3)
        assert est.std_error == pytest.approx(se)
        assert est.metadata["relative_lift"] == pytest.approx(1.0)

    @pytest.mark.parametrize("values", [[0, 2, 1], [1, 1, 1]])
    def test_rejects_non_binary_or_single_arm_treatment(self, values):
        df = pd.DataFrame({"treatment": values, "conversion": [0, 1, 0]})
        with pytest.raises(ValueError):
            difference_in_means(df)


class TestAIPW:
    def test_recovers_truth_known_propensity(self, rct, truth):
        est = aipw(rct, FEATURES, propensity=0.85)
        assert covers(est, truth)

    def test_more_precise_than_difference_in_means(self, rct):
        """Covariate adjustment should not widen the interval in an RCT."""
        assert aipw(rct, FEATURES, propensity=0.85).std_error <= difference_in_means(rct).std_error * 1.02

    def test_corrects_confounding(self, rct, truth):
        conf = make_confounded(rct, strength=2.0, seed=3)
        naive = difference_in_means(conf).ate
        adjusted = aipw(conf, FEATURES).ate
        assert abs(adjusted - truth) < abs(naive - truth)


class TestPSM:
    def test_balances_and_recovers_att_on_confounded_data(self, rct, truth):
        conf = make_confounded(rct, strength=2.0, seed=3)
        est = PropensityScoreMatching(n_bootstrap=15).estimate(conf, FEATURES)
        assert est.estimand == "ATT"
        assert est.metadata["max_smd_before"] > 0.3
        assert est.metadata["max_smd_after"] < 0.1
        assert abs(est.ate - truth) < abs(difference_in_means(conf).ate - truth)
        assert est.ci_lower < est.ci_upper

    def test_keeps_nearly_all_treated_units(self, rct):
        """Matching with replacement: unlike greedy 1:1 without replacement,
        treated units are not dropped just because controls are scarce."""
        est = PropensityScoreMatching(n_bootstrap=2).estimate(rct, FEATURES)
        assert est.metadata["match_rate"] > 0.95

    def test_smd(self):
        a = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        b = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        assert standardized_mean_differences(a, b, ["x"]) == {"x": 0.0}


class TestDiD:
    def test_recovers_effect_where_post_only_comparison_fails(self):
        panel = simulate_panel(effect=2.0, group_gap=5.0, seed=4)
        est = DifferenceInDifferences().estimate(panel, "y", "treated", "post")
        post = panel[panel.post == 1]
        naive = post[post.treated == 1].y.mean() - post[post.treated == 0].y.mean()
        assert covers(est, 2.0)
        assert naive > 6
        assert est.metadata["baseline_gap"] == pytest.approx(5.0, abs=0.2)

    def test_requires_both_periods(self):
        panel = simulate_panel(seed=4)
        with pytest.raises(ValueError):
            DifferenceInDifferences().estimate(panel[panel.post == 1], "y", "treated", "post")

    def test_with_covariates(self):
        panel = simulate_panel(seed=5)
        panel["z"] = np.random.default_rng(0).standard_normal(len(panel))
        est = DifferenceInDifferences().estimate(panel, "y", "treated", "post", ["z"])
        assert "z" in est.metadata["formula"]
