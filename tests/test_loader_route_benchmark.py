"""Loader (Parquet cache, validation), /estimate route, benchmark script."""

import json
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from liftlab.api.main import app
from liftlab.api.routes import estimate as estimate_route
from liftlab.causal.simulate import simulate_rct
from liftlab.data.loaders import load_criteo
from liftlab.db.session import Base, engine

FEATS = [f"f{i}" for i in range(12)]


@pytest.fixture
def criteo_csv(tmp_path):
    df = simulate_rct(3_000, base_rate=0.1, base_effect=0.03, seed=5).drop(columns="_true_cate")
    path = tmp_path / "criteo_uplift.csv"
    df.to_csv(path, index=False)
    return path


# ── loader ────────────────────────────────────────────────────────────────────

class TestLoader:
    def test_parses_once_then_uses_parquet_cache(self, criteo_csv):
        first = load_criteo(path=criteo_csv)
        cache = criteo_csv.with_name("criteo_uplift.parquet")
        assert cache.exists()
        assert first["f0"].dtype == "float32" and first["treatment"].dtype == "int8"
        stamp = cache.stat().st_mtime
        time.sleep(0.01)
        second = load_criteo(path=criteo_csv)
        assert cache.stat().st_mtime == stamp
        pd.testing.assert_frame_equal(first, second)

    def test_sampling_is_deterministic(self, criteo_csv):
        a = load_criteo(sample_frac=0.1, seed=1, path=criteo_csv)
        b = load_criteo(sample_frac=0.1, seed=1, path=criteo_csv)
        assert len(a) == 300
        pd.testing.assert_frame_equal(a, b)

    def test_missing_file_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="criteo-uplift"):
            load_criteo(path=tmp_path / "nope.csv")

    def test_rejects_non_binary_treatment(self, criteo_csv):
        df = pd.read_csv(criteo_csv)
        df.loc[0, "treatment"] = 2
        df.to_csv(criteo_csv, index=False)
        with pytest.raises(ValueError, match="treatment"):
            load_criteo(path=criteo_csv, use_cache=False)

    def test_rejects_missing_columns(self, criteo_csv):
        pd.read_csv(criteo_csv).drop(columns="visit").to_csv(criteo_csv, index=False)
        with pytest.raises(ValueError):
            load_criteo(path=criteo_csv, use_cache=False)

    @pytest.mark.parametrize("frac", [0, 1.5])
    def test_rejects_bad_sample_frac(self, criteo_csv, frac):
        with pytest.raises(ValueError):
            load_criteo(sample_frac=frac, path=criteo_csv)


# ── /estimate route ───────────────────────────────────────────────────────────

class FakeExperiment:
    def __init__(self, estimator):
        self.estimator, self.feature_cols = estimator, FEATS
        self.treatment_col, self.outcome_col = "treatment", "conversion"


@pytest.fixture(scope="module")
def sim():
    return simulate_rct(16_000, base_rate=0.2, base_effect=0.03, responder_multiplier=6, seed=21)


@pytest.mark.parametrize("name", ["difference_in_means", "aipw", "psm"])
def test_average_effect_estimators(sim, name):
    r = estimate_route._run_estimator(FakeExperiment(name), sim)
    assert r["estimand"] == ("ATT" if name == "psm" else "ATE")
    assert r["ci_lower"] < r["ate"] < r["ci_upper"] or name == "psm"
    assert "qini_coefficient" not in r


@pytest.mark.parametrize("name", ["tlearner", "xlearner", "causalforest"])
def test_cate_estimators_report_held_out_metrics(sim, name):
    r = estimate_route._run_estimator(FakeExperiment(name), sim)
    assert r["qini_coefficient"] is not None
    assert len(r["segment_effects"]) == 4
    assert r["segment_effects"][0]["uplift"] > r["segment_effects"][3]["uplift"]
    json.dumps(r["segment_effects"])  # JSON-safe (no NaN)


def test_unknown_estimator(sim):
    with pytest.raises(ValueError):
        estimate_route._run_estimator(FakeExperiment("did"), sim)


@pytest.fixture
def db_tables():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def _create(client, name, estimator):
    resp = client.post("/experiments/", json={
        "name": name, "feature_cols": FEATS, "estimator": estimator, "sample_frac": 0.5})
    assert resp.status_code in (200, 201), resp.text
    return resp.json()["id"]


def test_estimate_endpoint_end_to_end(db_tables, sim, monkeypatch):
    monkeypatch.setattr(estimate_route, "load_criteo", lambda sample_frac: sim)
    monkeypatch.setattr(estimate_route, "log_causal_run", lambda **kw: None)
    client = TestClient(app)
    exp_id = _create(client, "e2e_causalforest", "causalforest")
    resp = client.post("/estimate/", json={"experiment_id": exp_id})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["estimand"] == "ATE" and body["qini_coefficient"] is not None
    assert len(body["segment_effects"]) == 4 and body["feature_importance"]


def test_estimate_endpoint_hides_internal_errors(db_tables, monkeypatch):
    def boom(sample_frac):
        raise RuntimeError("password=hunter2 host=db.internal")
    monkeypatch.setattr(estimate_route, "load_criteo", boom)
    client = TestClient(app)
    exp_id = _create(client, "e2e_failure", "difference_in_means")
    resp = client.post("/estimate/", json={"experiment_id": exp_id})
    assert resp.status_code == 500
    assert "hunter2" not in resp.text and "db.internal" not in resp.text


def test_estimate_unknown_experiment(db_tables):
    resp = TestClient(app).post("/estimate/", json={"experiment_id": "00000000-0000-0000-0000-000000000000"})
    assert resp.status_code == 404


def test_did_not_offered_by_api():
    from pydantic import ValidationError

    from liftlab.api.schemas import ExperimentCreate
    with pytest.raises(ValidationError):
        ExperimentCreate(name="x_did", feature_cols=FEATS, estimator="did")


# ── benchmark script ──────────────────────────────────────────────────────────

def test_benchmark_script_end_to_end(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "bench", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_criteo.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    data = tmp_path / "criteo.csv"
    simulate_rct(20_000, base_rate=0.1, base_effect=0.02, seed=9).drop(columns="_true_cate").to_csv(data, index=False)
    out = tmp_path / "bench.json"
    assert bench.main(["--data", str(data), "--out", str(out), "--model-frac", "0.5",
                       "--cf-trees", "8", "--psm-bootstrap", "3"]) == 0
    sections = json.loads(out.read_text())["sections"]
    assert set(sections) == {"data", "rct_ground_truth", "f8_groups_observed", "sample_estimators",
                             "confounded_test", "cate_models_holdout", "refutation"}
    assert sections["data"]["rows"] == 20_000
    truth = sections["rct_ground_truth"]["conversion"]
    assert truth["ci"][0] < truth["estimate"] < truth["ci"][1]
    assert set(sections["cate_models_holdout"]["visit"]) >= {"tlearner", "xlearner", "causalforest"}
