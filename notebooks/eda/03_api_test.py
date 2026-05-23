# notebooks/eda/03_api_test.py
"""
API smoke test — creates an experiment, runs estimation, checks results.
"""
import httpx
import json

BASE = "http://localhost:8080"

print("=" * 60)
print("Milestone 4 — API Smoke Test")
print("=" * 60)

# 1. Health check
print("\n[1] Health check...")
r = httpx.get(f"{BASE}/health", timeout=10.0)
print(f"    Status: {r.status_code}")
print(f"    Body  : {json.dumps(r.json(), indent=4)}")
assert r.status_code == 200, f"Health check failed: {r.text}"

# 2. Create experiment — or reuse if already exists
print("\n[2] Creating experiment...")
payload = {
    "name": "criteo_causalforest_v1",
    "description": "CausalForest on Criteo 2% sample",
    "treatment_col": "treatment",
    "outcome_col": "conversion",
    "feature_cols": [f"f{i}" for i in range(12)],
    "estimator": "causalforest",
    "sample_frac": 0.02,
}
r = httpx.post(f"{BASE}/experiments/", json=payload)

if r.status_code == 409:
    print("    Already exists — fetching existing experiment...")
    r2 = httpx.get(f"{BASE}/experiments/")
    experiments = r2.json()
    exp = next(e for e in experiments if e["name"] == payload["name"])
else:
    assert r.status_code == 201, f"Create experiment failed: {r.text}"
    exp = r.json()

exp_id = exp["id"]
print(f"    Experiment ID : {exp_id}")
print(f"    Estimator     : {exp['estimator']}")
print(f"    Sample frac   : {exp['sample_frac']}")

# 3. List experiments
print("\n[3] Listing experiments...")
r = httpx.get(f"{BASE}/experiments/")
assert r.status_code == 200
print(f"    Found {len(r.json())} experiment(s)")

# 4. Run estimation
print(f"\n[4] Running causal estimation (this takes ~2-3 min)...")
r = httpx.post(
    f"{BASE}/estimate/",
    json={"experiment_id": exp_id},
    timeout=300.0,
)
print(f"    Status: {r.status_code}")
assert r.status_code == 200, f"Estimation failed: {r.text}"
result = r.json()

print(f"\n    ── Causal Results ──────────────────────")
print(f"    ATE             : {result['ate']:.6f}")
print(f"    95% CI          : [{result['ci_lower']:.6f}, {result['ci_upper']:.6f}]")
print(f"    CI Width        : {result['ci_width']:.6f}")
print(f"    N treated       : {result['n_treated']:,}")
print(f"    N control       : {result['n_control']:,}")
print(f"    N total         : {result['n_total']:,}")
print(f"    MLflow run ID   : {result['mlflow_run_id']}")
print(f"    Duration        : {result['duration_seconds']:.1f}s")

if result.get("shap_importance"):
    top3 = sorted(result["shap_importance"].items(), key=lambda x: -x[1])[:3]
    print(f"\n    ── Top 3 SHAP Features ─────────────────")
    for feat, imp in top3:
        print(f"    {feat:<6} : {imp:.6f}")

if result.get("segment_effects"):
    print(f"\n    ── Segment Effects ─────────────────────")
    for seg in result["segment_effects"]:
        print(f"    Bin {seg['bin']}: CATE={seg['mean_cate']:.6f} | "
              f"n={seg['n']:,} | lift={seg['lift_vs_average']:.2f}x")

print("\n" + "=" * 60)
print("Milestone 4 complete ✅")
print("=" * 60)
