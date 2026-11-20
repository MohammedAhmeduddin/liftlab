# LiftLab: causal inference on 14M real ad-campaign users

[![CI](https://github.com/MohammedAhmeduddin/liftlab/actions/workflows/ci.yml/badge.svg)](https://github.com/MohammedAhmeduddin/liftlab/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Tests](https://img.shields.io/badge/tests-99%20passed-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen.svg)

LiftLab measures how much an ad campaign **caused** users to visit and convert, and **which users** it moved. It runs on the full [Criteo Uplift v2.1](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) dataset: 13,979,592 users, 85% shown ads, 15% held out.

Every estimator is first checked against simulated data where the true effect is known, then run on Criteo. The benchmark found a problem in the data itself: the "randomized" groups are not quite balanced. That changes the headline answer, and it is the main finding below.

---

## Key results

All numbers come from `results/criteo_benchmark.json`, produced by `python scripts/benchmark_criteo.py` on all 13,979,592 rows (2.6 min on a MacBook Pro).

### 1. The randomization is slightly broken, and it matters

Criteo describes this data as a randomized trial, so the textbook answer is the treated-minus-control difference. LiftLab checks that assumption before trusting it:

| Check | Result | Under a clean randomization |
|---|---|---|
| Can a model predict who was treated? (held-out AUC) | **0.5101** | ≤ 0.5014 (max over 20 shuffled-label runs) |
| Largest feature difference between groups (\|SMD\|, f3) | **0.0488** | ~0.0008 (noise level at this size): **65× larger** |
| AIPW vs raw difference, 5 independent 2% samples (visit) | AIPW **3–5 SE lower every time** | agree within ~1 SE |

The imbalance is tiny by usual standards (SMD < 0.1). But at 14M rows it is real, and the features that differ also predict visits. So the raw difference is biased, and the reference is **cross-fitted AIPW** (doubly robust), which adjusts for the features:

| Outcome | Raw difference (biased) | **AIPW reference** (95% CI) | Relative lift |
|---|---|---|---|
| Conversion | 0.001152 (+59.4%) | **0.000997** [0.000930, 0.001064] | **+51.4%** |
| Visit | 0.010342 (+27.1%) | **0.007459** [0.007210, 0.007709] | **+19.5%** |

**The raw comparison overstates the visit effect by 39%, and the two confidence intervals do not overlap.**

### 2. Who responds: f8 splits users into responders and non-responders

`f8` has 3,743 distinct values, but one value covers 52.2% of users, so it supports two groups, not quartiles. These are effects on all rows, read directly from the AIPW scores with no extra model:

| f8 value range | Users | Conversion effect (AIPW) | vs average | Visit vs average |
|---|---|---|---|---|
| 3.635 – 3.911 | 3,539,789 (25%) | **0.003686** [0.003430, 0.003942] | **3.70×** | 3.42× |
| 3.911 – 3.972 | 10,439,803 (75%) | 0.000085 [0.000064, 0.000106] | 0.09× | 0.18× |

A quarter of users produce nearly all of the incremental conversions. Showing ads only to that group keeps most of the lift for about a quarter of the impressions.

### 3. Estimators on a 2% sample (279,592 rows) vs the all-rows reference

| Estimator | Estimand | Conversion (95% CI) | Visit (95% CI) | CI contains reference |
|---|---|---|---|---|
| Difference in means | ATE | 0.001303 [0.000818, 0.001788] | 0.008490 [0.006425, 0.010556] | ✅ ✅ |
| AIPW (cross-fitted) | ATE | 0.001008 [0.000550, 0.001467] | 0.006101 [0.004364, 0.007838] | ✅ ✅ |
| Propensity score matching | ATT | 0.001020 [0.000431, 0.001608] | 0.006734 [0.004195, 0.009273] | ✅ ✅ |

PSM matched 237,311 of 237,543 treated users (99.9%). No control was reused more than 50 times.

### 4. Confounded-sample test: can the estimators undo a known bias?

I bias the sample on purpose. Untreated users are dropped depending on the feature that best predicts the outcome, so the control share falls from 15.0% to 9.6%. Treated users are untouched, so the right answer is still the reference effect.

| Outcome | Biased through | Naive difference | AIPW (ATT) | PSM (ATT) | Target |
|---|---|---|---|---|---|
| Visit | f9 | **−0.010405** ❌ (sign flipped) | 0.005739 ✅ | 0.007125 ✅ | 0.007459 |
| Conversion | f11 | 0.001873 ❌ (≈ 2× too high) | −0.000070 ✅ (CI ±0.0012) | 0.000847 ✅ | 0.000997 |

✅ means the 95% CI contains the target. The naive comparison fails both times. For conversion, AIPW's interval contains the target only because it is very wide: with 0.3% positives and 9.6% controls, there is little signal left. PSM is the more precise estimator in that case.

### 5. Targeting models judged on held-out users

T-, X-Learner and Causal Forest (EconML) are trained on 139,796 users and scored on a separate 139,796. The scores use **observed** outcomes of the held-out users, not the models' own predictions:

| Model | Conversion Qini | Conversion top-25% lift | Visit Qini | Visit top-25% lift |
|---|---|---|---|---|
| T-Learner | 0.137 | 2.47× | 0.045 | 1.92× |
| X-Learner | 0.104 | 2.47× | 0.144 | 2.31× |
| **Causal Forest (DML)** | **0.192** | **2.89×** | **0.339** | **3.23×** |

"Top-25% lift" is the observed uplift among the 25% of held-out users the model ranks highest, divided by the overall uplift. Causal Forest is best on both outcomes. One caveat: these held-out uplifts use a within-segment treated-minus-control comparison, which carries the small imbalance from section 1.

### 6. Refutation

On conversion, all three checks pass:
- **Placebo treatment:** treatment labels shuffled 50 times; the effect disappears.
- **Random common cause:** adding a random feature leaves AIPW unchanged.
- **Subset stability:** the effect is stable across 10 random subsets.

---

## Bugs found and fixed during this audit

Each fix has a regression test.

| Problem | Effect | Fix |
|---|---|---|
| Treated Criteo as clean randomization | Headline effect overstated (visit by 39%) | Randomization check (C2ST + SMD vs its noise level); AIPW reference |
| Criteo file is stored ordered by treatment, and 52% of users share one f8 value | Rank-based quartiles split tied users by file position, giving fake segments (3.61×, 0.14×, NaN, 0.02×) | Value-based bins for features; random tie-breaking for model scores |
| Tied propensity scores in PSM | One control reused 147,395 times (reproduction); CI ±0.09 on a 0.001 effect | Random tie-breaking smaller than any score gap |
| Logistic propensity on skewed features | Left ~50% of injected bias; CIs covered the truth 27% of the time | Gradient-boosted propensity with early stopping |
| Bootstrap CI for matching with replacement | Invalid (Abadie & Imbens 2008); an estimate fell outside its own CI | Abadie–Imbens style variance with a control-reuse term; 97.5% coverage in 40 simulations |
| Difference-in-differences on Criteo with a fabricated time column | A meaningless number | DiD kept for real panel data, validated on simulation only |
| T/X-Learner CIs from bootstraps that never refit | Overconfident intervals | Models judged on held-out data instead |
| Segment lift measured in-sample | Optimistic numbers | 50/50 stratified train/test split |
| MLflow check only tested if a port was open | On macOS, AirPlay answers on port 5000; each call hung ~4 min | Probe requires HTTP 200 from `/health` |

---

## How it works

```
Criteo CSV (3 GB) ──► Parquet cache (float32 / int8), validated columns
        │
        ├─ 0. Randomization check ── classifier two-sample test, |SMD| vs noise
        ├─ 1. Reference effect ───── cross-fitted AIPW on all 14M rows
        ├─ 2. Segments ───────────── subgroup means of the AIPW scores
        ├─ 3. Sample estimators ──── difference in means, AIPW, PSM
        ├─ 4. Confounded test ────── injected bias, ATT estimators vs reference
        ├─ 5. CATE models ────────── T / X / Causal Forest, held-out Qini
        └─ 6. Refutation ─────────── placebo, random common cause, subsets
                     │
                     ▼
     results/criteo_benchmark.json ──► Streamlit dashboard (Results page)

FastAPI ── POST /experiments, POST /estimate ──► PostgreSQL (runs) + MLflow (optional)
```

### Estimators

| Estimator | Estimand | Notes |
|---|---|---|
| Difference in means | ATE | Neyman standard error |
| AIPW | ATE or ATT | Cross-fitted; gradient-boosted outcome and propensity models; influence-function SE; reports the share of clipped propensities |
| Propensity score matching | ATT | 1:1 nearest neighbour with replacement on the logit propensity; 0.2 SD caliper; random tie-breaking; balance (SMD) before and after |
| Difference-in-differences | ATT | 2×2 with HC3 errors, for panel data; validated on simulated panels |
| T-Learner, X-Learner, Causal Forest (DML) | CATE | EconML; judged on held-out data with Qini and top-k uplift |

Each estimator is tested against `liftlab.causal.simulate`, which generates randomized trials, confounded samples and panels with **known** effects.

---

## Quickstart

```bash
git clone https://github.com/MohammedAhmeduddin/liftlab.git
cd liftlab
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Download the Criteo Uplift v2.1 CSV from the [Criteo AI Lab page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) and save it as `data/criteo_uplift.csv`. The first load writes a Parquet cache next to it, so later loads take seconds.

```bash
python scripts/check_randomization.py     # balance, C2ST, DM vs AIPW on 5 samples
python scripts/benchmark_criteo.py        # full benchmark → results/criteo_benchmark.json
streamlit run src/liftlab/dashboard/app.py
```

### API (optional)

```bash
cp .env.example .env
docker compose up -d                      # PostgreSQL :5432, MLflow :5001
uvicorn liftlab.api.main:app --port 8080
```

```bash
curl -X POST http://localhost:8080/experiments/ -H "Content-Type: application/json" \
  -d '{"name": "criteo_cf", "feature_cols": ["f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11"],
       "estimator": "causalforest", "sample_frac": 0.02}'

curl -X POST http://localhost:8080/estimate/ -H "Content-Type: application/json" \
  -d '{"experiment_id": "<id returned by the first call>"}'
```

Estimators available through the API: `difference_in_means`, `aipw`, `psm`, `tlearner`, `xlearner`, `causalforest`. CATE runs return the AIPW average effect, the held-out Qini, and observed uplift per predicted-effect quartile. MLflow is optional: if it is unreachable, runs are still stored in PostgreSQL.

### Tests

```bash
pytest -q                                 # 99 tests, ~40 s, needs PostgreSQL for API/DB tests
pytest -q --deselect tests/test_api.py --deselect tests/test_session.py \
       -k "not endpoint and not unknown_experiment"    # without a database
```

CI runs the full suite against a PostgreSQL service, enforces 85% coverage (currently 94%), and lints with ruff.

---

## Project structure

```
src/liftlab/
  causal/
    estimators.py    difference in means, AIPW (ATE/ATT), PSM, DiD
    diagnostics.py   randomization check: classifier two-sample test, balance
    uplift.py        T / X / Causal Forest with held-out evaluation
    evaluation.py    Qini, uplift at k, tie-safe ranking and value bins
    refutation.py    placebo, random common cause, subset stability
    graph.py         DoWhy causal graph (randomized vs observational)
    simulate.py      data with known effects for testing
  data/loaders.py    Criteo loader, validation, Parquet cache
  api/               FastAPI app: experiments and estimation runs
  db/                SQLAlchemy models and session (psycopg 3)
  monitoring/        population drift (PSI, MMD) and alerts
  dashboard/app.py   Streamlit: run estimators, benchmark results, drift demo
  mlflow_logger.py   optional run logging with a fast health probe
scripts/
  benchmark_criteo.py      the full benchmark
  check_randomization.py   standalone randomization diagnostics
```

---

## Limitations

- **AIPW is the reference, not ground truth.** It removes bias from the measured features. If treatment also depends on something not in the data, some bias remains.
- **Criteo features are anonymized**, so the f8 segment can be found but not explained.
- **Models are fit on a 2% sample** (279,592 rows) to keep the benchmark under 3 minutes. The reference effects and f8 segments use all 13,979,592 rows.
- **The drift monitor demo uses simulated populations**, because Criteo has no time dimension.
- **Difference-in-differences is validated only on simulated panels**, for the same reason.
