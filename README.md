# 🧪 LiftLab — Production Causal Inference Engine

> **Distinguish genuine causal lift from correlation, seasonal effects, and selection bias — at production scale.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![EconML](https://img.shields.io/badge/EconML-0.15-orange.svg)](https://econml.azurewebsites.net)
[![Tests](https://img.shields.io/badge/tests-53%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-83%25-brightgreen.svg)](htmlcov/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The $2M Problem

E-commerce growth teams run 50+ promotions per quarter. Standard A/B tools return p-values — but when randomization is impossible, observational data is confounded, or populations shift between experiments, those p-values are measuring **correlation, not causation**.

The result: teams scale promotions that show positive correlation but **zero true causal lift**, misallocating millions in promotion spend.

LiftLab solves this with a production-grade causal inference engine that combines rigorous identification strategies, heterogeneous treatment effect estimation, and automated drift detection.

---

## Live Demo

| Service                    | URL                                               | Description                      |
| -------------------------- | ------------------------------------------------- | -------------------------------- |
| 🎨 **Streamlit Dashboard** | [localhost:8501](http://localhost:8501)           | Growth team experiment hub       |
| 🚀 **FastAPI Docs**        | [localhost:8080/docs](http://localhost:8080/docs) | Interactive API documentation    |
| 📊 **MLflow UI**           | [localhost:5001](http://localhost:5001)           | Experiment tracking & versioning |

---

## How It Works

Criteo (14M rows) + Olist (100K orders) ↓ Feature Engineering (dbt SQL) ↓ DoWhy Causal Graph → Identification Strategy ↓ ┌─────────────────────────────────────┐ │ Estimator Selection │ │ ├── PSM (observational data) │ │ ├── DiD (time-based rollouts) │ │ └── CausalForest DML (CATE) │ └─────────────────────────────────────┘ ↓ SHAP → Treatment Effect Heterogeneity ↓ MLflow → ATE, CATE, CI, refutation p-values ↓ Drift Monitor (PSI + MMD) → Retraining trigger ↓ Streamlit Growth Team Dashboard

---

## Supported Estimators

| Estimator        | Method                     | Best For                            | Implementation                                      |
| ---------------- | -------------------------- | ----------------------------------- | --------------------------------------------------- |
| **PSM**          | Propensity Score Matching  | Observational data, cross-sectional | `liftlab.causal.estimators.PropensityScoreMatching` |
| **DiD**          | Difference-in-Differences  | Time-based rollouts, pre/post data  | `liftlab.causal.estimators.DifferenceInDifferences` |
| **T-Learner**    | Meta-learner baseline      | Balanced treatment groups           | `liftlab.causal.uplift.TLearnerUplift`              |
| **X-Learner**    | Cross-fitting meta-learner | Imbalanced treatment (85/15 split)  | `liftlab.causal.uplift.XLearnerUplift`              |
| **CausalForest** | Double ML + forest         | Heterogeneous effects, main model   | `liftlab.causal.uplift.CausalForestUplift`          |

---

## Sample Results

All estimators run on the [Criteo Uplift Dataset](https://ailab.criteo.com/ressources/criteo-uplift-prediction-dataset/) (14M rows, treatment/control labels):

### Estimator Convergence

| Estimator    | ATE          | 95% CI Lower | 95% CI Upper | Duration |
| ------------ | ------------ | ------------ | ------------ | -------- |
| PSM          | **0.001189** | 0.000640     | 0.001772     | 18s      |
| DiD          | **0.001131** | 0.000424     | 0.001839     | 2s       |
| T-Learner    | **0.001243** | 0.001119     | 0.001362     | 19s      |
| X-Learner    | **0.001242** | 0.001144     | 0.001346     | 28s      |
| CausalForest | **0.000962** | -0.006259    | 0.008182     | 52s      |

> ✅ All five independent estimators converge on **ATE ≈ +0.001** — a strong robustness signal across fundamentally different identification strategies.

### Heterogeneous Treatment Effects (SHAP)

**Top 3 features driving treatment effect heterogeneity:**

| Feature | SHAP Importance | Interpretation                                   |
| ------- | --------------- | ------------------------------------------------ |
| `f8`    | 0.001788        | Strongest predictor of who responds to treatment |
| `f3`    | 0.001249        | Second-order heterogeneity driver                |
| `f2`    | 0.000886        | Third-order effect modifier                      |

### Customer Segmentation Results

| Segment                      | % of Users | CATE         | Lift vs Average |
| ---------------------------- | ---------- | ------------ | --------------- |
| **Low `f8`** (high-response) | 25%        | **0.003395** | **3.53x** 🎯    |
| **High `f8`** (low-response) | 75%        | 0.000129     | 0.13x           |

> 💡 **Business Impact:** Targeting only the high-response segment delivers 3.5x the average lift at 25% of the cost. This is the ROI of CATE over ATE.

---

## Tech Stack

| Layer                 | Tool              | Role                                          |
| --------------------- | ----------------- | --------------------------------------------- |
| **Causal Inference**  | EconML            | T-learner, X-learner, CausalForestDML         |
|                       | DoWhy             | Causal graph specification + identification   |
|                       | SHAP              | Treatment effect heterogeneity interpretation |
| **API & Data**        | FastAPI           | Experimentation platform REST API             |
|                       | PostgreSQL        | Experiment configs and results store          |
|                       | SQLAlchemy        | ORM + Alembic migrations                      |
| **ML Infrastructure** | MLflow            | Experiment tracking + model versioning        |
|                       | dbt               | SQL feature engineering pipelines             |
| **Monitoring**        | Evidently         | Drift detection (PSI, MMD)                    |
| **Dashboards**        | Streamlit         | Growth team experimentation UI                |
| **DevOps**            | Docker            | Containerized services                        |
|                       | GitHub Actions    | CI/CD + drift-triggered retraining            |
| **Base ML**           | Scikit-learn      | Base learners for meta-learners               |
|                       | SciPy/Statsmodels | DiD, power analysis, parallel trends test     |

---

## Quickstart

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- 4GB RAM minimum (EconML is memory-intensive)

### 1. Clone and install

git clone https://github.com/YOUR_USERNAME/liftlab.git
cd liftlab
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

## 2. Configure environment

cp .env.example .env

### Edit .env with your settings (optional — defaults work out of the box)

## 3. Start infrastructure

docker compose up -d

### Starts PostgreSQL (port 5432) and MLflow (port 5001)

### 4. Download datasets

Criteo Uplift Dataset (required)

Download from ailab.criteo.com
Place criteo-uplift-v2.1.csv.gz in data/
Decompress: gunzip data/criteo-uplift-v2.1.csv.gz && mv data/criteo-uplift-v2.1.csv data/criteo_uplift.csv
Olist E-Commerce Dataset (optional — for dbt feature engineering demo)

Download from Kaggle
Unzip to data/olist/

### 5. Start the API

uvicorn liftlab.api.main:app --reload --host 0.0.0.0 --port 8080
API docs available at http://localhost:8080/docs

### 6. Launch the dashboard

streamlit run src/liftlab/dashboard/app.py
Dashboard opens at http://localhost:8501

### 7. Run your first experiment

python notebooks/eda/03_api_test.py

### Expected output:

[1] Health check... ✅
[2] Creating experiment... ✅ Experiment ID: dc041bbb-...
[4] Running causal estimation (~2 min)... ✅
ATE : 0.000962
95% CI : [-0.006259, 0.008182]
Duration : 51.6s
── Top 3 SHAP Features ─────────────────
f8 : 0.000906
...

## Architecture

### System Flow

┌─────────────────────────────────────────────────────────┐
│ Data Layer │
│ ├─ Criteo CSV (14M rows) → load_criteo() │
│ └─ Olist CSVs (100K orders) → load_olist() │
└─────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│ Feature Engineering (dbt) │
│ ├─ models/features/behavioral_features.sql │
│ ├─ models/features/rfm_scores.sql │
│ └─ models/experiments/experiment_populations.sql │
└─────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│ Causal Inference Engine │
│ ├─ DoWhy: Causal graph → identification │
│ ├─ PSM: Propensity score matching │
│ ├─ DiD: Difference-in-differences │
│ └─ EconML: T/X-Learner, CausalForest │
└─────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│ Interpretability Layer │
│ └─ SHAP: Feature importance for treatment effects │
└─────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│ Persistence & Monitoring │
│ ├─ PostgreSQL: Experiment configs + results │
│ ├─ MLflow: ATE, CATE, CIs, refutation p-values │
│ └─ Drift Monitor: PSI + MMD → retraining trigger │
└─────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────┐
│ User Interfaces │
│ ├─ FastAPI: POST /estimate, GET /experiments │
│ └─ Streamlit: Experiment comparison + segment viz │
└─────────────────────────────────────────────────────────┘

## Project Structure

liftlab/
├── src/liftlab/
│ ├── causal/
│ │ ├── graph.py # DoWhy causal DAG
│ │ ├── estimators.py # PSM, DiD
│ │ ├── uplift.py # T/X-Learner, CausalForest
│ │ └── refutation.py # DoWhy refutation tests
│ ├── features/
│ │ └── engineering.py # RFM, behavioral features
│ ├── monitoring/
│ │ ├── drift.py # PSI, MMD drift detection
│ │ └── alerts.py # Slack alerting
│ ├── api/
│ │ ├── main.py # FastAPI app
│ │ ├── schemas.py # Pydantic request/response models
│ │ └── routes/
│ │ ├── estimate.py # POST /estimate endpoint
│ │ └── experiments.py # CRUD endpoints
│ ├── db/
│ │ ├── models.py # SQLAlchemy ORM
│ │ └── session.py # DB session management
│ ├── dashboard/
│ │ └── app.py # Streamlit dashboard
│ ├── data/
│ │ └── loaders.py # Criteo + Olist data loaders
│ ├── config.py # Pydantic settings
│ └── mlflow_logger.py # MLflow tracking utilities
├── dbt/
│ └── models/
│ ├── features/ # Behavioral feature SQL models
│ └── experiments/ # Experiment population models
├── tests/ # 53 tests, 83% coverage
│ ├── test_estimators.py
│ ├── test_drift.py
│ ├── test_api.py
│ └── ...
├── notebooks/eda/ # Validation scripts
│ ├── 00_data_validation.py
│ ├── 01_causal_estimators_test.py
│ ├── 02_uplift_models_test.py
│ ├── 03_api_test.py
│ └── 04_drift_detection_test.py
├── .github/workflows/
│ ├── ci.yml # Test + lint on push
│ └── retrain.yml # Drift-triggered retraining
├── docker-compose.yml
├── pyproject.toml
└── README.md

### MLOps: Drift Detection Pipeline

Population drift invalidates past causal estimates — a promotion targeting segment A produces incorrect lift estimates when applied to segment B.

LiftLab monitors two metrics weekly:

Population Stability Index (PSI)
PSI = Σ (current% - reference%) \* ln(current% / reference%)

Thresholds (industry standard):

PSI < 0.10 → ✅ Stable, no action
PSI < 0.20 → 🟡 Moderate shift, monitor
PSI ≥ 0.20 → 🔴 Significant shift, trigger retraining
Maximum Mean Discrepancy (MMD)

Multivariate covariate shift detection using RBF kernel:
MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

Threshold: MMD ≥ 0.05 → population shift detected

Automated Retraining

When thresholds are exceeded:

GitHub Actions workflow_dispatch triggers retrain.yml
CausalForest re-trains on current population sample
New ATE + CATE estimates logged to MLflow
Slack alert sent to growth team with updated estimates

### API Usage

### Create an experiment

curl -X POST http://localhost:8080/experiments/ \
 -H "Content-Type: application/json" \
 -d '{
"name": "q4_promo_test",
"description": "Q4 20% discount causal analysis",
"treatment_col": "treatment",
"outcome_col": "conversion",
"feature_cols": ["f0", "f1", "f2", "f3", "f4", "f5"],
"estimator": "causalforest",
"sample_frac": 0.05
}'

### Run estimation

curl -X POST http://localhost:8080/estimate/ \
 -H "Content-Type: application/json" \
 -d '{
"experiment_id": "dc041bbb-f512-4e1d-9c4f-85e30bdf3173"
}'

### Response:

{
"run_id": "a3f2...",
"ate": 0.000962,
"ci_lower": -0.006259,
"ci_upper": 0.008182,
"n_treated": 237543,
"n_control": 42049,
"shap_importance": {
"f8": 0.000906,
"f2": 0.000569,
...
},
"segment_effects": [
{"bin": 0, "mean_cate": 0.003395, "lift_vs_average": 3.53},
{"bin": 1, "mean_cate": 0.000129, "lift_vs_average": 0.13}
],
"duration_seconds": 51.6
}

### Testing

# Run full test suite

pytest tests/ -v

### With coverage report

pytest tests/ --cov=liftlab --cov-report=html

### Run specific test file

pytest tests/test_estimators.py -v

Test Coverage: 59 tests, 87% coverage

Key test categories:

Causal estimators (PSM, DiD, T/X-Learner, CausalForest)
Drift detection (PSI, MMD)
API endpoints (CRUD + estimation)
Data loaders and schemas
Edge cases and error handling

## Limitations

When causal inference assumptions break:

1. Unconfoundedness

PSM and CausalForest assume no unobserved confounders. If Criteo's targeting uses features not in f0-f11, estimates are biased.

Solution: Collect more features or use IV/front-door adjustment if instruments available.

2. Parallel Trends (DiD)

Requires treatment and control groups to trend identically pre-treatment. Violated when groups are inherently different.

Solution: Run parallel trends test (included) or use synthetic control methods.

3. SUTVA

Assumes no interference between units. Violated in social/network settings where one user's treatment affects another.

Solution: Use cluster-randomized designs or spillover-aware estimators.

4. Overlap

PSM requires common support — if propensity scores don't overlap, matching fails. Our 85/15 treatment split creates narrow overlap.

Solution: Trim extremes or use IPW (inverse propensity weighting) instead.

## Contributing

This is a portfolio project, but contributions are welcome:
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request

### License

MIT License - see LICENSE file for details

### Acknowledgments

Criteo AI Lab for the Criteo Uplift Prediction Dataset
Microsoft Research for EconML library
PyWhy team for DoWhy causal inference framework
Olist for the Brazilian E-Commerce Public Dataset

Contact

Your Name

📧 your.email@example.com
🔗 linkedin.com/in/yourprofile
💻 github.com/yourusername
