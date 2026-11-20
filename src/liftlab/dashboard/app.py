# src/liftlab/dashboard/app.py
"""
LiftLab — Growth Team Experimentation Dashboard

Run with:
    streamlit run src/liftlab/dashboard/app.py
"""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import httpx
from datetime import datetime


# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="LiftLab — Causal Inference Engine",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for visibility ─────────────────────────────
st.markdown("""
<style>
    /* Ensure text is visible in both light and dark mode */
    .stTextInput > label, .stSelectbox > label, .stSlider > label,
    .stTextArea > label, .stNumberInput > label {
        color: inherit !important;
        font-weight: 600 !important;
    }

    /* Form labels */
    [data-testid="stFormSubmitButton"] {
        width: 100%;
    }

    /* Improve mobile responsiveness */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }

    /* Success/error messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        padding: 1rem !important;
        border-radius: 0.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

API_BASE = os.environ.get("LIFTLAB_API", "http://localhost:8080")
RESULTS_PATH = Path(__file__).resolve().parents[3] / "results" / "criteo_benchmark.json"

# ── Helpers ───────────────────────────────────────────────
def api_get(endpoint: str) -> dict | list | None:
    try:
        r = httpx.get(f"{API_BASE}{endpoint}", timeout=10.0)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
    return None


def api_post(endpoint: str, payload: dict, timeout: float = 300.0) -> dict | None:
    try:
        r = httpx.post(f"{API_BASE}{endpoint}", json=payload, timeout=timeout)
        if r.status_code in (200, 201):
            return r.json()
        st.error(f"API error {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"API error: {e}")
    return None


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🧪 LiftLab")
    st.caption("Production Causal Inference Engine")
    st.divider()

    # Health check
    health = api_get("/health")
    if health:
        st.success(f"**API Status:** {health['status'].upper()}")
        db_icon = "🟢" if health['database'] == 'healthy' else "🔴"
        st.caption(f"{db_icon} Database: {health['database']}")
        mlflow_icon = "🟢" if health['mlflow'] == 'healthy' else "🟡"
        st.caption(f"{mlflow_icon} MLflow: {health['mlflow']}")
    else:
        st.error("⚠️ API unreachable")

    st.divider()
    page = st.radio(
        "**Navigate**",
        ["📊 Experiments", "🚀 Run Estimation", "📈 Results", "🔍 Drift Monitor"],
        label_visibility="collapsed"
    )

# ── Page: Experiments ─────────────────────────────────────
if page == "📊 Experiments":
    st.title("📊 Experiment Registry")
    st.markdown("*All registered causal experiments*")

    experiments = api_get("/experiments/") or []

    if not experiments:
        st.info("No experiments yet. Go to **🚀 Run Estimation** to create one.")
    else:
        df = pd.DataFrame(experiments)
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Experiments", len(df))
        col2.metric("Done", len(df[df["status"] == "done"]))
        col3.metric("Pending", len(df[df["status"] == "pending"]))

        st.dataframe(
            df[["name", "estimator", "sample_frac", "status", "created_at"]],
            width="stretch",
            hide_index=True,
        )

# ── Page: Run Estimation ──────────────────────────────────
elif page == "🚀 Run Estimation":
    st.title("🚀 Run Causal Estimation")
    st.caption("Criteo is a randomized trial, so the difference in means is unbiased; "
               "the other estimators are checked against it.")

    with st.form("new_experiment", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("**Experiment Name**",
                                 value=f"exp_{datetime.now().strftime('%Y%m%d_%H%M')}")
            estimator = st.selectbox(
                "**Estimator**",
                ["difference_in_means", "aipw", "psm", "causalforest", "xlearner", "tlearner"],
                help="difference_in_means / aipw / psm → average effect with 95% CI. "
                     "causalforest / xlearner / tlearner → who responds, judged on held-out data.",
            )
        with col2:
            sample_frac = st.slider("**Sample Fraction**", 0.01, 0.20, 0.02, 0.01,
                                    help="Fraction of the 14M-row Criteo dataset")
            outcome_col = st.selectbox("**Outcome**", ["conversion", "visit"])
        description = st.text_area("**Description (optional)**")
        submitted = st.form_submit_button("▶️ Launch Experiment", type="primary",
                                          width="stretch")

    if submitted:
        payload = {"name": name, "description": description or None,
                   "treatment_col": "treatment", "outcome_col": outcome_col,
                   "feature_cols": [f"f{i}" for i in range(12)],
                   "estimator": estimator, "sample_frac": sample_frac}
        with st.spinner("Creating experiment..."):
            exp = api_post("/experiments/", payload)
        if exp:
            with st.spinner(f"Running {estimator} (~1-3 min)..."):
                result = api_post("/estimate/", {"experiment_id": exp["id"]})
            if result:
                fmt = lambda v: "n/a" if v is None else f"{v:.6f}"  # noqa: E731
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"**{result['estimand']}**", fmt(result["ate"]))
                c2.metric("**95% CI**", f"[{fmt(result['ci_lower'])}, {fmt(result['ci_upper'])}]")
                c3.metric("**p-value**", "n/a" if result["p_value"] is None else f"{result['p_value']:.2g}")
                c4.metric("**Duration**", f"{result['duration_seconds']:.0f}s")

                if result.get("qini_coefficient") is not None:
                    st.subheader("Held-out evaluation")
                    c1, c2 = st.columns(2)
                    c1.metric("**Qini coefficient**", f"{result['qini_coefficient']:+.3f}",
                              help="0 = no better than random targeting")
                    c2.metric("**Observed lift, top 25%**", f"{result['top_25pct_lift']:.2f}x")

                if result.get("segment_effects"):
                    seg = pd.DataFrame(result["segment_effects"])
                    fig = go.Figure(go.Bar(
                        x=[f"Q{g + 1}" for g in seg["group"]], y=seg["uplift"],
                        error_y=dict(type="data", symmetric=False,
                                     array=seg["ci_upper"] - seg["uplift"],
                                     arrayminus=seg["uplift"] - seg["ci_lower"]),
                    ))
                    fig.update_layout(title="Observed uplift by quartile of predicted effect "
                                            "(held-out, Q1 = highest predicted)",
                                      yaxis_title="Observed uplift (treated − control)")
                    st.plotly_chart(fig, width="stretch")

                if result.get("feature_importance"):
                    imp = pd.DataFrame(list(result["feature_importance"].items()),
                                       columns=["Feature", "Importance"]).sort_values("Importance")
                    st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                                           title="Features driving effect heterogeneity"),
                                    width="stretch")

elif page == "📈 Results":
    st.title("📈 Criteo Benchmark Results")
    if not RESULTS_PATH.exists():
        st.info("No benchmark results yet. Run `python scripts/benchmark_criteo.py` to create "
                "`results/criteo_benchmark.json`.")
    else:
        bench = json.loads(RESULTS_PATH.read_text())["sections"]
        eff = bench["full_data_effects"]["conversion"]
        ref, raw = eff["aipw"], eff["difference_in_means"]
        c2 = bench.get("randomization_check", {}).get("c2st", {})
        col1, col2 = st.columns(2)
        col1.metric("**Reference (AIPW, all rows): conversion ATE**", f"{ref['estimate']:.6f}",
                    delta=f"{ref['relative_lift']:+.1%} relative lift")
        col2.metric("**Raw difference in means**", f"{raw['estimate']:.6f}",
                    delta=f"{raw['relative_lift']:+.1%} relative lift", delta_color="off")
        st.caption(f"{bench['data']['rows']:,} rows. AIPW 95% CI [{ref['ci'][0]:.6f}, {ref['ci'][1]:.6f}]. "
                   + (f"Randomization check: a classifier predicts treatment with AUC {c2['auc']:.3f} "
                      f"(shuffled labels reach {c2['null_auc_max']:.3f}). "
                      + ("Imbalance detected, so the raw difference is not used as ground truth."
                         if c2["detected"] else "No imbalance detected.") if c2 else ""))

        rows = []
        for key, r in bench["sample_estimators"]["conversion"].items():
            if isinstance(r, dict):
                rows.append({"name": f"{r['method']} ({r['estimand']})", "est": r["estimate"],
                             "lo": r["ci"][0], "hi": r["ci"][1]})
        conf = bench.get("confounded_test", {}).get("conversion", {})
        for key in ("naive_difference_in_means", "aipw_att", "psm"):
            if key in conf:
                r = conf[key]
                rows.append({"name": f"Confounded: {r['method']}", "est": r["estimate"],
                             "lo": r["ci"][0], "hi": r["ci"][1]})
        df = pd.DataFrame(rows)
        fig = go.Figure(go.Scatter(
            x=df["name"], y=df["est"], mode="markers", marker=dict(size=12),
            error_y=dict(type="data", symmetric=False, array=df["hi"] - df["est"],
                         arrayminus=df["est"] - df["lo"])))
        fig.add_hline(y=ref["estimate"], line_dash="dash", annotation_text="AIPW reference (all rows)")
        fig.update_layout(title="Estimators on a 2% sample vs the all-rows reference (95% CI)",
                          yaxis_title="Effect on conversion")
        st.plotly_chart(fig, width="stretch")

        st.subheader("Who responds? CATE models on held-out data")
        cate = bench["cate_models_holdout"]
        table = [{"outcome": y, "model": m["method"], "Qini": m["qini_coefficient"],
                  "top-25% observed lift": m["top_25pct"]["lift_vs_overall"]}
                 for y, models in cate.items() if isinstance(models, dict)
                 for m in models.values() if isinstance(m, dict)]
        st.dataframe(pd.DataFrame(table).round(3), width="stretch", hide_index=True)

        f8_section = bench.get("f8_groups")
        if f8_section:
            st.subheader("Effect by f8 value group (all rows): raw vs AIPW")
            f8 = pd.DataFrame(f8_section["conversion"])
            st.dataframe(f8[["value_from", "value_to", "n", "uplift", "aipw", "aipw_ci_lower",
                             "aipw_ci_upper", "aipw_lift_vs_overall"]].round(6),
                         width="stretch", hide_index=True)

elif page == "🔍 Drift Monitor":
    st.title("🔍 Population Drift Monitor")
    st.caption("PSI per feature and multivariate MMD between a reference population and a "
               "current one. This demo uses simulated populations; set a shift to see alerts.")
    from liftlab.causal.simulate import FEATURES, simulate_rct
    from liftlab.monitoring.drift import DriftDetector

    shift = st.slider("Simulated shift in f0–f5 (standard deviations)", 0.0, 1.5, 0.0, 0.1)
    ref = simulate_rct(5_000, seed=1)
    cur = simulate_rct(5_000, seed=2)
    cur[FEATURES[:6]] = cur[FEATURES[:6]] + shift
    report = DriftDetector().detect(ref, cur, FEATURES)

    psi_df = pd.DataFrame(list(report.psi_scores.items()), columns=["Feature", "PSI"])
    psi_df["Status"] = psi_df["PSI"].apply(
        lambda x: "🔴 Severe" if x >= 0.20 else ("🟡 Monitor" if x >= 0.10 else "✅ OK"))
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(psi_df, x="Feature", y="PSI", color="Status",
                     color_discrete_map={"✅ OK": "green", "🟡 Monitor": "orange", "🔴 Severe": "red"})
        fig.add_hline(y=0.20, line_dash="dash", line_color="red")
        fig.add_hline(y=0.10, line_dash="dash", line_color="orange")
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.metric("**MMD**", f"{report.mmd_score:.4f}",
                  delta="flagged" if report.mmd_flagged else "below threshold",
                  delta_color="inverse" if report.mmd_flagged else "normal")
        st.metric("**PSI flagged features**", f"{len(report.psi_flagged)} / {len(FEATURES)}")
        st.metric("**Severity**", report.severity.upper())
        st.info(report.recommendation)
