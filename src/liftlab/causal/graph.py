# src/liftlab/causal/graph.py
"""
DoWhy causal graph specification for the Criteo uplift setting.

Causal DAG:
  User features (f0-f11) → Treatment (ad exposure)
  User features (f0-f11) → Conversion (outcome)
  Treatment               → Conversion
  Treatment               → Visit → Conversion

This reflects the reality that:
- Criteo's ad targeting system uses user features to decide who sees the ad
- Those same features also independently drive conversion probability
- So features are CONFOUNDERS — we must adjust for them
"""
import networkx as nx
import dowhy
from dowhy import CausalModel
import pandas as pd
from loguru import logger
from typing import Optional


# Feature columns in the Criteo dataset
CRITEO_FEATURES = [f"f{i}" for i in range(12)]  # f0 through f11
TREATMENT_COL = "treatment"
OUTCOME_COL = "conversion"


def build_criteo_causal_graph() -> str:
    """
    Build the causal graph as a GML string for DoWhy.

    The DAG encodes:
        - f0..f11 → treatment  (ad system uses features for targeting)
        - f0..f11 → conversion (features also drive organic conversion)
        - treatment → conversion (the causal effect we want to estimate)
        - treatment → visit → conversion (mediation path)
    """
    gml_graph = """
    graph [
        directed 1
        node [ id "f0"         label "f0" ]
        node [ id "f1"         label "f1" ]
        node [ id "f2"         label "f2" ]
        node [ id "f3"         label "f3" ]
        node [ id "f4"         label "f4" ]
        node [ id "f5"         label "f5" ]
        node [ id "f6"         label "f6" ]
        node [ id "f7"         label "f7" ]
        node [ id "f8"         label "f8" ]
        node [ id "f9"         label "f9" ]
        node [ id "f10"        label "f10" ]
        node [ id "f11"        label "f11" ]
        node [ id "treatment"  label "treatment" ]
        node [ id "visit"      label "visit" ]
        node [ id "conversion" label "conversion" ]

        edge [ source "f0"  target "treatment" ]
        edge [ source "f1"  target "treatment" ]
        edge [ source "f2"  target "treatment" ]
        edge [ source "f3"  target "treatment" ]
        edge [ source "f4"  target "treatment" ]
        edge [ source "f5"  target "treatment" ]
        edge [ source "f6"  target "treatment" ]
        edge [ source "f7"  target "treatment" ]
        edge [ source "f8"  target "treatment" ]
        edge [ source "f9"  target "treatment" ]
        edge [ source "f10" target "treatment" ]
        edge [ source "f11" target "treatment" ]

        edge [ source "f0"  target "conversion" ]
        edge [ source "f1"  target "conversion" ]
        edge [ source "f2"  target "conversion" ]
        edge [ source "f3"  target "conversion" ]
        edge [ source "f4"  target "conversion" ]
        edge [ source "f5"  target "conversion" ]
        edge [ source "f6"  target "conversion" ]
        edge [ source "f7"  target "conversion" ]
        edge [ source "f8"  target "conversion" ]
        edge [ source "f9"  target "conversion" ]
        edge [ source "f10" target "conversion" ]
        edge [ source "f11" target "conversion" ]

        edge [ source "treatment"  target "conversion" ]
        edge [ source "treatment"  target "visit" ]
        edge [ source "visit"      target "conversion" ]
    ]
    """
    return gml_graph


def build_causal_model(
    df: pd.DataFrame,
    outcome_col: str = OUTCOME_COL,
    treatment_col: str = TREATMENT_COL,
) -> CausalModel:
    """
    Instantiate a DoWhy CausalModel with the Criteo DAG.

    Args:
        df: DataFrame with Criteo features + treatment + outcome
        outcome_col: target variable column name
        treatment_col: treatment indicator column name

    Returns:
        Configured DoWhy CausalModel ready for identification + estimation
    """
    logger.info("Building DoWhy CausalModel with Criteo DAG...")

    model = CausalModel(
        data=df,
        treatment=treatment_col,
        outcome=outcome_col,
        graph=build_criteo_causal_graph(),
    )

    logger.info(f"CausalModel built | treatment='{treatment_col}' | outcome='{outcome_col}'")
    return model


def identify_estimand(model: CausalModel) -> dowhy.causal_identifier.IdentifiedEstimand:
    """
    Run DoWhy's identification step — determines WHICH statistical quantity
    maps to the causal estimand (ATE) given our DAG.

    DoWhy will find the backdoor adjustment set: {f0..f11}
    This means: conditioning on all features blocks all confounding paths.
    """
    logger.info("Identifying causal estimand...")
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    logger.info(f"Identified estimand:\n{estimand}")
    return estimand
