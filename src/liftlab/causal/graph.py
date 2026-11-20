"""
DoWhy causal graphs for the Criteo uplift setting.

Criteo's uplift dataset comes from randomized incrementality tests:
treatment was assigned at random (~85% treated), independent of the user
features. So in the true DAG nothing points into ``treatment``:

    f0..f11 ──► visit ──► conversion
    f0..f11 ──────────────► conversion
    treatment ─► visit, treatment ─► conversion

The backdoor adjustment set is therefore EMPTY: a difference in means is
unbiased, and adjusting for features only improves precision.

``randomized=False`` gives the observational version (features ─►
treatment), used with confounded samples: there the features become a
required adjustment set. ``visit`` is a mediator and must never be
adjusted for when estimating the total effect on conversion.
"""

from __future__ import annotations

import dowhy
import pandas as pd
from dowhy import CausalModel
from loguru import logger

CRITEO_FEATURES = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
OUTCOME_COL = "conversion"


def build_criteo_causal_graph(randomized: bool = True, features: list[str] = CRITEO_FEATURES) -> str:
    """GML string for DoWhy."""
    nodes = features + [TREATMENT_COL, "visit", OUTCOME_COL]
    edges = [(f, OUTCOME_COL) for f in features] + [(f, "visit") for f in features]
    edges += [(TREATMENT_COL, "visit"), ("visit", OUTCOME_COL), (TREATMENT_COL, OUTCOME_COL)]
    if not randomized:
        edges += [(f, TREATMENT_COL) for f in features]
    lines = ["graph [", "  directed 1"]
    lines += [f'  node [ id "{n}" label "{n}" ]' for n in nodes]
    lines += [f'  edge [ source "{a}" target "{b}" ]' for a, b in edges]
    lines.append("]")
    return "\n".join(lines)


def build_causal_model(
    df: pd.DataFrame,
    outcome_col: str = OUTCOME_COL,
    treatment_col: str = TREATMENT_COL,
    randomized: bool = True,
) -> CausalModel:
    logger.info(f"Building DoWhy CausalModel (randomized={randomized})")
    return CausalModel(data=df, treatment=treatment_col, outcome=outcome_col,
                       graph=build_criteo_causal_graph(randomized))


def identify_estimand(model: CausalModel) -> dowhy.causal_identifier.IdentifiedEstimand:
    """DoWhy identification. Randomized graph -> empty backdoor set;
    observational graph -> {f0..f11}."""
    return model.identify_effect(proceed_when_unidentifiable=True)


def backdoor_set(model: CausalModel) -> list[str]:
    """Variables that must be adjusted for (sorted)."""
    estimand = identify_estimand(model)
    return sorted(estimand.get_backdoor_variables() or [])
