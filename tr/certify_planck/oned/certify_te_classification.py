# ============================================================
# certify_planck/oned/certify_te_classification.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module implements Thermal Energy (TE) phase classification
# for the Unified Thermal Relativity Boltzmann Solver.
#
# TE phases are defined as κ_C–local, cube–local strain regimes
# arising from thermal chase bookkeeping. They introduce no new
# physical entities and carry no independent dynamics.
#
# Phase classification is performed purely as post hoc labeling
# of recorded causal history. No solver state is modified and no
# causal influence is introduced.
#
# This module is non-causal and strictly read-only with respect
# to solver evolution.
#
# Architectural guarantees:
# --------------------------
# - Uses only recorded 1D history quantities (ε, κ_C)
# - Performs no solver evolution or state mutation
# - Introduces no thresholds beyond defined phase bins
# - Writes phase labels to history for downstream diagnostics only
#
# Governing role:
# ---------------
# - Classifies cube-local κ_C strain into discrete TE phases
# - Counts phase transition events over causal history
# - Provides authoritative phase labels for structure,
#   visibility, projection, and timeline certification
#
# TE phases are bookkeeping labels, not physical substances.
#
# ============================================================

import numpy as np
from typing import Dict, Any

# --------------------------------------------------
# Universal TE phase bins (κC-local strain)
# --------------------------------------------------

TE_PHASE_BINS = [
    ("THERMON",   0.00, 0.20),
    ("PLASMA",    0.20, 0.45),
    ("LIQUID",    0.45, 0.70),
    ("SOLID",     0.70, 0.90),
    ("THERMALUS", 0.90, 1.01),
]

PHASE_INDEX = {
    "THERMON": 0,
    "PLASMA": 1,
    "LIQUID": 2,
    "SOLID": 3,
    "THERMALUS": 4,
}


def classify_phase(strain: float) -> str:
    for name, lo, hi in TE_PHASE_BINS:
        if lo <= strain < hi:
            return name
    return "UNDEFINED"


def certify_te_classification_1d(
    history: dict,
    *,
    modes=None,        # pipeline requires it
    kappa_C: float = 1.0,
) -> Dict[str, Any]:
    """
    Tier–TE  Thermal Energy Classification (TR-native)

    • κC always enforced
    • Phase = cube-local κC chase strain
    • Pure bookkeeping (non-causal)
    """

    epsilon = np.asarray(history["epsilon"], dtype=float)

    if epsilon.size == 0:
        return {
            "num_phase_transitions": 0,
            "final_phase": "THERMON",
            "final_te_phase_index": 0,
            "unique_phases": [],
        }

    # --------------------------------------------------
    # Cube-local κC strain
    # --------------------------------------------------
    strain = epsilon / max(kappa_C, 1e-30)

    # --------------------------------------------------
    # Phase classification
    # --------------------------------------------------
    phases = np.array([classify_phase(s) for s in strain])

    transitions = int(np.sum(phases[1:] != phases[:-1]))
    unique_phases = list(dict.fromkeys(phases))

    history["_te_phases"] = phases.tolist()

    final_phase = phases[-1]
    final_index = PHASE_INDEX.get(final_phase, 0)

    return {
        # ==================================================
        # REQUIRED BY certify_1d.py (DO NOT RENAME)
        # ==================================================
        "num_phase_transitions": int(transitions),
        "final_phase": str(final_phase),
        "final_te_phase_index": int(final_index),
        "unique_phases": list(unique_phases),

        # ==================================================
        # REQUIRED FOR DOWNSTREAM CERTS
        # ==================================================
        # Visibility, projection, growth, etc. read this
        "te_phases": phases,

        # ==================================================
        # OPTIONAL DIAGNOSTICS (SAFE EXTRAS)
        # ==================================================
        "strain": strain,                  # κC geometric strain
        "volume_ratio": strain,            
    }
