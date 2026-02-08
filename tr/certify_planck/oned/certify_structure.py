# tr/certify_planck/oned/certify_structure.py

import numpy as np
from typing import Dict, Any


MATTER_PHASES = {"PLASMA", "LIQUID", "SOLID", "THERMALUS"}


def certify_structure_1d(history: dict, *, tol: float = 1e-12) -> Dict[str, Any]:
    """
    Structure certification (TR-native, cube-local)

    NEW ONTOLOGY:
      • No interior / moat
      • No fluids
      • No fields

    Matter exists iff cubes realize non-THERMON κC regimes.
    """

    eta    = np.asarray(history["eta"], dtype=float)
    N_C    = np.asarray(history["N_C"], dtype=float)
    V_univ = np.asarray(history["V_univ"], dtype=float)

    # TE classification must already have run
    phases = history.get("_te_phases", None)

    if phases is None:
        # Structure cannot be assessed yet
        return {
            "structure_regime": "unclassified",
            "matter_exists": False,
            "first_matter_eta": None,
            "max_additivity_error": 0.0,
            "bounds_ok": True,
            "min_dVint": 0.0,
        }

    phases = np.asarray(phases)

    # --------------------------------------------------
    # Matter existence
    # --------------------------------------------------
    matter_mask = np.isin(phases, list(MATTER_PHASES))
    matter_exists = bool(np.any(matter_mask))

    first_matter_eta = (
        float(eta[np.where(matter_mask)[0][0]])
        if matter_exists else None
    )

    # --------------------------------------------------
    # Additivity (cube identity)
    # --------------------------------------------------
    add_err = V_univ - N_C
    max_add_err = float(np.max(np.abs(add_err))) if len(add_err) else 0.0

    # --------------------------------------------------
    # Irreversibility (cube monotonicity)
    # --------------------------------------------------
    dN = np.diff(N_C)
    min_dN = float(np.min(dN)) if len(dN) else 0.0

    bounds_ok = (
        np.all(N_C >= -tol) and
        np.all(V_univ >= -tol) and
        np.all(V_univ <= N_C + tol)
    )

    return {
        "structure_regime": "cube-local",
        "matter_exists": matter_exists,
        "first_matter_eta": first_matter_eta,

        "max_additivity_error": max_add_err,
        "bounds_ok": bounds_ok,
        "min_dVint": min_dN,  # reinterpretation: cube irreversibility
    }
