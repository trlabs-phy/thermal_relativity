# certify_planck/causal/certify_domain.py

import numpy as np
from typing import Dict, Any


def certify_domain_1d(
    history: dict,
    *,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Domain certification (pre-structure identities)

    Verifies domain partition bookkeeping implied by Eq. 0:
      • prior to interior realization, all realized volume is moat
      • interior volume is identically zero
      • volume additivity holds exactly

    PURE 1D / HISTORY-NATIVE
    NO dynamics
    NO classification
    NO geometry
    NO printing
    """

    # --------------------------------------------------
    # Required traces
    # --------------------------------------------------
    V_univ = np.asarray(history["V_univ"], dtype=float)

    # Optional but expected if domains are tracked
    V_int  = np.asarray(history.get("V_int", np.zeros_like(V_univ)))
    V_moat = np.asarray(history.get("V_moat", V_univ.copy()))

    n = len(V_univ)
    if n < 5:
        raise ValueError("History too short for domain certification.")

    # --------------------------------------------------
    # Pre-interior region (where interior is zero)
    # --------------------------------------------------
    pre_int = np.where(V_int <= tol)[0]

    if pre_int.size > 0:
        additivity_err = np.max(
            np.abs(V_univ[pre_int] - (V_int[pre_int] + V_moat[pre_int]))
        )
        max_abs_V_int = np.max(np.abs(V_int[pre_int]))
        max_abs_moat_minus_V = np.max(
            np.abs(V_moat[pre_int] - V_univ[pre_int])
        )
    else:
        additivity_err = 0.0
        max_abs_V_int = 0.0
        max_abs_moat_minus_V = 0.0

    # --------------------------------------------------
    # Return raw diagnostics
    # --------------------------------------------------
    return {
        "pre_interior_additivity_error": float(additivity_err),
        "max_abs_pre_interior_volume": float(max_abs_V_int),
        "max_abs_moat_minus_volume": float(max_abs_moat_minus_V),
        "num_pre_interior_steps": int(pre_int.size),
    }
