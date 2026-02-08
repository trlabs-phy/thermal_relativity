# certify_planck/causal/certify_modes.py

import numpy as np
from typing import Dict, Any


def certify_modes_1d(history: dict) -> Dict[str, Any]:
    """
    Mode / eigen-structure certification (DERIVED)

    Modes are computed from history — not stored.
    """

    eta = np.asarray(history["eta"], dtype=float)
    TE_P = np.asarray(history["TE_P"], dtype=float)

    n = len(eta)
    if n < 3:
        raise ValueError("History too short for mode certification.")

    # --------------------------------------------------
    # DERIVE eigenmode from history
    # Example rule (replace with your real one):
    # --------------------------------------------------
    # eig_id = number of filled cubes (monotone by construction)
    eig = np.asarray(history["N_C"], dtype=int)

    # --------------------------------------------------
    # A) Validity
    # --------------------------------------------------
    eig_finite = np.all(np.isfinite(eig))
    eig_nonneg = np.all(eig >= 0)
    eig_valid = eig_finite and eig_nonneg

    # --------------------------------------------------
    # B) Monotonicity
    # --------------------------------------------------
    eig_monotone = bool(np.all(eig[1:] >= eig[:-1]))

    # --------------------------------------------------
    # C) Transitions
    # --------------------------------------------------
    transitions = np.where(np.diff(eig) > 0)[0]

    return {
        "eig_defined": bool(eig_valid),
        "eig_finite": bool(eig_finite),
        "eig_nonnegative": bool(eig_nonneg),
        "eig_monotone": eig_monotone,
        "num_mode_transitions": int(len(transitions)),
        "final_eig_id": int(eig[-1]),
    }
