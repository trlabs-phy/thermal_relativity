# certify_planck/causal/certify_inflation.py

import numpy as np
from typing import Dict, Any


def certify_inflation_1d(
    history: dict,
    *,
    tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    Inflation certification (global ordering event)

    • PURE 1D / HISTORY-NATIVE
    • No geometry
    • No modes
    • No causation
    • No printing
    • Returns raw diagnostics only

    Interprets inflation exhaustion as the first point
    where thermal activation efficiency drops below unity.
    """

    # --------------------------------------------------
    # Required traces
    # --------------------------------------------------
    eta  = np.asarray(history["eta"], dtype=float)
    TE_P = np.asarray(history["TE_P"], dtype=float)
    B_T  = np.asarray(history.get("B_T", []), dtype=float)

    n = len(eta)
    if n < 5:
        raise ValueError("History too short for inflation certification.")

    # --------------------------------------------------
    # Inflation exhaustion detection (diagnostic)
    # --------------------------------------------------
    infl_end_index = None
    infl_end_eta   = None

    if len(B_T) == n:
        for i in range(1, n):
            dE = TE_P[i] - TE_P[i - 1]
            dB = B_T[i] - B_T[i - 1]

            if dB > tol:
                Q = dE / dB
                if Q < 1.0:
                    infl_end_index = i
                    infl_end_eta   = float(eta[i])
                    break

    # --------------------------------------------------
    # Return raw diagnostics
    # --------------------------------------------------
    return {
        "inflation_end_detected": infl_end_index is not None,
        "inflation_end_index": infl_end_index,
        "inflation_end_eta": infl_end_eta,
    }
