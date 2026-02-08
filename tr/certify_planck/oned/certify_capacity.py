# ============================================================
# certify_planck/causal/capacity.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module implements one–dimensional capacity and weak–field
# diagnostics for the Unified Thermal Relativity Boltzmann Solver.
#
# It evaluates Equation 0 bookkeeping directly from the recorded
# causal history and returns raw numerical diagnostics describing
# capacity closure, irreversibility, scaling behavior, and
# weak–field indicators.
#
# This module is strictly causal-history–native and read-only.
# It introduces no geometry, no tier logic, no thresholds, and
# no pass/fail decisions. Interpretation and reporting are
# delegated entirely to downstream certification and output
# layers.
#
# Architectural guarantees:
# --------------------------
# - Uses only Eq. 0 quantities recorded in history
# - Performs no solver evolution or state mutation
# - Introduces no tuning, fitting, or corrective feedback
# - Returns transparent, numeric diagnostics only
#
# Governing role:
# ---------------
# - Verifies capacity closure (fill + chase consistency)
# - Confirms ordering irreversibility (cube monotonicity)
# - Measures homogeneous capacity scaling (3D proxy)
# - Reports weak–field indicators without classification
#
# This module measures capacity behavior; it does not judge it.
#
# ============================================================

import numpy as np
import math
from typing import Dict, Any

def certify_capacity_1d(
    history: dict,
    *,
    kappa_C: float,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """
    Capacity & weak-field certification (Eq. 0 bookkeeping)

    PURE 1D / HISTORY-NATIVE

    • Uses ONLY Eq. 0 quantities
    • No geometry
    • No Tier logic
    • No printing
    • No pass/fail
    • Returns raw diagnostics for output.py
    """

    # --------------------------------------------------
    # Required history traces (authoritative)
    # --------------------------------------------------
    eta     = np.asarray(history["eta"], dtype=float)
    TE_P    = np.asarray(history["TE_P"], dtype=float)
    N_C     = np.asarray(history["N_C"], dtype=float)
    epsilon = np.asarray(history["epsilon"], dtype=float)
    V       = np.asarray(history["V_univ"], dtype=float)

    n = len(eta)
    if n < 5:
        raise ValueError("History too short for capacity certification.")

    # --------------------------------------------------
    # Shape sanity
    # --------------------------------------------------
    def _assert_len(name, arr):
        if len(arr) != n:
            raise ValueError(
                f"capacity certify: length mismatch for {name} "
                f"(len={len(arr)}, expected {n})"
            )

    for name, arr in [
        ("TE_P", TE_P),
        ("N_C", N_C),
        ("epsilon", epsilon),
        ("V_univ", V),
    ]:
        _assert_len(name, arr)

    # ==================================================
    # A) Capacity closure (Fill + Chase)
    # ==================================================
    # Eq. 0 authority:
    #   TE_U_eff = κC·N_C + ε
    #   TE_F     = max(TE_P − TE_U_eff, 0)
    TE_U_eff = kappa_C * N_C + epsilon
    TE_F     = np.maximum(TE_P - TE_U_eff, 0.0)

    closure_residual = TE_P - (TE_U_eff + TE_F)

    # ==================================================
    # B) Ordering irreversibility (cube monotonicity)
    # ==================================================
    dNC = np.diff(N_C)

    # ==================================================
    # C) Homogeneous capacity scaling (3D proxy)
    # ==================================================
    # V_univ ∝ N_C  ⇒  a ∝ N_C^(1/3)
    etas = []
    slopes = []

    for i in range(1, n):
        if V[i] > 0 and V[i - 1] > 0:
            a_i   = V[i] ** (1.0 / 3.0)
            a_im1 = V[i - 1] ** (1.0 / 3.0)

            if a_i > 0 and a_im1 > 0:
                dlna = math.log(a_i) - math.log(a_im1)
                if abs(dlna) > tol:
                    slope = (math.log(V[i]) - math.log(V[i - 1])) / dlna
                    slopes.append(slope)
                    etas.append(eta[i])

    slope_median = float(np.median(slopes)) if slopes else None

    # ==================================================
    # D) Weak-field diagnostics (Eq. 0 regime tests)
    # ==================================================
    # These are NOT judgments — just indicators

    ordering_active = bool(np.any(np.diff(TE_P) > tol))
    free_energy_persists = bool(np.any(TE_F > tol))

    # Where free energy first appears (useful for plots / papers)
    first_TE_F_index = int(np.argmax(TE_F > tol)) if free_energy_persists else None
    first_TE_F_eta   = float(eta[first_TE_F_index]) if first_TE_F_index is not None else None

    # ==================================================
    # Return raw diagnostics (NO PASS/FAIL)
    # ==================================================
    return {
        # --- capacity bookkeeping ---
        "capacity_closure_max_residual": float(
            np.max(np.abs(closure_residual))
        ),
        "TE_U_eff": TE_U_eff,
        "TE_F": TE_F,

        # --- irreversibility ---
        "NC_min_delta": float(np.min(dNC)) if len(dNC) else 0.0,

        # --- scaling ---
        "scaling_eta": etas,
        "scaling_slopes": slopes,
        "homogeneous_scaling_median": slope_median,
        "homogeneous_scaling_target": 3.0,

        # --- weak-field indicators ---
        "ordering_active": ordering_active,
        "free_energy_persists": free_energy_persists,
        "first_TE_F_eta": first_TE_F_eta,
    }
