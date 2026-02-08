# ============================================================
# certify_planck/threed/certify_ell_window.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module implements three–dimensional ℓ–window localization
# diagnostics for the Unified Thermal Relativity Boltzmann Solver.
#
# It evaluates whether the projected angular response derived
# from a certified causal history is localized within a finite
# ℓ–window when realized in spatial projection.
#
# No sky model, pixelization, spherical harmonics, or a_{ℓm}
# decomposition is assumed or constructed.
#
# This module is strictly non-causal and read-only with respect
# to solver evolution.
#
# Architectural guarantees:
# --------------------------
# - Uses only certified causal history quantities
# - Performs no solver evolution or state mutation
# - Treats ℓ as a projection trajectory, not a spectrum
# - Introduces no observational fitting or interpretation
#
# Governing role:
# ---------------
# - Constructs a projected ℓ–trajectory from volume and scale
# - Diagnoses localization of the angular response
# - Verifies finite ℓ–window width consistency
# - Reports diagnostic bounds and provenance
#
# ℓ–window certification tests localization, not cosmological fitting.
#
# ============================================================

from typing import Dict, Any, Optional
import numpy as np


def certify_ell_window_3d(
    history: dict,
    acoustic_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """
    ℓ-window certification (3D projection, TR-native)

    • No sky
    • No pixels
    • No a_lm
    • Diagnoses localization of projected angular response
    """

    # -----------------------------
    # Required inputs
    # -----------------------------
    if "V_univ" not in history:
        raise ValueError("ℓ-window certification requires V_univ history")

    V = np.asarray(history["V_univ"], dtype=float)

    # -----------------------------
    # Effective size proxy
    # -----------------------------
    R_eff = V ** (1.0 / 3.0)

    # -----------------------------
    # Acoustic scale source
    # -----------------------------
    if acoustic_scale is not None and acoustic_scale > 0.0:
        r_s = float(acoustic_scale)
        acoustic_source = "external"
    elif "r_acoustic" in history:
        r_s_arr = np.asarray(history["r_acoustic"], dtype=float)
        r_s = float(np.nanmax(r_s_arr))
        acoustic_source = "history"
    else:
        r_s = None
        acoustic_source = "missing"

    # -----------------------------
    # ℓ trajectory
    # -----------------------------
    if r_s is not None and r_s > 0.0:
        ell_traj = np.pi * (R_eff / r_s)
    else:
        ell_traj = None

    # -----------------------------
    # Window diagnostics
    # -----------------------------
    if ell_traj is not None:
        ell_peak = float(np.nanmax(ell_traj))
        peak_index = int(np.nanargmax(ell_traj))

        # Define window around peak (±5 steps)
        i0 = max(0, peak_index - 5)
        i1 = min(len(ell_traj), peak_index + 6)

        ell_window = ell_traj[i0:i1]

        window_mean = float(np.mean(ell_window))
        window_std = float(np.std(ell_window))

        if window_mean != 0.0:
            window_frac_width = window_std / abs(window_mean)
        else:
            window_frac_width = np.nan
    else:
        ell_peak = None
        window_mean = None
        window_std = None
        window_frac_width = None

    max_width = 3e-2
    width = window_frac_width

    return {
        # Core observable
        "ell_peak": float(ell_peak),
        "window_fractional_width": float(width),

        # Acceptance gate (THIS is the cert)
        "ell_window_pass": bool(np.isfinite(width) and width <= max_width),

        # Bound metadata (not a target!)
        "max_window_fractional_width": float(max_width),

        # Optional diagnostics
        "window_mean": window_mean,
        "window_std": window_std,

        # Provenance
        "acoustic_scale_used": r_s,
        "acoustic_source": acoustic_source,
        "num_window_samples": None if ell_traj is None else (i1 - i0),
    }
