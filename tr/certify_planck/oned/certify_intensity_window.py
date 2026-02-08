# ============================================================
# certify_planck/oned/certify_intensity_window.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module implements one–dimensional pressure statistics
# diagnostics for the Unified Thermal Relativity Boltzmann Solver.
#
# It replaces the legacy concept of an "intensity window" with a
# TR–native interpretation based on κ_C–closure stress observed
# during the certified visibility interval.
#
# Pressure is treated here as a bookkeeping statistic derived
# from cube-local κ_C chase rates. It introduces no force law,
# no dynamics, and no causal influence.
#
# This module is strictly non-causal and read-only with respect
# to solver evolution.
#
# Architectural guarantees:
# --------------------------
# - Uses only recorded 1D history quantities
# - Performs no solver evolution or state mutation
# - Applies no pass/fail gating or thresholds
# - Returns raw statistical summaries only
#
# Governing role:
# ---------------
# - Computes κ_C–closure stress statistics during visibility
# - Provides mean, max, and RMS pressure diagnostics
# - Maintains legacy compatibility without semantic leakage
#
# Pressure here is an observational statistic, not a dynamical variable.
#
# ============================================================

from typing import Dict, Any
import numpy as np


def certify_intensity_window_1d(
    history: dict,
    *,
    atol: float = 1e-12,
) -> Dict[str, Any]:
    """
    Tier–IW  Pressure Statistics (TR-native)

    Legacy hook replacement for "intensity window".

    Physical meaning:
      Pressure ≡ κC-closure stress during visibility

    • No pass/fail gating
    • Visibility already guarantees relevance
    • Reports stress statistics only
    • Non-causal, pure bookkeeping
    """

    report: Dict[str, Any] = {}

    # --------------------------------------------------
    # Required signals
    # --------------------------------------------------
    eps = np.asarray(history.get("epsilon", []), dtype=float)
    TEF = np.asarray(history.get("TE_F", []), dtype=float)
    N_C = np.asarray(history.get("N_C", []), dtype=float)
    eta = np.asarray(history.get("eta", []), dtype=float)
    vis = np.asarray(history.get("VisFlag", []), dtype=int)

    if eps.size < 2 or vis.size == 0:
        return {
            "intensity_window_present": False,
            "pressure_defined": False,
            "reason": "missing_data",
        }

    # --------------------------------------------------
    # Visibility window
    # --------------------------------------------------
    vis_idx = np.where(vis == 1)[0]

    if vis_idx.size == 0:
        return {
            "intensity_window_present": False,
            "pressure_defined": False,
            "reason": "no_visibility",
        }

    i0 = int(vis_idx[0])
    i1 = int(vis_idx[-1] + 1)

    # --------------------------------------------------
    # κC-closure rate (pressure proxy)
    # --------------------------------------------------
    deps = np.diff(eps)
    deps = np.maximum(deps, 0.0)

    deps_win = deps[i0:i1] if i1 > i0 else np.array([])
    TEF_win  = TEF[i0:i1] if i1 > i0 else np.array([])
    NC_win   = N_C[i0:i1] if i1 > i0 else np.array([])

    # --------------------------------------------------
    # Pressure statistics
    # --------------------------------------------------
    pressure_defined = deps_win.size > 0

    mean_pressure = float(np.mean(deps_win)) if pressure_defined else 0.0
    max_pressure  = float(np.max(deps_win))  if pressure_defined else 0.0
    rms_pressure  = float(np.sqrt(np.mean(deps_win**2))) if pressure_defined else 0.0

    mean_TEF = float(np.mean(TEF_win)) if TEF_win.size else 0.0
    max_TEF  = float(np.max(TEF_win))  if TEF_win.size else 0.0

    # --------------------------------------------------
    # Legacy compatibility flag
    # --------------------------------------------------
    # Always TRUE if visibility exists
    intensity_window_present = True

    # --------------------------------------------------
    # Report
    # --------------------------------------------------
    report.update({
        # REQUIRED legacy key
        "intensity_window_present": intensity_window_present,

        # New semantics
        "pressure_defined": pressure_defined,

        # Pressure statistics
        "pressure_mean": mean_pressure,
        "pressure_max": max_pressure,
        "pressure_rms": rms_pressure,

        # Supporting diagnostics
        "mean_TEF": mean_TEF,
        "max_TEF": max_TEF,
        "mean_NC": float(np.mean(NC_win)) if NC_win.size else 0.0,

        # Window info
        "iw_eta_start": float(eta[i0]),
        "iw_eta_end": float(eta[i1 - 1]),
        "iw_duration": float(eta[i1 - 1] - eta[i0]),
    })

    return report
