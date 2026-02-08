
# ============================================================
# certify_planck/oned/certify_weak_field.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module implements one–dimensional weak–field certification
# for the Unified Thermal Relativity Boltzmann Solver.
#
# Weak–field behavior is detected directly from causal history
# without invoking ordering identities, proper–time lag, or
# post–weak-field bookkeeping.
#
# This module enforces a strict separation of concerns:
# it tests for the *existence* of weak–field phenomena only,
# not their closure or interpretation.
#
# Architectural guarantees:
# --------------------------
# - Uses only recorded 1D causal history quantities
# - Performs no solver evolution or state mutation
# - Computes no TE_L, no n, and no ordering identities
# - Returns raw existence diagnostics only
#
# Governing role:
# ---------------
# - Detects gravitational time dilation (τ ≠ η)
# - Confirms presence of a free–energy source (TE_F)
# - Verifies nontrivial proper–time response
#
# This module detects weak–field physics; it does not finalize it.
#
# ============================================================

from typing import Dict, Any
import numpy as np


def _series(history: dict, key: str) -> np.ndarray:
    a = history.get(key, None)
    if a is None:
        return np.asarray([], dtype=float)
    return np.asarray(a, dtype=float)


def _pad(a: np.ndarray, N: int) -> np.ndarray:
    if a.size == N:
        return a
    out = np.zeros(N, dtype=float)
    out[:a.size] = a
    return out


def certify_weak_field_1d(
    history: dict,
    *,
    eps: float = 0.0,
) -> Dict[str, Any]:
    """
    Tier–WF Weak Field Certification (TR-native)

    STRICT RULE:
    - NO n calculation
    - NO TE_L calculation
    - NO identities

    ONLY real weak-field tests.
    """

    eta = _series(history, "eta")
    tau = _series(history, "tau")
    TE_F = _series(history, "TE_F")

    N = max(eta.size, tau.size, TE_F.size)
    if N == 0:
        return {"weak_field_defined": False}

    eta = _pad(eta, N)
    tau = _pad(tau, N)
    TE_F = _pad(TE_F, N)

    # --------------------------------------------------
    # Test 1: gravitational time dilation exists
    # --------------------------------------------------
    d_eta = np.zeros(N)
    d_tau = np.zeros(N)
    d_eta[1:] = eta[1:] - eta[:-1]
    d_tau[1:] = tau[1:] - tau[:-1]

    time_dilation_present = bool(np.any(np.abs(d_tau - d_eta) > eps))

    # --------------------------------------------------
    # Test 2: source exists (free energy / gas)
    # --------------------------------------------------
    source_present = bool(np.any(TE_F > eps))

    # --------------------------------------------------
    # Test 3: nontrivial τ-response (not noise)
    # --------------------------------------------------
    tau_response_present = bool(np.any(d_tau != 0.0))

    return {
        "weak_field_defined": True,
        "time_dilation_present": time_dilation_present,
        "source_present": source_present,
        "tau_response_present": tau_response_present,
    }
