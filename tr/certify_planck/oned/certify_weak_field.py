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
