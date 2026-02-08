import numpy as np
from typing import Dict, Any


def certify_ordering_ledger_finalize_1d(
    history: dict,
    *,
    tol_sum: float = 1e-12,
) -> Dict[str, Any]:
    """
    POST-WEAK-FIELD: Proper Time & Ordering Finalization

    This is the ONLY place where:
    - TE_L is computed
    - n is computed
    - n = TE_F + TE_L is certified
    """

    eta = np.asarray(history.get("eta", []), dtype=float)
    tau = np.asarray(history.get("tau", []), dtype=float)
    TE_F = np.asarray(history.get("TE_F", []), dtype=float)

    N = max(eta.size, tau.size, TE_F.size)
    if N == 0:
        return {"ledger_defined": False}

    def _pad(a):
        if a.size == N:
            return a
        out = np.zeros(N, dtype=float)
        out[:a.size] = a
        return out

    eta, tau, TE_F = map(_pad, (eta, tau, TE_F))

    # --------------------------------------------------
    # Proper time response
    # --------------------------------------------------
    d_tau = np.zeros(N)
    d_tau[1:] = tau[1:] - tau[:-1]

    # --------------------------------------------------
    # Proper-time lag (PHYSICS)
    # --------------------------------------------------
    TE_L = np.maximum(0.0, TE_F - d_tau)

    # --------------------------------------------------
    # Ordering rate
    # --------------------------------------------------
    n = TE_F + TE_L

    history["TE_L"] = TE_L.tolist()
    history["n"] = n.tolist()

    # --------------------------------------------------
    # Identity check
    # --------------------------------------------------
    mismatch = np.abs((TE_F + TE_L) - n)
    max_mismatch = float(np.max(mismatch)) if mismatch.size else 0.0
    identity_ok = bool(max_mismatch <= tol_sum)

    return {
        "ledger_defined": True,
        "ordering_release_total": float(np.sum(TE_F)),
        "ordering_lag_total": float(np.sum(TE_L)),
        "n_total": float(np.sum(n)),
        "ordering_sum_identity_ok": float(identity_ok),
        "ordering_max_sum_mismatch": float(max_mismatch),
    }
