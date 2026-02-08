# tr/certify_planck/oned/certify_ordering_ledger.py
import numpy as np
from typing import Dict, Any

def certify_ordering_ledger_1d(history: dict) -> Dict[str, Any]:
    """
    PRE-WEAKFIELD ordering ledger seed.
    Stores only what is legitimately available before τ-response derivation.
    """
    eta = np.asarray(history.get("eta", []), dtype=float)
    TE_P = np.asarray(history.get("TE_P", []), dtype=float)  # optional

    N = int(max(eta.size, TE_P.size))
    if N == 0:
        history["_ordering_ledger"] = {"stage": "seed", "defined": False}
        return {"ledger_defined": False, "ordering_steps": 0}

    # pad safely
    def _pad(a):
        if a.size == N:
            return a
        out = np.zeros(N, dtype=float)
        out[:a.size] = a
        return out

    eta = _pad(eta)
    TE_P = _pad(TE_P)
    events = int(np.sum(eta[1:] != eta[:-1]))

    history["_ordering_ledger"] = {
        "stage": "seed",
        "defined": True,
        "eta": eta.tolist(),
        "TE_P": TE_P.tolist(),
        "events": events,
        # TE_F/TE_L/n intentionally absent here
    }

    return {
        "ledger_defined": True,
        "ordering_steps": N,
        "ordering_events": events,
    }
