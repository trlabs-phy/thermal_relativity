# ============================================================
# certify_planck/causal/certify_channels.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module implements one–dimensional channel and visibility
# coherence diagnostics for the Unified Thermal Relativity
# Boltzmann Solver.
#
# Channel identifiers and visibility flags are treated strictly
# as diagnostic bookkeeping fields. This module verifies that
# they remain binary and structurally consistent once interior
# volume is realized.
#
# This module is strictly causal-history–native and read-only.
# It introduces no dynamics, no geometry, and no solver control
# logic.
#
# Architectural guarantees:
# --------------------------
# - Uses only recorded 1D history quantities
# - Performs no solver evolution or state mutation
# - Applies no thresholds beyond numerical tolerance
# - Returns raw, boolean diagnostics only
#
# Governing role:
# ---------------
# - Verifies binary integrity of channel identifiers
# - Verifies binary integrity of visibility flags
# - Confirms channel/visibility consistency after structure onset
#
# This module enforces channel coherence; it does not create it.
#
# ============================================================

import numpy as np
from typing import Dict, Any

def certify_channels_1d(history: dict, *, tol: float = 1e-10) -> Dict[str, Any]:
    """
    Channel & visibility coherence after interior formation
    """

    V_int   = np.asarray(history["V_int"], dtype=float)
    ChanID  = np.asarray(history["ChanID"], dtype=int)
    VisFlag = np.asarray(history["VisFlag"], dtype=int)

    n = len(V_int)

    chan_binary = bool(np.all(np.isin(ChanID, [0, 1])))
    vis_binary  = bool(np.all(np.isin(VisFlag, [0, 1])))

    structural_ok = True
    for i in range(n):
        if V_int[i] > tol:
            if ChanID[i] != 1 or VisFlag[i] != 1:
                structural_ok = False
                break

    return {
        "chan_binary": chan_binary,
        "vis_binary": vis_binary,
        "channel_structural_consistency": structural_ok,
    }
