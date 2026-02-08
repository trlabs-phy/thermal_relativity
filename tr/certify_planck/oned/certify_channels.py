# certify_planck/causal/certify_channels.py
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
