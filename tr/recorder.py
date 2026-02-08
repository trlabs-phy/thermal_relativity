# recorder.py
"""
History Recorder (Non-Causal)

• Observes HistoryState ONLY
• No mutation
• No thresholds
• No classification
• No solver control logic

This module is pure bookkeeping.
"""

from collections import defaultdict
from typing import Dict, List
from state import HistoryState


# ============================================================
# Recorder initialization
# ============================================================

def init_recorder(cfg=None) -> Dict[str, List[float]]:
    rec = defaultdict(list)

    # --------------------------------------------------
    # Tier–0 structural constants (NON-EVOLVING)
    # --------------------------------------------------
    if cfg is not None:
        rec["_struct"] = {
            "c_AT": float(cfg.c_AT),
            "d_eta": float(cfg.d_eta),
            "kappa_C": float(cfg.kappa_C),
        }

    # Ordering
    rec["eta"]

    # Exposure / Primordium
    rec["V_AT"]
    rec["V_prim"]

    # Budget
    rec["B_T"]

    # Thermal energy bookkeeping
    rec["TE_P"]
    rec["TE_U"]
    rec["TE_F"]
    rec["epsilon"]

    # Cube / volume bookkeeping
    rec["N_C"]
    rec["V_univ"]

    # Mobility
    rec["mu"]

    return rec


# ============================================================
# Record one step
# ============================================================

def record_step(rec: Dict[str, List[float]], state: HistoryState) -> None:
    """
    Record one solver step.

    STRICT RULES:
    • Read-only
    • Append exactly once per key
    • No derived thresholds
    """

    rec["eta"].append(state.eta)

    rec["V_AT"].append(state.V_AT)
    rec["V_prim"].append(state.V_prim)

    rec["B_T"].append(state.B_T)

    rec["TE_P"].append(state.TE_P)
    rec["TE_U"].append(state.TE_U)
    rec["TE_F"].append(state.TE_F)
    rec["epsilon"].append(state.epsilon)

    rec["N_C"].append(state.N_C)
    rec["V_univ"].append(state.V_univ)

    rec["mu"].append(state.mu)
