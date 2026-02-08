# ============================================================
# certify_timeline.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module constructs the certification timeline for the
# Unified Thermal Relativity Boltzmann Solver.
#
# The certification timeline provides a unified, η-indexed
# temporal spine used exclusively for diagnostic reporting
# and certification summaries. It does not participate in
# solver evolution and introduces no causal influence.
#
# All quantities assembled here are derived from recorded
# solver history and certification outputs. Indices are
# cached for convenience only; the ordering coordinate η
# remains the sole authoritative causal parameter.
#
# This module is strictly non-causal.
#
# Architectural guarantees:
# --------------------------
# - Reads recorded history and certification outputs only
# - Performs no state mutation
# - Introduces no thresholds or solver control logic
# - Does not affect forward evolution in any way
#
# Governing role:
# ---------------
# - Synthesizes key certification events onto a single η timeline
# - Identifies visibility, thermal phase, and domain handoff markers
# - Provides a consistent temporal reference for reports and plots
#
# ============================================================

from typing import Dict, Any, Optional
import numpy as np


def build_certification_timeline(
    *,
    history: Dict[str, Any],
    te: Dict[str, Any],
    structure: Dict[str, Any],
    visibility: Dict[str, Any],
) -> Dict[str, Any]:
    """
    MASTER TEMPORAL SPINE (η-indexed).

    η is the ONLY causal coordinate.
    Indices are cached conveniences only.
    """

    # --------------------------------------------------
    # Required series
    # --------------------------------------------------
    eta = np.asarray(history.get("eta", []), dtype=float)
    V_univ = np.asarray(history.get("V_univ", []), dtype=float)

    if eta.size == 0 or V_univ.size == 0:
        raise RuntimeError("History must contain eta and V_univ")

    if eta.size != V_univ.size:
        raise RuntimeError("eta / V_univ length mismatch")

    n_steps = int(eta.size)

    timeline: Dict[str, Any] = {
        "n_steps": n_steps,
        "final_index": n_steps - 1,
        "final_eta": float(eta[-1]),
    }

    # ==================================================
    # Visibility (observational, NOT causal)
    # ==================================================
    vis_exists = bool(visibility.get("visibility_exists", False))

    vis_start = visibility.get("vis_start_index")
    vis_end   = visibility.get("vis_end_index")

    if history.get("VisIndex_anchor") is not None:
        i_vis = int(history["VisIndex_anchor"])
        reason = "visibility_anchor"
    elif history.get("VisIndex_peak") is not None:
        i_vis = int(history["VisIndex_peak"])
        reason = "visibility_peak"
    elif vis_exists and vis_start is not None and vis_end is not None:
        i_vis = int((int(vis_start) + int(vis_end)) // 2)
        reason = "visibility_window_mid"
    else:
        i_vis = None
        reason = "none"

    timeline.update({
        "visibility_exists": vis_exists,
        "visibility_index": i_vis,
        "visibility_eta": float(eta[i_vis]) if i_vis is not None else None,
        "visibility_reason": reason,
    })

    # ==================================================
    # Thermalus latch (AUTHORITATIVE)
    # ==================================================
    i_thermalus: Optional[int] = None

    phases = history.get("_te_phases")
    if phases is not None:
        for i, p in enumerate(phases):
            if p == "THERMALUS":
                i_thermalus = int(i)
                break

    if i_thermalus is None:
        phase_idx = history.get("te_phase_index")
        phase_map = history.get("_te_phase_map")
        if phase_idx is not None and phase_map is not None:
            inv = {v: k for k, v in dict(phase_map).items()}
            if "THERMALUS" in inv:
                code = int(inv["THERMALUS"])
                hits = np.where(np.asarray(phase_idx, dtype=int) == code)[0]
                if hits.size:
                    i_thermalus = int(hits[0])

    thermalus_exists = i_thermalus is not None

    timeline.update({
        "thermalus_exists": thermalus_exists,
        "thermalus_index": i_thermalus,
        "thermalus_eta": float(eta[i_thermalus]) if thermalus_exists else None,
    })

    # ==================================================
    # Domain handoff = Thermalus (ONLY)
    # ==================================================
    if thermalus_exists:
        i_handoff = i_thermalus
        timeline.update({
            "domain_handoff_index": i_handoff,
            "domain_handoff_eta": float(eta[i_handoff]),
            "domain_handoff_reason": "thermalus",
            "V_moat_frozen": float(V_univ[i_handoff]),
        })
    else:
        timeline.update({
            "domain_handoff_index": None,
            "domain_handoff_eta": None,
            "domain_handoff_reason": "none",
            "V_moat_frozen": float(V_univ[-1]),
        })

    # ==================================================
    # Matter onset (DIAGNOSTIC ONLY)
    # ==================================================
    if isinstance(structure, dict) and structure.get("matter_exists", False):
        i_matter = structure.get("first_matter_index")
        i_matter = int(i_matter) if i_matter is not None else None
    else:
        i_matter = None

    timeline.update({
        "matter_exists": i_matter is not None,
        "matter_index": i_matter,
        "matter_eta": float(eta[i_matter]) if i_matter is not None else None,
    })

    return timeline
