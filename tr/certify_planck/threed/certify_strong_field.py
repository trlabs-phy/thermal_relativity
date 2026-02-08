"""
Strong-field (boundary) diagnostics — TR-native certification

Structural, snapshot-based only.

• NO dynamics
• NO transport
• NO GR
• NO spacetime assumptions

Pure geometry + bookkeeping observables.
"""

from typing import Dict, Any
import numpy as np


# =====================================================
# Geometry helpers (pure math)
# =====================================================

def _surface_mask(mask: np.ndarray) -> np.ndarray:
    """
    Identify surface voxels using 6-neighbor topology.
    """
    inside = mask.astype(bool)

    neighbors = [
        np.roll(inside,  1, axis=0), np.roll(inside, -1, axis=0),
        np.roll(inside,  1, axis=1), np.roll(inside, -1, axis=1),
        np.roll(inside,  1, axis=2), np.roll(inside, -1, axis=2),
    ]

    interior = inside.copy()
    for n in neighbors:
        interior &= n

    return inside & (~interior)


def _count_outward_contacts(mask_bubble: np.ndarray) -> int:
    """
    Count surface-to-outside neighbor contacts (6-connectivity).

    Measures *possible* outward channels, not active ones.
    """
    inside = mask_bubble.astype(bool)
    surface = _surface_mask(inside)

    shifts = [
        np.roll(inside,  1, axis=0), np.roll(inside, -1, axis=0),
        np.roll(inside,  1, axis=1), np.roll(inside, -1, axis=1),
        np.roll(inside,  1, axis=2), np.roll(inside, -1, axis=2),
    ]

    contacts = 0
    for nb in shifts:
        contacts += np.count_nonzero(surface & (~nb))

    return int(contacts)


# =====================================================
# Certification (numeric + diagnostic only)
# =====================================================

def certify_strong_field_from_snapshot(
    snap,
    *,
    atol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Strong-field (boundary) diagnostics from a spatial snapshot.

    Expected snapshot fields
    ------------------------
    mask_bubble : bool (N,N,N)
    mask_conf   : optional bool (N,N,N)
    E_cell      : float (N,N,N)
    meta        : dict with provenance scalars
    """

    # -----------------------------
    # Sanity
    # -----------------------------
    mb = np.asarray(snap.mask_bubble, dtype=bool)
    Ec = np.asarray(snap.E_cell, dtype=float)

    if mb.shape != Ec.shape:
        raise ValueError("Snapshot arrays have incompatible shapes")

    surface = _surface_mask(mb)

    # -----------------------------
    # SF-0: Energy bookkeeping
    # -----------------------------
    E_total_realized = float(np.sum(Ec[mb]))
    E_wall = float(np.sum(Ec[surface]))
    E_bulk = float(np.sum(Ec[mb & (~surface)]))

    E_meta = float(
        snap.meta.get("TE_U_input", E_total_realized)
    )

    additivity_error = abs((E_wall + E_bulk) - E_total_realized)
    meta_match_error = abs(E_total_realized - E_meta)

    # Optional confined region
    if hasattr(snap, "mask_conf") and snap.mask_conf is not None:
        mc = np.asarray(snap.mask_conf, dtype=bool)
        E_confined = float(np.sum(Ec[mc]))
    else:
        E_confined = None

    # -----------------------------
    # SF-1: Boundary channel geometry
    # -----------------------------
    outward_contacts = _count_outward_contacts(mb)

    # -----------------------------
    # SF-2: Wall sanity
    # -----------------------------
    wall_nonnegative = bool(E_wall >= -atol)
    wall_le_total = bool(E_wall <= E_total_realized + atol)

    # -----------------------------
    # Return diagnostics
    # -----------------------------
    return {
        # -------------------------
        # Canonical bookkeeping
        # -------------------------
        "additivity_error": additivity_error,
        "meta_match_error": meta_match_error,

        # Aliases for existing pipeline
        "energy_additivity_error": additivity_error,
        "energy_meta_match_error": meta_match_error,

        # Tolerances
        "additivity_tolerance": atol,
        "meta_match_tolerance": atol,

        # Boundary structure
        "outward_neighbor_contacts": outward_contacts,
        "requires_outward_channel_gating": True,

        # Wall sanity
        "wall_energy_nonnegative": wall_nonnegative,
        "wall_energy_le_total": wall_le_total,

        # Diagnostics (non-gating)
        "E_total_realized": E_total_realized,
        "E_total_input": E_meta,
        "E_wall": E_wall,
        "E_bulk": E_bulk,
        "E_confined": E_confined,

        # Snapshot echo
        "dx": snap.dx,
        "N": snap.N,
    }
