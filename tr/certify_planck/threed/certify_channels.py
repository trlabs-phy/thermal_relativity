# ============================================================
# certify_planck/threed/certify_channels.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module implements three–dimensional channel structure
# diagnostics for the Unified Thermal Relativity Boltzmann Solver.
#
# Channel structure is evaluated exclusively from static spatial
# snapshots derived from a certified causal history. No dynamics,
# transport, or spacetime assumptions are introduced.
#
# Channels are treated as topological and bookkeeping constructs,
# not physical fields or forces. All results reported here are
# numeric diagnostics only and carry no pass/fail authority.
#
# This module is strictly non-causal and read-only with respect
# to solver evolution.
#
# Architectural guarantees:
# --------------------------
# - Uses only static 3D snapshot data
# - Performs no solver evolution or state mutation
# - Introduces no transport, GR, or spacetime concepts
# - Reports structural observables without interpretation
#
# Governing role:
# ---------------
# - Verifies existence and topology of η and η² channels
# - Confirms subset and containment relations
# - Diagnoses wall-contact adjacency for ordering channels
# - Provides snapshot-level channel structure observables
#
# Channels are structural labels, not dynamical entities.
#
# ============================================================

import numpy as np
from typing import Dict, Any

def certify_eta_ordering_wall_contact(
    snap,
) -> dict:
    """
    Diagnostic: does η-ordering reach the Primordium wall?

    This certifies *ordering adjacency*, not storage or confinement.

    • η may touch the wall
    • η² must not
    • Informational only (never fails)
    """

    # Wall geometry
    wall = snap.mask_surface.astype(bool)

    # η exists wherever ordering exists inside the bubble
    # (ordering fills the activated lattice)
    eta_region = snap.mask_bubble.astype(bool)

    # Adjacency test: η region touching wall
    eta_contacts_wall = bool(np.any(eta_region & wall))

    return {
        "eta_contacts_wall": eta_contacts_wall,
        "interpretation": (
            "η-ordering reaches wall (allowed)"
            if eta_contacts_wall
            else "η-ordering does not reach wall"
        ),
    }

# =====================================================
# Helper
# =====================================================

def _has_outside_neighbor(
    mask_inside: np.ndarray,
    region: np.ndarray,
) -> bool:
    """
    Returns True if any cell in `region` has a 6-neighbor outside `mask_inside`.
    """
    inside = mask_inside.astype(bool)
    reg = region.astype(bool)

    shifts = [
        np.roll(inside,  1, axis=0), np.roll(inside, -1, axis=0),
        np.roll(inside,  1, axis=1), np.roll(inside, -1, axis=1),
        np.roll(inside,  1, axis=2), np.roll(inside, -1, axis=2),
    ]

    for nb in shifts:
        if np.any(reg & (~nb)):
            return True
    return False


# =====================================================
# Certification (new-style)
# =====================================================
def certify_channels_from_snapshot(snap) -> Dict[str, Any]:
    """
    Channel structure certification (TR-native)

    η  : out-channel (exists throughout bubble, touches wall)
    η² : in-channel  (moat shell, touches wall, subset of η)

    Structural only. No dynamics.
    """

    # --------------------------------------------------
    # Required masks (UPDATED)
    # --------------------------------------------------
    for name in ["mask_eta", "mask_eta2", "mask_bubble", "mask_surface"]:
        if not hasattr(snap, name):
            raise AttributeError(f"Snapshot missing required mask: {name}")

    eta   = np.asarray(snap.mask_eta, dtype=bool)
    eta2  = np.asarray(snap.mask_eta2, dtype=bool)
    bub   = np.asarray(snap.mask_bubble, dtype=bool)
    surf  = np.asarray(snap.mask_surface, dtype=bool)

    # --------------------------------------------------
    # Channel existence
    # --------------------------------------------------
    eta_exists  = bool(np.any(eta))
    eta2_exists = bool(np.any(eta2))

    # --------------------------------------------------
    # Structural relations
    # --------------------------------------------------
    eta2_subset_of_eta = bool(np.all(eta2 <= eta))
    eta2_prohibited_outside = bool(np.all(~eta2 | bub))

    # --------------------------------------------------
    # Wall contact (THIS is the key ontology point)
    # --------------------------------------------------
    eta_contacts_wall  = bool(np.any(eta  & surf))
    eta2_contacts_wall = bool(np.any(eta2 & surf))

    return {
        "eta_exists": float(eta_exists),
        "eta2_exists": float(eta2_exists),

        "eta2_subset_of_eta": float(eta2_subset_of_eta),
        "eta2_prohibited_outside": float(eta2_prohibited_outside),

        # IMPORTANT: these are NOT pass/fail physics claims
        "eta_contacts_wall": float(eta_contacts_wall),
        "eta2_contacts_wall": float(eta2_contacts_wall),
    }
