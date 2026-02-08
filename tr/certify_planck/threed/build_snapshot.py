"""
Translation layer: causal → spatial (3D realization)

Constructs one static 3D snapshot from a certified 1D causal history.

• NO dynamics
• NO evolution
• NO feedback
• NO interpretation

Pure embedding + bookkeeping.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np

# =====================================================
# Snapshot container
# =====================================================

@dataclass
class BubbleSnapshot:
    N: int
    dx: float

    # bubble + surface
    mask_bubble: np.ndarray    # bool (N,N,N)
    mask_surface: np.ndarray   # bool (N,N,N)

    # channel identity
    mask_eta: np.ndarray       # bool (N,N,N)
    mask_eta2: np.ndarray      # bool (N,N,N)

    # domains
    mask_int: np.ndarray       # bool (N,N,N)
    mask_moat: np.ndarray      # bool (N,N,N)

    # bookkeeping fields (float64 for certification stability)
    E_cell: np.ndarray         # float (N,N,N)
    C_conf_cell: np.ndarray    # float (N,N,N)

    # frozen imprint field
    escape_weight: np.ndarray  # float (N,N,N)

    # metadata
    meta: Dict[str, Any]


# =====================================================
# Geometry helpers (pure math)
# =====================================================

def _sphere_mask_and_r(N: int, R_index: float) -> Tuple[np.ndarray, np.ndarray]:
    c = (N - 1) / 2.0
    x = np.arange(N) - c
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)
    mask = (r <= R_index)
    return mask, r


def _surface_mask(mask: np.ndarray) -> np.ndarray:
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


def choose_snapshot_index_for_3d(history: dict) -> Tuple[int, str]:
    if history.get("VisIndex_anchor", None) is not None:
        return int(history["VisIndex_anchor"]), "visibility_anchor"
    if history.get("VisIndex_peak", None) is not None:
        return int(history["VisIndex_peak"]), "visibility_peak"
    if "VisFlag" in history:
        vis = np.where(np.asarray(history["VisFlag"], dtype=int) == 1)[0]
        if vis.size > 0:
            return int(vis[0]), "visibility_flag_first"
    # final fallback
    return max(len(history.get("V_univ", [])) - 1, 0), "final_state_fallback"


# =====================================================
# Snapshot builder
# =====================================================

def build_bubble_snapshot(
    history: dict,
    idx: Optional[int] = None,
    *,
    N: int = 96,
    TE_U_key: str = "TE_U",
) -> BubbleSnapshot:
    """
    Spatial snapshot at a single η-index.

    TR ontology:
      • Geometry only
      • No dynamics
      • No gas / pressure
      • No gradients
      • Wall imprint = TEₚ (η-eigenmode at freeze-out)
      • Cubes are born at the center and spiral outward (ordering only)
    """

    # --------------------------------------------------
    # Snapshot index + η
    # --------------------------------------------------
    if idx is None:
        idx, snap_reason = choose_snapshot_index_for_3d(history)
    else:
        snap_reason = "explicit_idx"

    eta_series = history.get("eta")
    eta = float(eta_series[idx]) if eta_series is not None else None

    V_univ = float(history["V_univ"][idx])
    TE_U   = float(history.get(TE_U_key, [0.0])[idx])

    # --------------------------------------------------
    # Geometry (pure math, cubic lattice)
    # --------------------------------------------------
    center = (N // 2, N // 2, N // 2)

    R_index = 0.45 * N
    mask_bubble, r = _sphere_mask_and_r(N, R_index)
    mask_surface   = _surface_mask(mask_bubble)

    bubble_indices = np.argwhere(mask_bubble)
    n_bubble_cells = bubble_indices.shape[0]

    dx = (V_univ / n_bubble_cells) ** (1/3) if V_univ > 0 else 0.0

    # --------------------------------------------------
    # Domains (neutral default)
    # --------------------------------------------------
    mask_int  = np.zeros_like(mask_bubble, dtype=bool)
    mask_moat = mask_bubble.copy()

    # --------------------------------------------------
    # Energy bookkeeping (unchanged, uniform)
    # --------------------------------------------------
    E_cell = np.zeros((N, N, N), dtype=np.float64)
    if n_bubble_cells > 0:
        E_cell[mask_bubble] = TE_U / n_bubble_cells

    # --------------------------------------------------
    # TEₚ WALL IMPRINT (authoritative observable)
    # --------------------------------------------------
    escape_weight = np.zeros((N, N, N), dtype=np.float64)

    timeline = history.get("_cert_timeline", {})
    i_freeze = timeline.get("domain_handoff_index")

    if i_freeze is None:
        raise RuntimeError("Cannot build TR snapshot: no domain handoff index")

    mode_series = history.get("N_C")
    if mode_series is None:
        raise RuntimeError("N_C eigenmode history missing")

    rho_freeze = float(mode_series[int(i_freeze)])

    # Stamp TEₚ uniformly on the bubble wall
    escape_weight[mask_surface] = rho_freeze

    # --------------------------------------------------
    # OPTIONAL: spiral / shell ordering diagnostic
    # (NOT physics, NOT used for causation)
    # --------------------------------------------------
    spiral_index = np.zeros((N, N, N), dtype=np.int32)
    if dx > 0.0:
        spiral_index[mask_bubble] = np.round(r[mask_bubble] / dx).astype(np.int32)

    # --------------------------------------------------
    # Metadata (honest, minimal, explicit)
    # --------------------------------------------------
    meta = {
        "idx": int(idx),
        "eta": eta,
        "snapshot_reason": snap_reason,
        "snapshot_role": "TR_TEp_FREEZE",

        # center = cube birth origin
        "center_index": center,
        "center_role": "cube_birth_origin",

        # freeze-out info
        "freeze_index": int(i_freeze),
        "freeze_eta": float(history["eta"][i_freeze]),
        "rho_freeze": rho_freeze,

        # geometry
        "V_univ": V_univ,
        "n_bubble_cells": int(n_bubble_cells),
        "dx": float(dx),
    }

    return BubbleSnapshot(
        N=N,
        dx=float(dx),

        mask_bubble=mask_bubble,
        mask_surface=mask_surface,

        # channel masks (unchanged)
        mask_eta=mask_bubble.copy(),
        mask_eta2=np.zeros_like(mask_bubble),

        mask_int=mask_int,
        mask_moat=mask_moat,

        E_cell=E_cell,
        C_conf_cell=np.zeros_like(E_cell),

        # observable carrier
        escape_weight=escape_weight,

        # diagnostics only
        spiral_index=spiral_index,

        meta=meta,
    )

def choose_structural_snapshot_index(
    history: dict,
    *,
    thermalus_name: str = "THERMALUS",
    fallback: str = "last",
) -> Tuple[int, str]:
    """
    Choose the snapshot index where interior structure is physically allowed.

    Priority:
      1) First Thermalus occurrence (TE classification)
      2) First matter existence
      3) Final state fallback
    """

    # ----------------------------------
    # 1) Thermalus latch (preferred)
    # ----------------------------------
    phases = history.get("_te_phases", None)
    if phases is not None:
        for i, p in enumerate(phases):
            if p == thermalus_name:
                return int(i), "thermalus_latch"

    # Numeric phase index fallback
    phase_idx = history.get("te_phase_index", None)
    phase_map = history.get("_te_phase_map", None)
    if phase_idx is not None and phase_map is not None:
        inv = {v: k for k, v in dict(phase_map).items()}
        if thermalus_name in inv:
            code = inv[thermalus_name]
            for i, c in enumerate(phase_idx):
                if c == code:
                    return int(i), "thermalus_latch"

    # ----------------------------------
    # 2) Matter existence fallback
    # ----------------------------------
    if "matter_exists" in history:
        for i, m in enumerate(history["matter_exists"]):
            if m:
                return int(i), "matter_onset"

    # ----------------------------------
    # 3) Final fallback
    # ----------------------------------
    if fallback == "last":
        return max(len(history.get("V_univ", [])) - 1, 0), "final_state_fallback"

    raise RuntimeError("Unable to determine structural snapshot index")

def build_structural_bubble_snapshot(
    history: dict,
    *,
    N: int = 96,
    thermalus_name: str = "THERMALUS",
) -> BubbleSnapshot:
    """
    Build the post-Thermalus / post-matter structural snapshot.

    This snapshot is used for:
      • domain finalization (V_int / V_moat)
      • interior existence
      • late-time structure diagnostics
    """

    idx, reason = choose_structural_snapshot_index(
        history,
        thermalus_name=thermalus_name,
    )

    snap = build_bubble_snapshot(
        history,
        idx,
        N=N,
    )

    # annotate clearly (no ambiguity later)
    snap.meta["structural_snapshot"] = True
    snap.meta["structural_reason"] = reason

    return snap

def choose_post_latch_snapshot_index(
    history: dict,
    *,
    latch_index: int,
    delta: int = 1,
) -> Tuple[int, str]:
    """
    Choose a snapshot index AFTER the matter / Thermalus latch.

    This snapshot is guaranteed to allow nonzero interior volume
    if the theory permits it.
    """

    V_univ = history.get("V_univ", [])
    if not V_univ:
        raise RuntimeError("Missing V_univ history")

    idx = min(latch_index + delta, len(V_univ) - 1)

    return int(idx), f"post_latch_plus_{delta}"

def build_interior_bubble_snapshot(
    history: dict,
    *,
    N: int = 96,
    thermalus_name: str = "THERMALUS",
    delta: int = 1,
) -> BubbleSnapshot:
    """
    Build the post-latch interior snapshot.

    Ontology:
      • Matter has already latched
      • Moat is frozen
      • Interior volume is allowed to be nonzero
      • Used ONLY for interior/domain structure diagnostics
    """

    # Step 1: find latch (authoritative)
    latch_idx, latch_reason = choose_structural_snapshot_index(
        history,
        thermalus_name=thermalus_name,
    )

    # Step 2: move forward causally
    idx, reason = choose_post_latch_snapshot_index(
        history,
        latch_index=latch_idx,
        delta=delta,
    )

    # --- REQUIRED: η stamping (authoritative causal coordinate)
    eta_series = history.get("eta")
    eta = float(eta_series[idx]) if eta_series is not None else None

    # Step 3: build snapshot
    snap = build_bubble_snapshot(
        history,
        idx,
        N=N,
    )

    # Step 4: ontology annotations (THIS MATTERS FOR THE PAPER)
    snap.meta.update({
        "snapshot_role": "INTERIOR_STRUCTURE",
        "snapshot_class": "POST_LATCH",
        "latch_index": int(latch_idx),
        "latch_reason": latch_reason,
        "post_latch_delta": int(delta),
        "structural_snapshot": True,
        "structural_reason": reason,
        "interior_exists_diagnostic": float(mask_int.any()),
    })

    return snap
