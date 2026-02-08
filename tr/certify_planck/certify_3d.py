# ============================================================
# certify_planck/certify_3d.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module implements the three–dimensional certification
# suite for the Unified Thermal Relativity Boltzmann Solver.
#
# Three–dimensional certification evaluates whether a solver
# history that has already passed one–dimensional causal
# certification remains physically consistent when realized
# in full spatial geometry.
#
# All tests performed here are snapshot-based and non-causal.
# No time evolution, no solver dynamics, and no feedback into
# the history integrator are introduced at this stage.
#
# Causal authority:
# -----------------
# - All causal structure, ordering, and timelines originate
#   exclusively from certify_1d
# - The 3D certification operates only on certified history
#   and static spatial embeddings
#
# Architectural guarantees:
# --------------------------
# - No solver evolution or state mutation
# - No causal logic or ordering updates
# - No parameter tuning or corrective feedback
# - Deterministic, reproducible evaluation
#
# Governing role:
# ---------------
# - Realizes certified 1D history into static 3D snapshots
# - Validates strong-field energy bookkeeping
# - Certifies channel topology and matter localization
# - Verifies light bending and time dilation consistency
# - Tests rotational invariance and statistical isotropy
# - Confirms angular transfer, projection, and ℓ-window localization
# - Produces the authoritative 3D certification output
#
# This module validates spatial consistency; it does not
# introduce new physics or dynamics.
#
# ============================================================

from typing import Dict, Any, Tuple
import numpy as np

from utils.build_snapshot import build_bubble_snapshot
from .threed.certify_strong_field import certify_strong_field_from_snapshot
from .threed.certify_channels import certify_channels_from_snapshot, certify_eta_ordering_wall_contact
from .threed.certify_matter_spectrum import certify_matter_localization_spectrum
from .threed.certify_light_bending import certify_light_bending_3d
from .threed.certify_domain_finalize import certify_domain_finalize_3d

from .threed.certify_angular_modes import certify_angular_modes_3d
from .threed.certify_rotational_robustness import certify_rotational_robustness_3d
from .threed.certify_statistical_isotropy import certify_statistical_isotropy_3d

from .threed.certify_transfer_projection import certify_transfer_projection_3d
from .threed.certify_ell_window import certify_ell_window_3d

from utils.output import OutputCollector


def select_visibility_snapshot(history: dict) -> Tuple[int, str]:
    """
    Select a snapshot index for 3D spatial realization.

    Priority:
      1. Explicit visibility peak index
      2. First visibility-flagged index (if any)
      3. Final state (graceful fallback)

    Returns
    -------
    idx : int
        Snapshot index
    reason : str
        Human-readable selection reason
    """

    # 1. Explicit peak (strongest anchor)
    if "VisIndex_peak" in history and history["VisIndex_peak"] is not None:
        return int(history["VisIndex_peak"]), "visibility_peak"

    # 2. Visibility flag (if it actually fired)
    if "VisFlag" in history:
        vis = np.where(np.asarray(history["VisFlag"]) == 1)[0]
        if vis.size > 0:
            return int(vis[0]), "visibility_flag_first"

        # IMPORTANT: do NOT error here
        return len(history["V_univ"]) - 1, "no_visibility_flag_fallback"

    # 3. Last-resort fallback
    return len(history["V_univ"]) - 1, "final_state_fallback"


def certify_3d(history: dict, cert1d: dict, *, cfg) -> Dict[str, Any]:
    """
    3D certification (TR-native)

    • Projection + angular realization (no lattice)
    • Spatial / structural realization (snapshot-based)
    • No time evolution
    • No causal feedback
    """

    out = OutputCollector()

    # ==================================================
    # MASTER TIMELINE (AUTHORITATIVE — FROM 1D)
    # ==================================================
    timeline = cert1d.get("timeline", None)
    if timeline is None:
        raise RuntimeError("certify_3d requires timeline from certify_1d")

    # Canonical indices
    i_vis = int(timeline["visibility_index"])

    i_struct = int(
        timeline["domain_handoff_index"]
        if timeline["domain_handoff_index"] is not None
        else i_vis
    )

    # ----------------------------------
    # VISIBILITY SNAPSHOT (Planck anchor)
    # ----------------------------------
    snap_vis = build_bubble_snapshot(
        history,
        i_vis,
        N=getattr(cfg, "build_N", 96),
    )

    snap_vis.meta.update({
        "snapshot_id": "VIS",
        "snapshot_index": i_vis,
        "snapshot_role": "visibility",
    })

    out.add(
        group="snapshot",
        name="vis_snapshot_index",
        tr_value=float(i_vis),
        ref_value=None,
        ref_sigma=None,
    )

    # ----------------------------------
    # Strong-field boundary bookkeeping
    # ----------------------------------
    sf = certify_strong_field_from_snapshot(snap_vis)

    out.add(
        group="strong_field",
        name="energy_additivity_error",
        tr_value=sf["additivity_error"],
        ref_value=0.0,
        ref_sigma=sf["additivity_tolerance"],
    )

    out.add(
        group="strong_field",
        name="energy_meta_match_error",
        tr_value=sf["meta_match_error"],
        ref_value=0.0,
        ref_sigma=sf["meta_match_tolerance"],
    )

    out.add(
        group="strong_field",
        name="outward_neighbor_contacts",
        tr_value=sf["outward_neighbor_contacts"],
        ref_value=None,
        ref_sigma=None,
    )

    out.add(
        group="strong_field",
        name="wall_energy_nonnegative",
        tr_value=float(sf["wall_energy_nonnegative"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    # ----------------------------------
    # Channel structure (3D)
    # ----------------------------------
    ch = certify_channels_from_snapshot(snap_vis)

    out.add(
        group="channels",
        name="eta_exists",
        tr_value=float(ch["eta_exists"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    out.add(
        group="channels",
        name="eta2_subset_of_eta",
        tr_value=float(ch["eta2_subset_of_eta"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    out.add(
        group="channels",
        name="eta2_prohibited_outside",
        tr_value=float(ch["eta2_prohibited_outside"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    out.add(
        group="channels",
        name="eta2_contacts_wall",
        tr_value=float(ch["eta2_contacts_wall"]),
        ref_value=0.0,
        ref_sigma=None,
    )

    eta_wall = certify_eta_ordering_wall_contact(snap_vis)

    out.add(
        group="channels",
        name="eta_contacts_wall",
        tr_value=float(eta_wall["eta_contacts_wall"]),
        ref_value=None,      # informational
        ref_sigma=None,
    )

    # ----------------------------------
    # Matter localization spectrum (TR-native, channel-defined)
    # ----------------------------------
    phase_masks = {
        "MATTER_CHANNEL": snap_vis.mask_bubble & (~snap_vis.mask_eta2),
    }

    matter_report = certify_matter_localization_spectrum(
        snap_vis,
        phase_masks=phase_masks,
        expected_matter_like=["MATTER_CHANNEL"],
    )

    pr = matter_report["phases"]["MATTER_CHANNEL"]

    # --- Diagnostics (no ref; these are not "matches")
    out.add(
        group="matter_channel",
        name="cells",
        tr_value=pr["cells"],
        ref_value=None,
        ref_sigma=None,
    )

    out.add(
        group="matter_channel",
        name="interface_fraction",
        tr_value=pr["interface_fraction"],
        ref_value=None,          
        ref_sigma=None,
    )

    out.add(
        group="matter_channel",
        name="sigma_R",
        tr_value=pr["sigma_R"],
        ref_value=None,
        ref_sigma=None,
    )

    # --- Threshold / certification lines (this is the "story")
    min_if = float(matter_report["min_interface_frac"])
    if_val = float(pr["interface_fraction"]) if np.isfinite(pr["interface_fraction"]) else float("nan")

    out.add(
        group="matter_channel",
        name="interface_fraction_min_required",
        tr_value=min_if,
        ref_value=None,
        ref_sigma=None,
    )

    out.add(
        group="matter_channel",
        name="interface_fraction_margin",
        tr_value=(if_val - min_if) if np.isfinite(if_val) else float("nan"),
        ref_value=0.0,           # margin should be >= 0
        ref_sigma=None,
    )

    out.add(
        group="matter",
        name="channel_matter_present",
        tr_value=float(pr["cells"] >= matter_report["min_cells"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    out.add(
        group="matter",
        name="channel_interface_ok",
        tr_value=float(np.isfinite(if_val) and if_val >= min_if),
        ref_value=1.0,
        ref_sigma=None,
    )

    # ----------------------------------
    # Light bending (3D, weak-field)
    # ----------------------------------
    lb = certify_light_bending_3d(
        history,
        snap_vis,
        cfg=cfg,
    )

    out.add(
        group="light_bending",
        name="light_bending_defined",
        tr_value=float(lb["defined"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    # -----------------------------
    # GR numeric sanity check
    # -----------------------------
    out.add(
        group="light_bending",
        name="gr_closed_fractional_error",
        tr_value=float(lb["gr_closed_fractional_error"]),
        ref_value=0.0,
        ref_sigma=None,
    )

    out.add(
        group="light_bending",
        name="gr_numeric_ok",
        tr_value=float(lb["gr_ok"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    # -----------------------------
    # PR ↔ GR agreement (CRITICAL)
    # -----------------------------
    out.add(
        group="light_bending",
        name="pr_over_gr_ratio",
        tr_value=float(lb["pr_over_gr_ratio"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    out.add(
        group="light_bending",
        name="pr_matches_gr",
        tr_value=float(lb["pr_ok"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    # -----------------------------
    # Natural-angle diagnostics
    # (NO wall, NO observer)
    # -----------------------------
    out.add(
        group="light_bending",
        name="theta_midpoint",
        tr_value=float(lb["theta_mid"]),
        ref_value=None,     # informational
        ref_sigma=None,
    )

    out.add(
        group="light_bending",
        name="max_dtheta_dx",
        tr_value=float(lb["max_abs_dtheta_dx"]),
        ref_value=None,     # informational
        ref_sigma=None,
    )

    # -----------------------------
    # Time dilation consistency
    # -----------------------------
    out.add(
        group="light_bending",
        name="time_dilation_fractional_error",
        tr_value=float(lb["time_dilation_fractional_error"]),
        ref_value=0.0,
        ref_sigma=None,
    )

    out.add(
        group="light_bending",
        name="time_dilation_ok",
        tr_value=float(lb["time_dilation_ok"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    # ----------------------------------
    # Domain Finalization (3D ONLY)
    # ----------------------------------
    # Phases that are allowed to count as "matter" in TR
    expected_matter_like = [
        "THERMON",
        "PLASMA",
        "LIQUID",
        "SOLID",
        "THERMALUS",
        ]

    matter_report = certify_matter_localization_spectrum(
        snap_vis,
        phase_masks=phase_masks,
        expected_matter_like=expected_matter_like,
    )

    # ----------------------------------
    # STRUCTURAL SNAPSHOT (domains)
    # ----------------------------------
    timeline = history["_cert_timeline"]
    i_handoff = timeline["domain_handoff_index"]

    # move forward causally
    i_struct = min(i_handoff + 1, len(history["V_univ"]) - 1)

    snap_struct = build_bubble_snapshot(
        history,
        i_struct,
    )

    snap_struct.meta.update({
        "snapshot_role": "STRUCTURAL",
        "domain_snapshot_eta": history["eta"][i_struct],
    })


    # ----------------------------------
    # Stamp snapshot with authoritative η
    # ----------------------------------
    timeline = history["_cert_timeline"]

    # ----------------------------------
    # Structural snapshot index (AUTHORITATIVE)
    # ----------------------------------
    i_struct = min(
        timeline["domain_handoff_index"] + 1,
        timeline["final_index"],
    )

    if i_struct is None:
        raise RuntimeError("Domain handoff index not defined in certification timeline")

    i_struct = int(i_struct)

    snap_struct = build_bubble_snapshot(
        history,
        i_struct,
        N=getattr(cfg, "build_N", 96),
    )

    # ----------------------------------
    # Domain Finalization (AUTHORITATIVE)
    # ----------------------------------
    dom = certify_domain_finalize_3d(history, snap_struct)

    # REQUIRED: domain bookkeeping executed
    out.add(
        group="domains",
        name="domain_defined",
        tr_value=float(dom["domain_defined"]),
        ref_value=1.0,
    )

    if dom["domain_defined"]:

        # --- causal handoff η (from timeline / domain logic)
        out.add(
            group="domains",
            name="domain_handoff_eta",
            tr_value=float(dom["eta_handoff"]),
            ref_value=None,
            ref_sigma=None,
        )

        # --- snapshot realization η
        out.add(
            group="domains",
            name="domain_snapshot_eta",
            tr_value=float(snap_struct.meta["eta"]),
            ref_value=None,
            ref_sigma=None,
        )

        # --- interior existence (diagnostic, not required)
        out.add(
            group="domains",
            name="interior_exists",
            tr_value=float(dom["interior_exists"]),
            ref_value=None,
            ref_sigma=None,
        )

        # --- diagnostic ratio only
        out.add(
            group="domains_diagnostic",
            name="final_volume_ratio_int_to_moat",
            tr_value=float(dom["final_volume_ratio_int_to_moat"]),
            ref_value=None,
            ref_sigma=None,
        )


    # ----------------------------------
    # Angular mode existence
    # ----------------------------------
    am = certify_angular_modes_3d(history)

    out.add(
        group="projection_modes",
        name="angular_modes_defined",
        tr_value=float(am["modes_defined"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    if am.get("ell_max_supported") is not None:
        out.add(
            group="projection_modes",
            name="ell_max_supported",
            tr_value=float(am["ell_max_supported"]),
            ref_value=None,
            ref_sigma=None,
        )

    # ----------------------------------
    # Rotational robustness
    # ----------------------------------
    rot = certify_rotational_robustness_3d(history)

    out.add(
        group="rotation",
        name="rotation_fractional_variation",
        tr_value=rot["rotation_fractional_variation"],
        ref_value=0.0,                  # invariance target
        ref_sigma=rot["expected_sigma"] # tolerance only
    )

    out.add(
        group="rotation",
        name="num_rotations",
        tr_value=rot["num_rotations"],
        ref_value=None,
        ref_sigma=None,
    )

    # ----------------------------------
    # Statistical isotropy (projection consistency)
    # ----------------------------------
    iso = certify_statistical_isotropy_3d(history)

    out.add(
        group="isotropy",
        name="fractional_projection_variation",
        tr_value=float(iso["fractional_variation"]),
        ref_value=0.0,          # isotropy = no directional bias
        ref_sigma=iso.get("atol", 0.0),
    )

    # ----------------------------------
    # Transfer projection (TR-native)
    # ----------------------------------
    tp = certify_transfer_projection_3d(snap_vis)

    out.add(
        group="transfer",
        name="transfer_defined",
        tr_value=float(tp["transfer_defined"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    if tp["transfer_defined"]:
        out.add(
            group="transfer",
            name="projected_peak_ell",
            tr_value=tp["projected_peak_ell"],
            ref_value=None,          # diagnostic only
            ref_sigma=None,
        )

        out.add(
            group="transfer",
            name="ell_fractional_variation",
            tr_value=tp["ell_fractional_variation"],
            ref_value=0.0,           
            ref_sigma=1e-2,          
        )

    # ==================================================
    # MASTER TIMELINE (from 1D, authoritative)
    # ==================================================
    timeline = cert1d["timeline"]

    i_vis = int(timeline["visibility_index"])
    i_struct = int(
        timeline["domain_handoff_index"]
        if timeline["domain_handoff_index"] is not None
        else timeline["visibility_index"]
    )

    # ----------------------------------
    # ℓ-window localization
    # ----------------------------------
    lw = certify_ell_window_3d(
        history,
        acoustic_scale=cert1d["capacity"]["homogeneous_scaling_target"],
    )

    out.add(
        group="ell_window",
        name="window_fractional_width",
        tr_value=lw["window_fractional_width"],
        ref_value=None,
        ref_sigma=None,
    )

    out.add(
        group="ell_window",
        name="max_window_fractional_width",
        tr_value=lw["max_window_fractional_width"],
        ref_value=None,
        ref_sigma=None,
    )

    out.add(
        group="ell_window",
        name="ell_window_pass",
        tr_value=float(lw["ell_window_pass"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    timeline = cert1d["timeline"]

    i_vis = timeline["visibility_index"]
    i_struct = (
        timeline["domain_handoff_index"]
        if timeline["domain_handoff_index"] is not None
        else i_vis
    )

    return {
        # ----------------------------------
        # High-level structure
        # ----------------------------------
        "projection": {
            "angular_modes": am,
            "statistical_isotropy": iso,
            "transfer_projection": tp,
            "ell_window": lw,
            "rotational_robustness": rot,
        },

        # ----------------------------------
        # Snapshot provenance (MULTI-SNAPSHOT)
        # ----------------------------------
        "snapshots": {
            "visibility": {
                "index": snap_vis.meta.get("snapshot_index"),
                "role": snap_vis.meta.get("snapshot_role"),
                "reason": snap_vis.meta.get("snapshot_reason"),
                "N": snap_vis.N,
                "dx": snap_vis.dx,
            },
            "structural": {
                "index": snap_struct.meta.get("snapshot_index"),
                "role": snap_struct.meta.get("snapshot_role"),
                "reason": snap_struct.meta.get("snapshot_reason"),
                "N": snap_struct.N,
                "dx": snap_struct.dx,
            },
        },

        # ----------------------------------
        # Spatial diagnostics (raw, Tier–IV)
        # ----------------------------------
        "spatial": {
            "strong_field": sf,
            "matter_localization": matter_report,
            "channels": ch,
        },

        # ----------------------------------
        # Canonical output
        # ----------------------------------
        "output": out,
    }

