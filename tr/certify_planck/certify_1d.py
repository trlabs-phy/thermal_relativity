# tr/certify_planck/certify_1d.py

from typing import Dict, Any

from utils.certify_timeline import build_certification_timeline
from .oned.certify_ordering_ledger import certify_ordering_ledger_1d
from .oned.certify_weak_field import certify_weak_field_1d
from .oned.certify_ordering_ledger_finalize import certify_ordering_ledger_finalize_1d
from .oned.certify_capacity import certify_capacity_1d
from .oned.certify_inflation import certify_inflation_1d
from .oned.certify_domain import certify_domain_1d
from .oned.certify_modes import certify_modes_1d
from .oned.certify_te_classification import certify_te_classification_1d
from .oned.certify_structure import certify_structure_1d 
from .oned.certify_channels import certify_channels_1d
from .oned.certify_visibility_window import certify_visibility_window_1d
from utils.output import OutputCollector


def certify_1d(history: dict, *, cfg) -> Dict[str, Any]:
    """
    1D causal certification (Eq. 0 only)

    • No geometry
    • No sky
    • No structure
    """

    out = OutputCollector()

    # ----------------------------------
    # Ordering ledger (SEED, PRE–WEAKFIELD)
    # ----------------------------------
    ledger_seed = certify_ordering_ledger_1d(history)

    out.add(
        group="ordering",
        name="ledger_defined",
        tr_value=float(ledger_seed["ledger_defined"]),
        ref_value=1.0,
    )

    out.add(
        group="ordering",
        name="ordering_steps",
        tr_value=ledger_seed["ordering_steps"],
    )

    out.add(
        group="ordering",
        name="ordering_events",
        tr_value=ledger_seed["ordering_events"],
    )

    # --------------------------------------------------
    # Capacity bookkeeping (Eq. 0)
    # --------------------------------------------------
    cap = certify_capacity_1d(
        history,
        kappa_C=cfg.kappa_C,
    )

    # --------------------------------------------------
    # Record observables (numeric, transparent)
    # --------------------------------------------------
    out.add(
        group="capacity",
        name="max_fill_chase_residual",
        tr_value=cap["capacity_closure_max_residual"],
        ref_value=0.0,
        ref_sigma=1e-9,
    )

    # ----------------------------------
    # Capacity observables
    # ----------------------------------
    out.add(
        group="capacity",
        name="median_volume_scaling_exponent",
        tr_value=cap["homogeneous_scaling_median"],
        ref_value=cap["homogeneous_scaling_target"],
        ref_sigma=1e-3,
    )

    # ----------------------------------
    # Capacity scaling plot
    # ----------------------------------
    cap = certify_capacity_1d(history, kappa_C=cfg.kappa_C)

    out.plot_capacity_scaling(
        eta=cap["scaling_eta"],
        slopes=cap["scaling_slopes"],
        outdir="output/certify_planck",
        filename="capacity_scaling_1d.pdf",
    )

    # -----------------------------
    # Weak-field diagnostics (Eq. 0 regime)
    # -----------------------------
    wf = {
        "ordering_active": cap["ordering_active"],
        "free_energy_persists": cap["free_energy_persists"],
        "first_TE_F_eta": cap["first_TE_F_eta"],
    }

    out.add(
        group="weak_field",
        name="ordering_active",
        tr_value=float(cap["ordering_active"]),
        ref_value=1.0,          # expected TRUE
        ref_sigma=None,         # binary diagnostic
    )

    out.add(
        group="weak_field",
        name="free_energy_persists",
        tr_value=float(cap["free_energy_persists"]),
        ref_value=1.0,          # expected TRUE
        ref_sigma=None,
    )

    # ----------------------------------
    # Proper Time / Ordering Ledger FINAL
    # ----------------------------------
    ledger_final = certify_ordering_ledger_finalize_1d(history)

    out.add(
        group="proper_time",
        name="ordering_release_total",
        tr_value=ledger_final["ordering_release_total"],
    )

    out.add(
        group="proper_time",
        name="ordering_lag_total",
        tr_value=ledger_final["ordering_lag_total"],
    )

    out.add(
        group="proper_time",
        name="ordering_sum_identity_ok",
        tr_value=float(ledger_final["ordering_sum_identity_ok"]),
        ref_value=1.0,
    )

    out.add(
        group="proper_time",
        name="ordering_max_sum_mismatch",
        tr_value=ledger_final["ordering_max_sum_mismatch"],
        ref_value=0.0,
    )

    # --------------------------------------------------
    # Domains (moat / interior identity)
    # --------------------------------------------------
    dom = certify_domain_1d(history)

    out.add(
        group="domains",
        name="pre_interior_additivity_error",
        tr_value=dom["pre_interior_additivity_error"],
        ref_value=0.0,
        ref_sigma=1e-10,
    )

    out.add(
        group="domains",
        name="max_abs_pre_interior_volume",
        tr_value=dom["max_abs_pre_interior_volume"],
        ref_value=0.0,
        ref_sigma=1e-10,
    )

    out.add(
        group="domains",
        name="max_abs_moat_minus_volume",
        tr_value=dom["max_abs_moat_minus_volume"],
        ref_value=0.0,
        ref_sigma=1e-10,
    )

    # ----------------------------------
    # Inflation observables
    # ----------------------------------
    infl = certify_inflation_1d(history)

    out.add(
        group="inflation",
        name="inflation_end_detected",
        tr_value=float(infl["inflation_end_detected"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    if infl["inflation_end_eta"] is not None:
        out.add(
            group="inflation",
            name="inflation_end_eta",
            tr_value=infl["inflation_end_eta"],
        )

    # -----------------------------
    # Mode certification (NEW STEP)
    # -----------------------------
    modes = certify_modes_1d(history)

    out.add(
        group="modes",
        name="eigenmode_defined",
        tr_value=float(modes["eig_defined"]),
        ref_value=1.0,
    )

    out.add(
        group="modes",
        name="eigenmode_monotone",
        tr_value=float(modes["eig_monotone"]),
        ref_value=1.0,
    )

    out.add(
        group="modes",
        name="num_mode_transitions",
        tr_value=modes["num_mode_transitions"],
        ref_value=None,   # diagnostic, not a target
    )

    # ----------------------------------
    # TE Classification
    # ----------------------------------
    te = certify_te_classification_1d(
        history,
        modes=modes,
    )

    out.add(
        group="te_classification",
        name="num_te_phase_transitions",
        tr_value=te["num_phase_transitions"],
    )

    out.add(
        group="te_classification",
        name="final_te_phase_index",
        tr_value=float(
            ["THERMON","PLASMA","LIQUID","SOLID","THERMALUS"].index(
                te["final_phase"]
            )
        ),
    )

    # ----------------------------------
    # STRUCTURE observables
    # ----------------------------------
    struct = certify_structure_1d(history)
    
    out.add(
        group="structure",
        name="matter_exists",
        tr_value=float(struct["matter_exists"]),
        ref_value=1.0,   # ← UPDATED expectation
        ref_sigma=None,
    )

    if struct["first_matter_eta"] is not None:
        out.add(
            group="structure",
            name="first_matter_eta",
            tr_value=struct["first_matter_eta"],
            ref_value=None,
            ref_sigma=None,
        )

    out.add(
        group="structure",
        name="max_volume_additivity_error",
        tr_value=struct["max_additivity_error"],
        ref_value=0.0,
        ref_sigma=1e-10,
    )

    out.add(
        group="structure",
        name="interior_bounds_ok",
        tr_value=float(struct["bounds_ok"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    out.add(
        group="structure",
        name="min_dVint",
        tr_value=struct["min_dVint"],
        ref_value=0.0,
        ref_sigma=None,
    )

    # ----------------------------------
    # Channel coherence observables
    # ----------------------------------
    channels = certify_channels_1d(history)

    out.add(
        group="channels",
        name="channel_id_binary",
        tr_value=float(channels["chan_binary"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    out.add(
        group="channels",
        name="visibility_flag_binary",
        tr_value=float(channels["vis_binary"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    out.add(
        group="channels",
        name="channel_structural_consistency",
        tr_value=float(channels["channel_structural_consistency"]),
        ref_value=1.0,
        ref_sigma=None,
    )


    # ----------------------------------
    # Visibility window (causal observability gate)
    # ----------------------------------
    vis = certify_visibility_window_1d(history)

    out.add(
        group="visibility",
        name="visibility_exists",
        tr_value=float(vis["visibility_exists"]),
        ref_value=1.0,
        ref_sigma=None,
    )

    if vis["visibility_exists"]:
        out.add(
            group="visibility",
            name="visibility_duration",
            tr_value=vis["visibility_duration"],
            ref_value=None,
            ref_sigma=None,
        )

    if vis["visibility_exists"]:
        history["vis_start_index"] = int(vis["vis_start_index"])
        history["vis_end_index"]   = int(vis["vis_end_index"])
        history["VisIndex_anchor"] = int((vis["vis_start_index"] + vis["vis_end_index"]) // 2)


    # ==================================================
    # MASTER CERTIFICATION TIMELINE (AUTHORITATIVE)
    # ==================================================
    timeline = build_certification_timeline(
        history=history,
        te=te,
        structure=struct,
        visibility=vis,
    )

    # Persist for downstream consumers (3D + observations)
    history["_cert_timeline"] = timeline

    # --------------------------------------------------
    # Emit timeline into output (AUTHORITATIVE)
    # --------------------------------------------------
    out.report_timeline(timeline)


    return {
        "capacity": cap,
        "domains": dom,
        "weak_field": wf,
        "inflation": infl,
        "modes": modes,
        "te_classification": te,
        "te": te,
        "structure": struct,
        "channels": channels,
        "visibility": vis,
        "timeline": timeline,

        "output": out,
    }
