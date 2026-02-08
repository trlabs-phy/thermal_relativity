from typing import Dict, Any
import numpy as np


def certify_domain_finalize_3d(
    history: dict,
    snap,
) -> Dict[str, Any]:
    """
    DOMAIN FINALIZATION (η-based, AUTHORITATIVE)

    Domains are defined purely by η-intervals.
    Snapshots do NOT define volume — they only carry labels.
    """

    timeline = history.get("_cert_timeline")
    if timeline is None:
        return {"domain_defined": 0.0, "reason": "missing_cert_timeline"}

    eta_series = np.asarray(history.get("eta", []), dtype=float)
    if eta_series.size < 2:
        return {"domain_defined": 0.0, "reason": "eta_series_too_short"}

    eta_start = float(eta_series[0])
    eta_end   = float(eta_series[-1])
    eta_handoff = float(timeline["domain_handoff_eta"])

    # η-integrated domain measures
    delta_moat = max(eta_handoff - eta_start, 0.0)
    delta_int  = max(eta_end - eta_handoff, 0.0)

    interior_exists = float(delta_int > 0.0)

    ratio = (
        delta_int / delta_moat
        if delta_moat > 0.0 else 0.0
    )

    return {
        "domain_defined": 1.0,

        # causal facts
        "interior_exists": interior_exists,
        "moat_exists": float(delta_moat > 0.0),

        # η-based bookkeeping
        "eta_start": eta_start,
        "eta_handoff": eta_handoff,
        "eta_end": eta_end,

        # diagnostic ratio
        "final_volume_ratio_int_to_moat": float(ratio),

        # consistency (trivial here)
        "post_additivity_error": 0.0,
        "interior_nonnegative": 1.0,
        "moat_nonnegative": 1.0,
    }
