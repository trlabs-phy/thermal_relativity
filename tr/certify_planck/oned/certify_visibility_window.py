# tr/certify_planck/oned/certify_visibility_window.py

from typing import Dict, Any
import numpy as np


def certify_visibility_window_1d(
    history: dict,
    *,
    eta_start_value: float = 1.0,
    window: int = 256,
    eps_te_frac: float = 1e-8,     # TE_F threshold as fraction of kappa_C
    eps_eps_step: float = 1e-12,   # epsilon plateau threshold per step
) -> Dict[str, Any]:
    """
    TR-native visibility window.

    Meaning:
      Visibility exists while the system is actively producing *impressions*,
      i.e. while κC chase / cube creation / free-energy forcing is nontrivial.

    Start:
      first index where eta >= eta_start_value

    End:
      first index after start where, for `window` consecutive steps:
        - N_C is constant (no new cubes)
        - epsilon is plateaued (no κC chase in-flight)
        - TE_F is negligible (no forcing)
    """

    report: Dict[str, Any] = {}

    eta = np.asarray(history.get("eta", []), dtype=float)
    if eta.size == 0:
        return {"visibility_exists": False}

    N_C = np.asarray(history.get("N_C", []), dtype=float)
    TE_F = np.asarray(history.get("TE_F", []), dtype=float)
    eps = np.asarray(history.get("epsilon", []), dtype=float)

    if not (len(N_C) == len(eta) == len(TE_F) == len(eps)):
        raise ValueError("visibility cert requires eta, N_C, TE_F, epsilon of equal length")

    # kappa_C from struct if available, else assume 1.0
    kappa_C = float(history.get("_struct", {}).get("kappa_C", 1.0))
    te_thresh = eps_te_frac * max(kappa_C, 1e-30)

    # -----------------------------
    # Start index (eta >= 1)
    # -----------------------------
    start_candidates = np.where(eta >= eta_start_value)[0]
    if start_candidates.size == 0:
        return {
            "visibility_exists": False,
            "reason": "eta never reaches start value",
        }

    i_start = int(start_candidates[0])

    # -----------------------------
    # End detector
    # -----------------------------
    i_end = None
    n = len(eta)

    # need at least one full window after start
    if i_start + window >= n:
        return {
            "visibility_exists": False,
            "vis_start_index": i_start,
            "reason": "history too short for windowed end detection",
        }

    # precompute diffs
    d_eps = np.abs(np.diff(eps, prepend=eps[0]))

    for i in range(i_start + window, n):
        w0 = i - window
        w1 = i

        Nw = N_C[w0:w1]
        TEw = TE_F[w0:w1]
        dEw = d_eps[w0:w1]

        # (1) no cube creation
        cubes_quiet = (np.max(Nw) - np.min(Nw)) == 0.0

        # (2) epsilon plateau
        eps_quiet = np.max(dEw) < eps_eps_step

        # (3) no forcing
        te_quiet = np.max(TEw) < te_thresh

        if cubes_quiet and eps_quiet and te_quiet:
            i_end = i
            break

    eta_start = float(eta[i_start])
    eta_end   = float(eta[i_end])

    visibility_duration = eta_end - eta_start

    history["visibility_duration"] = visibility_duration

    if i_end is None:
        return {
            "visibility_exists": False,
            "vis_start_index": i_start,
            "reason": "no freeze-out detected",
        }

    return {
        "visibility_exists": True,
        "vis_start_index": i_start,
        "vis_end_index": int(i_end),
        "vis_start_eta": float(eta[i_start]),
        "vis_end_eta": float(eta[i_end]),
        "visibility_duration": float(eta[i_end] - eta[i_start]),

        # diagnostics
        "kappa_C": kappa_C,
        "TEF_threshold": float(te_thresh),
        "window": int(window),
    }
