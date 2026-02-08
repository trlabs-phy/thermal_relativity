# tr/certify_planck/threed/certify_angular_modes.py

from typing import Dict, Any
import numpy as np


def certify_angular_modes_3d(history: dict) -> Dict[str, Any]:
    """
    Angular mode certification (3D projection, TR-native)

    • No sky
    • No pixels
    • No spherical harmonics
    • Purely checks that angular modes are *defined and supported*
      by the causal + volume history.
    """

    # -----------------------------
    # Required inputs
    # -----------------------------
    if "V_univ" not in history:
        raise ValueError("Angular modes require V_univ history")

    if "eta" not in history:
        raise ValueError("Angular modes require eta history")

    V = np.asarray(history["V_univ"], dtype=float)
    eta = np.asarray(history["eta"], dtype=float)

    # -----------------------------
    # Basic sanity
    # -----------------------------
    finite_volume = np.all(np.isfinite(V)) and np.all(V > 0.0)
    monotone_volume = np.all(np.diff(V) >= 0.0)

    # -----------------------------
    # Effective radius proxy
    # -----------------------------
    # We never assume Euclidean geometry, but angular support
    # requires a characteristic size scale.
    #
    # Use volume-equivalent radius as a *projection proxy only*.
    #
    # R_eff ∝ V^(1/3)
    R_eff = V ** (1.0 / 3.0)

    finite_radius = np.all(np.isfinite(R_eff)) and np.all(R_eff > 0.0)

    # -----------------------------
    # Acoustic reach proxy
    # -----------------------------
    # We only need to know whether *some* finite causal scale exists.
    # If the user already certified acoustic scale earlier, this
    # just checks that it did not collapse or diverge.
    if "r_acoustic" in history:
        r_s = np.asarray(history["r_acoustic"], dtype=float)
        acoustic_finite = np.all(np.isfinite(r_s)) and np.any(r_s > 0.0)
        r_s_eff = np.nanmax(r_s)
    else:
        # fallback: treat as diagnostic only
        acoustic_finite = False
        r_s_eff = None

    # -----------------------------
    # ℓ support estimate
    # -----------------------------
    # ℓ_max is not a measurement here — it is a *support bound*.
    #
    # ℓ_max ~ π * (R_eff / r_s)
    if r_s_eff is not None and r_s_eff > 0.0:
        ell_max_supported = float(np.pi * (R_eff[-1] / r_s_eff))
    else:
        ell_max_supported = None

    # -----------------------------
    # Mode definition logic
    # -----------------------------
    modes_defined = bool(
        finite_volume
        and monotone_volume
        and finite_radius
        and (ell_max_supported is None or ell_max_supported > 1.0)
    )

    # -----------------------------
    # Diagnostics
    # -----------------------------
    return {
        # Core definition
        "modes_defined": modes_defined,

        # Volume / geometry proxies
        "finite_volume": finite_volume,
        "monotone_volume": monotone_volume,
        "finite_radius": finite_radius,

        # Acoustic coupling
        "acoustic_scale_present": acoustic_finite,

        # ℓ support
        "ell_max_supported": ell_max_supported,

        # Raw diagnostics (useful for plots later)
        "R_eff_final": float(R_eff[-1]),
        "R_eff_min": float(np.min(R_eff)),
        "R_eff_max": float(np.max(R_eff)),
    }
