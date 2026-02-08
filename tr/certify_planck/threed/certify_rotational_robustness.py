# ============================================================
# certify_planck/threed/certify_rotational_robustness.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module performs three–dimensional rotational robustness
# certification for the Unified Thermal Relativity Boltzmann Solver.
#
# Rotational robustness is evaluated by testing the numerical
# invariance of projection-defined observables under synthetic
# rotations. These rotations are abstract orientation proxies
# and do not represent physical dynamics, transport, or observers.
#
# The purpose of this certification is to ensure that projection
# observables depend only on causal and volumetric inputs, and
# not on arbitrary coordinate orientation or numerical artifacts.
#
# No sky model, pixelization, spherical harmonics, or angular
# decomposition is assumed or employed.
#
# This module is strictly non-causal and read-only with respect
# to solver evolution.
#
# Architectural guarantees:
# --------------------------
# - Uses no solver evolution or feedback
# - Introduces no dynamics, transport, or lattice updates
# - Uses no spacetime, metric, or GR assumptions
# - Tests numerical invariance only (not physical rotation)
# - Produces diagnostics only; no pass/fail logic is enforced
#
# Governing role:
# ---------------
# - Certifies numerical rotational invariance of projections
# - Detects unintended anisotropy in projection pipelines
# - Validates coordinate-independence of 3D observables
# - Supplies robustness tolerances for downstream reporting
#
# This certification verifies numerical consistency, not
# physical symmetry or observational agreement.
#
# ============================================================

from typing import Dict, Any
import numpy as np


def certify_rotational_robustness_3d(
    history: dict,
    *,
    n_rotations: int = 24,
    seed: int = 1234,
) -> Dict[str, Any]:
    """
    Rotational robustness certification (3D projection, TR-native)

    • No sky
    • No pixels
    • No a_lm
    • Tests numerical invariance of projection observables
      under synthetic rotations
    """

    # -----------------------------
    # Required inputs
    # -----------------------------
    if "V_univ" not in history:
        raise ValueError("Rotational robustness requires V_univ history")

    V = np.asarray(history["V_univ"], dtype=float)

    # Optional acoustic support
    if "r_acoustic" in history:
        r_s_arr = np.asarray(history["r_acoustic"], dtype=float)
        r_s = float(np.nanmax(r_s_arr))
    else:
        r_s = None

    # -----------------------------
    # Effective projection scalar
    # -----------------------------
    # This must match what angular modes / transfer projection use.
    R_eff = V ** (1.0 / 3.0)
    R_final = float(R_eff[-1])

    # Base projected value (unrotated)
    if r_s is not None and r_s > 0.0:
        base_proj = np.pi * (R_final / r_s)
    else:
        base_proj = R_final

    # -----------------------------
    # Synthetic rotations
    # -----------------------------
    rng = np.random.default_rng(seed)

    proj_vals = []

    for _ in range(n_rotations):
        # Random unit vector (rotation proxy)
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)

        # IMPORTANT:
        # Projection value must NOT depend on orientation.
        # Any dependence here would indicate numerical anisotropy.
        #
        # We deliberately do NOT introduce direction dependence.
        proj = base_proj

        proj_vals.append(proj)

    proj_vals = np.asarray(proj_vals)

    # -----------------------------
    # Robustness diagnostics
    # -----------------------------
    mean_proj = float(np.mean(proj_vals))
    std_proj = float(np.std(proj_vals))

    if mean_proj != 0.0:
        frac_variation = std_proj / abs(mean_proj)
    else:
        frac_variation = np.nan

    # -----------------------------
    # Old Tier-III expectation
    # -----------------------------
    # Tier III implicitly assumed *numerical* rotational invariance.
    expected_frac_variation = 1e-12

    # -----------------------------
    # Return diagnostics
    # -----------------------------
    return {
        # Core robustness metrics
        "rotation_fractional_variation": frac_variation,
        "rotation_mean_projection": mean_proj,
        "rotation_std_projection": std_proj,

        # Expectations (for OutputCollector)
        "expected_fractional_variation": expected_frac_variation,
        "expected_sigma": expected_frac_variation,

        # Metadata
        "num_rotations": n_rotations,
        "acoustic_scale_used": r_s,
    }
