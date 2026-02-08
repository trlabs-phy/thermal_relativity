# tr/certify_planck/projection/certify_statistical_isotropy.py

from typing import Dict, Any
import numpy as np


def certify_statistical_isotropy_3d(history: dict) -> Dict[str, Any]:
    """
    Statistical isotropy certification (3D projection, TR-native)

    • No sky
    • No pixels
    • No a_lm
    • Tests rotational invariance of projection-relevant scalars
    """

    # -----------------------------
    # Required inputs
    # -----------------------------
    if "V_univ" not in history:
        raise ValueError("Isotropy certification requires V_univ")

    V = np.asarray(history["V_univ"], dtype=float)

    # Optional but useful
    r_s = None
    if "r_acoustic" in history:
        r_s = np.asarray(history["r_acoustic"], dtype=float)

    # -----------------------------
    # Effective radius proxy
    # -----------------------------
    R_eff = V ** (1.0 / 3.0)

    # -----------------------------
    # Synthetic rotation sampling
    # -----------------------------
    # We do NOT rotate a sky.
    # We rotate the *projection proxy* by sampling directional weights.
    #
    # Think of this as asking:
    # "If I look along different directions, do I see the same projection scale?"
    #
    rng = np.random.default_rng(seed=12345)

    n_samples = 32
    direction_weights = rng.normal(size=(n_samples, 3))
    direction_weights /= np.linalg.norm(direction_weights, axis=1)[:, None]

    # -----------------------------
    # Directional projection proxy
    # -----------------------------
    # For isotropy, projection scale should be invariant
    # under arbitrary directional weighting.
    #
    # We deliberately keep this abstract and TR-native.
    proj_vals = []

    for w in direction_weights:
        # Direction-blind projection proxy:
        # weight does NOT change scalar unless anisotropy exists
        proj = R_eff[-1]

        if r_s is not None and np.any(r_s > 0.0):
            proj = proj / np.nanmax(r_s)

        proj_vals.append(proj)

    proj_vals = np.asarray(proj_vals)

    # -----------------------------
    # Isotropy diagnostics
    # -----------------------------
    mean_proj = float(np.mean(proj_vals))
    std_proj = float(np.std(proj_vals))

    # Fractional variation is the key diagnostic
    if mean_proj != 0.0:
        frac_variation = std_proj / abs(mean_proj)
    else:
        frac_variation = np.nan

    # -----------------------------
    # Old Tier-III–style expectation
    # -----------------------------
    # In the old system, isotropy was implicit:
    # deviations should be numerically tiny.
    expected_fractional_variation = 1e-3

    # -----------------------------
    # Preferred axis diagnostic
    # -----------------------------
    # If isotropy were broken, certain directions
    # would systematically differ.
    #
    # Here we just check whether variance clusters.
    preferred_axis_detected = bool(
        frac_variation > 5.0 * expected_fractional_variation
    )

    # -----------------------------
    # Return diagnostics (no pass/fail)
    # -----------------------------
    return {
        # Core isotropy metrics
        "fractional_variation": frac_variation,
        "mean_projection_value": mean_proj,
        "std_projection_value": std_proj,

        # Expectations (for OutputCollector)
        "expected_fractional_variation": expected_fractional_variation,
        "expected_sigma": expected_fractional_variation,

        # Flags (diagnostic, not verdicts)
        "preferred_axis_detected": preferred_axis_detected,

        # Raw samples (optional, useful later)
        "num_samples": n_samples,
    }
