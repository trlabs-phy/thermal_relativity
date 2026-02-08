import numpy as np
from typing import Dict, Any, Optional


def _surface_points(N: int, mask_surface: np.ndarray):
    """Return (x,y,z) coordinates of surface voxels centered at 0."""
    c = (N - 1) / 2.0
    idx = np.argwhere(mask_surface)
    x = idx[:, 0] - c
    y = idx[:, 1] - c
    z = idx[:, 2] - c
    return x, y, z


def _angles_from_xyz(x, y, z, eps=1e-30):
    r = np.sqrt(x*x + y*y + z*z) + eps
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))        # 0..pi
    phi = np.mod(np.arctan2(y, x), 2.0 * np.pi)         # 0..2pi
    return theta, phi


def certify_transfer_projection_3d(
    snap,
    *,
    n_theta: int = 64,
    n_phi: int = 256,
    min_samples: int = 256,
) -> Dict[str, Any]:
    """
    Transfer projection (TR-native, 3D)

    Meaning:
      Can we construct a stable angular decomposition of a surface “impression field”
      from the snapshot?

    No acoustic scale. No GR. No evolution.

    Output:
      - transfer_defined: bool
      - projected_peak_ell: proxy peak angular mode
      - ell_fractional_variation: stability proxy (ring-to-ring)
    """

    # --------------------------------------------------
    # Build surface impression field
    # --------------------------------------------------
    ms = np.asarray(snap.mask_surface, dtype=bool)
    Ec = np.asarray(snap.E_cell, dtype=float)

    if ms.shape != Ec.shape:
        raise ValueError("transfer: snapshot surface mask and E_cell shape mismatch")

    n_surf = int(np.count_nonzero(ms))
    if n_surf < min_samples:
        return {
            "transfer_defined": False,
            "reason": "insufficient_surface_samples",
            "n_surface_cells": n_surf,
        }

    N = ms.shape[0]
    x, y, z = _surface_points(N, ms)
    theta, phi = _angles_from_xyz(x, y, z)

    I = Ec[ms].astype(np.float64)
    I = I - np.mean(I)  # remove monopole

    # --------------------------------------------------
    # Bin to angular grid (theta rings × phi bins)
    # --------------------------------------------------
    th_edges = np.linspace(0.0, np.pi, n_theta + 1)
    ph_edges = np.linspace(0.0, 2.0*np.pi, n_phi + 1)

    # indices
    it = np.clip(np.digitize(theta, th_edges) - 1, 0, n_theta - 1)
    ip = np.clip(np.digitize(phi,   ph_edges) - 1, 0, n_phi   - 1)

    grid = np.zeros((n_theta, n_phi), dtype=np.float64)
    cnt  = np.zeros((n_theta, n_phi), dtype=np.int64)

    np.add.at(grid, (it, ip), I)
    np.add.at(cnt,  (it, ip), 1)

    # mean per bin (avoid /0)
    good = cnt > 0
    grid[good] /= cnt[good]
    grid[~good] = 0.0

    # --------------------------------------------------
    # “Angular spectrum proxy”: FFT along phi in each theta ring
    # --------------------------------------------------
    ft = np.fft.rfft(grid, axis=1)
    Pm = np.mean(np.abs(ft)**2, axis=0)  # average over theta rings
    # m index corresponds to phi-harmonics; treat it as an ℓ proxy for readiness
    m = np.arange(Pm.size, dtype=float)

    # ignore m=0 (already de-meaned, but safe)
    if Pm.size < 3:
        return {
            "transfer_defined": False,
            "reason": "insufficient_phi_resolution",
        }

    peak_idx = int(np.argmax(Pm[1:]) + 1)
    ell_peak_proxy = float(m[peak_idx])

    # ring-to-ring stability proxy
    ring_power = np.mean(np.abs(ft[:, peak_idx])**2)
    ring_pw = np.abs(ft[:, peak_idx])**2
    frac_var = float(np.std(ring_pw) / (np.mean(ring_pw) + 1e-30))

    return {
        "transfer_defined": True,
        "projected_peak_ell": ell_peak_proxy,
        "ell_fractional_variation": frac_var,

        # diagnostics
        "n_surface_cells": n_surf,
        "n_theta": int(n_theta),
        "n_phi": int(n_phi),
        "peak_m_index": int(peak_idx),
    }
