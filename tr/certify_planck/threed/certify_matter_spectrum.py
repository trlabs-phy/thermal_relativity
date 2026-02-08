"""
Matter localization (phase spectrum) — TR-native (new format)

Snapshot-based structural diagnostics only.

• NO dynamics
• NO GR
• NO spacetime assumptions

Reports numeric observables per phase; no verdict logic.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


# =====================================================
# Helpers
# =====================================================

def _surface_mask(mask: np.ndarray) -> np.ndarray:
    """6-neighborhood surface mask for a boolean region."""
    inside = mask.astype(bool)
    shifts = [
        np.roll(inside,  1, axis=0), np.roll(inside, -1, axis=0),
        np.roll(inside,  1, axis=1), np.roll(inside, -1, axis=1),
        np.roll(inside,  1, axis=2), np.roll(inside, -1, axis=2),
    ]
    interior = inside.copy()
    for s in shifts:
        interior &= s
    return inside & (~interior)


def _sigma_R_fft(
    field: np.ndarray,
    dx: float,
    R_phys: float,
    *,
    window: str = "gaussian",
) -> float:
    """
    σ_R-like variance via bandlimited smoothing.

    window:
      - "gaussian": exp(-0.5 (kR)^2)
      - "tophat" : spherical top-hat
    """
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(R_phys) or R_phys <= 0:
        return float("nan")

    N = field.shape[0]
    ft = np.fft.fftn(field)

    kfreq = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    k = np.sqrt(kx * kx + ky * ky + kz * kz)

    x = k * R_phys

    if window == "gaussian":
        W = np.exp(-0.5 * x * x)
    elif window == "tophat":
        W = np.ones_like(x)
        nz = x > 0
        xx = x[nz]
        W[nz] = 3.0 * (np.sin(xx) - xx * np.cos(xx)) / (xx ** 3)
    else:
        raise ValueError("window must be 'gaussian' or 'tophat'")

    sm = np.fft.ifftn(ft * W).real
    return float(np.sqrt(np.mean(sm ** 2)))


def _radial_power_spectrum(
    mask: np.ndarray,
    dx: float,
    *,
    k_bins: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isotropic power spectrum of a binary mask (mean-subtracted).
    Returns (k, Pk) in physical k units.
    """
    rho = mask.astype(np.float64)
    rho -= rho.mean()

    ft = np.fft.fftn(rho)
    Pk = np.abs(ft) ** 2

    N = rho.shape[0]
    kfreq = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)

    k_flat = kmag.ravel()
    P_flat = Pk.ravel()

    nz = k_flat > 0
    k_flat = k_flat[nz]
    P_flat = P_flat[nz]

    if k_flat.size == 0:
        return np.array([]), np.array([])

    edges = np.linspace(0.0, k_flat.max(), k_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    P_b = np.zeros(k_bins)
    cnt = np.zeros(k_bins, dtype=int)

    for i in range(k_bins):
        sel = (k_flat >= edges[i]) & (k_flat < edges[i + 1])
        if np.any(sel):
            P_b[i] = float(np.mean(P_flat[sel]))
            cnt[i] = int(np.count_nonzero(sel))

    good = cnt > 0
    return centers[good], P_b[good]


# =====================================================
# Certification (new-style)
# =====================================================

def certify_matter_localization_spectrum(
    snap,
    *,
    phase_masks: Dict[str, np.ndarray],
    expected_matter_like: Optional[List[str]] = None,
    sigma_R_cells: float = 8.0,
    sigma_window: str = "gaussian",
    min_cells: int = 64,
    min_interface_frac: float = 1e-3,
    k_bins: int = 64,
) -> Dict[str, Any]:
    """
    Phase-by-phase matter localization diagnostics.

    Returns numeric observables only; no verdict logic.
    """

    dx = float(snap.dx)
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("Snapshot must define a valid dx")

    bubble = np.asarray(snap.mask_bubble, dtype=bool)
    N = bubble.shape[0]

    if expected_matter_like is None:
        expected_matter_like = []

    report: Dict[str, Any] = {
        "dx": dx,
        "N": int(N),
        "sigma_R_cells": float(sigma_R_cells),
        "sigma_R_phys": float(sigma_R_cells * dx),
        "sigma_window": str(sigma_window),
        "min_cells": int(min_cells),
        "min_interface_frac": float(min_interface_frac),
        "phases": {},
    }

    for name, mask in phase_masks.items():
        m = np.asarray(mask, dtype=bool)
        if m.shape != bubble.shape:
            raise ValueError(f"Phase mask '{name}' has wrong shape")

        m = m & bubble
        n = int(np.count_nonzero(m))

        pr: Dict[str, Any] = {
            "cells": n,
            "exists": bool(n > 0),
            "expected_matter_like": bool(name in expected_matter_like),
        }

        if n >= min_cells:
            surf = _surface_mask(m)
            n_surf = int(np.count_nonzero(surf))
            interface_frac = float(n_surf / max(n, 1))

            rho = m.astype(np.float64)
            rho -= rho.mean()
            sigma_R = _sigma_R_fft(
                rho,
                dx=dx,
                R_phys=sigma_R_cells * dx,
                window=sigma_window,
            )

            k, Pk = _radial_power_spectrum(m, dx=dx, k_bins=k_bins)
            if len(Pk) > 8:
                dP = np.gradient(Pk, k)
                idx = np.where(dP < 0)[0]
                k_turnover = float(k[int(idx[0])]) if idx.size else float("nan")
                turnover_exists = bool(idx.size > 0)
            else:
                k_turnover = float("nan")
                turnover_exists = False
        else:
            interface_frac = float("nan")
            sigma_R = float("nan")
            k_turnover = float("nan")
            turnover_exists = False

        pr.update({
            "interface_fraction": interface_frac,
            "sigma_R": sigma_R,
            "k_turnover": k_turnover,
            "k_turnover_exists": turnover_exists,
        })

        report["phases"][name] = pr

    return report