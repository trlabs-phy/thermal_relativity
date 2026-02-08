# tr/certify_planck/threed/certify_light_bending.py
from typing import Dict, Any, Callable, Optional, Tuple
import numpy as np

# -----------------------------
# Constants (SI)
# -----------------------------
G = 6.674e-11
c = 3.0e8


# ============================================================
# GR / PR toy model builders (same as your toy, but packaged)
# ============================================================

def phi_gr(r: np.ndarray, *, M: float) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return -G * M / np.maximum(r, 1e-30)

def f_gr_weak(r: np.ndarray, *, M: float) -> np.ndarray:
    # first-order weak field: F = 1 + Phi/c^2
    return 1.0 + phi_gr(r, M=M) / (c**2)

def make_pr_model(*, M: float, eps: float = 0.01, p: float = 0.5,
                  S_inf: float = 1.0, T_inf: float = 3.0
                  ) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """
    Returns:
      F_PR(r)   : clock-rate function
      Phi_PR(r) : potential function
    """
    if eps <= 0.0:
        raise ValueError("eps must be > 0")

    r0 = G * M / (eps * c**2)
    alpha = eps

    def S_r(r):
        r = np.asarray(r, dtype=float)
        return S_inf * (r0 / np.maximum(r, 1e-30)) ** p

    def T_r(r):
        r = np.asarray(r, dtype=float)
        return T_inf * (r0 / np.maximum(r, 1e-30)) ** (1.0 - p)

    def F_PR(r):
        r = np.asarray(r, dtype=float)
        return 1.0 - alpha * (S_r(r) * T_r(r)) / (S_inf * T_inf)

    def Phi_PR(r):
        r = np.asarray(r, dtype=float)
        return c**2 * (F_PR(r) - 1.0)

    return F_PR, Phi_PR


# ============================================================
# Gradient-index bending engine (toy-certified)
# ============================================================

def _n_from_phi(phi: np.ndarray) -> np.ndarray:
    # n(r) = 1 - 2 Phi / c^2
    return 1.0 - 2.0 * phi / (c**2)

def _dndb_from_phi(phi_fn: Callable[[np.ndarray], np.ndarray],
                   x: np.ndarray,
                   b: float,
                   *,
                   r_min: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    ∂n/∂b = (dn/dr)*(∂r/∂b) with r = sqrt(x^2 + b^2)
    """
    r = np.sqrt(x*x + b*b)
    r = np.maximum(r, r_min)

    # numerical dn/dr (stable, matches your toy)
    dr = 1e-6 * r + 1e-3
    rp = r + dr
    rm = np.maximum(r - dr, r_min)

    phip = phi_fn(rp)
    phim = phi_fn(rm)

    np_ = _n_from_phi(phip)
    nm_ = _n_from_phi(phim)

    dn_dr = (np_ - nm_) / (rp - rm)

    # ∂r/∂b = b/r
    dr_db = b / r

    return dn_dr * dr_db, r

def bend_profile(phi_fn: Callable[[np.ndarray], np.ndarray],
                 *,
                 b: float,
                 X_factor: float = 300.0,
                 n_steps: int = 200000,
                 r_min: float = 1.0
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
      x, dtheta_dx, theta, meta
    """
    b = float(b)
    if b <= 0:
        raise ValueError("b must be > 0")

    X = float(X_factor) * b
    x = np.linspace(-X, X, int(n_steps), dtype=float)
    dx = x[1] - x[0]

    r = np.sqrt(x*x + b*b)
    r = np.maximum(r, r_min)

    phi = phi_fn(r)
    n = _n_from_phi(phi)

    dndb, r_used = _dndb_from_phi(phi_fn, x, b, r_min=r_min)
    dtheta_dx = (1.0 / n) * dndb

    theta = np.cumsum(dtheta_dx) * dx
    alpha = float(theta[-1] - theta[0])

    meta = {
        "b": float(b),
        "X": float(X),
        "X_factor": float(X_factor),
        "n_steps": int(n_steps),
        "r_min": float(r_min),
        "alpha": float(alpha),
        "alpha_abs": float(abs(alpha)),
        "n_min": float(np.min(n)),
        "n_max": float(np.max(n)),
        "r_min_used": float(np.min(r_used)),
    }
    return x, dtheta_dx, theta, meta


# ============================================================
# Certification entrypoint (3D)
# ============================================================

def certify_light_bending_3d(
    history: Dict[str, Any],
    snap,
    *,
    cfg=None,
    export_profile: bool = True,
) -> Dict[str, Any]:
    """
    3D certification: weak-field light bending + natural angles.

    - No wall
    - No observer
    - Returns: alpha agreement + (x, dtheta_dx, theta) profile
    - Optionally caches results into history["_cert_light_bending"]
      so Observations can reuse without recompute.
    """

    # -----------------------------
    # Config defaults
    # -----------------------------
    M = float(getattr(cfg, "lb_M", 1.989e30))                 # default solar mass
    b = float(getattr(cfg, "lb_b", 7.0e8))                    # default solar radius
    eps = float(getattr(cfg, "lb_eps", 0.01))
    p = float(getattr(cfg, "lb_p", 0.5))

    X_factor = float(getattr(cfg, "lb_X_factor", 300.0))
    n_steps = int(getattr(cfg, "lb_n_steps", 200000))
    r_min = float(getattr(cfg, "lb_r_min", 1.0))

    # tolerances
    tol_gr_numeric_vs_closed = float(getattr(cfg, "lb_tol_gr_closed_frac", 1e-4))
    tol_pr_vs_gr_ratio = float(getattr(cfg, "lb_tol_pr_gr_ratio", 1e-6))

    # -----------------------------
    # Build PR + GR potentials
    # -----------------------------
    F_PR, Phi_PR = make_pr_model(M=M, eps=eps, p=p)
    Phi_GR = (lambda r: phi_gr(r, M=M))

    # -----------------------------
    # Bending runs
    # -----------------------------
    xg, dthdx_g, th_g, meta_g = bend_profile(Phi_GR, b=b, X_factor=X_factor, n_steps=n_steps, r_min=r_min)
    xp, dthdx_p, th_p, meta_p = bend_profile(Phi_PR, b=b, X_factor=X_factor, n_steps=n_steps, r_min=r_min)

    # Closed-form GR weak-field benchmark
    alpha_gr_closed = 4.0 * G * M / (b * c**2)

    # Compare
    alpha_g = float(meta_g["alpha_abs"])
    alpha_p = float(meta_p["alpha_abs"])

    gr_frac_err = (alpha_g - alpha_gr_closed) / alpha_gr_closed if alpha_gr_closed != 0 else np.nan
    pr_gr_ratio = (alpha_p / alpha_g) if alpha_g != 0 else np.nan

    gr_ok = bool(np.isfinite(gr_frac_err) and abs(gr_frac_err) <= tol_gr_numeric_vs_closed)
    pr_ok = bool(np.isfinite(pr_gr_ratio) and abs(pr_gr_ratio - 1.0) <= tol_pr_vs_gr_ratio)

    # -----------------------------
    # Time dilation consistency (same toy logic)
    # -----------------------------
    r1 = float(getattr(cfg, "lb_td_r1", 7.0e8))
    r2 = float(getattr(cfg, "lb_td_r2", 7.0e10))

    dF_pr = float(F_PR(r1) - F_PR(r2))
    dF_gr = float(f_gr_weak(np.array([r1]), M=M)[0] - f_gr_weak(np.array([r2]), M=M)[0])
    td_frac_err = (dF_pr - dF_gr) / dF_gr if dF_gr != 0 else np.nan
    td_ok = bool(np.isfinite(td_frac_err) and abs(td_frac_err) <= float(getattr(cfg, "lb_tol_td_frac", 1e-10)))

    result: Dict[str, Any] = {
        "defined": True,

        # Weak-field validations
        "gr_numeric_alpha": alpha_g,
        "gr_closed_alpha": float(alpha_gr_closed),
        "gr_closed_fractional_error": float(gr_frac_err),
        "gr_ok": float(gr_ok),

        "pr_numeric_alpha": alpha_p,
        "pr_over_gr_ratio": float(pr_gr_ratio),
        "pr_ok": float(pr_ok),

        "time_dilation_dF_pr": dF_pr,
        "time_dilation_dF_gr": dF_gr,
        "time_dilation_fractional_error": float(td_frac_err),
        "time_dilation_ok": float(td_ok),

        # Natural angles (NO WALL)
        "x": xp if export_profile else None,
        "dtheta_dx": dthdx_p if export_profile else None,
        "theta": th_p if export_profile else None,

        # One-liners for downstream
        "theta_mid": float(th_p[len(th_p)//2]),
        "max_abs_dtheta_dx": float(np.max(np.abs(dthdx_p))),

        # config echo
        "b": b, "M": M, "eps": eps, "p": p,
        "X_factor": X_factor, "n_steps": n_steps, "r_min": r_min,
    }

    if export_profile:
        # cache for downstream usage (observations / binning / healpix adapter)
        history["_cert_light_bending"] = {
            "b": b,
            "alpha": alpha_p,
            "x": xp,
            "dtheta_dx": dthdx_p,
            "theta": th_p,
            "meta": {
                "theta_mid": result["theta_mid"],
                "max_abs_dtheta_dx": result["max_abs_dtheta_dx"],
            }
        }

    return result
