"""
backsolve.py
============

Primordium Boundary Utilities (Non-Causal)

This module provides ONLY pre-solver boundary conditions.
It contains no solver logic, no state mutation, and no imports
from solver modules.
"""

# ============================================================
# Backsolved thermal seed
# ============================================================

def backsolve_TE_F_seed(
    *,
    kappa_C: float,
    tau_target: float,
) -> float:
    """
    Backsolve the total thermal content required to realize
    a target final ordering extent.

    Definition:
        tau = TE_U / kappa_C

    Requiring tau → tau_target implies:
        TE_U(final) = kappa_C * tau_target

    Since all thermal energy originates at Primordium ignition:
        TE_P(seed) = TE_F(seed) = kappa_C * tau_target
    """
    return kappa_C * tau_target


def backsolve_initial_cube_state(
    *,
    kappa_C: float,
    tau_target: float,
):
    """
    Return a purely bookkeeping initial cube state.

    Returns
    -------
    N_C : int
        Number of fully filled cubes
    epsilon : float
        Partial fill of the next cube
    TE_P_seed : float
        Total thermal energy supplied to the solver
    """
    N_C = int(tau_target)
    frac = tau_target - N_C

    epsilon = frac * kappa_C
    TE_P_seed = kappa_C * tau_target

    return N_C, epsilon, TE_P_seed
