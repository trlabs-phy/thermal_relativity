# ============================================================
# backsolve.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module provides pre-solver boundary utilities for the
# Unified Thermal Relativity Boltzmann Solver.
#
# All functions in this file operate strictly in a non-causal
# bookkeeping role. They determine initial boundary conditions
# required for solver execution but do not participate in the
# forward causal evolution of the system.
#
# This module contains:
# - no solver logic,
# - no state mutation,
# - no imports from solver evolution modules.
#
# Backsolved quantities are inferred from closure requirements
# and conservation identities and are applied exactly once
# prior to solver execution.
#
# Governing role:
# ---------------
# - Determines the unique thermal seed consistent with a
#   target ordering extent
# - Produces an initial cube-fill bookkeeping state
# - Enforces solver closure without introducing tunable parameters
#
# ============================================================

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
