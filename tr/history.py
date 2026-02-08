# ============================================================
# history.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module implements the core history integrator for the
# Unified Thermal Relativity Boltzmann Solver.
#
# The history_step function advances the complete causal state
# of the system by a single increment of the universal ordering
# parameter η. All physical quantities are updated in strict
# vertical causal order, from exposure and thermal activation
# through capacity-limited volume realization.
#
# The solver is fully deterministic and contains no tunable
# parameters at runtime. Geometry, structure, and observables
# are not evolved independently; they arise solely from
# first-principles thermal energy bookkeeping under ordered
# activation.
#
# This file introduces no observational logic, no fitting, and
# no feedback from downstream quantities. All updates performed
# here are causal, local, and auditable step-by-step.
#
# Governing role:
# ---------------
# - Advances the universal ordering coordinate η
# - Applies Equation 0 (thermal activation)
# - Enforces κ_C capacity chasing and cube realization
# - Produces the complete background history used by all
#   downstream certification and observation modules
#
# ============================================================

import math
from dataclasses import dataclass
from state import HistoryState

# ============================================================
# Configuration (Eq. 0 only)
# ============================================================

@dataclass(frozen=True)
class HistoryConfig:
    d_eta: float
    c_AT: float
    rho_TPB0: float
    V_mu: float
    kappa_C: float = 1.0
    k_act: float = 1.0


# ============================================================
# History step (THE solver)
# ============================================================

def history_step(state: HistoryState, cfg: HistoryConfig) -> None:
    eps = 1e-30

    # --------------------------------------------------
    # 1) Ordering coordinate
    # --------------------------------------------------
    state.eta += cfg.d_eta

    # --------------------------------------------------
    # 2) Absolute-Time exposure (Primordium)
    # --------------------------------------------------
    state.V_AT += cfg.c_AT * cfg.d_eta
    state.V_prim = state.V_AT

    # --------------------------------------------------
    # 3) Thermal–Potential Budget
    # --------------------------------------------------
    state.B_T = cfg.rho_TPB0 * state.V_prim

    # --------------------------------------------------
    # 4) Mobility from exposure ONLY
    # --------------------------------------------------
    state.mu = 1.0 - math.exp(-state.V_AT / cfg.V_mu)
    state.mu = max(0.0, min(1.0, state.mu))

    # --------------------------------------------------
    # 5) Thermal activation (Eq. 0)
    # --------------------------------------------------
    dTE = cfg.k_act * state.mu * cfg.d_eta
    dTE = min(dTE, state.B_T - state.TE_P)
    dTE = max(dTE, 0.0)
    state.TE_P += dTE

    # --------------------------------------------------
    # 6) Cube-local κC chasing (volume-first)
    # --------------------------------------------------

    TE_U = cfg.kappa_C * state.N_C
    TE_F = max(state.TE_P - TE_U - state.epsilon, 0.0)

    if TE_F > eps:

        # --- local volume response (first) ---
        dV = min(TE_F, cfg.kappa_C * state.V_cube_eff)
        state.V_cube_eff += dV
        TE_F -= dV

        # --- if volume response saturates, create new cube ---
        if TE_F > eps:
            state.N_C += 1
            state.V_cube_eff = 1.0
            state.epsilon = 0.0

    # --------------------------------------------------
    # 7) Cube fill & chase (κC bookkeeping)
    # --------------------------------------------------
    TE_U = cfg.kappa_C * state.N_C
    TE_F = max(state.TE_P - TE_U - state.epsilon, 0.0)

    if TE_F > eps:
        d_eps = min(TE_F, cfg.kappa_C - state.epsilon)
        state.epsilon += d_eps

        if state.epsilon >= cfg.kappa_C - eps:
            state.N_C += 1
            state.epsilon = 0.0

    # --------------------------------------------------
    # 8) Derived bookkeeping (still causal)
    # --------------------------------------------------
    state.TE_U = cfg.kappa_C * state.N_C
    state.TE_F = max(state.TE_P - state.TE_U - state.epsilon, 0.0)

    state.V_univ = float(state.N_C)
