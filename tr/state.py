# ============================================================
# state.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module defines the immutable physical state container used
# by the Unified Thermal Relativity Boltzmann Solver.
#
# The HistoryState dataclass stores the complete set of causal
# variables required to advance the solver through the ordering
# coordinate η. It contains no logic, no dynamics, and no derived
# behavior of its own.
#
# All fields in this structure are updated exclusively by the
# history integrator and are treated as read-only by downstream
# modules such as recorders, certification tests, and observables.
#
# This separation enforces strict causal discipline: state is
# advanced in one location, observed everywhere else.
#
# Governing role:
# ---------------
# - Stores the instantaneous causal state of the solver
# - Contains ordering, exposure, thermal, and geometric quantities
# - Provides a single source of truth for the solver history
# - Prevents hidden state mutation or feedback
#
# ============================================================

from dataclasses import dataclass

@dataclass
class HistoryState:
    eta: float = 0.0
    V_AT: float = 0.0
    V_prim: float = 0.0
    B_T: float = 0.0
    mu: float = 0.0

    TE_P: float = 0.0
    TE_U: float = 0.0
    TE_F: float = 0.0

    N_C: int = 1          
    epsilon: float = 0.0

    V_cube_eff: float = 1.0   
    V_univ: float = 1.0

