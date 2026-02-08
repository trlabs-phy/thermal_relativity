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

    N_C: int = 1          # start with one cube
    epsilon: float = 0.0

    V_cube_eff: float = 1.0   # 🔹 NEW: local cube volume
    V_univ: float = 1.0

