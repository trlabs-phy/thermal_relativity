# ============================================================
# Imports
# ============================================================
import os
import json

from backsolve import backsolve_initial_cube_state
from history import HistoryConfig, history_step
from state import HistoryState
from recorder import init_recorder, record_step

from utils.certify_timeline import build_certification_timeline
from certify_planck.certify_1d import certify_1d
from certify_planck.certify_3d import certify_3d

# ============================================================
# Configuration
# ============================================================
TAU_TARGET = 24.0
ETA_MAX = 50.0

cfg = HistoryConfig(
    d_eta=1e-3,
    c_AT=1.0,
    rho_TPB0=1.0,
    V_mu=1.0,
    kappa_C=1.0,
)

steps = int(ETA_MAX / cfg.d_eta)


# ============================================================
# Backsolve boundary (ONCE)
# ============================================================
N_C, epsilon, TE_P_seed = backsolve_initial_cube_state(
    kappa_C=cfg.kappa_C,
    tau_target=TAU_TARGET,
)


# ============================================================
# Initialize solver state
# ============================================================
state = HistoryState(
    eta=0.0,
    V_AT=0.0,
    V_prim=0.0,
    B_T=0.0,
    mu=0.0,
    TE_P=TE_P_seed,
    TE_U=cfg.kappa_C * N_C,
    TE_F=0.0,
    N_C=N_C,
    epsilon=epsilon,
    V_univ=float(N_C),
)


# ============================================================
# Recorder
# ============================================================
rec = init_recorder(cfg)


# ============================================================
# Run solver (THE ONLY LOOP)
# ============================================================
for _ in range(steps):
    history_step(state, cfg)
    record_step(rec, state)


# ============================================================
# Persist raw history (optional)
# ============================================================
os.makedirs("output", exist_ok=True)

with open("output/history.json", "w") as f:
    json.dump(rec, f, indent=2)


# ============================================================
# Certification — 1D
# ============================================================
results_1d = certify_1d(rec, cfg=cfg)
out1d = results_1d["output"]

# ============================================================
# Certification — 1D
# ============================================================
results_1d = certify_1d(rec, cfg=cfg)
out1d = results_1d["output"]

out1d.print_report()
out1d.to_json("output/certify_planck/summary_1d.json")
out1d.to_csv("output/certify_planck/summary_1d.csv")
out1d.to_markdown("output/certify_planck/summary_1d.md")

print("Solver + Planck 1D certification complete")


# ============================================================
# Certification — 3D
# ============================================================
results_3d = certify_3d(
    history=rec,
    cert1d=results_1d,
    cfg=cfg,
)

out3d = results_3d["output"]

out3d.print_report()
out3d.to_json("output/certify_planck/summary_3d.json")
out3d.to_csv("output/certify_planck/summary_3d.csv")
out3d.to_markdown("output/certify_planck/summary_3d.md")

print("Solver + Planck 3D certification complete")

