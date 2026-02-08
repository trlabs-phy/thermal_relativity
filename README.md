# Thermal Relativity — Unified Boltzmann Solver

**Author:** T. Matthew Ressler  
**Framework:** Thermal Relativity (TR)  
**Language:** Python  
**Status:** Research / Certification-grade solver  

---

## Overview

This repository contains a **from-first-principles Boltzmann-style solver**
implementing the **Thermal Relativity (TR)** framework.

Unlike ΛCDM or GR-based cosmological solvers, this codebase:

- Treats **causal ordering (η)** as the sole fundamental driver  
- Models **proper time (τ)** as an accumulated physical response  
- Evolves **thermal energy bookkeeping exactly**, with zero tunable parameters  
- Separates **causal evolution (1D)** from **spatial realization (3D)** by design  

The solver is vertically certified:

> **Equation 0 → 1D causal consistency → 3D projection diagnostics**

No fitting to observational data is performed at any stage.

---

## What This Solver Demonstrates

- Exact **κ₍C₎ capacity closure** with zero residuals  
- Emergent **homogeneous 3D scaling** from strictly 1D causal bookkeeping  
- A measurable **ordering lag** (proper-time delay) as a physical remainder  
- Clean architectural separation of:
  - causality  
  - geometry  
  - observables  
- Weak-field agreement with GR **without invoking spacetime or metrics**

All results follow from bookkeeping identities and certified constraints,
not model tuning.

---

## Certified Result: 1D Capacity Scaling

The solver produces a certified 1D capacity-scaling result:

- **Median slope:** 3.0  
- **Residuals:** 0 (within numerical tolerance)

This confirms that **homogeneous 3D volume scaling emerges purely from causal
thermal-energy bookkeeping**, not from geometric assumptions or expansion laws.

The corresponding plot is generated during certification and saved to:

output/certify_planck/capacity_scaling_1d.pdf

This plot shows the instantaneous scaling exponent  
\(\mathrm{d}\ln V / \mathrm{d}\ln a\) converging to **3** with zero residual.

## Solver Architecture (High Level)

HistoryConfig
  |
  v
history_step (Eq. 0 — the sole causal evolution)
  |
  v
HistoryState
  |
  v
Recorder (read-only history capture)
  |
  v
certify_1d  -> causal consistency, ordering, visibility
  |
  v
certify_3d  -> snapshot projection, structure, light bending

Architectural guarantees:

- Exactly **one solver loop**
- **No feedback** from certification layers
- **No geometry** in 1D causal evolution
- **No causality** in 3D spatial realization

## Repository Structure (Core)

tr/
|-- history.py              # Core Eq. 0 solver (THE solver)
|-- state.py                # HistoryState container
|-- recorder.py             # Read-only history capture
|-- run_solver.py           # Main entrypoint
|
|-- certify_planck/
|   |-- certify_1d.py       # Causal (1D) certification
|   |-- certify_3d.py       # 3D projection certification
|   |-- causal/             # Pure 1D diagnostics
|   |-- threed/             # Snapshot-based diagnostics
|
|-- utils/
|   |-- build_snapshot.py
|   |-- output.py
|
|-- output/
|   |-- certify_planck/

## Requirements

- **Python ≥ 3.10** (recommended: 3.11)
- NumPy
- Matplotlib

```bash
pip install -r requirements.txt
```
Run the solver from the repository root:

```bash
pip install -r requirements.txt
python run_solver.py
```
**Outputs include:**

- Console certification reports  
- JSON / CSV / Markdown summaries  
- Plots written to `output/certify_planck/`

---

## **SECTION 10 — Reproducibility**

## Reproducibility

- No stochastic processes unless explicitly seeded  
- All certification thresholds are deterministic  
- No tunable cosmological parameters  
- Results are reproducible across runs and platforms

## What This Solver Does *Not* Do

- No ΛCDM fitting  
- No parameter optimization  
- No spacetime dynamics  
- No metric assumptions  
- No observational tuning  

This solver tests **physical consistency**, not cosmological preference.

## Citation

If you use this solver or its certification methodology, please cite:

> Ressler, T. Matthew. *Thermal Relativity: Unified Boltzmann Solver*. 2026.  
> GitHub repository: https://github.com/yourname/yourrepo

```bibtex
@misc{ressler2026trsolver,
  author       = {Ressler, T. Matthew},
  title        = {Thermal Relativity: Unified Boltzmann Solver},
  year         = {2026},
  howpublished = {\url{https://github.com/yourname/yourrepo}},
  note         = {Causal-first cosmological solver},
}
```
---

## **SECTION 13 — Author & License**

## Author

**T. Matthew Ressler**  
Independent Researcher  
Troy, MI  
matt.ressler@protonmail.com  

## License

This project is licensed under the MIT License.  
See `LICENSE` for details.
```
