# ============================================================
# certify_planck/output.py
# ============================================================
#
# Author: T. Matthew Ressler
#
# Description:
# ------------
# This module defines the output and reporting infrastructure
# for the Unified Thermal Relativity Boltzmann Solver certification
# pipeline.
#
# It provides standardized mechanisms for collecting, formatting,
# exporting, and visualizing certification observables produced by
# non-causal certification routines (1D and 3D).
#
# This module contains no solver logic and introduces no physical
# dynamics. All quantities handled here are read-only results derived
# from previously recorded solver history and certification analyses.
#
# Architectural guarantees:
# --------------------------
# - No solver evolution or state mutation
# - No thresholds, tuning, or corrective feedback
# - No causal influence on history or certification logic
# - Pure aggregation, presentation, and persistence
#
# Governing role:
# ---------------
# - Defines the canonical Observable data structure
# - Accumulates certification results across test suites
# - Produces human-readable certification reports
# - Exports results to JSON, CSV, Markdown, and plots
# - Provides standardized visualization utilities
#
# This module is strictly non-causal and presentation-only.
#
# ============================================================

import json
import os
import csv
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import numpy as np

def write_ttcls_spectrum(
    ell,
    ring_strength,
    *,
    path: str = "output/ttcls_spectrum.csv",
    header: str = "ell,Cl",
):
    """
    Write TTCLS angular spectrum to CSV.

    Parameters
    ----------
    ell : array-like
        Angular mode indices (ℓ or m — whatever the solver produces).
    ring_strength : array-like
        Corresponding frozen-imprint spectrum values.
    path : str
        Output CSV path.
    header : str
        CSV header line.
    """

    ell = np.asarray(ell, dtype=float)
    ring_strength  = np.asarray(ring_strength, dtype=float)

    if ell.shape != ring_strength.shape:
        raise ValueError("write_ttcls_spectrum: ell and Cl must have same shape")

    # ensure directory exists
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    np.savetxt(
        path,
        np.column_stack([ell, ring_strength]),
        delimiter=",",
        header=header,
        comments="",
    )

    return path

def plot_ttcls(
    ell,
    Cl,
    *,
    title="TTCLS Imprint Spectrum",
    path=None,
    show=False,
):
    ell = np.asarray(ell)
    Cl  = np.asarray(Cl)

    finite = np.isfinite(Cl)
    ell = ell[finite]
    Cl  = Cl[finite]

    plt.figure(figsize=(7, 4))
    plt.plot(ell, Cl, lw=2)
    plt.xlabel("Angular mode index")
    plt.ylabel("Frozen imprint amplitude")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=160)

    if show:
        plt.show()

    plt.close()

# ============================================================
# Observable record
# ============================================================

@dataclass
class Observable:
    group: str
    name: str
    tr_value: float
    ref_value: Optional[float] = None
    ref_sigma: Optional[float] = None

    def delta(self) -> Optional[float]:
        if self.ref_value is None:
            return None
        return self.tr_value - self.ref_value

    def rel_error(self) -> Optional[float]:
        if self.ref_value in (None, 0.0):
            return None
        return self.delta() / self.ref_value

    def sigma_units(self) -> Optional[float]:
        if self.ref_sigma in (None, 0.0):
            return None
        return self.delta() / self.ref_sigma


# ============================================================
# Output accumulator
# ============================================================

class OutputCollector:
    """
    Central authority for all certification outputs.
    """

    def __init__(self):
        self._observables: List[Observable] = []

    # --------------------------------------------------
    # Add observable
    # --------------------------------------------------
    def add(
        self,
        *,
        group: str,
        name: str,
        tr_value: float,
        ref_value: Optional[float] = None,
        ref_sigma: Optional[float] = None,
    ):
        self._observables.append(
            Observable(
                group=group,
                name=name,
                tr_value=float(tr_value),
                ref_value=None if ref_value is None else float(ref_value),
                ref_sigma=None if ref_sigma is None else float(ref_sigma),
            )
        )

    # --------------------------------------------------
    # Grouped view
    # --------------------------------------------------
    def grouped(self) -> Dict[str, List[Observable]]:
        groups: Dict[str, List[Observable]] = {}
        for obs in self._observables:
            groups.setdefault(obs.group, []).append(obs)
        return groups

    # ==================================================
    # Human-readable printing
    # ==================================================
    def print_report(self):
        print("\n================ CERTIFICATION REPORT ================\n")

        for group, items in self.grouped().items():
            print(f"[{group.upper()}]")
            print("-" * (len(group) + 2))

            for obs in items:
                line = f"{obs.name:30s} : {obs.tr_value:.6e}"

                if obs.ref_value is not None:
                    line += f"   ref={obs.ref_value:.6e}"

                    d = obs.delta()
                    if d is not None:
                        line += f"   Δ={d:+.3e}"

                    rel = obs.rel_error()
                    if rel is not None:
                        line += f"   rel={rel:+.3e}"

                    sig = obs.sigma_units()
                    if sig is not None:
                        line += f"   σ={sig:+.2f}"

                print(line)

            print()

        print("======================================================\n")

    def report_timeline(out, timeline):
        """
        Timeline reporting (η-authoritative)

        Never print sentinel values.
        Never coerce None → -1.
        """

        def emit(name):
            val = timeline.get(name)
            if val is None:
                return
            out.add(
                group="timeline",
                name=name,
                tr_value=float(val),
                ref_value=None,
                ref_sigma=None,
            )

        # Authoritative causal events
        emit("thermalus_index")
        emit("thermalus_eta")

        emit("domain_handoff_index")
        emit("domain_handoff_eta")
        emit("V_moat_frozen")

        # Observational anchors
        emit("visibility_index")
        emit("visibility_eta")

        # Diagnostic only
        emit("matter_index")
        emit("matter_eta")

    # ==================================================
    # JSON export (canonical archive)
    # ==================================================
    def to_json(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                [asdict(obs) for obs in self._observables],
                f,
                indent=2,
            )

    # ==================================================
    # CSV export (analysis / plotting)
    # ==================================================
    def to_csv(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "group", "name", "tr_value",
                "ref_value", "ref_sigma",
                "delta", "rel_error", "sigma_units"
            ])

            for obs in self._observables:
                writer.writerow([
                    obs.group,
                    obs.name,
                    obs.tr_value,
                    obs.ref_value,
                    obs.ref_sigma,
                    obs.delta(),
                    obs.rel_error(),
                    obs.sigma_units(),
                ])

    # ==================================================
    # Markdown export (human / paper-facing)
    # ==================================================
    def to_markdown(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            f.write("# Certification Summary (Planck)\n\n")

            for group, items in self.grouped().items():
                f.write(f"## {group.capitalize()}\n\n")
                f.write("| Observable | TR | Ref | Δ | σ |\n")
                f.write("|------------|----|-----|----|---|\n")

                for o in items:
                    sig = o.sigma_units()
                    f.write(
                        f"| {o.name} | {o.tr_value:.6e} | "
                        f"{'' if o.ref_value is None else f'{o.ref_value:.6e}'} | "
                        f"{'' if o.delta() is None else f'{o.delta():+.2e}'} | "
                        f"{'' if sig is None else f'{sig:+.2f}'} |\n"
                    )
                f.write("\n")

    # ==================================================
    # Plot: capacity scaling consistency
    # ==================================================
    def plot_capacity_scaling(
        self,
        *,
        eta,
        slopes,
        outdir: str,
        filename: str = "capacity_scaling_1d.pdf",
        target: float = 3.0,
        show: bool = False,
    ):
        """
        Plot instantaneous scaling exponent dlnV/dlna vs eta.
        """

        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, filename)

        plt.figure(figsize=(8, 4))

        plt.plot(
            eta,
            slopes,
            lw=1.5,
            label="Measured scaling exponent",
        )

        plt.axhline(
            target,
            color="k",
            ls="--",
            lw=1.0,
            label="Homogeneous 3D target",
        )

        plt.xlabel(r"Causal ordering $\eta$")
        plt.ylabel(r"$d\ln V / d\ln a$")
        plt.title("1D Capacity Scaling Consistency Check")

        plt.legend()
        plt.tight_layout()

        plt.savefig(path)
        if show:
            plt.show()

        plt.savefig(path)

        png_path = os.path.splitext(path)[0] + ".png"
        plt.savefig(png_path, dpi=160)

        plt.close()



