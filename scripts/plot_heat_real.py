#!/usr/bin/env python3
"""
Real grid-convergence of the heat solver, measured by Richardson comparison
(no analytic solution needed): the finest grid serves as the reference, coarser
grids are compared to it on their common (nested) nodes, and the L2 error vs
grid step h is plotted on log-log axes together with the O(h^2) slope.

All runs must be at the SAME physical time (use run_heat_conv.sh, which picks
steps ~ N^2 so that sim_time is equal across grids).

Usage:
    python3 scripts/plot_heat_real.py \
        64:results/heat_N64/output_0001.vtk \
        128:results/heat_N128/output_0001.vtk \
        256:results/heat_N256/output_0001.vtk \
        --ref 512:results/heat_N512/output_0001.vtk \
        --out diploma/figures/heat_convergence.pdf
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from plot_snapshot import read_vtk


def field(path):
    _, _, T = read_vtk(path, "Temperature")
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="N:path to coarse-grid VTK")
    ap.add_argument("--ref", required=True, help="Nref:path to finest-grid VTK")
    ap.add_argument("--out", default="diploma/figures/heat_convergence.pdf")
    args = ap.parse_args()

    Nref, ref_path = args.ref.split(":")
    Nref = int(Nref)
    Tref = field(ref_path)

    hs, errs = [], []
    print(f"{'N':>6} {'h':>10} {'L2':>14} {'p':>6}")
    prev_e = prev_h = None
    for spec in args.runs:
        N, path = spec.split(":")
        N = int(N)
        T = field(path)
        k = Nref // N
        ref_sub = Tref[::k, ::k]
        ny = min(T.shape[0], ref_sub.shape[0])
        nx = min(T.shape[1], ref_sub.shape[1])
        h = 1.0 / N
        diff = T[:ny, :nx] - ref_sub[:ny, :nx]
        L2 = np.sqrt(np.sum(diff**2) * h * h)
        p = "---" if prev_e is None else f"{np.log(prev_e/L2)/np.log(prev_h/h):.2f}"
        print(f"{N:>6} {h:>10.5f} {L2:>14.4e} {p:>6}")
        hs.append(h); errs.append(L2)
        prev_e, prev_h = L2, h

    hs = np.array(hs); errs = np.array(errs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(hs, errs, "o-", color="C1", label="численная ошибка")
    ref_line = errs[0] * (hs / hs[0])**2
    ax.loglog(hs, ref_line, "--", color="gray", label=r"$O(h^2)$ эталон")
    ax.set_xlabel(r"$h$ (шаг сетки)")
    ax.set_ylabel(r"$\|e_h\|_{L_2}$")
    ax.set_title("Сходимость уравнения теплопроводности")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    ax.invert_xaxis()
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("saved", args.out)


if __name__ == "__main__":
    main()
