#!/usr/bin/env python3
"""
Real Sod shock-tube plot from the solver's own VTK output (legacy binary
RECTILINEAR_GRID, the format this code actually writes).

Two modes:
  1) Density profile vs exact solution (same style as before):
       python3 scripts/plot_sod_real.py profile results/sod_N400/output_0001.vtk \
               --t 0.2 --out diploma/figures/sod_density.pdf
     --t is the physical time printed by the solver as "sim_time=...".

  2) Grid convergence table (L1 density error vs exact) over several runs:
       python3 scripts/plot_sod_real.py conv \
               100:results/sod_N100/output_0001.vtk:0.2 \
               200:results/sod_N200/output_0001.vtk:0.2 ...
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

GAMMA = 1.4


def sod_exact(x, t, gamma=GAMMA):
    """Exact density of the Sod problem at time t (rhoL=1,pL=1 | rhoR=0.125,pR=0.1)."""
    rhoL, uL, pL = 1.0, 0.0, 1.0
    rhoR, uR, pR = 0.125, 0.0, 0.1
    x0 = 0.5
    cL = np.sqrt(gamma * pL / rhoL)
    cR = np.sqrt(gamma * pR / rhoR)

    def f(p, pK, rhoK, cK):
        if p > pK:
            A = 2.0 / ((gamma + 1) * rhoK)
            B = (gamma - 1) / (gamma + 1) * pK
            return (p - pK) * np.sqrt(A / (p + B))
        return 2 * cK / (gamma - 1) * ((p / pK) ** ((gamma - 1) / (2 * gamma)) - 1)

    def F(p):
        return f(p, pL, rhoL, cL) + f(p, pR, rhoR, cR) + (uR - uL)

    lo, hi = 1e-6, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if F(mid) > 0:
            hi = mid
        else:
            lo = mid
    p_star = 0.5 * (lo + hi)
    u_star = 0.5 * (uL + uR) + 0.5 * (f(p_star, pR, rhoR, cR) - f(p_star, pL, rhoL, cL))

    rho = np.zeros_like(x)
    s = (x - x0) / t

    # Left: rarefaction (p_star < pL)
    rho_starL = rhoL * (p_star / pL) ** (1 / gamma)
    c_starL = cL * (p_star / pL) ** ((gamma - 1) / (2 * gamma))
    SHL = uL - cL
    STL = u_star - c_starL
    lw = s < SHL
    lf = (s >= SHL) & (s < STL)
    ls = (s >= STL) & (s < u_star)
    rho[lw] = rhoL
    c_fan = 2 / (gamma + 1) * (cL - (gamma - 1) / 2 * (uL - s[lf]))
    rho[lf] = rhoL * (c_fan / cL) ** (2 / (gamma - 1))
    rho[ls] = rho_starL

    # Right: shock (p_star > pR)
    rho_starR = rhoR * ((p_star / pR + (gamma - 1) / (gamma + 1))
                        / ((gamma - 1) / (gamma + 1) * p_star / pR + 1))
    SR = uR + cR * np.sqrt((gamma + 1) / (2 * gamma) * p_star / pR
                           + (gamma - 1) / (2 * gamma))
    rs = (s >= u_star) & (s < SR)
    rw = s >= SR
    rho[rs] = rho_starR
    rho[rw] = rhoR
    return rho


def numeric_line(vtk_path):
    """Density along a horizontal line (Sod is 1D, any interior row works)."""
    x, y, d = read_vtk(vtk_path, "Density")
    row = d[d.shape[0] // 2, :]          # central row
    return x, row


def cmd_profile(args):
    xn, rhon = numeric_line(args.vtk)
    xe = np.linspace(0, 1, 2001)
    rhoe = sod_exact(xe, args.t)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xe, rhoe, "k-", lw=1.6, label="точное")
    ax.plot(xn, rhon, "o", ms=3.0, mfc="none", mec="C3",
            label=f"численное, N={xn.size}")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho$")
    ax.set_title(f"Задача Sod, $t={args.t}$")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print("saved", args.out)


def cmd_conv(args):
    rows = []
    prev_e = prev_h = None
    print(f"{'N':>6} {'h':>10} {'L1':>14} {'p':>6}")
    for spec in args.runs:
        N, path, t = spec.split(":")
        N = int(N); t = float(t)
        x, rho = numeric_line(path)
        rho_e = sod_exact(x, t)
        h = 1.0 / N
        L1 = np.sum(np.abs(rho - rho_e)) * h
        p = "---" if prev_e is None else f"{np.log(prev_e/L1)/np.log(prev_h/h):.2f}"
        print(f"{N:>6} {h:>10.5f} {L1:>14.4e} {p:>6}")
        rows.append((N, h, L1, p))
        prev_e, prev_h = L1, h
    print("\nLaTeX rows:")
    for N, h, L1, p in rows:
        m, e = f"{L1:.2e}".split("e")
        print(f"{N} & ${m}\\cdot10^{{{int(e)}}}$ & {p} \\\\")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("profile")
    pp.add_argument("vtk")
    pp.add_argument("--t", type=float, required=True)
    pp.add_argument("--out", default="diploma/figures/sod_density.pdf")
    pp.set_defaults(func=cmd_profile)
    pc = sub.add_parser("conv")
    pc.add_argument("runs", nargs="+", help="N:path/to/output.vtk:t")
    pc.set_defaults(func=cmd_conv)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
