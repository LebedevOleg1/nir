#!/usr/bin/env python3
"""
Isentropic vortex convergence analysis.

Reads the final VTK snapshot from each grid run, computes L2 error in density
vs the exact solution, and plots convergence order for HLL (1st order) and
MUSCL (2nd order).

The exact solution at t_final: vortex center has moved by (u_inf * t_final, 0).
Since the domain is periodic and t_final ≈ one period (Lx/u_inf = 10), the
exact solution is close to the initial condition. We compare the final
density field to the analytical vortex formula evaluated at t_final.

VTK format: binary RECTILINEAR_GRID (output_NNNN.vtk) — matches VTKWriter.hpp.

Usage: python3 scripts/plot_vortex_convergence.py [--device gpu|cpu] [--basedir results/vortex_conv]
Then copy: cp results/vortex_conv/vortex_convergence.pdf diploma/figures/
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ─── Vortex parameters (must match Physics.hpp / solver_impl.inl) ────────────
GAMMA   = 1.4
U_INF   = 0.0   # stationary vortex: exact solution = initial condition
V_INF   = 0.0
RHO_INF = 1.0
T_INF   = 1.0
EPS     = 5.0
XMIN, XMAX = -5.0, 5.0
YMIN, YMAX = -5.0, 5.0
LX = XMAX - XMIN
LY = YMAX - YMIN

GRIDS = [32, 64, 128, 256]
STEPS_FACTOR = 1   # must match run_vortex_convergence.sh


# ─── Binary VTK reader (RECTILINEAR_GRID, big-endian float32) ────────────────

def _read_line(f):
    """Read one ASCII line from a binary file."""
    line = b""
    while True:
        ch = f.read(1)
        if not ch or ch == b"\n":
            return line.decode("ascii", errors="replace").strip()
        line += ch


def read_vtk_density(vtk_path):
    """Read density field from a binary RECTILINEAR_GRID VTK file."""
    with open(str(vtk_path), "rb") as f:
        nx = ny = None
        for _ in range(20):
            line = _read_line(f)
            if line.startswith("DIMENSIONS"):
                parts = line.split()
                nx, ny = int(parts[1]), int(parts[2])
                break
        if nx is None:
            raise RuntimeError(f"No DIMENSIONS header in {vtk_path}")

        npts = nx * ny

        _read_line(f)  # X_COORDINATES nx float
        f.read(nx * 4)  # skip x coords
        f.read(1)       # newline

        _read_line(f)  # Y_COORDINATES ny float
        f.read(ny * 4)  # skip y coords
        f.read(1)

        _read_line(f)  # Z_COORDINATES 1 float
        f.read(4 + 1)

        _read_line(f)  # POINT_DATA npts

        while True:
            line = _read_line(f)
            if not line:
                break
            if line.startswith("SCALARS"):
                name = line.split()[1]
                _read_line(f)  # LOOKUP_TABLE default
                raw = f.read(npts * 4)
                if len(raw) < npts * 4:
                    break
                if name == "Density":
                    arr = np.frombuffer(raw, dtype=">f4").astype(float)
                    return arr.reshape((ny, nx)), nx, ny
                f.read(1)  # newline after non-density field

    raise RuntimeError(f"No 'Density' field in {vtk_path}")


# ─── Exact solution ───────────────────────────────────────────────────────────

def vortex_rho_exact(x, y, t):
    """Exact density at time t (vortex advects at u_inf, returns after t=Lx)."""
    xc = 0.0  # initial center x (half domain = 0)
    yc = 0.0  # initial center y
    xc_t = xc + U_INF * t
    yc_t = yc + V_INF * t
    dx = x - xc_t
    dy = y - yc_t
    # Minimum-image convention (periodic domain)
    dx -= LX * np.floor(dx / LX + 0.5)
    dy -= LY * np.floor(dy / LY + 0.5)
    r2 = dx**2 + dy**2
    dT = -(GAMMA - 1) * EPS**2 / (8 * GAMMA * np.pi**2) * np.exp(1 - r2)
    T  = T_INF + dT
    return RHO_INF * (T / T_INF) ** (1 / (GAMMA - 1))


def compute_l2_error(rho_num, nx, ny, t_final):
    hx = LX / nx
    hy = LY / ny
    xs = XMIN + (np.arange(nx) + 0.5) * hx
    ys = YMIN + (np.arange(ny) + 0.5) * hy
    X, Y = np.meshgrid(xs, ys)
    rho_ex = vortex_rho_exact(X, Y, t_final)
    err    = rho_num - rho_ex
    return np.sqrt(np.sum(err**2) * hx * hy)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='gpu')
    parser.add_argument('--basedir', default='results/vortex_conv')
    args = parser.parse_args()

    basedir = Path(args.basedir)

    errors_hll   = []
    errors_muscl = []
    hs           = []

    # Estimate c_inf from background state
    p_inf = RHO_INF ** GAMMA / GAMMA
    c_inf = (GAMMA * p_inf / RHO_INF) ** 0.5

    for N in GRIDS:
        h = LX / N
        hs.append(h)
        steps     = STEPS_FACTOR * N
        dt_approx = 0.4 * h / (U_INF + c_inf)
        t_final   = steps * dt_approx

        # --- HLL ---
        d = basedir / f"N{N}_{args.device}"
        vtk_files = sorted(d.glob("*.vtk"))
        if not vtk_files:
            print(f"  WARNING: no .vtk files in {d}, skipping N={N}")
            errors_hll.append(None)
        else:
            rho, nx_r, ny_r = read_vtk_density(vtk_files[-1])
            e = compute_l2_error(rho, nx_r, ny_r, t_final)
            errors_hll.append(e)
            print(f"HLL   N={N:4d}: h={h:.4f}, t_final={t_final:.3f}, L2={e:.4e}")

        # --- MUSCL ---
        d = basedir / f"N{N}_{args.device}_muscl"
        vtk_files = sorted(d.glob("*.vtk"))
        if not vtk_files:
            print(f"  WARNING: no .vtk files in {d}, skipping N={N}")
            errors_muscl.append(None)
        else:
            rho, nx_r, ny_r = read_vtk_density(vtk_files[-1])
            e = compute_l2_error(rho, nx_r, ny_r, t_final)
            errors_muscl.append(e)
            print(f"MUSCL N={N:4d}: h={h:.4f}, t_final={t_final:.3f}, L2={e:.4e}")

    # ─── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))

    valid_hll   = [(hs[i], errors_hll[i])   for i in range(len(GRIDS)) if errors_hll[i]]
    valid_muscl = [(hs[i], errors_muscl[i]) for i in range(len(GRIDS)) if errors_muscl[i]]

    if valid_hll:
        hv, ev = zip(*valid_hll)
        ax.loglog(hv, ev, 'o-', label='HLL (1-й порядок)', color='steelblue', linewidth=1.5)
    if valid_muscl:
        hv, ev = zip(*valid_muscl)
        ax.loglog(hv, ev, 's-', label='MUSCL+SSP-RK2 (2-й порядок)', color='tomato', linewidth=1.5)

    # Reference slopes
    h_ref = np.array([hs[0], hs[-1]])
    if valid_hll:
        e0 = valid_hll[0][1]
        ax.loglog(h_ref, e0 * (h_ref / h_ref[0])**1, '--', color='steelblue',
                  alpha=0.5, label='наклон 1')
    if valid_muscl:
        e0 = valid_muscl[0][1]
        ax.loglog(h_ref, e0 * (h_ref / h_ref[0])**2, '--', color='tomato',
                  alpha=0.5, label='наклон 2')

    ax.set_xlabel('$h$ (шаг сетки)', fontsize=12)
    ax.set_ylabel('$\\|e\\|_{L_2}$ (плотность)', fontsize=12)
    ax.set_title('Сходимость: изотропный вихрь Йи–Сандхама', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.tight_layout()

    outpath = basedir / 'vortex_convergence.pdf'
    plt.savefig(str(outpath), dpi=150)
    print(f"\nPlot saved: {outpath}")
    print(f"Copy to diploma: cp {outpath} diploma/figures/vortex_convergence.pdf")

    # ─── Print table ──────────────────────────────────────────────────────────
    print("\n=== Таблица сходимости (скопируйте в chapter5_experiment.tex) ===")
    print(f"{'N':>6}  {'h':>8}  {'L2_HLL':>12}  {'p_HLL':>7}  {'L2_MUSCL':>12}  {'p_MUSCL':>8}")
    prev_e_hll = prev_e_muscl = None
    prev_h_hll = prev_h_muscl = None
    for i, N in enumerate(GRIDS):
        h  = hs[i]
        eh = errors_hll[i]
        em = errors_muscl[i]
        ph = '---' if (prev_e_hll is None or eh is None) else \
             f'{np.log(prev_e_hll / eh) / np.log(prev_h_hll / h):.2f}'
        pm = '---' if (prev_e_muscl is None or em is None) else \
             f'{np.log(prev_e_muscl / em) / np.log(prev_h_muscl / h):.2f}'
        eh_s = f'{eh:.4e}' if eh else '---'
        em_s = f'{em:.4e}' if em else '---'
        print(f"{N:>6}  {h:>8.5f}  {eh_s:>12}  {ph:>7}  {em_s:>12}  {pm:>8}")
        if eh:
            prev_e_hll, prev_h_hll = eh, h
        if em:
            prev_e_muscl, prev_h_muscl = em, h


if __name__ == '__main__':
    main()
