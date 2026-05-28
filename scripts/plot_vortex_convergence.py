#!/usr/bin/env python3
"""
Isentropic vortex convergence analysis.

Reads the final VTK snapshot from each grid run, computes L2 error in density
vs the exact solution, and plots convergence order for HLL (1st order) and
MUSCL (2nd order).

The exact solution at t_final: vortex center has moved by (u_inf * t_final, 0).
Since the domain is periodic and t_final ≈ one period (Lx/u_inf = 10), the
exact solution is just the initial condition. We therefore compare the final
density field to the analytical vortex formula evaluated at t_final.

Usage: python3 scripts/plot_vortex_convergence.py [--device gpu|cpu]
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xml.etree.ElementTree as ET
import struct
import base64

# ─── Vortex parameters (must match Physics.hpp) ─────────────────────────────
GAMMA   = 1.4
U_INF   = 1.0
V_INF   = 0.0
RHO_INF = 1.0
T_INF   = 1.0
EPS     = 5.0
XMIN, XMAX = -5.0, 5.0
YMIN, YMAX = -5.0, 5.0
LX = XMAX - XMIN
LY = YMAX - YMIN

GRIDS = [32, 64, 128, 256]
STEPS_FACTOR = 6  # must match run script


def vortex_rho_exact(x, y, t):
    """Exact density at time t."""
    xc = 0.5 * (XMIN + XMAX) + U_INF * t
    yc = 0.5 * (YMIN + YMAX) + V_INF * t
    # Wrap to domain with minimum image
    dx = x - xc
    dy = y - yc
    dx -= LX * np.floor(dx / LX + 0.5)
    dy -= LY * np.floor(dy / LY + 0.5)
    r2 = dx**2 + dy**2
    dT = -(GAMMA - 1) * EPS**2 / (8 * GAMMA * np.pi**2) * np.exp(1 - r2)
    T  = T_INF + dT
    return RHO_INF * (T / T_INF) ** (1 / (GAMMA - 1))


def read_vtk_density(vtk_path):
    """Read density field from a VTK XML (.vti) file produced by our solver."""
    tree = ET.parse(vtk_path)
    root = tree.getroot()

    # Get grid dimensions
    piece = root.find('.//Piece')
    ext   = piece.get('Extent').split()
    # Extent: x0 x1 y0 y1 0 0 (node indices)
    nx = int(ext[1]) - int(ext[0])
    ny = int(ext[3]) - int(ext[2])

    # Find density array
    for da in root.iter('DataArray'):
        if da.get('Name') == 'rho':
            raw = base64.b64decode(da.text.strip())
            # First 8 bytes = uint64 byte count
            n_bytes = struct.unpack_from('<Q', raw, 0)[0]
            arr = np.frombuffer(raw, dtype=np.float32, offset=8, count=nx * ny)
            return arr.reshape((ny, nx)), nx, ny

    raise RuntimeError(f"Could not find 'rho' in {vtk_path}")


def compute_l2_error(rho_num, nx, ny, t_final):
    hx = LX / nx
    hy = LY / ny
    xs = XMIN + (np.arange(nx) + 0.5) * hx
    ys = YMIN + (np.arange(ny) + 0.5) * hy
    X, Y = np.meshgrid(xs, ys)
    rho_ex = vortex_rho_exact(X, Y, t_final)
    err    = rho_num - rho_ex
    return np.sqrt(np.sum(err**2) * hx * hy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='gpu')
    parser.add_argument('--basedir', default='results/vortex_conv')
    args = parser.parse_args()

    basedir = Path(args.basedir)

    errors_hll   = []
    errors_muscl = []
    hs           = []

    for N in GRIDS:
        h = LX / N
        hs.append(h)
        steps    = STEPS_FACTOR * N
        # Approximate dt at this N for c_inf estimation
        p_inf  = RHO_INF ** GAMMA / GAMMA
        c_inf  = (GAMMA * p_inf / RHO_INF) ** 0.5
        dt_approx = 0.4 * h / (U_INF + c_inf)
        t_final   = steps * dt_approx

        # --- HLL ---
        d = basedir / f"N{N}_{args.device}"
        vtk_files = sorted(d.glob("*.vti"))
        if not vtk_files:
            print(f"  WARNING: no VTK files in {d}, skipping N={N}")
            errors_hll.append(None)
        else:
            rho, nx_r, ny_r = read_vtk_density(vtk_files[-1])
            e = compute_l2_error(rho, nx_r, ny_r, t_final)
            errors_hll.append(e)
            print(f"HLL   N={N:4d}: h={h:.4f}, t_final={t_final:.3f}, L2={e:.4e}")

        # --- MUSCL ---
        d = basedir / f"N{N}_{args.device}_muscl"
        vtk_files = sorted(d.glob("*.vti"))
        if not vtk_files:
            print(f"  WARNING: no VTK files in {d}, skipping N={N}")
            errors_muscl.append(None)
        else:
            rho, nx_r, ny_r = read_vtk_density(vtk_files[-1])
            e = compute_l2_error(rho, nx_r, ny_r, t_final)
            errors_muscl.append(e)
            print(f"MUSCL N={N:4d}: h={h:.4f}, t_final={t_final:.3f}, L2={e:.4e}")

    # Compute convergence orders
    def order(e, h):
        orders = [None]
        for i in range(1, len(e)):
            if e[i] and e[i - 1]:
                orders.append(np.log(e[i - 1] / e[i]) / np.log(h[i - 1] / h[i]))
            else:
                orders.append(None)
        return orders

    hs_arr = np.array(hs)
    e_hll   = [x for x in errors_hll   if x is not None]
    e_muscl = [x for x in errors_muscl if x is not None]

    # ─── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.5))

    valid_hll   = [(hs[i], errors_hll[i])   for i in range(len(GRIDS)) if errors_hll[i]]
    valid_muscl = [(hs[i], errors_muscl[i]) for i in range(len(GRIDS)) if errors_muscl[i]]

    if valid_hll:
        hv, ev = zip(*valid_hll)
        ax.loglog(hv, ev, 'o-', label='HLL (1-й порядок)', color='steelblue')
    if valid_muscl:
        hv, ev = zip(*valid_muscl)
        ax.loglog(hv, ev, 's-', label='MUSCL (2-й порядок)', color='tomato')

    # Reference slopes
    h_ref = np.array([hs[0], hs[-1]])
    if valid_hll:
        e0 = valid_hll[0][1]
        ax.loglog(h_ref, e0 * (h_ref / h_ref[0])**1, '--', color='steelblue',
                  alpha=0.5, label='slope 1')
    if valid_muscl:
        e0 = valid_muscl[0][1]
        ax.loglog(h_ref, e0 * (h_ref / h_ref[0])**2, '--', color='tomato',
                  alpha=0.5, label='slope 2')

    ax.set_xlabel('$h$ (шаг сетки)', fontsize=12)
    ax.set_ylabel('$L_2$-ошибка плотности', fontsize=12)
    ax.set_title('Сходимость: вихрь Йи–Сандхама', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.tight_layout()

    outpath = basedir / 'vortex_convergence.pdf'
    plt.savefig(outpath, dpi=150)
    print(f"\nPlot saved to {outpath}")

    # ─── Print table ─────────────────────────────────────────────────────────
    print("\nТаблица сходимости:")
    print(f"{'N':>6}  {'h':>8}  {'L2_HLL':>12}  {'p_HLL':>7}  {'L2_MUSCL':>12}  {'p_MUSCL':>8}")
    prev_h_hll = prev_h_muscl = None
    prev_e_hll = prev_e_muscl = None
    for i, N in enumerate(GRIDS):
        h  = hs[i]
        eh = errors_hll[i]
        em = errors_muscl[i]
        ph = '---' if (prev_e_hll is None or eh is None) else \
             f'{np.log(prev_e_hll/eh)/np.log(prev_h_hll/h):.2f}'
        pm = '---' if (prev_e_muscl is None or em is None) else \
             f'{np.log(prev_e_muscl/em)/np.log(prev_h_muscl/h):.2f}'
        print(f"{N:>6}  {h:>8.5f}  "
              f"{eh if eh else '---':>12}  {ph:>7}  "
              f"{em if em else '---':>12}  {pm:>8}")
        if eh: prev_e_hll,   prev_h_hll   = eh, h
        if em: prev_e_muscl, prev_h_muscl = em, h


if __name__ == '__main__':
    main()
