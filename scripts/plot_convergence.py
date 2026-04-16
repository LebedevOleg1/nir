"""
Convergence study for the Sod shock tube problem.
Runs the solver at several resolutions and plots L2 density error vs h.

Usage:
    python3 plot_convergence.py --exec ./build/problems/sod_shock/sod
    python3 plot_convergence.py --exec ./build/problems/sod_shock/sod --device gpu

Expected convergence: ~1st order (dominant error near the shock discontinuity).
"""

import subprocess
import numpy as np
import struct
import glob
import os
import argparse
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# =============================================================================
# Exact Riemann solver for the Sod shock tube (ported from Exact.hpp)
# Reference: Toro, "Riemann Solvers and Numerical Methods", Ch. 4
# =============================================================================

def sod_exact_density(x_arr, t,
                      gamma=1.4,
                      rho_L=1.0,  u_L=0.0, p_L=1.0,
                      rho_R=0.125, u_R=0.0, p_R=0.1,
                      x0=0.5):
    """
    Vectorized exact density for the Sod problem at physical time t.
    x_arr : 1D numpy array of cell-centre x-positions
    """
    if t < 1e-12:
        return np.where(x_arr < x0, rho_L, rho_R)

    g  = gamma
    g1 = (g - 1.0) / (2.0 * g)
    g2 = (g + 1.0) / (2.0 * g)
    g4 = 2.0 / (g - 1.0)
    g5 = 2.0 / (g + 1.0)
    g6 = (g - 1.0) / (g + 1.0)
    g7 = (g - 1.0) / 2.0

    cL = np.sqrt(g * p_L / rho_L)
    cR = np.sqrt(g * p_R / rho_R)

    # Newton-Raphson for p*
    p_star = 0.5 * (p_L + p_R)
    for _ in range(100):
        if p_star <= p_L:
            r   = p_star / p_L
            fL  = g4 * cL * (r**g1 - 1.0)
            dfL = (1.0 / (rho_L * cL)) * r**(-g2)
        else:
            AL  = g5 / rho_L
            BL  = g6 * p_L
            sqt = np.sqrt(AL / (p_star + BL))
            fL  = (p_star - p_L) * sqt
            dfL = sqt * (1.0 - (p_star - p_L) / (2.0 * (p_star + BL)))

        if p_star <= p_R:
            r   = p_star / p_R
            fR  = g4 * cR * (r**g1 - 1.0)
            dfR = (1.0 / (rho_R * cR)) * r**(-g2)
        else:
            AR  = g5 / rho_R
            BR  = g6 * p_R
            sqt = np.sqrt(AR / (p_star + BR))
            fR  = (p_star - p_R) * sqt
            dfR = sqt * (1.0 - (p_star - p_R) / (2.0 * (p_star + BR)))

        dp = -(fL + fR + u_R - u_L) / (dfL + dfR)
        p_star += dp
        if abs(dp) < 1e-8 * p_star:
            break

    # u*
    fR_star = (g4 * cR * ((p_star / p_R)**g1 - 1.0) if p_star <= p_R
               else (p_star - p_R) * np.sqrt((g5 / rho_R) / (p_star + g6 * p_R)))
    fL_star = (g4 * cL * ((p_star / p_L)**g1 - 1.0) if p_star <= p_L
               else (p_star - p_L) * np.sqrt((g5 / rho_L) / (p_star + g6 * p_L)))
    u_star  = 0.5 * (u_L + u_R + fR_star - fL_star)

    xi = (x_arr - x0) / t   # characteristic variable (vectorized)

    rho_out = np.full_like(x_arr, rho_R)

    # Left undisturbed
    rho_out[xi <= u_L - cL] = rho_L

    if p_star <= p_L:
        cL_star = cL * (p_star / p_L)**g1
        # Rarefaction fan
        fan = (xi > u_L - cL) & (xi <= u_star - cL_star)
        rho_out[fan] = rho_L * (g5 + g6 / cL * (u_L - xi[fan]))**g4
        # Left star region
        lstar = (xi > u_star - cL_star) & (xi <= u_star)
        rho_out[lstar] = rho_L * (p_star / p_L)**(1.0 / g)
    else:
        rho_Ls = rho_L * ((g + 1) * p_star + (g - 1) * p_L) / \
                         ((g - 1) * p_star + (g + 1) * p_L)
        S_L = u_L - cL * np.sqrt((g + 1) / (2 * g) * p_star / p_L + g1)
        rho_out[(xi > u_L - cL) & (xi <= S_L)]  = rho_L
        rho_out[(xi > S_L)      & (xi <= u_star)] = rho_Ls

    # Right star region
    rho_Rs = (rho_R * (p_star / p_R)**(1.0 / g) if p_star <= p_R
              else rho_R * ((g + 1) * p_star + (g - 1) * p_R) /
                           ((g - 1) * p_star + (g + 1) * p_R))
    if p_star <= p_R:
        cR_star = cR * (p_star / p_R)**g1
        rho_out[(xi > u_star) & (xi <= u_star + cR_star)] = rho_Rs
        # Right rarefaction fan
        rfan = (xi > u_star + cR_star) & (xi < u_R + cR)
        rho_out[rfan] = rho_R * (g5 - g6 / cR * (u_R - xi[rfan]))**g4
    else:
        S_R = u_R + cR * np.sqrt((g + 1) / (2 * g) * p_star / p_R + g1)
        rho_out[(xi > u_star) & (xi < S_R)] = rho_Rs

    # Right undisturbed
    rho_out[xi >= u_R + cR] = rho_R

    return rho_out


# =============================================================================
# VTK reader: returns x_coords and density (first row)
# =============================================================================

def _rl(f):
    line = b""
    while True:
        ch = f.read(1)
        if not ch or ch == b"\n":
            return line.decode("ascii", errors="replace").strip()
        line += ch


def read_vtk_density(filename):
    """Return (x_coords_1d, density_1d) from a VTK RECTILINEAR_GRID file."""
    with open(filename, "rb") as f:
        nx = ny = None
        for _ in range(20):
            line = _rl(f)
            if line.startswith("DIMENSIONS"):
                nx, ny = int(line.split()[1]), int(line.split()[2])
                break
        if nx is None:
            raise ValueError(f"No DIMENSIONS in {filename}")

        npts = nx * ny

        _rl(f)  # X_COORDINATES
        x_coords = np.frombuffer(f.read(nx * 4), dtype=">f4").astype(np.float64)
        f.read(1)

        _rl(f)  # Y_COORDINATES
        f.read(ny * 4 + 1)

        _rl(f)  # Z_COORDINATES
        f.read(4 + 1)

        _rl(f)  # POINT_DATA

        while True:
            line = _rl(f)
            if not line:
                break
            if line.startswith("SCALARS Density"):
                _rl(f)  # LOOKUP_TABLE
                data = np.frombuffer(f.read(npts * 4), dtype=">f4").astype(np.float64)
                return x_coords, data[:nx]   # first y-row is representative

    return None, None


# =============================================================================
# Sod constants (must match problems/sod_shock/Physics.hpp)
# =============================================================================
GAMMA   = 1.4
C_MAX   = float(np.sqrt(GAMMA * 1.0 / 1.0))  # sqrt(gamma * p_L / rho_L) ≈ 1.1832
T_FINAL = 0.20


def steps_to_reach(nx, t=T_FINAL, cfl=0.4, xmax=1.0):
    h  = xmax / nx
    dt = cfl * h / C_MAX
    return max(1, int(np.ceil(t / dt)))


def run_resolution(exec_path, nx, device, ny=4):
    outdir = f"/tmp/sod_conv_nx{nx}"
    os.makedirs(outdir, exist_ok=True)

    n_steps  = steps_to_reach(nx)
    t_est    = n_steps * 0.4 / (nx * C_MAX)
    cmd = [
        "mpirun", "-np", "1", exec_path,
        f"--nx={nx}", f"--ny={ny}",
        f"--steps={n_steps}", f"--save-every={n_steps}",
        f"--device={device}",
        "--xmin=0", "--xmax=1", "--ymin=0", "--ymax=0.01",
        "--cfl=0.4", "--gamma=1.4",
    ]
    print(f"  nx={nx:4d}  steps={n_steps}  t_est={t_est:.4f}  running...", end=" ", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=outdir)
    if result.returncode != 0:
        print(f"ERROR\n  {result.stderr[:300]}")
        return None, None
    print("done")
    vtk_files = sorted(glob.glob(os.path.join(outdir, "output_*.vtk")))
    return (vtk_files[-1] if vtk_files else None), t_est


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec",   default="./build/problems/sod_shock/sod")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--out",    default="convergence.png")
    args = parser.parse_args()

    resolutions = [50, 100, 200, 400, 800]
    h_values, l2_errors = [], []

    print("=== Convergence study: Sod shock tube ===")
    print(f"Target physical time: t ≈ {T_FINAL}")
    for nx in resolutions:
        vtk, t_actual = run_resolution(args.exec, nx, args.device)
        if vtk is None:
            continue
        try:
            x_arr, rho_num = read_vtk_density(vtk)
        except Exception as e:
            print(f"  Could not read {vtk}: {e}")
            continue
        if x_arr is None:
            continue

        rho_ex = sod_exact_density(x_arr, t_actual)
        h  = 1.0 / nx
        l2 = float(np.sqrt(np.mean((rho_num - rho_ex)**2)))
        print(f"  nx={nx:4d}  h={h:.5f}  L2_rho={l2:.6f}")
        h_values.append(h)
        l2_errors.append(l2)

    if len(h_values) < 2:
        print("Not enough data points.")
        return

    h_arr  = np.array(h_values)
    l2_arr = np.array(l2_errors)
    orders = np.log(l2_arr[:-1] / l2_arr[1:]) / np.log(h_arr[:-1] / h_arr[1:])
    print(f"\nConvergence orders: {np.round(orders, 3)}")
    print(f"Average order: {np.mean(orders):.2f}")

    if HAS_PLT:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(h_arr, l2_arr, 'o-', color='steelblue', linewidth=2,
                  markersize=7, label='L2 density error')
        ax.loglog(h_arr, l2_arr[0] * (h_arr / h_arr[0])**1.0,
                  'k--', alpha=0.5, label='O(h)  — 1st order')
        ax.loglog(h_arr, l2_arr[0] * (h_arr / h_arr[0])**2.0,
                  'k:',  alpha=0.5, label='O(h²) — 2nd order')
        ax.set_xlabel("Grid spacing  h = 1/nx")
        ax.set_ylabel(r"L2 density error  $\|\rho - \rho_\mathrm{exact}\|_2$")
        ax.set_title(f"Sod shock tube — spatial convergence  (t = {T_FINAL})")
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"Plot saved: {args.out}")


if __name__ == "__main__":
    main()
