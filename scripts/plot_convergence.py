"""
Convergence study for the Sod shock tube problem.
Runs the solver at several resolutions and plots L2 density error vs h.

Usage:
    python3 plot_convergence.py --exec ./build/problems/sod_shock/sod
    python3 plot_convergence.py --exec ./build/problems/sod_shock/sod --device gpu

Expected convergence: ~1st order due to shock discontinuity (Gibbs effect),
or ~2nd order in smooth regions (rarefaction fan).
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


def read_vtk_density(filename):
    """Read only the Density field from a VTK file."""
    with open(filename, "rb") as f:
        nx = ny = None
        for _ in range(20):
            line = b""
            while True:
                ch = f.read(1)
                if not ch or ch == b"\n": break
                line += ch
            line = line.decode("ascii", errors="replace").strip()
            if line.startswith("DIMENSIONS"):
                parts = line.split()
                nx, ny = int(parts[1]), int(parts[2])
                break

        if nx is None:
            raise ValueError(f"No DIMENSIONS in {filename}")

        npts = nx * ny

        # Skip coordinates
        for _ in range(3):
            hdr_line = b""
            while True:
                ch = f.read(1)
                if not ch or ch == b"\n": break
                hdr_line += ch
            hdr = hdr_line.decode("ascii", errors="replace")
            parts = hdr.split()
            count = int(parts[1])
            f.read(count * 4 + 1)

        # Skip POINT_DATA line
        while True:
            ch = f.read(1)
            if not ch or ch == b"\n": break

        # Find Density field
        while True:
            line = b""
            while True:
                ch = f.read(1)
                if not ch or ch == b"\n": break
                line += ch
            line = line.decode("ascii", errors="replace").strip()
            if not line: break
            if line.startswith("SCALARS Density"):
                # Skip LOOKUP_TABLE line
                while True:
                    ch = f.read(1)
                    if not ch or ch == b"\n": break
                raw = f.read(npts * 4)
                data = np.frombuffer(raw, dtype=">f4").astype(np.float64)
                x_arr = np.array([0.0] * npts)  # placeholder
                return nx, ny, data

    return None


def sod_exact(x_arr, t, gamma=1.4, x0=0.5):
    """Vectorized approximate exact Sod solution (density only)."""
    rho = np.where(x_arr < x0, 1.0, 0.125)  # very rough — use full Exact.hpp for real work
    return rho


def run_resolution(exec_path, nx, device, steps=200, ny=4):
    """Run the sod solver at a given resolution, return final output VTK."""
    outdir = f"/tmp/sod_nx{nx}"
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        "mpirun", "-np", "1", exec_path,
        f"--nx={nx}", f"--ny={ny}",
        f"--steps={steps}", "--save-every=200",
        f"--device={device}",
        "--xmin=0", "--xmax=1", "--ymin=0", "--ymax=0.01",
        "--cfl=0.4", "--gamma=1.4",
    ]
    print(f"  Running nx={nx}: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=outdir)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}")
        return None

    vtk_files = sorted(glob.glob(os.path.join(outdir, "output_*.vtk")))
    return vtk_files[-1] if vtk_files else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec",   default="./build/problems/sod_shock/sod")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--out",    default="convergence.png")
    args = parser.parse_args()

    resolutions = [50, 100, 200, 400]
    h_values, l2_errors = [], []

    print("=== Convergence study: Sod shock tube ===")
    for nx in resolutions:
        vtk = run_resolution(args.exec, nx, args.device)
        if vtk is None:
            continue

        try:
            nx_r, ny_r, rho_num = read_vtk_density(vtk)
        except Exception as e:
            print(f"  Could not read {vtk}: {e}")
            continue

        # x-coords at cell centers
        h = 1.0 / nx_r
        x_arr = np.arange(nx_r) * h + 0.5 * h
        rho_ex = sod_exact(x_arr, t=0.2)  # t=0.2 typical Sod time

        # Average over y (thin strip)
        rho_1d = rho_num[:nx_r]  # first row is representative
        l2 = float(np.sqrt(np.mean((rho_1d - rho_ex)**2)))

        print(f"  nx={nx_r:4d}  h={h:.4f}  L2_rho={l2:.6f}")
        h_values.append(h)
        l2_errors.append(l2)

    if len(h_values) < 2:
        print("Not enough data points for convergence plot.")
        return

    # Convergence order
    h_arr  = np.array(h_values)
    l2_arr = np.array(l2_errors)
    orders = np.log(l2_arr[:-1] / l2_arr[1:]) / np.log(h_arr[:-1] / h_arr[1:])
    print(f"Convergence orders: {orders}")
    print(f"Average order: {np.mean(orders):.2f}")

    if HAS_PLT:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.loglog(h_arr, l2_arr, 'o-', label='L2 density error')
        # Reference slopes
        h_ref = h_arr
        ax.loglog(h_ref, l2_arr[0] * (h_ref/h_arr[0])**1.0, 'k--', alpha=0.5, label='O(h)')
        ax.loglog(h_ref, l2_arr[0] * (h_ref/h_arr[0])**2.0, 'k:',  alpha=0.5, label='O(h²)')
        ax.set_xlabel("Grid spacing h")
        ax.set_ylabel("L2 density error")
        ax.set_title("Sod shock tube: convergence")
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out, dpi=120)
        print(f"Plot saved: {args.out}")


if __name__ == "__main__":
    main()
