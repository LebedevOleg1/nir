"""
Universal FVM solver animation script.
Auto-detects available fields from VTK output; works for all physics types.

Usage:
    python3 animate.py                           # auto-pick first field, kh_animation.gif
    python3 animate.py --field Density           # explicit field
    python3 animate.py --field Temperature       # heat problem
    python3 animate.py --out rt.gif --fps 20
    python3 animate.py --out kh.mp4             # MP4 (requires ffmpeg)
    python3 animate.py --vmin 0.8 --vmax 2.2    # fixed color range
    python3 animate.py --dir /path/to/vtks       # look in specific dir

Available fields (auto-detected from VTK files):
    Euler:    Density, VelocityX, VelocityY, Pressure, Mach
    Heat:     Temperature
    Diffusion: Concentration
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import glob
import os
import sys
import argparse


# =============================================================================
# Binary VTK reader (RECTILINEAR_GRID, big-endian floats)
# =============================================================================

def _read_line(f):
    line = b""
    while True:
        ch = f.read(1)
        if not ch or ch == b"\n":
            return line.decode("ascii", errors="replace").strip()
        line += ch


def read_vtk(filename):
    """Return (x_coords, y_coords, {field_name: 2D array})."""
    with open(filename, "rb") as f:
        nx = ny = None
        for _ in range(20):
            line = _read_line(f)
            if line.startswith("DIMENSIONS"):
                parts = line.split()
                nx, ny = int(parts[1]), int(parts[2])
                break
        if nx is None:
            raise ValueError(f"No DIMENSIONS header in {filename}")

        npts = nx * ny

        _read_line(f)                       # X_COORDINATES N float
        x_coords = np.frombuffer(f.read(nx * 4), dtype=">f4").astype(np.float32)
        f.read(1)

        _read_line(f)                       # Y_COORDINATES N float
        y_coords = np.frombuffer(f.read(ny * 4), dtype=">f4").astype(np.float32)
        f.read(1)

        _read_line(f)                       # Z_COORDINATES 1 float
        f.read(4 + 1)

        _read_line(f)                       # POINT_DATA N

        fields = {}
        while True:
            line = _read_line(f)
            if not line:
                break
            if line.startswith("SCALARS"):
                name = line.split()[1]
                _read_line(f)               # LOOKUP_TABLE default
                raw = f.read(npts * 4)
                if len(raw) < npts * 4:
                    break
                data = np.frombuffer(raw, dtype=">f4").astype(np.float32)
                fields[name] = data.reshape(ny, nx)
                f.read(1)

    return x_coords, y_coords, fields


# =============================================================================
# Colormap / label tables
# =============================================================================

_CMAPS = {
    "Density":       "plasma",
    "VelocityX":     "RdBu_r",
    "VelocityY":     "RdBu_r",
    "Pressure":      "inferno",
    "Mach":          "viridis",
    "Temperature":   "hot",
    "Concentration": "Blues",
}

_LABELS = {
    "Density":       "Density  ρ",
    "VelocityX":     "Velocity  u",
    "VelocityY":     "Velocity  v",
    "Pressure":      "Pressure  p",
    "Mach":          "Mach number",
    "Temperature":   "Temperature  T",
    "Concentration": "Concentration  c",
}

_PROBLEM_TITLES = {
    "Density":     "Kelvin–Helmholtz / Rayleigh–Taylor instability",
    "Temperature": "Heat equation with Gaussian source",
    "Pressure":    "Euler — pressure field",
    "Mach":        "Euler — Mach number",
}


# =============================================================================
# Animation builder
# =============================================================================

def make_animation(vtk_files, field, out_file, fps, dpi, vmin_arg, vmax_arg):
    x, y, flds = read_vtk(vtk_files[0])

    if field is None:
        field = next(iter(flds))
        print(f"Auto-selected field: {field}")

    if field not in flds:
        avail = list(flds.keys())
        print(f"Error: field '{field}' not found.  Available: {avail}")
        sys.exit(1)

    nx_pts, ny_pts = len(x), len(y)
    print(f"Field: {field}  |  grid: {nx_pts} x {ny_pts}  |  frames: {len(vtk_files)}")

    # --- colour range ---
    if vmin_arg is not None and vmax_arg is not None:
        vmin, vmax = vmin_arg, vmax_arg
    else:
        print("Scanning files for colour range...", end=" ", flush=True)
        vmin, vmax = np.inf, -np.inf
        for fname in vtk_files:
            _, _, flds_i = read_vtk(fname)
            d = flds_i.get(field)
            if d is None:
                continue
            if d.min() < vmin:
                vmin = float(d.min())
            if d.max() > vmax:
                vmax = float(d.max())
        print(f"[{vmin:.4f}, {vmax:.4f}]")

    colormap = _CMAPS.get(field, "viridis")
    if field in ("VelocityX", "VelocityY"):
        abs_max = max(abs(vmin), abs(vmax))
        vmin, vmax = -abs_max, abs_max

    # --- figure ---
    aspect = nx_pts / ny_pts
    fig_w  = 8.0
    fig_h  = fig_w / aspect + 0.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    norm = Normalize(vmin=vmin, vmax=vmax)
    _, _, flds = read_vtk(vtk_files[0])
    im = ax.imshow(flds[field], origin="lower",
                   extent=[x[0], x[-1], y[0], y[-1]],
                   cmap=colormap, norm=norm, aspect="auto",
                   interpolation="bilinear")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=9)
    cbar.outline.set_edgecolor("#555")
    cbar.set_label(_LABELS.get(field, field), color="white", fontsize=10)

    title = ax.set_title("", color="white", fontsize=11, pad=6)
    ax.set_xlabel("x", color="#aaa", fontsize=9)
    ax.set_ylabel("y", color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.tight_layout()

    def update(idx):
        _, _, flds_i = read_vtk(vtk_files[idx])
        im.set_data(flds_i[field])
        step_tag = os.path.splitext(os.path.basename(vtk_files[idx]))[0]
        step_tag = step_tag.replace("output_", "step ")
        title.set_text(f"{_LABELS.get(field, field)}  |  {step_tag}")
        return [im, title]

    print(f"Rendering {len(vtk_files)} frames...", end=" ", flush=True)
    ani = animation.FuncAnimation(fig, update, frames=len(vtk_files),
                                  interval=1000 // fps, blit=True)

    ext = os.path.splitext(out_file)[1].lower()
    if ext in (".mp4", ".mkv", ".avi"):
        ani.save(out_file,
                 writer=animation.FFMpegWriter(fps=fps, bitrate=4000,
                                               extra_args=["-vcodec", "libx264"]),
                 dpi=dpi)
    else:
        ani.save(out_file, writer=animation.PillowWriter(fps=fps), dpi=dpi)

    plt.close(fig)
    size_mb = os.path.getsize(out_file) / 1024 / 1024
    print(f"done  →  {out_file}  ({size_mb:.1f} MB)")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Animate FVM solver VTK output. Field is auto-detected.")
    p.add_argument("--field",  default=None,
                   help="Field name (Density/Temperature/Pressure/Mach/VelocityX/VelocityY). "
                        "Auto-detected from file if omitted.")
    p.add_argument("--out",    default="animation.gif",
                   help="Output filename (.gif or .mp4)")
    p.add_argument("--fps",    type=int,   default=15)
    p.add_argument("--dpi",    type=int,   default=100)
    p.add_argument("--vmin",   type=float, default=None)
    p.add_argument("--vmax",   type=float, default=None)
    p.add_argument("--dir",    default=".",
                   help="Directory containing output_*.vtk files")
    args = p.parse_args()

    vtk_files = sorted(glob.glob(os.path.join(args.dir, "output_*.vtk")))
    if not vtk_files:
        print(f"Error: no output_*.vtk files found in '{args.dir}'")
        sys.exit(1)

    print(f"Found {len(vtk_files)} VTK files in '{args.dir}'")
    make_animation(vtk_files, args.field, args.out,
                   args.fps, args.dpi, args.vmin, args.vmax)


if __name__ == "__main__":
    main()
