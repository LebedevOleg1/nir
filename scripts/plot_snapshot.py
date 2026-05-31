#!/usr/bin/env python3
"""
Generate a static density PNG snapshot from the last VTK output file.

Reads the last output_*.vtk in the given directory, plots the density field,
and saves a publication-quality PNG.

Usage:
    python3 scripts/plot_snapshot.py <vtk_dir> <out_png> [--field Density] [--cmap plasma]

Examples (run from project root after cluster runs):
    python3 scripts/plot_snapshot.py results/kh_mcnally_512   diploma/figures/kh_mcnally_512.png
    python3 scripts/plot_snapshot.py results/rt_liska_64x384  diploma/figures/rt_liska_64x384.png
    python3 scripts/plot_snapshot.py results/rt_liska_120x480 diploma/figures/rt_liska_120x480.png
"""
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ─── Binary VTK reader ───────────────────────────────────────────────────────

def _read_line(f):
    line = b""
    while True:
        ch = f.read(1)
        if not ch or ch == b"\n":
            return line.decode("ascii", errors="replace").strip()
        line += ch


def read_vtk(vtk_path, field_name="Density"):
    """Return (x_coords, y_coords, field_2d) from a binary RECTILINEAR_GRID file."""
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

        _read_line(f)
        x_coords = np.frombuffer(f.read(nx * 4), dtype=">f4").astype(float)
        f.read(1)

        _read_line(f)
        y_coords = np.frombuffer(f.read(ny * 4), dtype=">f4").astype(float)
        f.read(1)

        _read_line(f)   # Z_COORDINATES 1 float
        f.read(4 + 1)

        _read_line(f)   # POINT_DATA npts

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
                if name == field_name:
                    arr = np.frombuffer(raw, dtype=">f4").astype(float)
                    return x_coords, y_coords, arr.reshape((ny, nx))
                f.read(1)

    raise RuntimeError(f"Field '{field_name}' not found in {vtk_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

_CMAPS = {
    "Density":   "plasma",
    "Pressure":  "inferno",
    "VelocityX": "RdBu_r",
    "VelocityY": "RdBu_r",
    "Mach":      "viridis",
}

_LABELS = {
    "Density":   r"$\rho$",
    "Pressure":  r"$p$",
    "VelocityX": r"$u$",
    "VelocityY": r"$v$",
    "Mach":      "Mach",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vtk_path",
                        help="Directory with output_*.vtk files (uses last frame), "
                             "or a specific .vtk file")
    parser.add_argument("out_png", help="Output PNG path")
    parser.add_argument("--field", default="Density",
                        choices=list(_CMAPS.keys()))
    parser.add_argument("--cmap", default=None,
                        help="Override colormap (default: field-specific)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-colorbar", action="store_true")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    p = Path(args.vtk_path)
    if p.is_file() and p.suffix == ".vtk":
        last_vtk = p
    else:
        vtk_files = sorted(p.glob("output_*.vtk"))
        if not vtk_files:
            print(f"Error: no output_*.vtk files in '{p}'")
            sys.exit(1)
        last_vtk = vtk_files[-1]
    print(f"Reading {last_vtk} ...")

    x, y, field = read_vtk(last_vtk, args.field)
    nx, ny = len(x), len(y)
    print(f"Grid: {nx} x {ny},  "
          f"{args.field} range [{field.min():.4f}, {field.max():.4f}]")

    cmap = args.cmap or _CMAPS.get(args.field, "viridis")

    # Aspect ratio: physical domain proportions
    domain_w = x[-1] - x[0]
    domain_h = y[-1] - y[0]
    fig_h = 5.0
    fig_w = fig_h * (domain_w / domain_h)
    fig_w = max(2.5, min(fig_w, 10.0))  # clamp

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        field, origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        cmap=cmap, aspect="equal",
        interpolation="bilinear",
    )

    if not args.no_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(_LABELS.get(args.field, args.field), fontsize=11)

    step_str = last_vtk.stem.replace("output_", "")
    title = args.title or f"{args.field},  step {step_str}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("$x$", fontsize=10)
    ax.set_ylabel("$y$", fontsize=10)

    plt.tight_layout()
    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
