#!/usr/bin/env python3
"""
Diagnostic: print per-frame statistics of a field across all VTK snapshots
in a directory. Shows whether an instability is growing (std rising) or
the field is essentially frozen (std flat).

Usage:
    python3 scripts/diag_field.py <vtk_dir> [field]

Examples:
    python3 scripts/diag_field.py results/rt_liska_120x480
    python3 scripts/diag_field.py results/kh_mcnally_512 Density
    python3 scripts/diag_field.py results/kh_mcnally_512 VelocityY
"""
import sys
import glob
import numpy as np

sys.path.insert(0, 'scripts')
from plot_snapshot import read_vtk


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    vtk_dir = sys.argv[1].rstrip('/')
    field = sys.argv[2] if len(sys.argv) > 2 else 'Density'

    files = sorted(glob.glob(vtk_dir + '/output_*.vtk'))
    if not files:
        print('No output_*.vtk files in', vtk_dir)
        sys.exit(1)

    print('Field:', field, '| frames:', len(files))
    print('%-22s %10s %10s %10s %10s' % ('file', 'min', 'max', 'mean', 'std'))
    for f in files:
        try:
            x, y, d = read_vtk(f, field)
        except RuntimeError as e:
            print(f.split('/')[-1], 'ERROR:', e)
            continue
        name = f.split('/')[-1]
        print('%-22s %10.4f %10.4f %10.4f %10.4f'
              % (name, d.min(), d.max(), d.mean(), d.std()))


if __name__ == '__main__':
    main()
