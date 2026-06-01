#!/bin/bash
# MPI correctness test: run the same Sod problem on 1 and on 2 MPI ranks with
# an identical full grid, then compare the final density fields. A correct
# domain decomposition + ghost exchange must reproduce the single-rank result
# up to rounding. Uses CPU so it does not depend on GPU availability.
#
# Run from the project root:
#   ./scripts/run_mpi_check.sh
set -e
ROOT=$(pwd)
BIN="$ROOT/build/problems/sod_shock/sod"
export OMP_NUM_THREADS=1

mkdir -p "$ROOT/results/mpi_check"
cd "$ROOT/results/mpi_check"

run() {  # nranks outdir
    local np=$1 out=$2
    mkdir -p "$out"; (cd "$out" && rm -f output_*.vtk &&
        mpirun -np $np "$BIN" \
            --device=cpu --muscl=true --hllc=true \
            --nx=400 --ny=8 --steps=200 --save-every=200 \
            --xmin=0 --xmax=1 --ymin=0 --ymax=0.02 \
            > run.log 2>&1)
}

echo "Running Sod on 1 rank ..."; run 1 np1
echo "Running Sod on 2 ranks ..."; run 2 np2

echo "Comparing final density fields (max abs difference):"
python3 - <<'PY'
import sys, glob
sys.path.insert(0, 'scripts' if __import__('os').path.isdir('scripts') else '../../scripts')
from plot_snapshot import read_vtk
import numpy as np
f1 = sorted(glob.glob('np1/output_*.vtk'))[-1]
f2 = sorted(glob.glob('np2/output_*.vtk'))[-1]
_, _, d1 = read_vtk(f1, 'Density')
_, _, d2 = read_vtk(f2, 'Density')
print('  1 rank :', f1, d1.shape)
print('  2 ranks:', f2, d2.shape)
diff = np.max(np.abs(d1 - d2))
print('  max|rho_1 - rho_2| = %.3e' % diff)
print('  => MPI decomposition is %s' % ('CORRECT (fields match)' if diff < 1e-5 else 'SUSPECT'))
PY
