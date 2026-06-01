#!/bin/bash
# MPI strong-scaling test on CPU: fixed problem size, vary the number of MPI
# ranks (1, 2, 4). Each rank is single-threaded (OMP_NUM_THREADS=1) so that the
# measured speed-up reflects the MPI domain decomposition alone, not threading.
# CPU is used because it provides as many independent execution units as needed,
# independent of GPU availability on the shared node.
#
# Run from the project root:
#   ./scripts/run_mpi_scaling.sh
set -e
ROOT=$(pwd)
BIN="$ROOT/build/problems/kelvin_helmholtz/kh"
export OMP_NUM_THREADS=1            # isolate MPI scaling from OpenMP

mkdir -p "$ROOT/results/mpi_scaling"
cd "$ROOT/results/mpi_scaling"

NX=512; NY=512; STEPS=40
for NP in 1 2 4; do
    echo "######## MPI ranks = $NP, grid ${NX}x${NY} (CPU, 1 thread/rank), steps=$STEPS"
    rm -f output_*.vtk
    mpirun -np $NP "$BIN" \
        --device=cpu --muscl=true --hllc=true \
        --nx=$NX --ny=$NY --steps=$STEPS --save-every=$((STEPS + 1)) \
        --xmin=0 --xmax=1 --ymin=0 --ymax=1 2>&1 \
      | grep -E "MPI:|Done!|step_cpu" || true
    echo
done

echo "Strong-scaling speed-up = T(1 rank) / T(N ranks); efficiency = speed-up / N."
echo "Use the 'Done!' wall time of each run."
