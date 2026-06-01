#!/bin/bash
# OpenMP thread-scaling test: single MPI process, vary OMP_NUM_THREADS.
# Fixed problem size (512x512 Euler, MUSCL+HLLC) on the node CPU.
# --bind-to none lets the OpenMP threads spread over physical cores.
#
# Run from the project root:
#   ./scripts/run_omp_scaling.sh
set -e
ROOT=$(pwd)
BIN="$ROOT/build/problems/sod_shock/sod"

mkdir -p "$ROOT/results/omp_scaling"
cd "$ROOT/results/omp_scaling"

NX=512; NY=512; STEPS=20
for T in 1 2 4 8; do
    echo "######## OMP_NUM_THREADS=$T, grid ${NX}x${NY} (CPU), steps=$STEPS"
    rm -f output_*.vtk
    OMP_NUM_THREADS=$T mpirun --bind-to none -np 1 "$BIN" \
        --device=cpu --muscl=true --hllc=true \
        --nx=$NX --ny=$NY --steps=$STEPS --save-every=$((STEPS + 1)) \
        --xmin=0 --xmax=1 --ymin=0 --ymax=1 2>&1 \
      | grep -E "Done|step_cpu" || true
    echo
done

echo "Speed-up = T(1 thread) / T(N threads); efficiency = speed-up / N."
echo "Use the kernel Avg(ms) from each Timing Report."
