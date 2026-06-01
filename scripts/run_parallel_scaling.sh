#!/bin/bash
# Combined parallel scaling on CPU, run back-to-back under identical conditions
# so the single-unit baseline (1 thread == 1 rank) is consistent between the
# OpenMP and MPI tables. Same binary, grid, step count; --bind-to none in both
# so the OS schedules execution units onto free cores of the shared node.
#
# Run from the project root:
#   ./scripts/run_parallel_scaling.sh
set -e
ROOT=$(pwd)
BIN="$ROOT/build/problems/sod_shock/sod"

mkdir -p "$ROOT/results/parallel"
cd "$ROOT/results/parallel"

NX=512; NY=512; STEPS=20
ARGS="--device=cpu --muscl=true --hllc=true --nx=$NX --ny=$NY \
      --steps=$STEPS --save-every=$((STEPS + 1)) --xmin=0 --xmax=1 --ymin=0 --ymax=1"

echo "================= OpenMP scaling (1 process, N threads) ================="
for T in 1 2 4 8; do
    echo -n "threads=$T  "
    rm -f output_*.vtk
    OMP_NUM_THREADS=$T mpirun --bind-to none -np 1 "$BIN" $ARGS 2>&1 \
      | grep "step_cpu" || true
done

echo "================= MPI scaling (N processes, 1 thread/rank) ================="
for P in 1 2 4; do
    echo -n "ranks=$P  "
    rm -f output_*.vtk
    OMP_NUM_THREADS=1 mpirun --bind-to none -np $P "$BIN" $ARGS 2>&1 \
      | grep "step_cpu" || true
done

echo
echo "Speed-up = T(1 unit) / T(N units). Baselines (threads=1, ranks=1) should match."
