#!/bin/bash
# MPI strong-scaling test: fixed problem size (1024x1024 Kelvin-Helmholtz),
# vary the number of MPI ranks (1 and 2), one GPU per rank.
#
# Requires TWO free GPUs. Do NOT pre-set CUDA_VISIBLE_DEVICES — each rank
# selects its GPU as (rank % device_count); with two visible GPUs rank 0
# uses GPU 0 and rank 1 uses GPU 1.
#
# Run from the project root:
#   ./scripts/run_mpi_scaling.sh
set -e
ROOT=$(pwd)
BIN="$ROOT/build/problems/kelvin_helmholtz/kh"

# Make both GPUs visible to the ranks.
unset CUDA_VISIBLE_DEVICES

mkdir -p "$ROOT/results/mpi_scaling"
cd "$ROOT/results/mpi_scaling"

NX=1024; NY=1024; STEPS=300
for NP in 1 2; do
    echo "######## MPI ranks = $NP, grid ${NX}x${NY}, steps=$STEPS"
    rm -f output_*.vtk
    mpirun -np $NP "$BIN" \
        --device=gpu --muscl=true --hllc=true \
        --nx=$NX --ny=$NY --steps=$STEPS --save-every=$((STEPS + 1)) \
        --xmin=0 --xmax=1 --ymin=0 --ymax=1 2>&1 \
      | grep -E "MPI:|Done!|step_gpu" || true
    echo
done

echo "Strong-scaling efficiency = T(1 rank) / (2 * T(2 ranks)) * 100%."
echo "Use the 'Done!' wall time of each run."
