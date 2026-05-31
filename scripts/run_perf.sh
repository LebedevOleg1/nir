#!/bin/bash
# Performance benchmark: per-step wall time for the Euler solver on several
# grids, GPU (HLLC, 1st-order and MUSCL+SSP-RK2) and single/multi-thread CPU.
# Prints the "Done!" line (wall time, steps/s) and the kernel Timing Report
# for each configuration, from which the real per-step time is taken.
#
# Run from the project root:
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_perf.sh
set -e
ROOT=$(pwd)
BIN="$ROOT/build/problems/kelvin_helmholtz/kh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

mkdir -p "$ROOT/results/perf"
cd "$ROOT/results/perf"

run() {
    # label device muscl nx ny steps [omp_threads]
    local label=$1 dev=$2 muscl=$3 nx=$4 ny=$5 steps=$6 omp=$7
    echo "######## $label : ${nx}x${ny} device=$dev muscl=$muscl steps=$steps ${omp:+OMP=$omp}"
    rm -f output_*.vtk
    OMP_NUM_THREADS=${omp:-0} mpirun -np 1 "$BIN" \
        --device=$dev --muscl=$muscl --hllc=true \
        --nx=$nx --ny=$ny --steps=$steps --save-every=$((steps + 1)) \
        --xmin=0 --xmax=1 --ymin=0 --ymax=1 2>&1 \
      | grep -E "Done!|step_(gpu|cpu)" || true
    echo
}

echo "================= GPU (HLLC) ================="
run "GPU 1st  512"  gpu false 512  512  300
run "GPU MUSCL 512" gpu true  512  512  300
run "GPU 1st  1024" gpu false 1024 1024 200
run "GPU MUSCL 1024" gpu true 1024 1024 200
run "GPU 1st  2048" gpu false 2048 2048 100
run "GPU MUSCL 2048" gpu true 2048 2048 100

echo "================= CPU (OpenMP, HLLC 1st order) ================="
run "CPU 1-thread 512"   cpu false 512 512 20 1
run "CPU all-threads 512" cpu false 512 512 20

echo "Done. Per-step time = (wall time)/steps, or use kernel Avg(ms) from Timing Report."
