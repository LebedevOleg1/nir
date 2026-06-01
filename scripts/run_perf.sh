#!/bin/bash
# Unified performance benchmark, single consistent series:
#   - GPU (NVIDIA V100, free card): per-step time on 512/1024/2048, 1st order and MUSCL.
#   - CPU baseline (single thread, --bind-to none): 512/1024, 1st order and MUSCL.
# Same binary (sod / Euler), same step counts per grid, so all tables (GPU,
# CPU baseline, OpenMP, MPI) share a consistent single-unit baseline.
#
# Run from the project root:
#   ./scripts/run_perf.sh
set -e
ROOT=$(pwd)
BIN="$ROOT/build/problems/sod_shock/sod"

mkdir -p "$ROOT/results/perf"
cd "$ROOT/results/perf"

bench() {  # device muscl nx ny steps  [omp_threads]
    local dev=$1 muscl=$2 nx=$3 ny=$4 steps=$5 omp=${6:-1}
    rm -f output_*.vtk
    local vis=""
    [ "$dev" = "gpu" ] && vis="CUDA_VISIBLE_DEVICES=1"
    echo -n "  ${dev} muscl=${muscl} ${nx}x${ny}: "
    env $vis OMP_NUM_THREADS=$omp mpirun --bind-to none -np 1 "$BIN" \
        --device=$dev --muscl=$muscl --hllc=true \
        --nx=$nx --ny=$ny --steps=$steps --save-every=$((steps + 1)) \
        --xmin=0 --xmax=1 --ymin=0 --ymax=1 2>&1 \
      | grep -E "step_(gpu|cpu)" || true
}

echo "================= GPU (V100), per-step kernel time ================="
bench gpu false 512  512  300
bench gpu true  512  512  300
bench gpu false 1024 1024 200
bench gpu true  1024 1024 200
bench gpu false 2048 2048 100
bench gpu true  2048 2048 100

echo "================= CPU baseline (1 thread, --bind-to none) ================="
bench cpu false 512  512  20
bench cpu true  512  512  20
bench cpu false 1024 1024 10
bench cpu true  1024 1024 10

echo
echo "Use the Avg(ms) column. GPU-vs-CPU speed-up = T(CPU 1 thread) / T(GPU), same grid & scheme."
