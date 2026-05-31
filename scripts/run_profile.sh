#!/bin/bash
# Profile the Euler flux/update kernel with NVIDIA Nsight Compute (ncu).
# Captures a few launches of the grid-stride kernel on a 1024x1024 MUSCL run
# and prints the key metrics: achieved occupancy, registers/thread, compute
# and memory throughput, DRAM throughput, kernel duration.
#
# Run from project root:
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_profile.sh
set -e
ROOT=$(pwd)
BIN="$ROOT/build/problems/kelvin_helmholtz/kh"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

mkdir -p "$ROOT/results/perf"
cd "$ROOT/results/perf"

echo "=== ncu profile: flux/update kernel, 1024x1024, MUSCL+HLLC ==="
# --launch-skip 20: skip warm-up steps; --launch-count 2: profile 2 launches.
ncu --launch-skip 20 --launch-count 2 \
    --section LaunchStats \
    --section Occupancy \
    --section SpeedOfLight \
    --kernel-name regex:"fvm_parallel_for_kernel" \
    "$BIN" --device=gpu --muscl=true --hllc=true \
    --nx=1024 --ny=1024 --steps=40 --save-every=999 \
    --xmin=0 --xmax=1 --ymin=0 --ymax=1 2>&1 | tee ncu_kh_1024.txt | \
  grep -E "Registers Per Thread|Achieved Occupancy|Compute \(SM\)|Memory Throughput|DRAM Throughput|Duration|Block Size|Grid Size" || true

echo
echo "Full report saved to results/perf/ncu_kh_1024.txt"
