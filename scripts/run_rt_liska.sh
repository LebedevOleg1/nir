#!/bin/bash
# Run Rayleigh-Taylor instability using Liska & Wendroff (2003) parameters.
# Compare density finger pattern at t=8.9 with Liska & Wendroff (2003) Fig.4.4.
#
# Usage: ./scripts/run_rt_liska.sh [64x384 | 128x768]
set -e
NX=${1:-64}
NY=${2:-384}
BUILD=${BUILD:-build}
BIN=${BUILD}/problems/rayleigh_taylor/rt

# Steps: t_final=8.9, h_y=1/NY, c~sqrt(5/3*2.5/1)~2.04
# dt~cfl*h/(c) ~ 0.3*(1/NY)/2.04 => steps ~ 8.9*NY*2.04/0.3
STEPS=$(python3 -c "import math; print(int(8.9 * $NY * 2.04 / 0.3) + 200)")

OUTDIR="results/rt_liska_${NX}x${NY}"
mkdir -p "$OUTDIR"
echo "=== RT Liska 2003: ${NX}x${NY}, steps=$STEPS ==="

(cd "$OUTDIR" && cp -r ../../problems/rayleigh_taylor/inputs . &&
 mpirun -np 1 ../../${BIN} \
    --nx=$NX --ny=$NY --steps=$STEPS --save-every=$((STEPS/15)) \
    --device=gpu --muscl=true \
    --xmin=0 --xmax=0.16667 --ymin=0 --ymax=1 \
    --gravity=0.1) \
 2>&1 | tee rt_liska_${NX}x${NY}.log

echo "Done. Visualise VTK in ParaView; compare density at t≈8.9 with Liska (2003) Fig.4.4."
