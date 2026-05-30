#!/bin/bash
# Run Rayleigh-Taylor instability using Liska & Wendroff (2003) parameters.
# Compare density finger pattern at t=8.9 with Liska & Wendroff (2003) Fig.4.4.
#
# Liska & Wendroff 2003 use 120x480 grid on domain [0,1/6]x[0,1].
# Default here is 120x480 to match the reference for direct visual comparison.
#
# Usage: ./scripts/run_rt_liska.sh [NX [NY]]
#   ./scripts/run_rt_liska.sh          # 120x480 (matches Liska 2003)
#   ./scripts/run_rt_liska.sh 64 384   # coarser run for quick check
set -e
NX=${1:-120}
NY=${2:-480}
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

echo "Done. To generate diploma figure:"
echo "  python3 scripts/plot_snapshot.py $OUTDIR diploma/figures/rt_liska_${NX}x${NY}.png"
echo "Also place Liska & Wendroff (2003) Fig.4.4 as:"
echo "  diploma/figures/rt_liska_reference.png"
