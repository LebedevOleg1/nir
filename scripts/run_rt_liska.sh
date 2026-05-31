#!/bin/bash
# Run Rayleigh-Taylor instability using Liska & Wendroff (2003) parameters.
# Compare density at T=8.5 with Liska & Wendroff (2003) Fig.4.8.
#
# Liska & Wendroff 2003 use a 100x400 grid on domain [0,1/6]x[0,1], T=8.5.
# Default here matches the reference for the most correct comparison.
#
# Usage: ./scripts/run_rt_liska.sh [NX [NY]]
#   ./scripts/run_rt_liska.sh          # 100x400 (matches Liska 2003)
#   ./scripts/run_rt_liska.sh 64 256   # coarser run for quick check
set -e
NX=${1:-100}
NY=${2:-400}
BUILD=${BUILD:-build}
BIN=${BUILD}/problems/rayleigh_taylor/rt

# Steps to reach T_final=8.5. dt = cfl*h_min/c_max, h_min = min(hx,hy)
# where hx = (1/6)/NX, hy = 1/NY. The solver uses the SMALLER of the two,
# so steps must be computed from h_min (else we undershoot T_final).
# c_max ~ |v| + sqrt(gamma*p/rho), with growth use ~2.6.
STEPS=$(python3 -c "
nx=$NX; ny=$NY
hx=(1.0/6.0)/nx; hy=1.0/ny
h=min(hx,hy); cmax=2.6
print(int(8.5*cmax/(0.3*h))+1000)
")

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
echo "Compare with Liska & Wendroff (2003) Fig.4.8 (PPM panel)."
