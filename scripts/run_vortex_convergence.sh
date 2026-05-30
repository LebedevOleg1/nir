#!/bin/bash
# Run isentropic vortex convergence study.
# For each grid N: computes one full vortex period, saves VTK at t_final.
# Results in results/vortex_conv_N/
#
# Usage: ./scripts/run_vortex_convergence.sh [cpu|gpu]
set -e
DEVICE=${1:-gpu}
BUILD=${BUILD:-build}
BIN=${BUILD}/problems/isentropic_vortex/vortex

GRIDS="32 64 128 256"
GAMMA=1.4
# u_inf=1, c_inf ~ sqrt(1.4) ~ 1.183; one period t=10; dt ~ cfl*h/(u+c) ~ 0.4*(10/N)/2.18
# steps = t_final / dt = 10 / (0.4*(10/N)/2.18) = 10*N*2.18/(10*0.4) = N*5.45
STEPS_FACTOR=6   # steps = STEPS_FACTOR * N (conservative, overshoots t=10 slightly)

mkdir -p results/vortex_conv

for N in $GRIDS; do
    STEPS=$((STEPS_FACTOR * N))
    OUTDIR="results/vortex_conv/N${N}_${DEVICE}"
    mkdir -p "$OUTDIR"
    echo "=== Grid ${N}x${N}, steps=$STEPS, device=$DEVICE ==="

    # Without MUSCL (1st order HLL)
    (cd "$OUTDIR" && cp -r ../../problems/isentropic_vortex/inputs . &&
     mpirun -np 1 ../../${BIN} \
        --nx=$N --ny=$N --steps=$STEPS --save-every=$STEPS \
        --device=$DEVICE --muscl=false \
        --xmin=-5 --xmax=5 --ymin=-5 --ymax=5) \
     > results/vortex_conv/hll_${N}.log 2>&1

    # With MUSCL (2nd order)
    OUTDIRM="results/vortex_conv/N${N}_${DEVICE}_muscl"
    mkdir -p "$OUTDIRM"
    (cd "$OUTDIRM" && cp -r ../../problems/isentropic_vortex/inputs . &&
     mpirun -np 1 ../../${BIN} \
        --nx=$N --ny=$N --steps=$STEPS --save-every=$STEPS \
        --device=$DEVICE --muscl=true \
        --xmin=-5 --xmax=5 --ymin=-5 --ymax=5) \
     > results/vortex_conv/muscl_${N}.log 2>&1

    echo "  Done."
done

echo "Now run:"
echo "  python3 scripts/plot_vortex_convergence.py --device $DEVICE"
echo "  cp results/vortex_conv/vortex_convergence.pdf diploma/figures/"
