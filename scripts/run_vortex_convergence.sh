#!/bin/bash
# Run isentropic vortex convergence study.
# For each grid N: computes one full vortex period, saves VTK at t_final.
# Results in results/vortex_conv_N/
#
# Usage: ./scripts/run_vortex_convergence.sh [cpu|gpu]
set -e
DEVICE=${1:-gpu}
ROOT=$(pwd)                       # project root (run from there)
BUILD=${BUILD:-build}
BIN="${ROOT}/${BUILD}/problems/isentropic_vortex/vortex"
INPUTS="${ROOT}/problems/isentropic_vortex/inputs"

GRIDS="32 64 128 256"
# u_inf=1, c_inf ~ sqrt(1.4) ~ 1.183; one period t=10; dt ~ cfl*h/(u+c) ~ 0.4*(10/N)/2.18
# steps = t_final / dt = 10 / (0.4*(10/N)/2.18) = 10*N*2.18/(10*0.4) = N*5.45
STEPS_FACTOR=6   # steps = STEPS_FACTOR * N (conservative, overshoots t=10 slightly)

mkdir -p "${ROOT}/results/vortex_conv"

for N in $GRIDS; do
    STEPS=$((STEPS_FACTOR * N))
    OUTDIR="${ROOT}/results/vortex_conv/N${N}_${DEVICE}"
    mkdir -p "$OUTDIR"
    echo "=== Grid ${N}x${N}, steps=$STEPS, device=$DEVICE ==="

    # 1st order (no reconstruction, donor-cell)
    (cd "$OUTDIR" && cp -r "$INPUTS" . &&
     mpirun -np 1 "$BIN" \
        --nx=$N --ny=$N --steps=$STEPS --save-every=$STEPS \
        --device=$DEVICE --muscl=false \
        --xmin=-5 --xmax=5 --ymin=-5 --ymax=5) \
     > "${ROOT}/results/vortex_conv/hll_${N}.log" 2>&1

    # 2nd order (MUSCL + SSP-RK2)
    OUTDIRM="${ROOT}/results/vortex_conv/N${N}_${DEVICE}_muscl"
    mkdir -p "$OUTDIRM"
    (cd "$OUTDIRM" && cp -r "$INPUTS" . &&
     mpirun -np 1 "$BIN" \
        --nx=$N --ny=$N --steps=$STEPS --save-every=$STEPS \
        --device=$DEVICE --muscl=true \
        --xmin=-5 --xmax=5 --ymin=-5 --ymax=5) \
     > "${ROOT}/results/vortex_conv/muscl_${N}.log" 2>&1

    echo "  Done."
done

echo "Now run:"
echo "  python3 scripts/plot_vortex_convergence.py --device $DEVICE"
echo "  cp results/vortex_conv/vortex_convergence.pdf diploma/figures/"
