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
# Stationary vortex (u_inf=0). Integrate to a short fixed time (t~2) so the
# error is in the asymptotic regime (vortex not yet destroyed by dissipation).
# dt ~ cfl*h/c_max, c_max~1.8 (rotation+sound); steps = STEPS_FACTOR*N keeps
# t_final ~ const across grids (steps proportional to N, dt proportional to 1/N).
STEPS_FACTOR=1   # steps = STEPS_FACTOR * N  ->  t_final ~ 2

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
