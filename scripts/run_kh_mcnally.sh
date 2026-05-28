#!/bin/bash
# Run Kelvin-Helmholtz instability using McNally et al. (2012) parameters.
# Compare result at t=2 with McNally et al. (2012), ApJS 201, 18, Fig.1 (top-left).
#
# Usage: ./scripts/run_kh_mcnally.sh [512|1024]
set -e
N=${1:-512}
BUILD=${BUILD:-build}
BIN=${BUILD}/problems/kelvin_helmholtz/kh

# Steps: t_final=2, h=1/N, dt~cfl*h/(|u|+c), c=sqrt(5/3*2.5/2)=1.44, u=0.5
# dt ~ 0.4*(1/N)/(0.5+1.44) ~ 0.206/N => steps ~ 2*N/0.206 = 9.7*N
STEPS=$(python3 -c "import math; print(int(2.0 / (0.4 * (1.0/$N) / 1.94) + 50))")

OUTDIR="results/kh_mcnally_${N}"
mkdir -p "$OUTDIR"
echo "=== KH McNally 2012: ${N}x${N}, steps=$STEPS ==="

(cd "$OUTDIR" && cp -r ../../problems/kelvin_helmholtz/inputs . &&
 mpirun -np 1 ../../../${BIN} \
    --nx=$N --ny=$N --steps=$STEPS --save-every=$((STEPS/10)) \
    --device=gpu --muscl=true \
    --xmin=0 --xmax=1 --ymin=0 --ymax=1) \
 2>&1 | tee kh_mcnally_${N}.log

echo "Done. Visualise VTK in ParaView; compare density at t≈2 with McNally (2012) Fig.1."
