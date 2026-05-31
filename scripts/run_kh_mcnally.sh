#!/bin/bash
# Run Kelvin-Helmholtz instability using McNally et al. (2012) parameters.
# Compare result at t=1.5 with McNally et al. (2012), ApJS 201, 18, Fig.2
# (reference Pencil Code solution, density at t=1.5).
#
# Usage: ./scripts/run_kh_mcnally.sh [512|1024]
set -e
N=${1:-512}
BUILD=${BUILD:-build}
BIN=${BUILD}/problems/kelvin_helmholtz/kh

# Steps: t_final=1.5, h=1/N, dt~cfl*h/(|u|+c), c=sqrt(5/3*2.5/1)=2.04, u=0.5
# dt ~ 0.4*(1/N)/(0.5+2.04) ~ 0.157/N => steps ~ 1.5*N/0.157 = 9.5*N
STEPS=$(python3 -c "import math; print(int(1.5 / (0.4 * (1.0/$N) / 2.54) + 50))")

OUTDIR="results/kh_mcnally_${N}"
mkdir -p "$OUTDIR"
echo "=== KH McNally 2012: ${N}x${N}, steps=$STEPS ==="

(cd "$OUTDIR" && cp -r ../../problems/kelvin_helmholtz/inputs . &&
 mpirun -np 1 ../../${BIN} \
    --nx=$N --ny=$N --steps=$STEPS --save-every=$((STEPS/10)) \
    --device=gpu --muscl=true \
    --xmin=0 --xmax=1 --ymin=0 --ymax=1) \
 2>&1 | tee kh_mcnally_${N}.log

echo "Done. To generate diploma figure:"
echo "  python3 scripts/plot_snapshot.py $OUTDIR diploma/figures/kh_mcnally_${N}.png"
echo "Compare with McNally (2012) Fig.2 (reference density at t=1.5)."
