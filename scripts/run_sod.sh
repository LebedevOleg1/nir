#!/bin/bash
# Real Sod runs for the density profile and grid-convergence table.
# Thin strip (1D problem), HLLC+MUSCL. Each run prints "sim_time=..." — use it
# as --t for the exact solution.
#
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_sod.sh
set -e
ROOT=$(pwd)
BIN="$ROOT/build/problems/sod_shock/sod"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# Integrate every grid to the SAME physical time t=0.2 via --t-final
# (steps is just a generous upper bound).
TF=0.2
for N in 100 200 400 800; do
    OUT="$ROOT/results/sod_N$N"; mkdir -p "$OUT"
    echo "=== Sod N=$N, t_final=$TF ==="
    (cd "$OUT" && rm -f output_*.vtk && cp "$ROOT/problems/sod_shock/inputs" . &&
     mpirun -np 1 "$BIN" --nx=$N --ny=8 --steps=5000 --t-final=$TF --save-every=5000 \
        --device=gpu --muscl=true --hllc=true \
        --xmin=0 --xmax=1 --ymin=0 --ymax=0.02 2>&1 | grep -E "sim_time")
done

echo
echo "All grids are at t=0.2. Build the figures:"
echo "  python3 scripts/plot_sod_real.py profile results/sod_N400/output_0001.vtk --t 0.2 --out figures_out/sod_density.pdf"
echo "  python3 scripts/plot_sod_real.py conv 100:results/sod_N100/output_0001.vtk:0.2 200:results/sod_N200/output_0001.vtk:0.2 400:results/sod_N400/output_0001.vtk:0.2 800:results/sod_N800/output_0001.vtk:0.2"
