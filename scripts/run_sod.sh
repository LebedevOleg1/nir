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

for N in 100 200 400 800; do
    STEPS=$(python3 -c "print(int(0.9*$N))")   # ~ reaches t≈0.2
    OUT="$ROOT/results/sod_N$N"; mkdir -p "$OUT"
    echo "=== Sod N=$N, steps=$STEPS ==="
    (cd "$OUT" && rm -f output_*.vtk && cp "$ROOT/problems/sod_shock/inputs" . &&
     mpirun -np 1 "$BIN" --nx=$N --ny=8 --steps=$STEPS --save-every=$STEPS \
        --device=gpu --muscl=true --hllc=true \
        --xmin=0 --xmax=1 --ymin=0 --ymax=0.02 2>&1 | grep -E "sim_time")
done

echo
echo "Profile (use sim_time of N=400 as <t>):"
echo "  python3 scripts/plot_sod_real.py profile results/sod_N400/output_0001.vtk --t <t> --out diploma/figures/sod_density.pdf"
echo "Convergence table (use each run's sim_time):"
echo "  python3 scripts/plot_sod_real.py conv 100:results/sod_N100/output_0001.vtk:<t100> 200:results/sod_N200/output_0001.vtk:<t200> 400:results/sod_N400/output_0001.vtk:<t400> 800:results/sod_N800/output_0001.vtk:<t800>"
