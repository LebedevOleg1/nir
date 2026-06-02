#!/bin/bash
# Real heat-equation grid-convergence runs. Heat is diffusion-limited, so the
# stable dt ~ h^2; to compare all grids at the SAME physical time, steps ~ N^2.
# The finest grid (512) is the Richardson reference (see plot_heat_real.py).
#
#   CUDA_VISIBLE_DEVICES=1 ./scripts/run_heat_conv.sh
set -e
ROOT=$(pwd)
BIN="$ROOT/build/problems/heat_gaussian/heat"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

TFIX=0.5                       # common physical time for all grids
for N in 64 128 256 512; do
    # dt = cfl*0.25*h^2/kappa, h=10/N, cfl=0.4, kappa=1  => steps = TFIX*N^2/10
    STEPS=$(python3 -c "print(int($TFIX*$N*$N/10.0))")
    OUT="$ROOT/results/heat_N$N"; mkdir -p "$OUT"
    echo "=== Heat N=$N, steps=$STEPS ==="
    (cd "$OUT" && rm -f output_*.vtk && cp "$ROOT/problems/heat_gaussian/inputs" . &&
     mpirun -np 1 "$BIN" --nx=$N --ny=$N --steps=$STEPS --save-every=$STEPS \
        --device=gpu --kappa=1 \
        --xmin=0 --xmax=10 --ymin=0 --ymax=10 2>&1 | grep -E "sim_time")
done

echo
echo "Then build the convergence figure:"
echo "  python3 scripts/plot_heat_real.py \\"
echo "    64:results/heat_N64/output_0001.vtk 128:results/heat_N128/output_0001.vtk \\"
echo "    256:results/heat_N256/output_0001.vtk \\"
echo "    --ref 512:results/heat_N512/output_0001.vtk \\"
echo "    --out diploma/figures/heat_convergence.pdf"
