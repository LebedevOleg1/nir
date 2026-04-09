#!/bin/bash
#SBATCH --job-name=sod-verify
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=sod_%j.log

cd "$SLURM_SUBMIT_DIR"

echo "=== Sod shock tube verification ==="

rm -f output_*.vtk

for NX in 50 100 200 400; do
    echo "--- nx=$NX ---"
    mpirun -np 1 ./build/problems/sod_shock/sod \
        --device=gpu --nx=$NX --ny=4 --steps=200
done

echo "=== Run plot_convergence.py to see L2 error vs h ==="
