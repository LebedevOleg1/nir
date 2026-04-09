#!/bin/bash
#SBATCH --job-name=scaling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=scaling_%j.log

cd "$SLURM_SUBMIT_DIR"

echo "=== Strong scaling test (Kelvin-Helmholtz 512x512) ==="
echo "Fixed problem size, varying MPI ranks"

for NRANKS in 1 2 4; do
    echo "--- $NRANKS ranks ---"
    rm -f output_*.vtk
    time mpirun -np $NRANKS ./build/problems/kelvin_helmholtz/kh \
        --device=gpu --nx=512 --ny=512 --steps=100 --save-every=100
done

echo "=== Weak scaling test (per-rank 256x256) ==="
for NRANKS in 1 2 4; do
    NY=$((256 * NRANKS))
    echo "--- $NRANKS ranks, NY=$NY ---"
    rm -f output_*.vtk
    time mpirun -np $NRANKS ./build/problems/kelvin_helmholtz/kh \
        --device=gpu --nx=256 --ny=$NY --steps=100 --save-every=100
done

echo "=== Done ==="
