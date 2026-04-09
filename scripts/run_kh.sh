#!/bin/bash
#SBATCH --job-name=kh-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=kh_%j.log

cd "$SLURM_SUBMIT_DIR"

echo "=== Kelvin-Helmholtz (GPU + MPI) ==="
echo "Job $SLURM_JOB_ID on $(hostname), ranks=$SLURM_NTASKS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==="

rm -f output_*.vtk

mpirun -np $SLURM_NTASKS ./build/problems/kelvin_helmholtz/kh \
    --device=gpu --nx=512 --ny=512 --steps=3000 --save-every=30

echo "=== VTK files: $(ls output_*.vtk 2>/dev/null | wc -l) ==="
