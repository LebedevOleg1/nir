#!/bin/bash
#SBATCH --job-name=fvm-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=solver_gpu_%j.log

cd "$SLURM_SUBMIT_DIR"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "MPI ranks: $SLURM_NTASKS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "==="

mpirun -np $SLURM_NTASKS ./build/solver gpu \
    --physics=heat --nx=1000 --ny=1000 \
    --steps=400 --save-every=10 \
    --source=gaussian:5,5,0.5,1000

echo "=== Done ==="
