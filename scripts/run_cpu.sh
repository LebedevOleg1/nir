#!/bin/bash
#SBATCH --job-name=kh-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=kh_cpu_%j.log

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== Kelvin-Helmholtz (CPU + MPI + OpenMP) ==="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "MPI ranks=$SLURM_NTASKS, OMP threads=$OMP_NUM_THREADS"
echo "==="

rm -f output_*.vtk

mpirun -np $SLURM_NTASKS ./build/problems/kelvin_helmholtz/kh \
    --device=cpu --nx=512 --ny=512 --steps=3000 --save-every=30

echo "=== Done ==="
