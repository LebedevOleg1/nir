#!/bin/bash
#SBATCH --job-name=fvm-cpu
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --time=00:30:00
#SBATCH --output=solver_cpu_%j.log

cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "MPI ranks: $SLURM_NTASKS, OMP threads/rank: $OMP_NUM_THREADS"
echo "==="

mpirun -np $SLURM_NTASKS --bind-to socket --map-by socket ./build/solver \
    --physics=heat --nx=1000 --ny=1000 \
    --steps=400 --save-every=10 \
    --source=gaussian:5,5,0.5,1000

echo "=== Done ==="
