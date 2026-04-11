#!/bin/bash
# =============================================================================
# Benchmark Pipeline for FVM Solver
# Cluster: gpu (1 node, 18 cores, 2x V100), compute (2 nodes, 48 cores each)
#
# Usage:
#   cd /path/to/nir
#   bash benchmark/run_all.sh
#
# This script submits SLURM jobs for all benchmark scenarios.
# Results are appended to benchmark/results.csv
# Then run: python3 benchmark/plot_results.py
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOLVER="$PROJECT_DIR/build/solver"
RESULTS="$SCRIPT_DIR/results.csv"
LOGS="$SCRIPT_DIR/logs"

mkdir -p "$LOGS"

# Initialize CSV header
echo "scenario,device,physics,nx,ny,steps,mpi_ranks,omp_threads,time_s" > "$RESULTS"

# Common params
STEPS=200
IC="sod"
PHYSICS="euler"

# Helper: submit a single benchmark job
# Usage: submit_bench SCENARIO PARTITION DEVICE NX NY MPI_RANKS OMP_THREADS [PHYSICS] [EXTRA_ARGS]
submit_bench() {
    local SCENARIO="$1"
    local PARTITION="$2"
    local DEVICE="$3"
    local NX="$4"
    local NY="$5"
    local MPI_RANKS="$6"
    local OMP_THREADS="$7"
    local PHYS="${8:-$PHYSICS}"
    local EXTRA="${9:-}"

    local JOBNAME="bench_${SCENARIO}_${DEVICE}_${NX}x${NY}_mpi${MPI_RANKS}_omp${OMP_THREADS}"
    local LOGFILE="$LOGS/${JOBNAME}_%j.log"

    # Determine SLURM resources
    local NODES=1
    local GRES=""
    local CPUS_PER_TASK="$OMP_THREADS"

    if [ "$PARTITION" = "gpu" ]; then
        GRES="#SBATCH --gres=gpu:${MPI_RANKS}"
    fi

    # For MPI CPU jobs needing >48 cores, use 2 nodes
    local TOTAL_CORES=$((MPI_RANKS * OMP_THREADS))
    if [ "$PARTITION" = "compute" ] && [ "$TOTAL_CORES" -gt 48 ]; then
        NODES=2
    fi

    # IC args for Euler
    local IC_ARGS=""
    if [ "$PHYS" = "euler" ]; then
        IC_ARGS="--ic=$IC"
    elif [ "$PHYS" = "heat" ]; then
        IC_ARGS="--source=gaussian:5,5,0.5,1000"
    fi

    # Create and submit SLURM job
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOBNAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$MPI_RANKS
#SBATCH --cpus-per-task=$CPUS_PER_TASK
$GRES
#SBATCH --time=00:30:00
#SBATCH --output=$LOGFILE

cd "$PROJECT_DIR"

export OMP_NUM_THREADS=$OMP_THREADS
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "=== \$SLURM_JOB_ID | $JOBNAME ==="

mpirun -np $MPI_RANKS $SOLVER $DEVICE \
    --physics=$PHYS --nx=$NX --ny=$NY \
    --steps=$STEPS --benchmark \
    $IC_ARGS $EXTRA 2>&1 | tee /dev/stderr | \
    grep "^BENCH_RESULT," | while IFS= read -r line; do
        # line: BENCH_RESULT,device,physics,nx,ny,steps,mpi_size,time_s
        TAIL=\$(echo "\$line" | cut -d',' -f2-)
        echo "${SCENARIO},\${TAIL},${OMP_THREADS}" >> "$RESULTS"
    done

echo "=== Done ==="
EOF

    echo "  Submitted: $JOBNAME"
}

echo "============================================"
echo " FVM Solver Benchmark Suite"
echo " Results -> $RESULTS"
echo "============================================"

# =====================================================================
# A. CPU OpenMP Strong Scaling
# Fix grid, vary OMP threads (MPI=1)
# =====================================================================
echo ""
echo "--- A. CPU OpenMP Strong Scaling ---"
for NX in 1000 2000; do
    for THREADS in 1 2 4 8 16 24 48; do
        submit_bench "omp_scaling" "compute" "cpu" "$NX" "$NX" 1 "$THREADS"
    done
done

# =====================================================================
# B. GPU vs CPU across Grid Sizes
# =====================================================================
echo ""
echo "--- B. GPU vs CPU across Grid Sizes ---"
for NX in 100 200 500 1000 2000 4000; do
    # CPU with max threads
    submit_bench "grid_scaling" "compute" "cpu" "$NX" "$NX" 1 48
    # GPU single
    submit_bench "grid_scaling" "gpu" "gpu" "$NX" "$NX" 1 1
done

# =====================================================================
# C. MPI Strong Scaling (CPU)
# Fix grid 2000x2000, vary MPI ranks with 12 OMP threads each
# =====================================================================
echo ""
echo "--- C. MPI Strong Scaling (CPU) ---"
for MPI_RANKS in 1 2 4 8; do
    submit_bench "mpi_cpu" "compute" "cpu" 2000 2000 "$MPI_RANKS" 12
done

# =====================================================================
# D. MPI + GPU Scaling
# =====================================================================
echo ""
echo "--- D. MPI + GPU Scaling ---"
for MPI_RANKS in 1 2; do
    submit_bench "mpi_gpu" "gpu" "gpu" 2000 2000 "$MPI_RANKS" 1
done

# =====================================================================
# E. Physics Comparison (Heat vs Euler)
# =====================================================================
echo ""
echo "--- E. Physics Comparison ---"
for PHYS in heat euler; do
    submit_bench "physics" "compute" "cpu" 1000 1000 1 48 "$PHYS"
    submit_bench "physics" "gpu" "gpu" 1000 1000 1 1 "$PHYS"
done

# =====================================================================
# F. Weak Scaling (CPU)
# Base 500x500 per rank, grid grows with ranks
# =====================================================================
echo ""
echo "--- F. Weak Scaling (CPU) ---"
for MPI_RANKS in 1 2 4 8; do
    NY=$((500 * MPI_RANKS))
    submit_bench "weak_scaling" "compute" "cpu" 500 "$NY" "$MPI_RANKS" 12
done

echo ""
echo "============================================"
echo " All jobs submitted!"
echo " Monitor: squeue -u \$USER"
echo " When done: python3 benchmark/plot_results.py"
echo "============================================"
