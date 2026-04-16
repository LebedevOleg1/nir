#!/bin/bash
# =============================================================================
# run_demo.sh — полный демо-прогон всех задач + генерация анимаций.
#
# Запуск:   sbatch scripts/run_demo.sh
#           (из корня репозитория)
#
# Результаты:
#   results/kh/kh_density.gif         — неустойчивость Кельвина–Гельмгольца
#   results/rt/rt_density.gif          — неустойчивость Рэлея–Тейлора
#   results/heat/heat_temperature.gif  — уравнение теплопроводности
#   results/sod/convergence.png        — порядок сходимости (ударная труба Содa)
# =============================================================================
#SBATCH --job-name=fvm2d-demo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=demo_%j.log

set -e

REPO=$(cd "$(dirname "$0")/.." && pwd)
BUILD="$REPO/build"

echo "======================================================================"
echo "FVM2D Hybrid Solver — Demo Run"
echo "Repo:  $REPO"
echo "Build: $BUILD"
echo "Job:   ${SLURM_JOB_ID:-local}  on $(hostname)"
date
echo "======================================================================"

# Проверка сборки
if [ ! -f "$BUILD/problems/kelvin_helmholtz/kh" ]; then
    echo ">>> Build not found — building now..."
    cd "$REPO"
    cmake -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
    cmake --build build -j"$(nproc)" 2>&1 | tail -20
    echo ">>> Build done"
fi

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "(nvidia-smi not available)"

mkdir -p "$REPO/results/kh" "$REPO/results/rt" \
         "$REPO/results/heat" "$REPO/results/sod"

# ==========================================================================
# 1. Kelvin–Helmholtz instability (Euler, 2 GPU ranks, 256×256)
# ==========================================================================
echo ""
echo ">>> [1/4] Kelvin–Helmholtz instability  (Euler, 2×GPU, 256×256, 1000 steps)"
cd "$REPO/results/kh"
rm -f output_*.vtk

time mpirun -np 2 "$BUILD/problems/kelvin_helmholtz/kh" \
    --device=gpu --nx=256 --ny=256 --steps=1000 --save-every=10 \
    --cfl=0.4

echo "    VTK files: $(ls output_*.vtk 2>/dev/null | wc -l)"
python3 "$REPO/scripts/animate.py" --dir . --field Density \
    --out kh_density.gif --fps 20 --dpi 100
python3 "$REPO/scripts/animate.py" --dir . --field VelocityX \
    --out kh_velocity.gif --fps 20 --dpi 100

# ==========================================================================
# 2. Rayleigh–Taylor instability (Euler, 2 GPU ranks, 128×256)
# ==========================================================================
echo ""
echo ">>> [2/4] Rayleigh–Taylor instability   (Euler, 2×GPU, 128×256, 800 steps)"
cd "$REPO/results/rt"
rm -f output_*.vtk

time mpirun -np 2 "$BUILD/problems/rayleigh_taylor/rt" \
    --device=gpu --nx=128 --ny=256 --steps=800 --save-every=8 \
    --xmax=1.0 --ymax=2.0 --cfl=0.3 \
    --bc-left=wall --bc-right=wall --bc-bottom=wall --bc-top=wall

echo "    VTK files: $(ls output_*.vtk 2>/dev/null | wc -l)"
python3 "$REPO/scripts/animate.py" --dir . --field Density \
    --out rt_density.gif --fps 15 --dpi 100

# ==========================================================================
# 3. Heat equation with Gaussian source (single GPU, 200×200)
# ==========================================================================
echo ""
echo ">>> [3/4] Heat equation + Gaussian source  (GPU, 200×200, 400 steps)"
cd "$REPO/results/heat"
rm -f output_*.vtk

time mpirun -np 1 "$BUILD/problems/heat_gaussian/heat" \
    --device=gpu --nx=200 --ny=200 --steps=400 --save-every=4 \
    --cfl=0.4

echo "    VTK files: $(ls output_*.vtk 2>/dev/null | wc -l)"
python3 "$REPO/scripts/animate.py" --dir . --field Temperature \
    --out heat_temperature.gif --fps 20 --dpi 100

# ==========================================================================
# 4. Sod shock tube — convergence study
# ==========================================================================
echo ""
echo ">>> [4/4] Sod shock tube — convergence study"
cd "$REPO/results/sod"

python3 "$REPO/scripts/plot_convergence.py" \
    --exec "$BUILD/problems/sod_shock/sod" \
    --device gpu --out convergence.png

# ==========================================================================
# 5. Performance benchmark: GPU vs CPU (KH 256×256, 200 steps)
# ==========================================================================
echo ""
echo ">>> [5/5] Performance benchmark: GPU vs CPU  (KH 256×256, 200 steps)"
cd "$REPO/results"
rm -f bench_*.vtk output_*.vtk

echo "    --- GPU (2 ranks) ---"
time mpirun -np 2 "$BUILD/problems/kelvin_helmholtz/kh" \
    --device=gpu --nx=256 --ny=256 --steps=200 --save-every=200 \
    --cfl=0.4 2>&1 | grep -E "steps/s|Wall time|SOLVER"
rm -f output_*.vtk

echo "    --- CPU (2 ranks, ${SLURM_CPUS_PER_TASK:-4} threads/rank) ---"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
time mpirun -np 2 "$BUILD/problems/kelvin_helmholtz/kh" \
    --device=cpu --nx=256 --ny=256 --steps=200 --save-every=200 \
    --cfl=0.4 2>&1 | grep -E "steps/s|Wall time|SOLVER"
rm -f output_*.vtk

# ==========================================================================
# Summary
# ==========================================================================
echo ""
echo "======================================================================"
echo "DONE.  Results:"
for f in "$REPO/results/kh/kh_density.gif" \
          "$REPO/results/kh/kh_velocity.gif" \
          "$REPO/results/rt/rt_density.gif" \
          "$REPO/results/heat/heat_temperature.gif" \
          "$REPO/results/sod/convergence.png"; do
    if [ -f "$f" ]; then
        size=$(du -sh "$f" 2>/dev/null | cut -f1)
        echo "  OK  $size  $f"
    else
        echo "  MISSING  $f"
    fi
done
echo "======================================================================"
date
