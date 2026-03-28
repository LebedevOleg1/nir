#pragma once
#include "Mesh.hpp"
#include "GpuMesh.hpp"
#include "VTKWriter.hpp"
#include <iostream>
#include <chrono>

// ============================================================================
// Solver — единый решатель уравнения теплопроводности.
//
// Объединяет бывшие CpuSolver и CudaSolver. Хранит CPU-сетку (Mesh) и
// GPU-сетку (GpuMesh). В зависимости от флага use_gpu вызывает step_cpu()
// или step_gpu().
//
// Разделение по файлам:
//   solver.cpp — CPU-реализация (step_cpu, solve, update_source)
//   solver.cu  — CUDA-ядро и step_gpu
//
// Это работает потому что CMake компилирует .cpp через g++ и .cu через nvcc,
// а линкер собирает всё в один бинарник. Оба файла видят один Solver.hpp.
// ============================================================================
class Solver {
private:
    Mesh& mesh;           // CPU-сетка (данные живут в оперативной памяти)
    GpuMesh gpu_mesh;     // GPU-сетка (данные живут в видеопамяти)

    float_t alpha;        // коэффициент температуропроводности
    float_t dt;           // шаг по времени (вычисляется из CFL)
    bool use_gpu;

    // MPI-информация (для domain decomposition)
    int mpi_rank = 0;
    int mpi_size = 1;

    // --- CPU-часть (solver.cpp) ---
    void step_cpu();
    void update_dynamic_source(float_t power, float_t time);
    float_t compute_max_dt() const;

    // --- GPU-часть (solver.cu) ---
    void step_gpu();

public:
    Solver(Mesh& mesh_, float_t alpha_, bool use_gpu_,
           int rank = 0, int size = 1);

    // Основной цикл решения
    void solve(int total_steps, int save_every);
};
