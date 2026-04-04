#include "Types.hpp"
#include "Mesh.hpp"
#include "Solver.hpp"
#include "MpiDecomp.hpp"
#include <iostream>

// ============================================================================
// main — точка входа.
//
// Порядок работы:
// 1) MPI_Init    — запуск MPI-среды (создаёт коммуникатор MPI_COMM_WORLD)
// 2) MpiDecomp   — разбиваем глобальную сетку NY строк между ранками
// 3) Mesh        — каждый ранк создаёт свою ЛОКАЛЬНУЮ сетку (nx × local_ny)
//                  со сдвинутыми координатами по Y
// 4) Solver      — объединённый решатель (CPU или GPU, один класс)
// 5) MPI_Finalize — завершение MPI
//
// На кластере запуск:
//   mpirun -np 4 ./heat        (4 ранка на CPU)
//   mpirun -np 2 ./heat        (2 ранка, каждый на своём GPU)
//
// Каждый GPU на кластере cluster4 — отдельный CUDA device.
// cudaSetDevice(rank) привязывает ранк к конкретному GPU.
// ============================================================================
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MpiDecomp decomp;

    int_t global_nx = 1000, global_ny = 1000;

    // Разбиение: каждый ранк получает свою полосу строк
    decomp.init(global_nx, global_ny);

    // Выбор CPU/GPU через аргумент командной строки:
    //   mpirun -np 2 ./heat gpu    → CUDA
    //   mpirun -np 2 ./heat cpu    → CPU (по умолчанию)
    // Интерактивный stdin не подходит для MPI — все ранки должны
    // получить одинаковое значение, а stdin доступен только rank 0.
    bool use_gpu = false;
    if (argc > 1 && std::string(argv[1]) == "gpu") {
        use_gpu = true;
    }

    // Если GPU — привязываем каждый MPI-ранк к своему CUDA-устройству.
    // На cluster4 — 2 GPU, rank 0 → GPU 0, rank 1 → GPU 1.
    if (use_gpu) {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = decomp.rank % device_count;
            cudaSetDevice(device_id);
            if (decomp.rank == 0)
                std::cout << "CUDA devices: " << device_count << "\n";
        }
    }

    // Локальная сетка для этого ранка:
    // Полная область [0,10]×[0,10], этот ранк владеет полосой по Y
    float_t y_total = 10.0f;
    float_t hy_global = y_total / static_cast<float_t>(global_ny);
    float_t y_min = hy_global * decomp.j_start;
    float_t y_max = hy_global * (decomp.j_start + decomp.local_ny);

    Mesh mesh(global_nx, decomp.local_ny,
              Float3(0.0f, y_min, 0.0f),
              Float3(10.0f, y_max, 0.0f),
              decomp.size > 1);

    Solver solver(mesh, 1.0f, use_gpu, &decomp);

    solver.solve(400, 10);

    decomp.finalize();
    return 0;
}
