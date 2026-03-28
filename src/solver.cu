#include "Solver.hpp"
#include "HeatKernel.hpp"
#include <thrust/device_ptr.h>

// ============================================================================
// CUDA-ядро для FVM расчёта теплопроводности.
//
// Каждый CUDA-поток обрабатывает одну ячейку сетки. Потоки сгруппированы
// в блоки по 256 (оптимально для occupancy на большинстве GPU).
//
// threadIdx.x — номер потока внутри блока (0..255)
// blockIdx.x  — номер блока
// blockDim.x  — размер блока (256)
//
// Глобальный индекс ячейки: i = blockIdx.x * blockDim.x + threadIdx.x
//
// Функция calculate_heat_flux_core — та же самая, что и на CPU,
// скомпилированная nvcc с __host__ __device__. Один код — два бэкенда.
//
// SoA-аргументы (отдельные массивы face_owner, face_neighbor и т.д.)
// обеспечивают coalesced access: потоки warp'а читают соседние float'ы
// из одного массива → одна транзакция памяти на 32 потока.
// ============================================================================
__global__ void heat_fvm_kernel(
    const float* RESTRICT T_curr,
    float* RESTRICT T_next,
    const float* RESTRICT volumes,
    const int*   RESTRICT face_owner,
    const int*   RESTRICT face_neighbor,
    const float* RESTRICT face_area,
    const float* RESTRICT face_distance,
    const int*   RESTRICT cell_faces,
    const float* RESTRICT kappa_face,
    const float* RESTRICT source,
    const int ncells,
    const float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    calculate_heat_flux_core(
        i, T_curr, T_next, volumes,
        face_owner, face_neighbor, face_area, face_distance,
        cell_faces, kappa_face, source,
        ncells, dt
    );
}

// ============================================================================
// Solver::step_gpu — один шаг по времени на GPU.
//
// thrust::raw_pointer_cast(vec.data()) извлекает сырой указатель (float*)
// из thrust::device_vector. CUDA-ядра (__global__) не умеют работать
// с thrust-объектами напрямую — им нужны обычные указатели.
//
// Запуск ядра: <<<blocks, threads>>>
//   blocks  = ceil(ncells / 256) — сколько блоков запустить
//   threads = 256 — потоков в блоке
//
// cudaDeviceSynchronize() — ждём завершения ядра. Без него следующие
// операции на CPU могли бы начаться до окончания GPU-вычислений.
// ============================================================================
void Solver::step_gpu() {
    int n = gpu_mesh.ncells;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Извлекаем raw-указатели из thrust::device_vector
    float* d_T_curr     = thrust::raw_pointer_cast(gpu_mesh.T_curr.data());
    float* d_T_next     = thrust::raw_pointer_cast(gpu_mesh.T_next.data());
    float* d_volumes    = thrust::raw_pointer_cast(gpu_mesh.volumes.data());
    int*   d_f_owner    = thrust::raw_pointer_cast(gpu_mesh.faces.owner.data());
    int*   d_f_neighbor = thrust::raw_pointer_cast(gpu_mesh.faces.neighbor.data());
    float* d_f_area     = thrust::raw_pointer_cast(gpu_mesh.faces.area.data());
    float* d_f_distance = thrust::raw_pointer_cast(gpu_mesh.faces.distance.data());
    int*   d_cf         = thrust::raw_pointer_cast(gpu_mesh.cell_faces.data());
    float* d_kf         = thrust::raw_pointer_cast(gpu_mesh.kappa_face.data());
    float* d_src        = thrust::raw_pointer_cast(gpu_mesh.source.data());

    heat_fvm_kernel<<<blocks, threads>>>(
        d_T_curr, d_T_next, d_volumes,
        d_f_owner, d_f_neighbor, d_f_area, d_f_distance,
        d_cf, d_kf, d_src,
        n, dt
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaDeviceSynchronize();

    gpu_mesh.swap_buffers();
}
