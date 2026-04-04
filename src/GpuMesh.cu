#include "GpuMesh.hpp"

// ============================================================================
// GpuMesh::upload — копирование CPU → GPU.
//
// thrust::device_vector при присваивании от std::vector автоматически:
//   1) Вызывает cudaMalloc если размер не совпадает
//   2) Вызывает cudaMemcpy(hostPtr, devicePtr, size, HostToDevice)
// Поэтому весь upload — это просто присваивания.
// ============================================================================
void GpuMesh::upload(const Mesh& mesh) {
    ncells = mesh.get_ncells();
    nfaces = mesh.faces.count;

    // Температура: копируем начальное состояние в оба буфера
    T_curr = mesh.data.curr.T;
    T_next.resize(ncells, 0.0f);

    // Геометрия ячеек
    volumes = mesh.volumes;

    // Грани (SoA) — каждый массив копируется отдельно
    faces.owner    = mesh.faces.owner;
    faces.neighbor = mesh.faces.neighbor;
    faces.area     = mesh.faces.area;
    faces.distance = mesh.faces.distance;

    // Связность ячейка→грани
    cell_faces = mesh.cell_faces;

    // Физические поля
    kappa_face = mesh.kappa_face;
    source     = mesh.source;
}

// ============================================================================
// GpuMesh::download_T — копирование T_curr из GPU в CPU.
//
// thrust::copy из device_vector в host-итератор вызывает cudaMemcpy D→H.
// Нужно для:
//   - VTK-вывода (визуализация в ParaView)
//   - MPI halo exchange (пока без CUDA-aware MPI)
// ============================================================================
void GpuMesh::download_T(Mesh& mesh) {
    // thrust::copy выполняет cudaMemcpy Device→Host
    thrust::copy(T_curr.begin(), T_curr.end(), mesh.data.curr.T.begin());
}

void GpuMesh::upload_source(const std::vector<float>& src) {
    source = src;  // автоматический cudaMemcpy H→D через thrust
}

// ============================================================================
// swap_buffers — обмен указателей T_curr ↔ T_next.
//
// thrust::device_vector::swap — O(1), меняет только внутренние указатели,
// не копирует данные. Это стандартный паттерн «ping-pong buffer».
// ============================================================================
void GpuMesh::swap_buffers() {
    T_curr.swap(T_next);
}

// Копируем граничные реальные строки GPU → CPU для MPI halo exchange
void GpuMesh::download_halo_rows(Mesh& mesh, int nx, int real_ny) {
    float* d_ptr = thrust::raw_pointer_cast(T_curr.data());
    float* h_ptr = mesh.data.curr.T.data();
    // Строка 1 (первая реальная) → CPU
    cudaMemcpy(h_ptr + nx, d_ptr + nx, nx * sizeof(float), cudaMemcpyDeviceToHost);
    // Строка real_ny (последняя реальная) → CPU
    cudaMemcpy(h_ptr + real_ny * nx, d_ptr + real_ny * nx, nx * sizeof(float), cudaMemcpyDeviceToHost);
}

// Копируем ghost-строки CPU → GPU после MPI halo exchange
void GpuMesh::upload_halo_rows(const Mesh& mesh, int nx, int total_ny) {
    float* d_ptr = thrust::raw_pointer_cast(T_curr.data());
    const float* h_ptr = mesh.data.curr.T.data();
    // Строка 0 (ghost bottom) → GPU
    cudaMemcpy(d_ptr, h_ptr, nx * sizeof(float), cudaMemcpyHostToDevice);
    // Строка total_ny-1 (ghost top) → GPU
    cudaMemcpy(d_ptr + (total_ny - 1) * nx, h_ptr + (total_ny - 1) * nx,
               nx * sizeof(float), cudaMemcpyHostToDevice);
}

// Загрузить T_curr целиком из CPU в GPU
void GpuMesh::upload_T(const Mesh& mesh) {
    T_curr = mesh.data.curr.T;
}
