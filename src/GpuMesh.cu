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
