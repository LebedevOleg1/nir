#pragma once

#include "Mesh.hpp"
#include <thrust/device_vector.h>

// ============================================================================
// GpuMesh -- geometry-only GPU mirror of Mesh.
// Physics state is in GpuState<NVAR> (separate).
// ============================================================================
struct GpuFaces {
    thrust::device_vector<int>   owner;
    thrust::device_vector<int>   neighbor;
    thrust::device_vector<float> area;
    thrust::device_vector<float> normal_x;
    thrust::device_vector<float> normal_y;
    thrust::device_vector<float> distance;
};

struct GpuMesh {
    thrust::device_vector<float> volumes;
    thrust::device_vector<int>   cell_faces;
    thrust::device_vector<int>   face_boundary_id;

    GpuFaces faces;

    int ncells = 0;
    int ncells_total = 0;
    int nfaces = 0;

    void upload(const Mesh& mesh);
};

// ============================================================================
// GpuState<NVAR> -- physics state on GPU.
// SoA layout: U[var * ncells_total + cell].
// Double-buffered (curr/next) for explicit time stepping.
// ============================================================================
template<int NVAR>
struct GpuState {
    thrust::device_vector<float> curr;  // NVAR * ncells_total
    thrust::device_vector<float> next;  // NVAR * ncells_total
    thrust::device_vector<float> source; // ncells (for heat/diffusion source term)

    int ncells_total = 0;

    void upload(const float* cpu_curr, int n_total);
    void download(float* cpu_curr, int n_total);
    void upload_source(const std::vector<float>& src);
    void swap_buffers();

    // Targeted halo transfers (MPI: only boundary rows)
    void download_halo_rows(float* cpu_curr, int nx, int real_ny, int nvar);
    void upload_halo_rows(const float* cpu_curr, int nx, int total_ny, int nvar);
};
