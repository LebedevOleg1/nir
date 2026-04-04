#include "GpuMesh.hpp"

// ============================================================================
// GpuMesh::upload -- geometry only (no physics state)
// ============================================================================
void GpuMesh::upload(const Mesh& mesh) {
    ncells = mesh.get_ncells();
    ncells_total = mesh.get_ncells_total();
    nfaces = mesh.faces.count;

    volumes = mesh.volumes;

    faces.owner    = mesh.faces.owner;
    faces.neighbor = mesh.faces.neighbor;
    faces.area     = mesh.faces.area;
    faces.normal_x = mesh.faces.normal_x;
    faces.normal_y = mesh.faces.normal_y;
    faces.distance = mesh.faces.distance;

    cell_faces = mesh.cell_faces;
    face_boundary_id = mesh.face_boundary_id;
}

// ============================================================================
// GpuState<NVAR> -- physics state on GPU
// ============================================================================
template<int NVAR>
void GpuState<NVAR>::upload(const float* cpu_curr, int n_total) {
    ncells_total = n_total;
    curr.assign(cpu_curr, cpu_curr + NVAR * n_total);
    next.resize(NVAR * n_total, 0.0f);
}

template<int NVAR>
void GpuState<NVAR>::download(float* cpu_curr, int n_total) {
    thrust::copy(curr.begin(), curr.begin() + NVAR * n_total, cpu_curr);
}

template<int NVAR>
void GpuState<NVAR>::upload_source(const std::vector<float>& src) {
    source = src;
}

template<int NVAR>
void GpuState<NVAR>::swap_buffers() {
    curr.swap(next);
}

template<int NVAR>
void GpuState<NVAR>::download_halo_rows(float* cpu_curr, int nx, int real_ny, int nvar) {
    float* d_ptr = thrust::raw_pointer_cast(curr.data());
    int ncells = ncells_total;  // includes ghosts
    // For each variable, download rows 1 and real_ny
    for (int v = 0; v < nvar; ++v) {
        int offset = v * ncells;
        // Row 1 (first real row)
        cudaMemcpy(cpu_curr + offset + nx,
                   d_ptr + offset + nx,
                   nx * sizeof(float), cudaMemcpyDeviceToHost);
        // Row real_ny (last real row)
        cudaMemcpy(cpu_curr + offset + real_ny * nx,
                   d_ptr + offset + real_ny * nx,
                   nx * sizeof(float), cudaMemcpyDeviceToHost);
    }
}

template<int NVAR>
void GpuState<NVAR>::upload_halo_rows(const float* cpu_curr, int nx, int total_ny, int nvar) {
    float* d_ptr = thrust::raw_pointer_cast(curr.data());
    int ncells = ncells_total;
    for (int v = 0; v < nvar; ++v) {
        int offset = v * ncells;
        // Row 0 (ghost bottom)
        cudaMemcpy(d_ptr + offset,
                   cpu_curr + offset,
                   nx * sizeof(float), cudaMemcpyHostToDevice);
        // Row total_ny-1 (ghost top)
        cudaMemcpy(d_ptr + offset + (total_ny - 1) * nx,
                   cpu_curr + offset + (total_ny - 1) * nx,
                   nx * sizeof(float), cudaMemcpyHostToDevice);
    }
}

// Explicit template instantiations
template struct GpuState<1>;  // Heat, Diffusion
template struct GpuState<4>;  // Euler
