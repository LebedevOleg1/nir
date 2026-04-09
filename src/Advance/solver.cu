// solver.cu — single compilation unit for Solver<P>.
// nvcc compiles both the CPU template bodies (via solver_impl.inl)
// and the GPU-specific specializations below.

#include "Riemann/FluxKernels.hpp"
#include "Riemann/EulerUtils.hpp"
#include "Base/ParallelFor.hpp"
#include "Base/FieldView.hpp"
#include "Timing/Timer.hpp"
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

// All CPU template method bodies (constructor, step_cpu, solve, etc.)
#include "Advance/solver_impl.inl"

// ============================================================================
// GPU step — ParallelFor + FieldView (no explicit __global__ needed).
//
// The lambda captures raw device pointers (POD) annotated FVM_HOST_DEVICE,
// compiled by nvcc as a device-callable with --expt-extended-lambda.
// ============================================================================

template<>
void Solver<PhysicsType::Heat>::step_gpu() {
    int n  = gpu_mesh.ncells;
    int nt = gpu_mesh.ncells_total;

    float* d_curr = thrust::raw_pointer_cast(gpu_state.curr.data());
    float* d_next = thrust::raw_pointer_cast(gpu_state.next.data());
    float* d_vol  = thrust::raw_pointer_cast(gpu_mesh.volumes.data());
    int*   d_fo   = thrust::raw_pointer_cast(gpu_mesh.faces.owner.data());
    int*   d_fn   = thrust::raw_pointer_cast(gpu_mesh.faces.neighbor.data());
    float* d_fa   = thrust::raw_pointer_cast(gpu_mesh.faces.area.data());
    float* d_fd   = thrust::raw_pointer_cast(gpu_mesh.faces.distance.data());
    int*   d_cf   = thrust::raw_pointer_cast(gpu_mesh.cell_faces.data());
    float* d_src  = thrust::raw_pointer_cast(gpu_state.source.data());
    float  kappa  = config.kappa;
    float  dt_    = dt;

    ParallelFor(true, n, [=] FVM_HOST_DEVICE (int i) {
        compute_cell_update_heat(
            i, d_curr, d_next, d_vol,
            d_fo, d_fn, d_fa, d_fd,
            d_cf, d_src, n, nt, kappa, dt_);
    });

    cudaDeviceSynchronize();
    gpu_state.swap_buffers();
}

template<>
void Solver<PhysicsType::Diffusion>::step_gpu() {
    int n  = gpu_mesh.ncells;
    int nt = gpu_mesh.ncells_total;

    float* d_curr = thrust::raw_pointer_cast(gpu_state.curr.data());
    float* d_next = thrust::raw_pointer_cast(gpu_state.next.data());
    float* d_vol  = thrust::raw_pointer_cast(gpu_mesh.volumes.data());
    int*   d_fo   = thrust::raw_pointer_cast(gpu_mesh.faces.owner.data());
    int*   d_fn   = thrust::raw_pointer_cast(gpu_mesh.faces.neighbor.data());
    float* d_fa   = thrust::raw_pointer_cast(gpu_mesh.faces.area.data());
    float* d_fd   = thrust::raw_pointer_cast(gpu_mesh.faces.distance.data());
    int*   d_cf   = thrust::raw_pointer_cast(gpu_mesh.cell_faces.data());
    float* d_src  = thrust::raw_pointer_cast(gpu_state.source.data());
    float  D   = config.kappa;
    float  dt_ = dt;

    ParallelFor(true, n, [=] FVM_HOST_DEVICE (int i) {
        compute_cell_update_diffusion(
            i, d_curr, d_next, d_vol,
            d_fo, d_fn, d_fa, d_fd,
            d_cf, d_src, n, nt, D, dt_);
    });

    cudaDeviceSynchronize();
    gpu_state.swap_buffers();
}

template<>
void Solver<PhysicsType::Euler>::step_gpu() {
    int n  = gpu_mesh.ncells;
    int nt = gpu_mesh.ncells_total;

    float* d_curr = thrust::raw_pointer_cast(gpu_state.curr.data());
    float* d_next = thrust::raw_pointer_cast(gpu_state.next.data());
    float* d_vol  = thrust::raw_pointer_cast(gpu_mesh.volumes.data());
    int*   d_fo   = thrust::raw_pointer_cast(gpu_mesh.faces.owner.data());
    int*   d_fn   = thrust::raw_pointer_cast(gpu_mesh.faces.neighbor.data());
    float* d_fa   = thrust::raw_pointer_cast(gpu_mesh.faces.area.data());
    float* d_fd   = thrust::raw_pointer_cast(gpu_mesh.faces.distance.data());
    float* d_fnx  = thrust::raw_pointer_cast(gpu_mesh.faces.normal_x.data());
    float* d_fny  = thrust::raw_pointer_cast(gpu_mesh.faces.normal_y.data());
    int*   d_cf   = thrust::raw_pointer_cast(gpu_mesh.cell_faces.data());
    float  gamma_   = config.gamma;
    float  gravity_ = config.gravity;
    float  dt_      = dt;

    ParallelFor(true, n, [=] FVM_HOST_DEVICE (int i) {
        compute_cell_update_euler(
            i, d_curr, d_next, d_vol,
            d_fo, d_fn, d_fa, d_fd,
            d_fnx, d_fny, d_cf,
            n, nt, gamma_, dt_, gravity_);
    });

    cudaDeviceSynchronize();
    gpu_state.swap_buffers();
}

// ============================================================================
// GPU CFL for Euler — thrust::transform_reduce over all cells
// ============================================================================
struct WaveSpeedFunctor {
    const float* U;
    int nt;
    float gamma;

    __host__ __device__ float operator()(int i) const {
        return euler_max_wavespeed(
            U[0*nt+i], U[1*nt+i], U[2*nt+i], U[3*nt+i], gamma);
    }
};

template<>
float Solver<PhysicsType::Euler>::compute_dt_gpu() {
    float* d_curr = thrust::raw_pointer_cast(gpu_state.curr.data());
    int n  = gpu_mesh.ncells;
    int nt = gpu_mesh.ncells_total;

    WaveSpeedFunctor func{d_curr, nt, config.gamma};
    float max_speed = thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(n),
        func, 1e-10f, thrust::maximum<float>());

    if (mpi_size > 1) {
        float gmax;
        MPI_Allreduce(&max_speed, &gmax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        max_speed = gmax;
    }

    float h_min = fminf(mesh.get_hx(), mesh.get_hy());
    return config.cfl * h_min / max_speed;
}

template<>
float Solver<PhysicsType::Heat>::compute_dt_gpu() {
    float h = fminf(mesh.get_hx(), mesh.get_hy());
    return config.cfl * 0.25f * h * h / config.kappa;
}

template<>
float Solver<PhysicsType::Diffusion>::compute_dt_gpu() {
    float h = fminf(mesh.get_hx(), mesh.get_hy());
    return config.cfl * 0.25f * h * h / config.kappa;
}

// GPU BC stubs (BC applied on CPU: download→apply→upload in solve loop)
template<> void Solver<PhysicsType::Heat>::apply_bcs_gpu()      {}
template<> void Solver<PhysicsType::Diffusion>::apply_bcs_gpu() {}
template<> void Solver<PhysicsType::Euler>::apply_bcs_gpu()     {}

// ============================================================================
// Explicit template instantiations — must come AFTER all specializations
// so the linker sees the right (specialized) versions for GPU methods.
// ============================================================================
template class Solver<PhysicsType::Heat>;
template class Solver<PhysicsType::Diffusion>;
template class Solver<PhysicsType::Euler>;
