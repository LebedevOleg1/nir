#include "Solver.hpp"
#include "FluxKernels.hpp"
#include "EulerUtils.hpp"
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

// ============================================================================
// CUDA kernels -- one per physics type
// ============================================================================

__global__ void heat_kernel(
    const float* RESTRICT U_curr,
    float* RESTRICT U_next,
    const float* RESTRICT volumes,
    const int*   RESTRICT face_owner,
    const int*   RESTRICT face_neighbor,
    const float* RESTRICT face_area,
    const float* RESTRICT face_distance,
    const int*   RESTRICT cell_faces,
    const float* RESTRICT source,
    const int ncells,
    const int ncells_total,
    const float kappa,
    const float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    compute_cell_update_heat(
        i, U_curr, U_next, volumes,
        face_owner, face_neighbor, face_area, face_distance,
        cell_faces, source, ncells, ncells_total, kappa, dt);
}

__global__ void diffusion_kernel(
    const float* RESTRICT U_curr,
    float* RESTRICT U_next,
    const float* RESTRICT volumes,
    const int*   RESTRICT face_owner,
    const int*   RESTRICT face_neighbor,
    const float* RESTRICT face_area,
    const float* RESTRICT face_distance,
    const int*   RESTRICT cell_faces,
    const float* RESTRICT source,
    const int ncells,
    const int ncells_total,
    const float D_coeff,
    const float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    compute_cell_update_diffusion(
        i, U_curr, U_next, volumes,
        face_owner, face_neighbor, face_area, face_distance,
        cell_faces, source, ncells, ncells_total, D_coeff, dt);
}

__global__ void euler_kernel(
    const float* RESTRICT U_curr,
    float* RESTRICT U_next,
    const float* RESTRICT volumes,
    const int*   RESTRICT face_owner,
    const int*   RESTRICT face_neighbor,
    const float* RESTRICT face_area,
    const float* RESTRICT face_distance,
    const float* RESTRICT face_nx,
    const float* RESTRICT face_ny,
    const int*   RESTRICT cell_faces,
    const int ncells,
    const int ncells_total,
    const float gamma,
    const float dt,
    const float gravity)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    compute_cell_update_euler(
        i, U_curr, U_next, volumes,
        face_owner, face_neighbor, face_area, face_distance,
        face_nx, face_ny, cell_faces,
        ncells, ncells_total, gamma, dt, gravity);
}

// ============================================================================
// step_gpu -- launches the appropriate kernel
// ============================================================================
template<>
void Solver<PhysicsType::Heat>::step_gpu() {
    int n = gpu_mesh.ncells;
    int nt = gpu_mesh.ncells_total;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float* d_curr = thrust::raw_pointer_cast(gpu_state.curr.data());
    float* d_next = thrust::raw_pointer_cast(gpu_state.next.data());
    float* d_vol  = thrust::raw_pointer_cast(gpu_mesh.volumes.data());
    int*   d_fo   = thrust::raw_pointer_cast(gpu_mesh.faces.owner.data());
    int*   d_fn   = thrust::raw_pointer_cast(gpu_mesh.faces.neighbor.data());
    float* d_fa   = thrust::raw_pointer_cast(gpu_mesh.faces.area.data());
    float* d_fd   = thrust::raw_pointer_cast(gpu_mesh.faces.distance.data());
    int*   d_cf   = thrust::raw_pointer_cast(gpu_mesh.cell_faces.data());
    float* d_src  = thrust::raw_pointer_cast(gpu_state.source.data());

    heat_kernel<<<blocks, threads>>>(
        d_curr, d_next, d_vol,
        d_fo, d_fn, d_fa, d_fd,
        d_cf, d_src, n, nt, config.kappa, dt);

    cudaDeviceSynchronize();
    gpu_state.swap_buffers();
}

template<>
void Solver<PhysicsType::Diffusion>::step_gpu() {
    int n = gpu_mesh.ncells;
    int nt = gpu_mesh.ncells_total;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float* d_curr = thrust::raw_pointer_cast(gpu_state.curr.data());
    float* d_next = thrust::raw_pointer_cast(gpu_state.next.data());
    float* d_vol  = thrust::raw_pointer_cast(gpu_mesh.volumes.data());
    int*   d_fo   = thrust::raw_pointer_cast(gpu_mesh.faces.owner.data());
    int*   d_fn   = thrust::raw_pointer_cast(gpu_mesh.faces.neighbor.data());
    float* d_fa   = thrust::raw_pointer_cast(gpu_mesh.faces.area.data());
    float* d_fd   = thrust::raw_pointer_cast(gpu_mesh.faces.distance.data());
    int*   d_cf   = thrust::raw_pointer_cast(gpu_mesh.cell_faces.data());
    float* d_src  = thrust::raw_pointer_cast(gpu_state.source.data());

    diffusion_kernel<<<blocks, threads>>>(
        d_curr, d_next, d_vol,
        d_fo, d_fn, d_fa, d_fd,
        d_cf, d_src, n, nt, config.kappa, dt);

    cudaDeviceSynchronize();
    gpu_state.swap_buffers();
}

template<>
void Solver<PhysicsType::Euler>::step_gpu() {
    int n = gpu_mesh.ncells;
    int nt = gpu_mesh.ncells_total;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

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

    euler_kernel<<<blocks, threads>>>(
        d_curr, d_next, d_vol,
        d_fo, d_fn, d_fa, d_fd,
        d_fnx, d_fny, d_cf,
        n, nt, config.gamma, dt, config.gravity);

    cudaDeviceSynchronize();
    gpu_state.swap_buffers();
}

// ============================================================================
// GPU CFL for Euler (adaptive dt)
// ============================================================================
struct EulerWaveSpeedFunctor {
    const float* U;
    int ncells_total;
    float gamma;

    EulerWaveSpeedFunctor(const float* U_, int nt, float g)
        : U(U_), ncells_total(nt), gamma(g) {}

    __host__ __device__ float operator()(int i) const {
        float rho  = U[0 * ncells_total + i];
        float rhou = U[1 * ncells_total + i];
        float rhov = U[2 * ncells_total + i];
        float E    = U[3 * ncells_total + i];
        return euler_max_wavespeed(rho, rhou, rhov, E, gamma);
    }
};

template<>
float_t Solver<PhysicsType::Euler>::compute_dt_gpu() {
    int n = gpu_mesh.ncells;
    float* d_curr = thrust::raw_pointer_cast(gpu_state.curr.data());

    // Create counting iterator and transform-reduce
    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end(n);

    EulerWaveSpeedFunctor func(d_curr, gpu_mesh.ncells_total, config.gamma);
    float max_speed = thrust::transform_reduce(
        begin, end, func, 1e-10f, thrust::maximum<float>());

    if (mpi_size > 1) {
        float global_max;
        MPI_Allreduce(&max_speed, &global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
        max_speed = global_max;
    }

    float hx = mesh.get_hx();
    float hy = mesh.get_hy();
    float h_min = (hx < hy) ? hx : hy;
    return config.cfl * h_min / max_speed;
}

// Heat/Diffusion: fixed dt, same formula as CPU
template<>
float_t Solver<PhysicsType::Heat>::compute_dt_gpu() {
    float hx = mesh.get_hx();
    float hy = mesh.get_hy();
    float h_min = (hx < hy) ? hx : hy;
    return config.cfl * 0.25f * h_min * h_min / config.kappa;
}

template<>
float_t Solver<PhysicsType::Diffusion>::compute_dt_gpu() {
    float hx = mesh.get_hx();
    float hy = mesh.get_hy();
    float h_min = (hx < hy) ? hx : hy;
    return config.cfl * 0.25f * h_min * h_min / config.kappa;
}

// GPU BC application stubs (download → CPU apply → upload, handled in solve loop)
template<>
void Solver<PhysicsType::Heat>::apply_bcs_gpu() {}
template<>
void Solver<PhysicsType::Diffusion>::apply_bcs_gpu() {}
template<>
void Solver<PhysicsType::Euler>::apply_bcs_gpu() {}
