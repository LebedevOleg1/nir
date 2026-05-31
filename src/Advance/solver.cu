// solver.cu — single compilation unit for Solver<P>.
// nvcc compiles both the CPU template bodies (via solver_impl.inl)
// and the GPU-specific specializations below.

#include "Advance/Solver.hpp"
#include "Riemann/FluxKernels.hpp"
#include "Riemann/EulerUtils.hpp"
#include "Riemann/BCKernel.hpp"
#include "Base/ParallelFor.hpp"
#include "Base/FieldView.hpp"
#include "Timing/Timer.hpp"
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

// ============================================================================
// GPU boundary conditions for Euler.
//
// Fills ghost-cell values of a given state buffer directly on the GPU by
// looping over all faces and applying apply_bc_euler() to boundary faces.
// POD struct EulerBCSpecs is captured by value into the device lambda
// (plain arrays inside a struct copy cleanly into extended lambdas).
// ============================================================================
struct EulerBCSpecs {
    int   type[4];
    float rho[4], u[4], v[4], p[4];
};

static void apply_euler_bcs_gpu_buffer(
    float* d_U, const GpuMesh& gm, const SimConfig& cfg)
{
    int nfaces = gm.nfaces;
    int nt     = gm.ncells_total;

    const int*   d_bid = thrust::raw_pointer_cast(gm.face_boundary_id.data());
    const int*   d_fo  = thrust::raw_pointer_cast(gm.faces.owner.data());
    const int*   d_fn  = thrust::raw_pointer_cast(gm.faces.neighbor.data());
    const float* d_fnx = thrust::raw_pointer_cast(gm.faces.normal_x.data());
    const float* d_fny = thrust::raw_pointer_cast(gm.faces.normal_y.data());
    float gamma_ = cfg.gamma;

    EulerBCSpecs bcs;
    for (int b = 0; b < 4; ++b) {
        bcs.type[b] = (int)cfg.bc[b].type;
        bcs.rho[b]  = cfg.bc[b].inlet_rho;
        bcs.u[b]    = cfg.bc[b].inlet_u;
        bcs.v[b]    = cfg.bc[b].inlet_v;
        bcs.p[b]    = cfg.bc[b].inlet_p;
    }

    ParallelFor(true, nfaces, [=] FVM_HOST_DEVICE (int fi) {
        int bid = d_bid[fi];
        if (bid < 0) return;                 // not a BC ghost face
        int ghost    = d_fn[fi];
        int interior = d_fo[fi];
        apply_bc_euler(ghost, interior, d_U, nt,
                       (BCType)bcs.type[bid],
                       d_fnx[fi], d_fny[fi],
                       bcs.rho[bid], bcs.u[bid], bcs.v[bid], bcs.p[bid],
                       gamma_);
    });
    cudaDeviceSynchronize();
}

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

// Helper: extract all raw Euler kernel pointers from GpuMesh/GpuState
static inline void euler_gpu_ptrs(
    const GpuMesh& gm, GpuState<4>& gs,
    float*& d_curr, float*& d_next,
    const float*& d_vol, const int*& d_fo, const int*& d_fn,
    const float*& d_fa, const float*& d_fd,
    const float*& d_fnx, const float*& d_fny,
    const int*& d_cf, const int*& d_fso, const int*& d_fsn)
{
    d_curr = thrust::raw_pointer_cast(gs.curr.data());
    d_next = thrust::raw_pointer_cast(gs.next.data());
    d_vol  = thrust::raw_pointer_cast(gm.volumes.data());
    d_fo   = thrust::raw_pointer_cast(gm.faces.owner.data());
    d_fn   = thrust::raw_pointer_cast(gm.faces.neighbor.data());
    d_fa   = thrust::raw_pointer_cast(gm.faces.area.data());
    d_fd   = thrust::raw_pointer_cast(gm.faces.distance.data());
    d_fnx  = thrust::raw_pointer_cast(gm.faces.normal_x.data());
    d_fny  = thrust::raw_pointer_cast(gm.faces.normal_y.data());
    d_cf   = thrust::raw_pointer_cast(gm.cell_faces.data());
    d_fso  = thrust::raw_pointer_cast(gm.faces.stencil_owner.data());
    d_fsn  = thrust::raw_pointer_cast(gm.faces.stencil_neighbor.data());
}

template<>
void Solver<PhysicsType::Euler>::step_gpu() {
    int n  = gpu_mesh.ncells;
    int nt = gpu_mesh.ncells_total;

    float *d_curr, *d_next;
    const float *d_vol, *d_fa, *d_fd, *d_fnx, *d_fny;
    const int   *d_fo, *d_fn, *d_cf, *d_fso, *d_fsn;
    euler_gpu_ptrs(gpu_mesh, gpu_state,
                   d_curr, d_next, d_vol, d_fo, d_fn,
                   d_fa, d_fd, d_fnx, d_fny, d_cf, d_fso, d_fsn);

    float gamma_   = config.gamma;
    float gravity_ = config.gravity;
    float dt_      = dt;
    int   muscl_   = config.muscl ? 1 : 0;
    int   hllc_    = config.hllc ? 1 : 0;

    ParallelFor(true, n, [=] FVM_HOST_DEVICE (int i) {
        compute_cell_update_euler(
            i, d_curr, d_next, d_vol,
            d_fo, d_fn, d_fa, d_fd,
            d_fnx, d_fny, d_cf, d_fso, d_fsn,
            n, nt, gamma_, dt_, gravity_, muscl_, hllc_);
    });

    cudaDeviceSynchronize();
    gpu_state.swap_buffers();
}

// ============================================================================
// SSP-RK2 GPU step (2nd-order in time, designed for use with MUSCL)
//
// Stage 1: next  = U^n + dt*L(U^n)
// Stage 2: curr  = U* + dt*L(U*)  [reuses curr buffer, U^n saved in rk_aux]
// Blend:   curr  = 0.5*rk_aux + 0.5*curr = U^{n+1}
//
// Note: BCs are NOT applied between stages to avoid CPU round-trips.
// For periodic-BC problems (KH, isentropic vortex) this is exact.
// ============================================================================
template<>
void Solver<PhysicsType::Euler>::step_rk2_gpu() {
    int n  = gpu_mesh.ncells;
    int nt = gpu_mesh.ncells_total;
    int sz = 4 * nt;  // NVAR=4

    float *d_curr, *d_next;
    const float *d_vol, *d_fa, *d_fd, *d_fnx, *d_fny;
    const int   *d_fo, *d_fn, *d_cf, *d_fso, *d_fsn;
    euler_gpu_ptrs(gpu_mesh, gpu_state,
                   d_curr, d_next, d_vol, d_fo, d_fn,
                   d_fa, d_fd, d_fnx, d_fny, d_cf, d_fso, d_fsn);
    float* d_aux = thrust::raw_pointer_cast(gpu_state.rk_aux.data());

    float gamma_   = config.gamma;
    float gravity_ = config.gravity;
    float dt_      = dt;
    int   muscl_   = config.muscl ? 1 : 0;
    int   hllc_    = config.hllc ? 1 : 0;

    // Save U^n → rk_aux
    thrust::copy(gpu_state.curr.begin(), gpu_state.curr.end(), gpu_state.rk_aux.begin());

    // Stage 1: curr → next  (next = U*)
    ParallelFor(true, n, [=] FVM_HOST_DEVICE (int i) {
        compute_cell_update_euler(
            i, d_curr, d_next, d_vol,
            d_fo, d_fn, d_fa, d_fd,
            d_fnx, d_fny, d_cf, d_fso, d_fsn,
            n, nt, gamma_, dt_, gravity_, muscl_, hllc_);
    });
    cudaDeviceSynchronize();

    // After stage 1: curr = U^n (rk_aux), next = U*
    // The kernel only writes interior cells [0, ncells); ghost cells of the
    // next buffer are stale/garbage. Stage-2 MUSCL reconstruction reads ghost
    // cells at non-periodic boundaries (walls), so we must refresh them here.
    // For periodic problems (n_ghost_bc == 0) this is a no-op.
    if (mesh.get_n_ghost_bc() > 0)
        apply_euler_bcs_gpu_buffer(
            thrust::raw_pointer_cast(gpu_state.next.data()), gpu_mesh, config);

    // Stage 2: next → curr  (curr = U**)
    // Swap roles: input = next (U*), output = curr (U**)
    float* d_ustar = thrust::raw_pointer_cast(gpu_state.next.data());
    float* d_uout  = thrust::raw_pointer_cast(gpu_state.curr.data());

    ParallelFor(true, n, [=] FVM_HOST_DEVICE (int i) {
        compute_cell_update_euler(
            i, d_ustar, d_uout, d_vol,
            d_fo, d_fn, d_fa, d_fd,
            d_fnx, d_fny, d_cf, d_fso, d_fsn,
            n, nt, gamma_, dt_, gravity_, muscl_, hllc_);
    });
    cudaDeviceSynchronize();

    // Blend: curr = 0.5*rk_aux + 0.5*curr  →  U^{n+1} in curr
    thrust::transform(
        gpu_state.curr.begin(),   gpu_state.curr.begin() + sz,
        gpu_state.rk_aux.begin(),
        gpu_state.curr.begin(),
        [] __host__ __device__ (float a, float b) { return 0.5f * a + 0.5f * b; });
    cudaDeviceSynchronize();
    // curr = U^{n+1}; no swap needed
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

// GPU BC: Heat/Diffusion still use the CPU path (download→apply→upload).
template<> void Solver<PhysicsType::Heat>::apply_bcs_gpu()      {}
template<> void Solver<PhysicsType::Diffusion>::apply_bcs_gpu() {}

// Euler: fill ghost cells of the curr buffer directly on the GPU.
template<> void Solver<PhysicsType::Euler>::apply_bcs_gpu() {
    if (mesh.get_n_ghost_bc() == 0) return;   // periodic: nothing to do
    apply_euler_bcs_gpu_buffer(
        thrust::raw_pointer_cast(gpu_state.curr.data()), gpu_mesh, config);
}

// step_rk2_gpu stubs for non-Euler physics (RK2 is only needed for Euler)
template<> void Solver<PhysicsType::Heat>::step_rk2_gpu()      { step_gpu(); }
template<> void Solver<PhysicsType::Diffusion>::step_rk2_gpu() { step_gpu(); }

// ============================================================================
// Explicit template instantiations — must come AFTER all specializations
// so the linker sees the right (specialized) versions for GPU methods.
// ============================================================================
template class Solver<PhysicsType::Heat>;
template class Solver<PhysicsType::Diffusion>;
template class Solver<PhysicsType::Euler>;
