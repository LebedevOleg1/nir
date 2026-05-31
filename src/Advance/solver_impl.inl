#include "Advance/Solver.hpp"
#include "Riemann/FluxKernels.hpp"
#include "Riemann/BCKernel.hpp"
#include "Riemann/EulerUtils.hpp"
#include "Base/ParallelFor.hpp"
#include "Timing/Timer.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <mpi.h>
#include <cstring>

// ============================================================================
// Constructor
// ============================================================================
template<PhysicsType P>
Solver<P>::Solver(Mesh& mesh_, const SimConfig& config_, MpiDecomp* decomp_)
    : mesh(mesh_), config(config_), decomp(decomp_), use_gpu(config_.use_gpu)
{
    if (decomp) {
        mpi_rank = decomp->rank;
        mpi_size = decomp->size;
    }

    state.resize(mesh.get_ncells_total());
    source.assign(mesh.get_ncells(), 0.0f);

    set_initial_conditions();

    if (use_gpu) {
        // Assign GPU device: round-robin across available GPUs
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            cudaSetDevice(mpi_rank % device_count);
        }

        gpu_mesh.upload(mesh);
        gpu_state.upload(state.curr.data(), mesh.get_ncells_total());
        gpu_state.upload_source(source);
    }
}

// ============================================================================
// Initial conditions
// ============================================================================
template<PhysicsType P>
void Solver<P>::set_initial_conditions() {
    int ncells       = mesh.get_ncells();
    int ncells_total = mesh.get_ncells_total();

    if constexpr (P == PhysicsType::Heat || P == PhysicsType::Diffusion) {
        // Default: zero field; BCs and source will drive the solution.
    }
    else if constexpr (P == PhysicsType::Euler) {
        float gamma  = config.gamma;
        float Lx     = config.xmax - config.xmin;
        float Ly     = config.ymax - config.ymin;
        float x_off  = config.xmin;
        float y_off  = config.ymin;

        if (config.ic == "sod") {
            float x_mid = x_off + 0.5f * Lx;
            for (int i = 0; i < ncells; ++i) {
                float x = mesh.centers[i].x;
                float rho, u, v, p;
                if (x < x_mid) { rho=1.0f; u=0.0f; v=0.0f; p=1.0f; }
                else            { rho=0.125f; u=0.0f; v=0.0f; p=0.1f; }
                float E = p/(gamma-1.0f) + 0.5f*rho*(u*u+v*v);
                state.curr[0*ncells_total+i] = rho;
                state.curr[1*ncells_total+i] = rho*u;
                state.curr[2*ncells_total+i] = rho*v;
                state.curr[3*ncells_total+i] = E;
            }
        }
        else if (config.ic == "blast") {
            float cx = x_off + 0.5f*Lx, cy = y_off + 0.5f*Ly;
            float r  = 0.1f * Lx;
            for (int i = 0; i < ncells; ++i) {
                float dx = mesh.centers[i].x - cx;
                float dy = mesh.centers[i].y - cy;
                float p  = (dx*dx + dy*dy < r*r) ? 10.0f : 0.1f;
                float E  = p/(gamma-1.0f);
                state.curr[0*ncells_total+i] = 1.0f;
                state.curr[1*ncells_total+i] = 0.0f;
                state.curr[2*ncells_total+i] = 0.0f;
                state.curr[3*ncells_total+i] = E;
            }
        }
        else if (config.ic == "kh") {
            // Kelvin-Helmholtz: McNally, Lyra & Tassoul (2012) benchmark.
            // Smooth tanh density/velocity profiles, single-mode perturbation.
            // Domain [0,1]x[0,1]; rho1=1 (outer), rho2=2 (inner), p=2.5, gamma=5/3.
            // sigma = 0.05/sqrt(2), w0=0.1 (perturbation amplitude).
            float pi    = 3.14159265f;
            float p0    = 2.5f;
            float rho1  = 1.0f, rho2 = 2.0f;
            float u1    = -0.5f, u2  = 0.5f;
            // Use sigma from config or default
            float sigma = 0.05f / 1.41421356f;  // 0.05/sqrt(2)
            float w0    = 0.1f;
            float y1    = y_off + 0.25f * Ly;
            float y2    = y_off + 0.75f * Ly;

            for (int i = 0; i < ncells; ++i) {
                float x = mesh.centers[i].x;
                float y = mesh.centers[i].y;

                // McNally eq.(2)-(3): smooth tanh profiles
                float rho = rho1 + (rho2 - rho1) * 0.5f *
                            (tanhf((y - y1) / sigma) - tanhf((y - y2) / sigma));
                float u   = u1  + (u2  - u1)  * 0.5f *
                            (tanhf((y - y1) / sigma) - tanhf((y - y2) / sigma));

                // McNally eq.(4): single-mode v perturbation localised at interfaces
                float env1 = expf(-(y - y1) * (y - y1) / (2.0f * sigma * sigma));
                float env2 = expf(-(y - y2) * (y - y2) / (2.0f * sigma * sigma));
                float v    = w0 * sinf(2.0f * pi * x / Lx) * (env1 + env2);

                float E = p0 / (gamma - 1.0f) + 0.5f * rho * (u * u + v * v);
                state.curr[0 * ncells_total + i] = rho;
                state.curr[1 * ncells_total + i] = rho * u;
                state.curr[2 * ncells_total + i] = rho * v;
                state.curr[3 * ncells_total + i] = E;
            }
        }
        else if (config.ic == "vortex") {
            // Isentropic vortex (Yee et al. 1999): smooth exact solution for
            // convergence testing. Background: rho=1, u=1, v=0, T=1, gamma=1.4.
            float pi      = 3.14159265f;
            float u_inf   = 1.0f, v_inf = 0.0f;
            float rho_inf = 1.0f;
            float T_inf   = 1.0f;
            float eps     = 5.0f;   // vortex strength
            float x0      = 0.5f * (config.xmin + config.xmax);
            float y0      = 0.5f * (config.ymin + config.ymax);

            for (int i = 0; i < ncells; ++i) {
                float x   = mesh.centers[i].x;
                float y   = mesh.centers[i].y;
                float dx  = x - x0;
                float dy  = y - y0;
                float r2  = dx * dx + dy * dy;
                float f   = (1.0f - r2) / 2.0f;
                float du  = -eps / (2.0f * pi) * dy * expf(f);
                float dv  =  eps / (2.0f * pi) * dx * expf(f);
                float dT  = -(gamma - 1.0f) * eps * eps /
                             (8.0f * gamma * pi * pi) * expf(1.0f - r2);
                float T   = T_inf + dT;
                float rho = rho_inf * powf(T / T_inf, 1.0f / (gamma - 1.0f));
                float u   = u_inf + du;
                float v   = v_inf + dv;
                float p   = rho * T;  // ideal gas: p = rho * T (R=1)
                float E   = p / (gamma - 1.0f) + 0.5f * rho * (u * u + v * v);
                state.curr[0 * ncells_total + i] = rho;
                state.curr[1 * ncells_total + i] = rho * u;
                state.curr[2 * ncells_total + i] = rho * v;
                state.curr[3 * ncells_total + i] = E;
            }
        }
        else if (config.ic == "rt") {
            // Rayleigh-Taylor: Liska & Wendroff (2003) benchmark (SIAM J. Sci. Comput.).
            // Heavy fluid (rho=2) above y=0.5, light (rho=1) below.
            // Gravity g pointing in -y direction.
            // Hydrostatic pressure: p(y) = p_top + g*rho_heavy*(1-y) for y>0.5
            //                              p(y) = p_top + g*rho_heavy*0.5 + g*rho_light*(0.5-y) for y<0.5
            // Single-mode perturbation localized at the interface y=0.5:
            //   v = 0.01*(1+cos(2*pi*x/Lx))*(1+cos(2*pi*(y-0.5)/Ly))/4
            // One wavelength across the domain in x (one finger), smoothly
            // vanishing at the top/bottom walls in y.
            float pi       = 3.14159265f;
            float g        = (config.gravity > 0) ? config.gravity : 0.1f;
            float rho_h    = 2.0f, rho_l = 1.0f;
            float y_mid    = y_off + 0.5f * Ly;
            float p_top    = 2.5f;  // Liska 2003 boundary pressure

            // Pressure at interface y_mid (integrating hydrostatic from top)
            float p_mid    = p_top + g * rho_h * (y_off + Ly - y_mid);

            for (int i = 0; i < ncells; ++i) {
                float x = mesh.centers[i].x;
                float y = mesh.centers[i].y;

                float rho, p;
                if (y >= y_mid) {
                    rho = rho_h;
                    p   = p_top + g * rho_h * (y_off + Ly - y);
                } else {
                    rho = rho_l;
                    p   = p_mid + g * rho_l * (y_mid - y);
                }

                // Single-mode perturbation, localized at the interface y_mid.
                // (1+cos(2*pi*x/Lx)): one wavelength across the width.
                // (1+cos(2*pi*(y-y_mid)/Ly)): peaks at y_mid, zero at walls.
                float v = 0.01f * (1.0f + cosf(2.0f * pi * x / Lx))
                                * (1.0f + cosf(2.0f * pi * (y - y_mid) / Ly)) / 4.0f;

                float E = p / (gamma - 1.0f) + 0.5f * rho * v * v;
                state.curr[0 * ncells_total + i] = rho;
                state.curr[1 * ncells_total + i] = 0.0f;
                state.curr[2 * ncells_total + i] = rho * v;
                state.curr[3 * ncells_total + i] = E;
            }
        }
        else {
            // Uniform state from left-inlet BC
            const BCSpec& in = config.bc[(int)Boundary::Left];
            float rho0 = in.inlet_rho, u0 = in.inlet_u;
            float v0   = in.inlet_v,   p0 = in.inlet_p;
            float E0   = p0/(gamma-1.0f) + 0.5f*rho0*(u0*u0+v0*v0);
            for (int i = 0; i < ncells; ++i) {
                state.curr[0*ncells_total+i] = rho0;
                state.curr[1*ncells_total+i] = rho0*u0;
                state.curr[2*ncells_total+i] = rho0*v0;
                state.curr[3*ncells_total+i] = E0;
            }
        }
    }
}

// ============================================================================
// Stable dt (CFL condition)
// ============================================================================
template<PhysicsType P>
float Solver<P>::compute_dt() {
    float hx = mesh.get_hx(), hy = mesh.get_hy();
    float h_min = (hx < hy) ? hx : hy;

    if constexpr (P == PhysicsType::Heat || P == PhysicsType::Diffusion) {
        return config.cfl * 0.25f * h_min * h_min / config.kappa;
    }
    else if constexpr (P == PhysicsType::Euler) {
        int ncells       = mesh.get_ncells();
        int ncells_total = mesh.get_ncells_total();
        float max_speed  = 1e-10f;

        #pragma omp parallel for reduction(max:max_speed)
        for (int i = 0; i < ncells; ++i) {
            float s = euler_max_wavespeed(
                state.curr[0*ncells_total+i], state.curr[1*ncells_total+i],
                state.curr[2*ncells_total+i], state.curr[3*ncells_total+i],
                config.gamma);
            if (s > max_speed) max_speed = s;
        }

        if (mpi_size > 1) {
            float gmax;
            MPI_Allreduce(&max_speed, &gmax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
            max_speed = gmax;
        }
        return config.cfl * h_min / max_speed;
    }
    return 1e-4f;  // unreachable
}

// ============================================================================
// Source term
// ============================================================================
template<PhysicsType P>
void Solver<P>::update_source(float time) {
    if constexpr (P == PhysicsType::Heat || P == PhysicsType::Diffusion) {
        if (config.source.type != "gaussian") return;

        int nx    = mesh.get_nx();
        int start = mesh.is_mpi_mode() ? nx : 0;
        int end   = mesh.is_mpi_mode() ? mesh.get_ncells() - nx : mesh.get_ncells();

        float sx = config.source.x, sy = config.source.y;
        float r  = config.source.radius;
        float Ps = config.source.power;

        for (int c = start; c < end; ++c) {
            float dx = mesh.centers[c].x - sx;
            float dy = mesh.centers[c].y - sy;
            source[c] = (dx*dx + dy*dy < r*r)
                ? Ps * (1.0f + 0.5f * sinf(2.0f * 3.14159f * time / 5.0f))
                : 0.0f;
        }
    }
}

// ============================================================================
// Boundary conditions (CPU)
// ============================================================================
template<PhysicsType P>
void Solver<P>::apply_bcs_cpu() {
    if (mesh.get_n_ghost_bc() == 0) return;

    int ncells       = mesh.get_ncells();
    int ncells_total = mesh.get_ncells_total();
    float* U         = state.curr.data();

    for (int fi = 0; fi < mesh.faces.count; ++fi) {
        int bid = mesh.face_boundary_id[fi];
        if (bid < 0) continue;
        int owner = mesh.faces.owner[fi];
        int ghost = mesh.faces.neighbor[fi];
        if (ghost < ncells) continue;

        const BCSpec& bc = mesh.get_bc((Boundary)bid);

        if constexpr (P == PhysicsType::Heat || P == PhysicsType::Diffusion) {
            apply_bc_scalar(ghost, owner, U, ncells_total, bc.type, bc.value);
        }
        else if constexpr (P == PhysicsType::Euler) {
            apply_bc_euler(ghost, owner, U, ncells_total, bc.type,
                           mesh.faces.normal_x[fi], mesh.faces.normal_y[fi],
                           bc.inlet_rho, bc.inlet_u, bc.inlet_v, bc.inlet_p,
                           config.gamma);
        }
    }
}

// ============================================================================
// CPU step (OpenMP via ParallelFor)
// ============================================================================
template<PhysicsType P>
void Solver<P>::step_cpu() {
    int ncells       = mesh.get_ncells();
    int ncells_total = mesh.get_ncells_total();

    const float* U_curr = state.curr.data();
    float*       U_next = state.next.data();
    const float* vols   = mesh.volumes.data();
    const int*   f_owner    = mesh.faces.owner.data();
    const int*   f_neighbor = mesh.faces.neighbor.data();
    const float* f_area     = mesh.faces.area.data();
    const float* f_distance = mesh.faces.distance.data();
    const int*   cf         = mesh.cell_faces.data();

    if constexpr (P == PhysicsType::Heat) {
        const float* src   = source.data();
        float        kappa = config.kappa;
        float        dt_   = dt;
        ParallelFor(false, ncells, [=] FVM_HOST_DEVICE (int i) {
            compute_cell_update_heat(
                i, U_curr, U_next, vols,
                f_owner, f_neighbor, f_area, f_distance,
                cf, src, ncells, ncells_total, kappa, dt_);
        });
    }
    else if constexpr (P == PhysicsType::Diffusion) {
        const float* src = source.data();
        float D   = config.kappa;
        float dt_ = dt;
        ParallelFor(false, ncells, [=] FVM_HOST_DEVICE (int i) {
            compute_cell_update_diffusion(
                i, U_curr, U_next, vols,
                f_owner, f_neighbor, f_area, f_distance,
                cf, src, ncells, ncells_total, D, dt_);
        });
    }
    else if constexpr (P == PhysicsType::Euler) {
        const float* fnx = mesh.faces.normal_x.data();
        const float* fny = mesh.faces.normal_y.data();
        const int*   fso = mesh.face_stencil_owner.data();
        const int*   fsn = mesh.face_stencil_neighbor.data();
        float gamma_   = config.gamma;
        float gravity_ = config.gravity;
        float dt_      = dt;
        int   muscl_   = config.muscl ? 1 : 0;
        int   hllc_    = config.hllc ? 1 : 0;
        ParallelFor(false, ncells, [=] FVM_HOST_DEVICE (int i) {
            compute_cell_update_euler(
                i, U_curr, U_next, vols,
                f_owner, f_neighbor, f_area, f_distance,
                fnx, fny, cf, fso, fsn,
                ncells, ncells_total, gamma_, dt_, gravity_, muscl_, hllc_);
        });
    }

    state.swap_buffers();
}

// ============================================================================
// SSP-RK2 CPU step (2nd-order time integration, for use with MUSCL)
//
// Stage 1: U* = U^n + dt * L(U^n)
// Stage 2: U^{n+1} = 0.5*U^n + 0.5*(U* + dt*L(U*))
// ============================================================================
template<PhysicsType P>
void Solver<P>::step_rk2_cpu() {
    if constexpr (P != PhysicsType::Euler) {
        step_cpu();  // RK2 only needed for Euler
        return;
    }

    int ncells_total = mesh.get_ncells_total();
    int n_total_vars = NVAR * ncells_total;

    // Save U^n in rk_aux
    std::copy(state.curr.begin(), state.curr.end(), state.rk_aux.begin());

    // --- Stage 1: curr → next (forward Euler, next = U*) ---
    step_cpu();
    // After step_cpu: curr = U* (buffers were swapped), next = U^n (old curr)
    // Apply BCs to U* (now in state.curr)
    apply_bcs_cpu();

    // --- Stage 2: curr → next (forward Euler with U*, next = U**) ---
    step_cpu();
    // After: curr = U** (= U* + dt*L(U*)), next = U*

    // --- Blend: curr = 0.5*rk_aux + 0.5*curr ---
    for (int c = 0; c < n_total_vars; ++c)
        state.curr[c] = 0.5f * state.rk_aux[c] + 0.5f * state.curr[c];
}

// ============================================================================
// MPI halo exchange
// ============================================================================
template<PhysicsType P>
void Solver<P>::do_halo_exchange() {
    if (!decomp || mpi_size <= 1) return;
    Timer::Scope t("halo_exchange");

    int nx          = mesh.get_nx();
    int total_ny    = mesh.get_ny();
    int real_ny     = mesh.get_real_ny();
    int ncells_total = mesh.get_ncells_total();

    if (use_gpu) {
        gpu_state.download_halo_rows(state.curr.data(), nx, real_ny, NVAR);
        decomp->exchange_halos(state.curr.data(), nx, total_ny, NVAR, ncells_total);
        gpu_state.upload_halo_rows(state.curr.data(), nx, total_ny, NVAR);
    } else {
        decomp->exchange_halos(state.curr.data(), nx, total_ny, NVAR, ncells_total);
    }
}

// ============================================================================
// VTK output (gather across MPI ranks)
// ============================================================================
template<PhysicsType P>
void Solver<P>::gather_and_save_vtk(int step_index) {
    int nx           = mesh.get_nx();
    int ncells_total = mesh.get_ncells_total();

    if (mpi_size <= 1) {
        VTKWriter::save_fields<P>(state.curr.data(), ncells_total,
                                  &mesh, step_index, config.gamma);
        return;
    }

    int real_ny    = mesh.get_real_ny();
    int local_count = nx * real_ny;

    // Pack real rows (skip ghost row 0)
    std::vector<float> local_data(NVAR * local_count);
    for (int v = 0; v < NVAR; ++v) {
        std::memcpy(local_data.data() + v * local_count,
                    state.curr.data() + v * ncells_total + nx,
                    local_count * sizeof(float));
    }

    std::vector<int> counts(mpi_size), displs(mpi_size);
    MPI_Gather(&local_count, 1, MPI_INT,
               counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < mpi_size; ++r)
            displs[r] = displs[r-1] + counts[r-1];
        int total    = displs[mpi_size-1] + counts[mpi_size-1];
        int global_ny = total / nx;

        std::vector<float> global_data(NVAR * total);
        for (int v = 0; v < NVAR; ++v) {
            MPI_Gatherv(local_data.data() + v * local_count, local_count, MPI_FLOAT,
                        global_data.data() + v * total, counts.data(), displs.data(),
                        MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        float hy_g = (config.ymax - config.ymin) / float(global_ny);
        VTKWriter::save_raw_fields<P>(
            global_data.data(), total,
            nx, global_ny,
            Vec3(config.xmin, config.ymin),
            mesh.get_hx(), hy_g,
            step_index, config.gamma);
    } else {
        for (int v = 0; v < NVAR; ++v) {
            MPI_Gatherv(local_data.data() + v * local_count, local_count, MPI_FLOAT,
                        nullptr, nullptr, nullptr, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
    }
}

// ============================================================================
// Main solve loop
// ============================================================================
template<PhysicsType P>
void Solver<P>::solve() {
    if (mpi_rank == 0) {
        std::cout << ">>> SOLVER: " << PhysicsTraits<P>::name
                  << " | " << (use_gpu ? "GPU (CUDA)" : "CPU (OpenMP)")
                  << " | NVAR=" << NVAR << "\n";
    }

    dt = use_gpu ? compute_dt_gpu() : compute_dt();
    if (mpi_rank == 0)
        std::cout << "dt=" << dt << " mpi_size=" << mpi_size << "\n";

    apply_bcs_cpu();
    do_halo_exchange();

    if (use_gpu)
        gpu_state.upload(state.curr.data(), mesh.get_ncells_total());

    int saved = 0;
    gather_and_save_vtk(saved++);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 1; step <= config.steps; ++step) {
        float time = step * dt;
        update_source(time);
        if (use_gpu) {
            if constexpr (P == PhysicsType::Heat || P == PhysicsType::Diffusion)
                gpu_state.upload_source(source);
        }

        if constexpr (P == PhysicsType::Euler) {
            dt = use_gpu ? compute_dt_gpu() : compute_dt();
        }

        {
            Timer::Scope ts(use_gpu ? "step_gpu" : "step_cpu");
            if (use_gpu) {
                if (config.muscl) step_rk2_gpu();
                else              step_gpu();
            } else {
                if (config.muscl) step_rk2_cpu();
                else              step_cpu();
            }
        }

        do_halo_exchange();

        if (mesh.get_n_ghost_bc() > 0) {
            if (use_gpu) {
                gpu_state.download(state.curr.data(), mesh.get_ncells_total());
                apply_bcs_cpu();
                gpu_state.upload(state.curr.data(), mesh.get_ncells_total());
            } else {
                apply_bcs_cpu();
            }
        }

        if (step % config.save_every == 0) {
            if (use_gpu)
                gpu_state.download(state.curr.data(), mesh.get_ncells_total());

            if (mpi_rank == 0) print_diagnostics(step);
            gather_and_save_vtk(saved++);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = t_end - t_start;

    if (mpi_rank == 0) {
        std::cout << "Done! Wall time: " << dur.count() << "s"
                  << " | " << (config.steps / dur.count()) << " steps/s\n";
        Timer::get().report(mpi_rank);
    }
}

// ============================================================================
// Diagnostics helper (called from solve loop on rank 0)
// ============================================================================
template<PhysicsType P>
void Solver<P>::print_diagnostics(int step) {
    int ncells       = mesh.get_ncells();
    int ncells_total = mesh.get_ncells_total();

    if constexpr (P == PhysicsType::Heat || P == PhysicsType::Diffusion) {
        float tmin = state.curr[0], tmax = state.curr[0];
        int nans = 0;
        for (int c = 0; c < ncells; ++c) {
            float v = state.curr[c];
            if (std::isnan(v)||std::isinf(v)){++nans;continue;}
            if (v < tmin) tmin=v; if (v > tmax) tmax=v;
        }
        std::cout << "Step " << step << ": min=" << tmin << " max=" << tmax;
        if (nans) std::cout << " NaN=" << nans;
        std::cout << "\n";
    }
    else if constexpr (P == PhysicsType::Euler) {
        float rho_min=1e30f, rho_max=-1e30f, p_min=1e30f, p_max=-1e30f;
        for (int c = 0; c < ncells; ++c) {
            float rho  = state.curr[0*ncells_total+c];
            float rhou = state.curr[1*ncells_total+c];
            float rhov = state.curr[2*ncells_total+c];
            float E    = state.curr[3*ncells_total+c];
            float p    = euler_pressure(rho, rhou, rhov, E, config.gamma);
            if (rho<rho_min) rho_min=rho; if (rho>rho_max) rho_max=rho;
            if (p<p_min)     p_min=p;     if (p>p_max)     p_max=p;
        }
        std::cout << "Step " << step
                  << ": rho=[" << rho_min << "," << rho_max
                  << "] p=[" << p_min << "," << p_max << "]\n";
    }
}

// Explicit instantiations are in solver.cu (single compilation unit).
