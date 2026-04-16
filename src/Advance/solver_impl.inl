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
            // Kelvin-Helmholtz: smooth tanh interface, density/velocity jump.
            // Uses global coordinates for MPI correctness.
            float pi    = 3.14159265f;
            float p0    = 2.5f;
            float delta = 0.05f * Ly;
            float y1    = y_off + 0.25f * Ly;
            float y2    = y_off + 0.75f * Ly;

            for (int i = 0; i < ncells; ++i) {
                float x = mesh.centers[i].x;
                float y = mesh.centers[i].y;

                float s1   = 0.5f * (1.0f + tanhf((y - y1) / delta));
                float s2   = 0.5f * (1.0f + tanhf((y - y2) / delta));
                float frac = s1 * (1.0f - s2);

                float rho = 1.0f + frac;
                float u   = -0.5f + frac;

                float env1 = expf(-(y-y1)*(y-y1) / (2.0f*delta*delta));
                float env2 = expf(-(y-y2)*(y-y2) / (2.0f*delta*delta));
                float v    = 0.1f * sinf(2.0f*pi*x/Lx)
                           + 0.1f * sinf(4.0f*pi*x/Lx)
                           + 0.05f* sinf(6.0f*pi*x/Lx);
                v *= (env1 + env2);

                float E = p0/(gamma-1.0f) + 0.5f*rho*(u*u+v*v);
                state.curr[0*ncells_total+i] = rho;
                state.curr[1*ncells_total+i] = rho*u;
                state.curr[2*ncells_total+i] = rho*v;
                state.curr[3*ncells_total+i] = E;
            }
        }
        else if (config.ic == "rt") {
            // Rayleigh-Taylor: heavy fluid on top, light below, gravity drives it.
            float pi    = 3.14159265f;
            float g     = (config.gravity > 0) ? config.gravity : 1.0f;
            float y_mid = y_off + 0.5f * Ly;
            float delta = 0.05f * Ly;

            for (int i = 0; i < ncells; ++i) {
                float x   = mesh.centers[i].x;
                float y   = mesh.centers[i].y;
                float rho = 1.5f + 0.5f * tanhf((y - y_mid) / delta);
                float env = expf(-(y-y_mid)*(y-y_mid) / (2.0f*delta*delta));
                float v   = env * 0.01f * (sinf(2.0f*pi*x/Lx) +
                                           sinf(4.0f*pi*x/Lx) +
                                           0.5f*sinf(6.0f*pi*x/Lx));
                float p   = 10.0f + rho * g * (y_off + Ly - y);
                float E   = p/(gamma-1.0f) + 0.5f*rho*v*v;
                state.curr[0*ncells_total+i] = rho;
                state.curr[1*ncells_total+i] = 0.0f;
                state.curr[2*ncells_total+i] = rho*v;
                state.curr[3*ncells_total+i] = E;
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
        float gamma_   = config.gamma;
        float gravity_ = config.gravity;
        float dt_      = dt;
        ParallelFor(false, ncells, [=] FVM_HOST_DEVICE (int i) {
            compute_cell_update_euler(
                i, U_curr, U_next, vols,
                f_owner, f_neighbor, f_area, f_distance,
                fnx, fny, cf,
                ncells, ncells_total, gamma_, dt_, gravity_);
        });
    }

    state.swap_buffers();
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
            if (use_gpu) step_gpu();
            else         step_cpu();
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
