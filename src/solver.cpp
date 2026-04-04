#include "Solver.hpp"
#include "FluxKernels.hpp"
#include "BCKernel.hpp"
#include "EulerUtils.hpp"
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
    int ncells = mesh.get_ncells();
    int ncells_total = mesh.get_ncells_total();

    if constexpr (P == PhysicsType::Heat || P == PhysicsType::Diffusion) {
        // Default: T=0 everywhere (source will heat it up)
        // Dirichlet BCs will set boundary values via ghost cells
    }
    else if constexpr (P == PhysicsType::Euler) {
        int nx = mesh.get_nx();
        float gamma = config.gamma;

        if (config.ic == "sod") {
            // Sod shock tube: discontinuity at x = domain_midpoint
            // Left:  rho=1.0, u=0, v=0, p=1.0
            // Right: rho=0.125, u=0, v=0, p=0.1
            float x_mid = (mesh.get_vmin().x + mesh.get_vmax().x) * 0.5f;
            // In MPI mode, use global midpoint
            if (mesh.is_mpi_mode()) x_mid = 5.0f;

            for (int i = 0; i < ncells; ++i) {
                float x = mesh.centers[i].x;
                float rho, u, v, p;
                if (x < x_mid) {
                    rho = 1.0f; u = 0.0f; v = 0.0f; p = 1.0f;
                } else {
                    rho = 0.125f; u = 0.0f; v = 0.0f; p = 0.1f;
                }
                float E = p / (gamma - 1.0f) + 0.5f * rho * (u*u + v*v);
                state.curr[0 * ncells_total + i] = rho;
                state.curr[1 * ncells_total + i] = rho * u;
                state.curr[2 * ncells_total + i] = rho * v;
                state.curr[3 * ncells_total + i] = E;
            }
        }
        else if (config.ic == "blast") {
            // Circular blast wave: high pressure in center, low outside
            float cx = (mesh.get_vmin().x + mesh.get_vmax().x) * 0.5f;
            float cy = (mesh.get_vmin().y + mesh.get_vmax().y) * 0.5f;
            float r_blast = 0.1f * (mesh.get_vmax().x - mesh.get_vmin().x);

            for (int i = 0; i < ncells; ++i) {
                float x = mesh.centers[i].x;
                float y = mesh.centers[i].y;
                float dist = std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
                float rho = 1.0f;
                float u = 0.0f, v = 0.0f;
                float p = (dist < r_blast) ? 10.0f : 0.1f;
                float E = p / (gamma - 1.0f) + 0.5f * rho * (u*u + v*v);
                state.curr[0 * ncells_total + i] = rho;
                state.curr[1 * ncells_total + i] = rho * u;
                state.curr[2 * ncells_total + i] = rho * v;
                state.curr[3 * ncells_total + i] = E;
            }
        }
        else {
            // Default: uniform state from inlet BC
            const BCSpec& inlet = config.bc[(int)Boundary::Left];
            float rho0 = inlet.inlet_rho;
            float u0   = inlet.inlet_u;
            float v0   = inlet.inlet_v;
            float p0   = inlet.inlet_p;
            float E0 = p0 / (gamma - 1.0f) + 0.5f * rho0 * (u0*u0 + v0*v0);

            for (int i = 0; i < ncells; ++i) {
                state.curr[0 * ncells_total + i] = rho0;
                state.curr[1 * ncells_total + i] = rho0 * u0;
                state.curr[2 * ncells_total + i] = rho0 * v0;
                state.curr[3 * ncells_total + i] = E0;
            }
        }
    }
}

// ============================================================================
// Compute stable dt (CFL condition)
// ============================================================================
template<PhysicsType P>
float_t Solver<P>::compute_dt() {
    if constexpr (P == PhysicsType::Heat || P == PhysicsType::Diffusion) {
        // dt = cfl * min(dx^2, dy^2) / (2 * kappa)  [2D diffusion CFL]
        float hx = mesh.get_hx();
        float hy = mesh.get_hy();
        float h_min = (hx < hy) ? hx : hy;
        return config.cfl * 0.25f * h_min * h_min / config.kappa;
    }
    else if constexpr (P == PhysicsType::Euler) {
        // dt = cfl * min(dx, dy) / max_wavespeed
        int ncells = mesh.get_ncells();
        int ncells_total = mesh.get_ncells_total();
        float max_speed = 1e-10f;

        #pragma omp parallel for reduction(max:max_speed)
        for (int i = 0; i < ncells; ++i) {
            float rho  = state.curr[0 * ncells_total + i];
            float rhou = state.curr[1 * ncells_total + i];
            float rhov = state.curr[2 * ncells_total + i];
            float E    = state.curr[3 * ncells_total + i];
            float s = euler_max_wavespeed(rho, rhou, rhov, E, config.gamma);
            if (s > max_speed) max_speed = s;
        }

        // MPI global max
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
}

// ============================================================================
// Source term update
// ============================================================================
template<PhysicsType P>
void Solver<P>::update_source(float_t time) {
    if constexpr (P == PhysicsType::Heat || P == PhysicsType::Diffusion) {
        if (config.source.type == "none") return;
        if (config.source.type != "gaussian") return;

        int_t nx = mesh.get_nx();
        int_t start = mesh.is_mpi_mode() ? nx : 0;
        int_t end   = mesh.is_mpi_mode() ? mesh.get_ncells() - nx : mesh.get_ncells();

        float sx = config.source.x;
        float sy = config.source.y;
        float r  = config.source.radius;
        float P_src = config.source.power;

        for (int_t c = start; c < end; ++c) {
            Float3 pos = mesh.centers[c];
            float dist_sq = (pos.x - sx)*(pos.x - sx) + (pos.y - sy)*(pos.y - sy);
            if (dist_sq < r * r) {
                source[c] = P_src * (1.0f + 0.5f * std::sin(2.0f * 3.14159f * time / 5.0f));
            } else {
                source[c] = 0.0f;
            }
        }
    }
    // Euler: no source term by default
}

// ============================================================================
// Boundary condition application (CPU)
// ============================================================================
template<PhysicsType P>
void Solver<P>::apply_bcs_cpu() {
    int n_ghost = mesh.get_n_ghost_bc();
    if (n_ghost == 0) return;

    int ncells = mesh.get_ncells();
    int ncells_total = mesh.get_ncells_total();
    float* U = state.curr.data();

    // Iterate over boundary faces to find ghost cells
    int nfaces = mesh.faces.count;
    for (int fi = 0; fi < nfaces; ++fi) {
        int bid = mesh.face_boundary_id[fi];
        if (bid < 0) continue;  // interior face

        int owner = mesh.faces.owner[fi];
        int ghost = mesh.faces.neighbor[fi];
        if (ghost < ncells) continue;  // not a ghost cell

        const BCSpec& bc = mesh.get_bc((Boundary)bid);

        if constexpr (P == PhysicsType::Heat || P == PhysicsType::Diffusion) {
            apply_bc_scalar(ghost, owner, U, ncells_total, bc.type, bc.value);
        }
        else if constexpr (P == PhysicsType::Euler) {
            float nx = mesh.faces.normal_x[fi];
            float ny = mesh.faces.normal_y[fi];
            apply_bc_euler(ghost, owner, U, ncells_total, bc.type,
                          nx, ny,
                          bc.inlet_rho, bc.inlet_u, bc.inlet_v, bc.inlet_p,
                          config.gamma);
        }
    }
}

// ============================================================================
// CPU step
// ============================================================================
template<PhysicsType P>
void Solver<P>::step_cpu() {
    int ncells = mesh.get_ncells();
    int ncells_total = mesh.get_ncells_total();

    const float* U_curr     = state.curr.data();
    float*       U_next     = state.next.data();
    const float* vols       = mesh.volumes.data();
    const int*   f_owner    = mesh.faces.owner.data();
    const int*   f_neighbor = mesh.faces.neighbor.data();
    const float* f_area     = mesh.faces.area.data();
    const float* f_distance = mesh.faces.distance.data();
    const int*   cf         = mesh.cell_faces.data();

    if constexpr (P == PhysicsType::Heat) {
        const float* src = source.data();
        float kappa = config.kappa;
        #pragma omp parallel for
        for (int i = 0; i < ncells; ++i) {
            compute_cell_update_heat(
                i, U_curr, U_next, vols,
                f_owner, f_neighbor, f_area, f_distance,
                cf, src, ncells, ncells_total, kappa, dt);
        }
    }
    else if constexpr (P == PhysicsType::Diffusion) {
        const float* src = source.data();
        float D = config.kappa;
        #pragma omp parallel for
        for (int i = 0; i < ncells; ++i) {
            compute_cell_update_diffusion(
                i, U_curr, U_next, vols,
                f_owner, f_neighbor, f_area, f_distance,
                cf, src, ncells, ncells_total, D, dt);
        }
    }
    else if constexpr (P == PhysicsType::Euler) {
        const float* fnx = mesh.faces.normal_x.data();
        const float* fny = mesh.faces.normal_y.data();
        #pragma omp parallel for
        for (int i = 0; i < ncells; ++i) {
            compute_cell_update_euler(
                i, U_curr, U_next, vols,
                f_owner, f_neighbor, f_area, f_distance,
                fnx, fny, cf,
                ncells, ncells_total, config.gamma, dt);
        }
    }

    state.swap_buffers();
}

// ============================================================================
// MPI halo exchange
// ============================================================================
template<PhysicsType P>
void Solver<P>::do_halo_exchange() {
    if (!decomp || mpi_size <= 1) return;

    int nx = mesh.get_nx();
    int total_ny = mesh.get_ny();
    int real_ny = mesh.get_real_ny();
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
// VTK gather and save
// ============================================================================
template<PhysicsType P>
void Solver<P>::gather_and_save_vtk(int step_index) {
    int nx = mesh.get_nx();
    int ncells_total = mesh.get_ncells_total();

    if (mpi_size <= 1) {
        // Single rank: write directly
        // For NVAR=1: pass var 0. For Euler: pass all vars.
        VTKWriter::save_fields<P>(state.curr.data(), ncells_total,
                                  &mesh, step_index, config.gamma);
        return;
    }

    int real_ny = mesh.get_real_ny();
    int local_count = nx * real_ny;

    // For each variable, extract real rows (skip ghost row 0)
    std::vector<float> local_data(NVAR * local_count);
    for (int v = 0; v < NVAR; ++v) {
        std::memcpy(local_data.data() + v * local_count,
                    state.curr.data() + v * ncells_total + nx,
                    local_count * sizeof(float));
    }

    std::vector<int> counts(mpi_size), displs(mpi_size);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < mpi_size; ++r)
            displs[r] = displs[r-1] + counts[r-1];
        int global_total = displs[mpi_size-1] + counts[mpi_size-1];
        int global_ny = global_total / nx;

        // Gather each variable separately
        std::vector<float> global_data(NVAR * global_total);
        for (int v = 0; v < NVAR; ++v) {
            MPI_Gatherv(local_data.data() + v * local_count, local_count, MPI_FLOAT,
                        global_data.data() + v * global_total, counts.data(), displs.data(), MPI_FLOAT,
                        0, MPI_COMM_WORLD);
        }

        float_t hy_global = 10.0f / static_cast<float_t>(global_ny);
        VTKWriter::save_raw_fields<P>(global_data.data(), global_total,
                                       nx, global_ny,
                                       Float3(0.0f, 0.0f, 0.0f),
                                       mesh.get_hx(), hy_global,
                                       step_index, config.gamma);
    } else {
        for (int v = 0; v < NVAR; ++v) {
            MPI_Gatherv(local_data.data() + v * local_count, local_count, MPI_FLOAT,
                        nullptr, nullptr, nullptr, MPI_FLOAT,
                        0, MPI_COMM_WORLD);
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

    dt = compute_dt();
    if (mpi_rank == 0)
        std::cout << "dt=" << dt << " mpi_size=" << mpi_size << "\n";

    // Apply BCs and initial halo exchange
    apply_bcs_cpu();
    do_halo_exchange();

    if (use_gpu) {
        gpu_state.upload(state.curr.data(), mesh.get_ncells_total());
    }

    int saved = 0;
    gather_and_save_vtk(saved++);

    auto t_start = std::chrono::high_resolution_clock::now();

    int total_steps = config.steps;
    int save_every = config.save_every;

    for (int step = 1; step <= total_steps; ++step) {
        float_t time = step * dt;

        // Source term
        update_source(time);

        // Recompute dt for Euler (adaptive CFL)
        if constexpr (P == PhysicsType::Euler) {
            if (use_gpu) {
                dt = compute_dt_gpu();
            } else {
                dt = compute_dt();
            }
        }

        if (use_gpu) {
            if constexpr (P == PhysicsType::Heat || P == PhysicsType::Diffusion) {
                gpu_state.upload_source(source);
            }
            step_gpu();
        } else {
            step_cpu();
        }

        do_halo_exchange();

        // Apply BCs after halo exchange
        if (mesh.get_n_ghost_bc() > 0) {
            if (use_gpu) {
                // Download state, apply BCs on CPU, upload back
                // (optimization: apply_bcs_gpu for GPU path)
                gpu_state.download(state.curr.data(), mesh.get_ncells_total());
                apply_bcs_cpu();
                gpu_state.upload(state.curr.data(), mesh.get_ncells_total());
            } else {
                apply_bcs_cpu();
            }
        }

        if (step % save_every == 0) {
            if (use_gpu) {
                gpu_state.download(state.curr.data(), mesh.get_ncells_total());
            }

            // Diagnostics
            if (mpi_rank == 0) {
                int ncells = mesh.get_ncells();
                int ncells_total = mesh.get_ncells_total();
                if constexpr (P == PhysicsType::Heat || P == PhysicsType::Diffusion) {
                    float T_min = state.curr[0], T_max = state.curr[0];
                    int nan_count = 0;
                    for (int c = 0; c < ncells; ++c) {
                        float v = state.curr[c];
                        if (std::isnan(v) || std::isinf(v)) { ++nan_count; continue; }
                        if (v < T_min) T_min = v;
                        if (v > T_max) T_max = v;
                    }
                    std::cout << "Step " << step << ": min=" << T_min
                              << " max=" << T_max;
                    if (nan_count > 0) std::cout << " NaN/Inf=" << nan_count;
                    std::cout << "\n";
                }
                else if constexpr (P == PhysicsType::Euler) {
                    float rho_min = 1e30f, rho_max = -1e30f;
                    float p_min = 1e30f, p_max = -1e30f;
                    for (int c = 0; c < ncells; ++c) {
                        float rho  = state.curr[0 * ncells_total + c];
                        float rhou = state.curr[1 * ncells_total + c];
                        float rhov = state.curr[2 * ncells_total + c];
                        float E    = state.curr[3 * ncells_total + c];
                        float p = euler_pressure(rho, rhou, rhov, E, config.gamma);
                        if (rho < rho_min) rho_min = rho;
                        if (rho > rho_max) rho_max = rho;
                        if (p < p_min) p_min = p;
                        if (p > p_max) p_max = p;
                    }
                    std::cout << "Step " << step
                              << ": rho=[" << rho_min << "," << rho_max
                              << "] p=[" << p_min << "," << p_max << "]\n";
                }
            }

            gather_and_save_vtk(saved++);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = t_end - t_start;

    if (mpi_rank == 0) {
        std::cout << "Done! Time: " << dur.count() << "s | Speed: "
                  << (total_steps / dur.count()) << " steps/s\n";
    }
}

// Explicit template instantiations
template class Solver<PhysicsType::Heat>;
template class Solver<PhysicsType::Diffusion>;
template class Solver<PhysicsType::Euler>;
