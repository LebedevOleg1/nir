#include "Solver.hpp"
#include "HeatKernel.hpp"
#include <cmath>
#include <algorithm>
#include <limits>
#include <mpi.h>

Solver::Solver(Mesh& mesh_, float_t alpha_, bool use_gpu_, MpiDecomp* decomp_)
    : mesh(mesh_), alpha(alpha_), use_gpu(use_gpu_), decomp(decomp_)
{
    if (decomp) {
        mpi_rank = decomp->rank;
        mpi_size = decomp->size;
    }
    if (use_gpu) {
        gpu_mesh.upload(mesh);
    }
}

float_t Solver::compute_max_dt() const {
    float_t min_dist_sq = 1e10f;
    for (int_t fi = 0; fi < mesh.faces.count; ++fi) {
        float_t d = mesh.faces.distance[fi];
        if (d > 0 && d*d < min_dist_sq) min_dist_sq = d*d;
    }
    return 0.25f * min_dist_sq / alpha;
}

void Solver::update_dynamic_source(float_t power, float_t time) {
    Float3 source_pos(5.0f, 5.0f, 0.0f);
    float_t radius = 0.5f;

    // В MPI-режиме пропускаем ghost-строки (j=0 и j=ny-1)
    int_t nx = mesh.get_nx();
    int_t start = mesh.is_mpi_mode() ? nx : 0;
    int_t end   = mesh.is_mpi_mode() ? mesh.get_ncells() - nx : mesh.get_ncells();

    for (int_t c = start; c < end; ++c) {
        Float3 pos = mesh.centers[c];
        float_t dist_sq = (pos.x - source_pos.x)*(pos.x - source_pos.x) +
                          (pos.y - source_pos.y)*(pos.y - source_pos.y);
        if (dist_sq < radius * radius) {
            mesh.source[c] = power * (1.0f + 0.5f * std::sin(2.0f * 3.14159f * time / 5.0f));
        } else {
            mesh.source[c] = 0.0f;
        }
    }
}

void Solver::do_halo_exchange() {
    if (!decomp || mpi_size <= 1) return;

    int nx = mesh.get_nx();
    int total_ny = mesh.get_ny();
    int real_ny = mesh.get_real_ny();

    if (use_gpu) {
        // GPU → CPU (только 2 граничные строки)
        gpu_mesh.download_halo_rows(mesh, nx, real_ny);
        // MPI обмен
        decomp->exchange_halos(mesh.get_T_curr(), nx, total_ny);
        // CPU → GPU (только 2 ghost-строки)
        gpu_mesh.upload_halo_rows(mesh, nx, total_ny);
    } else {
        decomp->exchange_halos(mesh.get_T_curr(), nx, total_ny);
    }
}

void Solver::gather_and_save_vtk(int step_index) {
    if (mpi_size <= 1) {
        // Одиночный ранк — пишем напрямую (без ghost, т.к. их нет)
        VTKWriter::save(&mesh, step_index);
        return;
    }

    int nx = mesh.get_nx();
    int real_ny = mesh.get_real_ny();
    int local_count = nx * real_ny;

    // Извлекаем реальные строки (без ghost) из локального T
    std::vector<float> local_T(local_count);
    float_t* T = mesh.get_T_curr();
    // Реальные данные: строки 1..real_ny (пропускаем ghost строку 0)
    std::memcpy(local_T.data(), T + nx, local_count * sizeof(float));

    // Собираем размеры с каждого ранка
    std::vector<int> counts(mpi_size), displs(mpi_size);
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        displs[0] = 0;
        for (int r = 1; r < mpi_size; ++r)
            displs[r] = displs[r-1] + counts[r-1];
        int global_total = displs[mpi_size-1] + counts[mpi_size-1];
        int global_ny = global_total / nx;

        std::vector<float> global_T(global_total);
        MPI_Gatherv(local_T.data(), local_count, MPI_FLOAT,
                     global_T.data(), counts.data(), displs.data(), MPI_FLOAT,
                     0, MPI_COMM_WORLD);

        float_t hy_global = 10.0f / static_cast<float_t>(global_ny);
        VTKWriter::save_raw(global_T.data(), nx, global_ny,
                            Float3(0.0f, 0.0f, 0.0f),
                            mesh.get_hx(), hy_global, step_index);
    } else {
        MPI_Gatherv(local_T.data(), local_count, MPI_FLOAT,
                     nullptr, nullptr, nullptr, MPI_FLOAT,
                     0, MPI_COMM_WORLD);
    }
}

void Solver::step_cpu() {
    int ncells = mesh.get_ncells();

    const float* T_curr     = mesh.data.curr.T.data();
    float*       T_next     = mesh.data.next.T.data();
    const float* volumes    = mesh.volumes.data();
    const int*   f_owner    = mesh.faces.owner.data();
    const int*   f_neighbor = mesh.faces.neighbor.data();
    const float* f_area     = mesh.faces.area.data();
    const float* f_distance = mesh.faces.distance.data();
    const int*   cf         = mesh.cell_faces.data();
    const float* kf         = mesh.kappa_face.data();
    const float* src        = mesh.source.data();

    #pragma omp parallel for
    for (int i = 0; i < ncells; ++i) {
        calculate_heat_flux_core(
            i, T_curr, T_next, volumes,
            f_owner, f_neighbor, f_area, f_distance,
            cf, kf, src,
            ncells, dt
        );
    }

    mesh.data.swap_buffers();
}

void Solver::solve(int total_steps, int save_every) {
    if (use_gpu) {
        if (mpi_rank == 0)
            std::cout << ">>> SOLVER: GPU (CUDA + thrust)\n";
    } else {
        if (mpi_rank == 0)
            std::cout << ">>> SOLVER: CPU (OpenMP)\n";
    }

    dt = 0.5f * compute_max_dt();

    if (mpi_rank == 0)
        std::cout << "dt=" << dt << " mpi_size=" << mpi_size << "\n";

    // Начальный halo exchange (заполнить ghost-строки перед первым шагом)
    do_halo_exchange();

    int saved = 0;
    gather_and_save_vtk(saved++);

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 1; step <= total_steps; ++step) {
        float_t time = step * dt;

        update_dynamic_source(1000.0f, time);

        if (use_gpu) {
            gpu_mesh.upload_source(mesh.source);
            step_gpu();
        } else {
            step_cpu();
        }

        // Halo exchange после каждого шага
        do_halo_exchange();

        if (step % save_every == 0) {
            if (use_gpu) gpu_mesh.download_T(mesh);

            // Диагностика
            if (mpi_rank == 0) {
                float_t* T = mesh.get_T_curr();
                int ncells = mesh.get_ncells();
                float_t T_min = T[0], T_max = T[0];
                int nan_count = 0;
                for (int c = 0; c < ncells; ++c) {
                    if (std::isnan(T[c]) || std::isinf(T[c])) { ++nan_count; continue; }
                    if (T[c] < T_min) T_min = T[c];
                    if (T[c] > T_max) T_max = T[c];
                }
                std::cout << "Step " << step << ": T_min=" << T_min
                          << " T_max=" << T_max;
                if (nan_count > 0) std::cout << " NaN/Inf=" << nan_count;
                std::cout << "\n";
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
