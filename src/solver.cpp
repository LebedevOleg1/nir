#include "Solver.hpp"
#include "HeatKernel.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

// ============================================================================
// Конструктор Solver.
//
// Если use_gpu=true, загружаем всю сетку в видеопамять через GpuMesh::upload.
// Это единственный «тяжёлый» cudaMemcpy за весь запуск — дальше данные
// живут на GPU и обратно копируются только для VTK-вывода.
// ============================================================================
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

// ============================================================================
// compute_max_dt — условие CFL для устойчивости явной схемы.
//
// Для уравнения диффузии ∂T/∂t = α∇²T на равномерной сетке:
//   dt_max = h² / (4α)  (в 2D)
// Берём 0.25 * min(distance²) / alpha с запасом (коэффициент 0.25 < 0.5).
// Если dt > dt_max, схема Эйлера «взрывается» — температура осциллирует
// и уходит в бесконечность.
// ============================================================================
float_t Solver::compute_max_dt() const {
    float_t min_dist_sq = 1e10f;
    for (int_t fi = 0; fi < mesh.faces.count; ++fi) {
        float_t d = mesh.faces.distance[fi];
        if (d > 0 && d*d < min_dist_sq) min_dist_sq = d*d;
    }
    return 0.25f * min_dist_sq / alpha;
}

// ============================================================================
// update_dynamic_source — задаёт пространственно-временной источник тепла.
//
// Гауссов «пятачок» радиуса 0.5 в центре области (5,5), модулированный
// синусоидой по времени. Моделирует, например, лазерный нагрев с
// переменной мощностью.
// ============================================================================
void Solver::update_dynamic_source(float_t power, float_t time) {
    Float3 source_pos(5.0f, 5.0f, 0.0f);
    float_t radius = 0.5f;
    for (int_t c = 0; c < mesh.get_ncells(); ++c) {
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

// ============================================================================
// step_cpu — один шаг по времени на CPU.
//
// OpenMP параллелизм: каждый поток обрабатывает свой диапазон ячеек.
// calculate_heat_flux_core — inline-функция, общая для CPU и GPU (из
// HeatKernel.hpp), только на CPU она вызывается из обычного for-цикла,
// а на GPU — из CUDA-ядра (__global__ функции).
// ============================================================================
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

// ============================================================================
// solve — главный цикл по времени.
//
// 1) Вычисляем dt из CFL
// 2) На каждом шаге: обновляем источник → делаем шаг (CPU или GPU) → сохраняем VTK
// 3) В конце выводим время и скорость (steps/s)
//
// При GPU: источник загружается в видеопамять через thrust (upload_source),
// а температура скачивается обратно (download_T) только для VTK-вывода.
// ============================================================================
void Solver::solve(int total_steps, int save_every) {
    if (use_gpu) {
        if (mpi_rank == 0)
            std::cout << ">>> SOLVER: GPU (CUDA + thrust)\n";
    } else {
        if (mpi_rank == 0)
            std::cout << ">>> SOLVER: CPU (OpenMP)\n";
    }

    dt = 0.5f * compute_max_dt();

    int saved = 0;
    if (mpi_rank == 0)
        VTKWriter::save(&mesh, saved++);

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

        // Обмен ghost-строками между MPI-ранками
        if (decomp && mpi_size > 1) {
            if (use_gpu) gpu_mesh.download_T(mesh);
            int local_ny_with_ghosts = mesh.get_ny() + 2;
            decomp->exchange_halos(mesh.get_T_curr(),
                                   mesh.get_nx(), local_ny_with_ghosts);
            if (use_gpu) gpu_mesh.upload(mesh);
        }

        if (step % save_every == 0) {
            if (use_gpu) gpu_mesh.download_T(mesh);

            // Диагностика: проверяем температуру на nan/inf/взрыв
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

            if (mpi_rank == 0)
                VTKWriter::save(&mesh, saved++);
        }
    }

    if (use_gpu) gpu_mesh.download_T(mesh);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = t_end - t_start;

    if (mpi_rank == 0) {
        std::cout << "Done! Time: " << dur.count() << "s | Speed: "
                  << (total_steps / dur.count()) << " steps/s\n";
    }

    if (mpi_rank == 0)
        VTKWriter::writePVD(saved);
}
