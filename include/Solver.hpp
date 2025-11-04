#pragma once
#include "Mesh.hpp"
#include "VTKWriter.hpp"
#include "OpenCLSolver.hpp"
#include <iostream>
#include <chrono>

class Solver {
private:
    float_t alpha;
    float_t dt;
    OpenCLSolver ocl_solver;
    bool use_opencl;

    void update_dynamic_source(Mesh &mesh, float_t power, float_t time) {
        Float3 source_pos(5.0f, 5.0f, 0.0f);
        float_t radius = 0.5f;
        
        for (int_t c = 0; c < mesh.get_ncells(); ++c) {
            Float3 pos = mesh.centers[c];
            float_t dist_sq = (pos.x - source_pos.x)*(pos.x - source_pos.x) + 
                             (pos.y - source_pos.y)*(pos.y - source_pos.y);
            
            if (dist_sq < radius * radius) {
                // Пульсирующий источник (всегда положительный)
                mesh.source[c] = power * (1.0f + 0.5f * std::sin(2.0f * M_PI * time / 5.0f));
            } else {
                mesh.source[c] = 0.0f;
            }
        }
    }
    
public:
    Solver(float_t alpha_) : alpha(alpha_), use_opencl(false) {
        use_opencl = ocl_solver.is_available();
        if (!use_opencl) {
            std::cerr << "ERROR: OpenCL is not available! Cannot proceed.\n";
            std::cerr << "Please check your OpenCL installation.\n";
        }
    }
    
    // Вычисление максимального шага по времени (CFL)
    float_t compute_max_dt(const Mesh& mesh) {
        float_t min_dist_sq = 1e10f;
        for (const auto& face : mesh.faces) {
            float_t d = face.distance;
            if (d > 0 && d*d < min_dist_sq) min_dist_sq = d*d;
        }
        return 0.25f * min_dist_sq / alpha; // CFL для диффузии
    }
    
    // Основной цикл решения с OpenCL
    void solve(Mesh& mesh, int total_steps, int save_every) {
        if (!use_opencl) {
            std::cerr << "Cannot solve: OpenCL not available!\n";
            return;
        }
        
        // Инициализация буферов на GPU
        if (!ocl_solver.init_buffers(&mesh)) {
            std::cerr << "Failed to initialize OpenCL buffers!\n";
            return;
        }
        
        // Вычисляем dt
        dt = 0.5f * compute_max_dt(mesh);
        std::cout << "Solver using OpenCL\n";
        std::cout << "Parameters: alpha=" << alpha << ", dt=" << dt << "\n";
        std::cout << "Mesh: " << mesh.get_ncells() << " cells\n";
        
        int saved = 0;
        VTKWriter::save(&mesh, saved++);
        
        auto t_start = std::chrono::high_resolution_clock::now();
        
        for (int step = 1; step <= total_steps; ++step) {
            float_t time = step * dt;
            update_dynamic_source(mesh, 1000.0f, time);
            ocl_solver.update_source(&mesh);
            
            // Выполнение шага на GPU
            if (!ocl_solver.step_device(mesh.get_ncells(), alpha, dt)) {
                std::cerr << "OpenCL step failed at iteration " << step << "\n";
                break;
            }
            
            // Сохранение результатов
            if (step % save_every == 0) {
                // Копируем данные с GPU на CPU для сохранения
                if (!ocl_solver.read_current_to_host(&mesh)) {
                    std::cerr << "Failed to read data from GPU at step " << step << "\n";
                    break;
                }
                
                VTKWriter::save(&mesh, saved++);
                
                // Прогресс
                if (step % (save_every * 10) == 0) {
                    std::cout << "Progress: step " << step << "/" << total_steps 
                             << " (" << (100.0f * step / total_steps) << "%)\n";
                }
            }
        }
        
        // Финальное копирование с GPU
        ocl_solver.read_current_to_host(&mesh);
        
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = t_end - t_start;
        
        std::cout << "=====================================\n";
        std::cout << "Simulation completed!\n";
        std::cout << "Total steps: " << total_steps << "\n";
        std::cout << "Saved frames: " << saved << "\n";
        std::cout << "Compute time: " << dur.count() << " seconds\n";
        std::cout << "Performance: " << (total_steps / dur.count()) << " steps/second\n";
        std::cout << "=====================================\n";
        
        VTKWriter::writePVD(saved);
    }
};
