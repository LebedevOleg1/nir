#pragma once
#include "Mesh.hpp"
#include "VTKWriter.hpp"
#include "CudaSolver.hpp"
#include "CpuSolver.hpp"
#include <iostream>
#include <chrono>

class Solver {
private:
    float_t alpha;
    float_t dt;
    
    CudaSolver cuda_solver;
    CpuSolver cpu_solver;
    
    bool use_gpu;

    void update_dynamic_source(Mesh &mesh, float_t power, float_t time) {
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
    
public:
    Solver(float_t alpha_, bool use_gpu_) : alpha(alpha_), use_gpu(use_gpu_) {}
    
    float_t compute_max_dt(const Mesh& mesh) {
        float_t min_dist_sq = 1e10f;
        for (const auto& face : mesh.faces) {
            float_t d = face.distance;
            if (d > 0 && d*d < min_dist_sq) min_dist_sq = d*d;
        }
        return 0.25f * min_dist_sq / alpha; 
    }
    
    void solve(Mesh& mesh, int total_steps, int save_every) {
        if (use_gpu) {
            if (!cuda_solver.init_buffers(&mesh)) {
                std::cerr << "CUDA Init failed!\n"; return;
            }
            std::cout << ">>> SOLVER: GPU (CUDA RTX)\n";
        } else {
            cpu_solver.init(mesh);
            std::cout << ">>> SOLVER: CPU (Single Core / OpenMP)\n";
        }

        dt = 0.5f * compute_max_dt(mesh);
        
        int saved = 0;
        VTKWriter::save(&mesh, saved++);
        
        auto t_start = std::chrono::high_resolution_clock::now();
        
        for (int step = 1; step <= total_steps; ++step) {
            float_t time = step * dt;
            
            update_dynamic_source(mesh, 1000.0f, time);
            
            if (use_gpu) {
                cuda_solver.update_source(mesh.source);
                cuda_solver.step(alpha, dt);
            } else {
                cpu_solver.step(mesh, dt);
            }
            
            if (step % save_every == 0) {
                if (use_gpu) cuda_solver.read_current_to_host(&mesh);
                
                VTKWriter::save(&mesh, saved++);
                if (step % (save_every * 10) == 0) 
                     std::cout << "Step " << step << "/" << total_steps << "\n";
            }
        }
        
        if (use_gpu) cuda_solver.read_current_to_host(&mesh);
        
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = t_end - t_start;
        
        std::cout << "Done! Time: " << dur.count() << "s | Speed: " 
                  << (total_steps / dur.count()) << " steps/s\n";
        
        VTKWriter::writePVD(saved);
    }
};