#include "Types.hpp"
#include "Mesh.hpp"
#include "Solver.hpp"
#include <iostream>

// init_gaussians, init_layered_material ...


int main() {
    int_t nx = 1000, ny = 1000;
    

    std::cout << "\n1. GPU (CUDA)\n2. CPU\n> ";
    int choice;
    std::cin >> choice;
    bool use_gpu = (choice == 1);

    Mesh mesh(nx, ny, Float3(0.0f,0.0f,0.0f), Float3(10.0f,10.0f,0.0f));
    
    // init_gaussians(mesh, 100.0f, 0.8f);
    // init_layered_material(mesh);
    // add_constant_heat_source(mesh, 5.0f);
    
    Solver solver(1.0f, use_gpu);
    
    solver.solve(mesh, 400, 10);
    
    return 0;
}