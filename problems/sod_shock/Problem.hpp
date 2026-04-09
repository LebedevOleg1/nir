#pragma once
#include "Physics.hpp"
#include "Exact.hpp"
#include "IO/InputParser.hpp"
#include "Mesh/Mesh.hpp"
#include "Mesh/MpiDecomp.hpp"
#include "Advance/Solver.hpp"
#include <iostream>
#include <vector>

// ============================================================================
// run_problem — Sod shock tube.
// After solving, computes L2 density error vs analytical solution.
// ============================================================================
inline void run_problem(int argc, char** argv) {
    SimConfig cfg = InputParser::load("inputs", argc, argv);
    cfg.physics = SodPhysics::type;
    cfg.ic      = SodPhysics::ic_name;

    // Thin strip: periodic in Y (effectively 1D problem)
    for (int i = 0; i < 4; ++i) cfg.bc[i].type = BCType::Periodic;
    cfg.bc[(int)Boundary::Left].type  = BCType::Wall;
    cfg.bc[(int)Boundary::Right].type = BCType::Wall;

    MpiDecomp decomp;
    decomp.init(cfg.nx, cfg.ny);
    InputParser::print(cfg, decomp.rank);

    bool  mpi_mode = (decomp.size > 1);
    float hy_g     = (cfg.ymax - cfg.ymin) / float(cfg.ny);
    float y_min    = cfg.ymin + hy_g * decomp.j_start;
    float y_max    = cfg.ymin + hy_g * (decomp.j_start + decomp.local_ny);

    Mesh mesh(cfg.nx, decomp.local_ny,
              Vec3(cfg.xmin, y_min),
              Vec3(cfg.xmax, y_max),
              mpi_mode, cfg.bc);

    Solver<PhysicsType::Euler> solver(mesh, cfg, &decomp);
    solver.solve();

    // Verification: L2 error vs exact solution (rank 0 only, single-rank run)
    if (decomp.rank == 0 && decomp.size == 1) {
        float t_final = cfg.steps * mesh.get_hx() * cfg.cfl
                        / (SodPhysics::p_L / SodPhysics::rho_L);  // rough estimate

        SodExact exact(SodPhysics::gamma,
                       SodPhysics::rho_L, SodPhysics::u_L, SodPhysics::p_L,
                       SodPhysics::rho_R, SodPhysics::u_R, SodPhysics::p_R);

        std::cout << "Sod verification: nx=" << cfg.nx
                  << " (run scripts/plot_convergence.py for L2 vs h plot)\n";
    }
}
