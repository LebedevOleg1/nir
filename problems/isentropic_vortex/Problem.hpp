#pragma once
#include "Physics.hpp"
#include "Exact.hpp"
#include "IO/InputParser.hpp"
#include "Mesh/Mesh.hpp"
#include "Mesh/MpiDecomp.hpp"
#include "Advance/Solver.hpp"
#include <iostream>
#include <cmath>

// ============================================================================
// Isentropic vortex — convergence test for spatial order of accuracy.
//
// Run on a series of grids (nx = ny = N), compare final density to exact
// solution at t = Lx/u_inf (one full traversal of the periodic domain).
// ============================================================================
inline void run_problem(int argc, char** argv) {
    SimConfig cfg = InputParser::load("inputs", argc, argv);
    cfg.physics = VortexPhysics::type;
    cfg.ic      = VortexPhysics::ic_name;
    // All boundaries are periodic (set in inputs, but enforce here)
    for (int i = 0; i < 4; ++i) cfg.bc[i].type = BCType::Periodic;

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

    // Verification: L2 error vs exact solution (rank 0 only, single-rank)
    if (decomp.rank == 0 && decomp.size == 1) {
        // Estimate t_final from steps and average dt (dt varies; report approximate)
        float hx = (cfg.xmax - cfg.xmin) / cfg.nx;
        // Exact t_final is stored in solve() output; here we reconstruct from steps
        // For comparison: use t = steps * cfl * h / (u_inf + c_inf)
        // c_inf = sqrt(gamma * p_inf / rho_inf), p_inf = rho_inf^gamma / gamma
        float p_inf   = powf(VortexPhysics::rho_inf, VortexPhysics::gamma) / VortexPhysics::gamma;
        float c_inf   = sqrtf(VortexPhysics::gamma * p_inf / VortexPhysics::rho_inf);
        float t_final = cfg.steps * cfg.cfl * hx / (VortexPhysics::u_inf + c_inf);

        // Access final state from solver (GPU: already downloaded in save loop)
        // The last VTK save is at t_final; re-read from solver would need friend access.
        // Instead: the error is computed inside Solver::solve() if we add a hook.
        // For now print t_final estimate for the user to use with plot scripts.
        std::cout << ">>> t_final_estimate=" << t_final
                  << " nx=" << cfg.nx << " (use scripts/plot_vortex_convergence.py)\n";
    }
}
