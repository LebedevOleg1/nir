#pragma once
#include "Physics.hpp"
#include "IO/InputParser.hpp"
#include "Mesh/Mesh.hpp"
#include "Mesh/MpiDecomp.hpp"
#include "Advance/Solver.hpp"

inline void run_problem(int argc, char** argv) {
    SimConfig cfg = InputParser::load("inputs", argc, argv);
    cfg.physics = HeatPhysics::type;
    if (cfg.source.type == "none") {
        cfg.source.type   = "gaussian";
        cfg.source.x      = HeatPhysics::src_x;
        cfg.source.y      = HeatPhysics::src_y;
        cfg.source.radius = HeatPhysics::src_radius;
        cfg.source.power  = HeatPhysics::src_power;
    }

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

    Solver<PhysicsType::Heat> solver(mesh, cfg, &decomp);
    solver.solve();
}
