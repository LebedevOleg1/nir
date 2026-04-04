#include "Types.hpp"
#include "Mesh.hpp"
#include "Solver.hpp"
#include "MpiDecomp.hpp"
#include "Config.hpp"
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Parse CLI
    SimConfig config = parse_cli(argc, argv);

    MpiDecomp decomp;
    decomp.init(config.nx, config.ny);

    print_config(config, decomp.rank);

    // GPU setup
    if (config.use_gpu) {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = decomp.rank % device_count;
            cudaSetDevice(device_id);
            if (decomp.rank == 0)
                std::cout << "CUDA devices: " << device_count << "\n";
        }
    }

    // Local mesh for this rank
    float_t y_total = 10.0f;
    float_t hy_global = y_total / static_cast<float_t>(config.ny);
    float_t y_min = hy_global * decomp.j_start;
    float_t y_max = hy_global * (decomp.j_start + decomp.local_ny);

    bool mpi_mode = (decomp.size > 1);

    Mesh mesh(config.nx, decomp.local_ny,
              Float3(0.0f, y_min, 0.0f),
              Float3(10.0f, y_max, 0.0f),
              mpi_mode, config.bc);

    // Template dispatch on physics type
    switch (config.physics) {
        case PhysicsType::Heat: {
            Solver<PhysicsType::Heat> solver(mesh, config, &decomp);
            solver.solve();
            break;
        }
        case PhysicsType::Diffusion: {
            Solver<PhysicsType::Diffusion> solver(mesh, config, &decomp);
            solver.solve();
            break;
        }
        case PhysicsType::Euler: {
            Solver<PhysicsType::Euler> solver(mesh, config, &decomp);
            solver.solve();
            break;
        }
    }

    decomp.finalize();
    return 0;
}
