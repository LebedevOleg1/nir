#pragma once
#include "Mesh.hpp"
#include "GpuMesh.hpp"
#include "MpiDecomp.hpp"
#include "VTKWriter.hpp"
#include "Config.hpp"
#include "StateVector.hpp"
#include "PhysicsType.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

// ============================================================================
// Solver<P> -- unified FVM solver templated on physics type.
//
// PhysicsType::Heat      -- scalar heat conduction (NVAR=1)
// PhysicsType::Diffusion -- scalar diffusion (NVAR=1)
// PhysicsType::Euler     -- 2D Euler equations (NVAR=4)
//
// CPU path: solver.cpp (OpenMP)
// GPU path: solver.cu  (CUDA)
// ============================================================================
template<PhysicsType P>
class Solver {
    static constexpr int NVAR = PhysicsTraits<P>::NVAR;

private:
    Mesh& mesh;
    GpuMesh gpu_mesh;
    GpuState<NVAR> gpu_state;
    MpiDecomp* decomp;
    const SimConfig& config;

    StateVector<NVAR> state;
    std::vector<float> source;  // source term (for heat/diffusion)

    float_t dt;
    bool use_gpu;

    int mpi_rank = 0;
    int mpi_size = 1;

    // --- CPU path (solver.cpp) ---
    void step_cpu();
    void apply_bcs_cpu();
    void update_source(float_t time);
    float_t compute_dt();

    // --- GPU path (solver.cu) ---
    void step_gpu();
    void apply_bcs_gpu();
    float_t compute_dt_gpu();

    // --- MPI ---
    void do_halo_exchange();
    void gather_and_save_vtk(int step_index);

    // --- Initial conditions ---
    void set_initial_conditions();

public:
    Solver(Mesh& mesh_, const SimConfig& config_, MpiDecomp* decomp_ = nullptr);

    void solve();
};
