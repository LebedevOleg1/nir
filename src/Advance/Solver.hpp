#pragma once
#include "Mesh/Mesh.hpp"
#include "Mesh/GpuMesh.hpp"
#include "Mesh/MpiDecomp.hpp"
#include "IO/VTKWriter.hpp"
#include "IO/InputParser.hpp"
#include "Base/StateVector.hpp"
#include "Base/PhysicsType.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

// ============================================================================
// Solver<P> — unified FVM time-stepper templated on physics type.
//
//   PhysicsType::Heat      — scalar heat conduction  (NVAR=1)
//   PhysicsType::Diffusion — scalar diffusion        (NVAR=1)
//   PhysicsType::Euler     — 2D compressible Euler   (NVAR=4)
//
// CPU path: Advance/solver.cpp  (OpenMP via ParallelFor)
// GPU path: Advance/solver.cu   (CUDA via ParallelFor + FieldView)
// ============================================================================
template<PhysicsType P>
class Solver {
    static constexpr int NVAR = PhysicsTraits<P>::NVAR;

private:
    Mesh&           mesh;
    GpuMesh         gpu_mesh;
    GpuState<NVAR>  gpu_state;
    MpiDecomp*      decomp;
    const SimConfig& config;

    StateVector<NVAR> state;
    std::vector<float> source;

    float dt     = 0.0f;
    bool  use_gpu;
    int   mpi_rank = 0;
    int   mpi_size = 1;

    // CPU path (solver_impl.inl)
    void step_cpu();
    void apply_bcs_cpu();
    void update_source(float time);
    float compute_dt();

    // MPI + IO
    void do_halo_exchange();
    void gather_and_save_vtk(int step_index);

    // Initial conditions
    void set_initial_conditions();
    void print_diagnostics(int step);

public:
    Solver(Mesh& mesh_, const SimConfig& config_, MpiDecomp* decomp_ = nullptr);

    // GPU path (solver.cu) — public because CUDA extended lambdas
    // cannot be captured inside private member functions.
    void step_gpu();
    void apply_bcs_gpu();
    float compute_dt_gpu();

    void solve();
};
