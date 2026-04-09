#pragma once
#include "Base/PhysicsType.hpp"
#include "Mesh/BC.hpp"
#include <string>

// ============================================================================
// SimConfig — simulation parameters.
// ============================================================================
struct SourceSpec {
    std::string type = "none";  // "none" | "gaussian"
    float x = 5.0f, y = 5.0f;
    float radius = 0.5f;
    float power  = 1000.0f;
};

struct SimConfig {
    // Physics
    PhysicsType physics = PhysicsType::Heat;
    bool use_gpu        = false;

    // Domain
    float xmin = 0.0f, xmax = 10.0f;
    float ymin = 0.0f, ymax = 10.0f;

    // Grid
    int nx = 100, ny = 100;

    // Time
    int   steps      = 400;
    int   save_every = 10;
    float cfl        = 0.5f;

    // Boundary conditions [left, right, bottom, top]
    BCSpec bc[4];

    // Heat / Diffusion
    float kappa = 1.0f;

    // Euler
    float gamma   = 1.4f;
    float gravity = 0.0f;

    // Initial condition name
    std::string ic = "default";

    // Source term (Heat / Diffusion)
    SourceSpec source;
};

// ============================================================================
// InputParser
//
// Reads an "inputs" file (key = value, # comments) and/or CLI flags.
// CLI flags override file values.
//
// Supported keys (same names as CLI flags without --):
//   physics, device, nx, ny, steps, save-every, cfl, gamma, gravity,
//   kappa, ic, xmin, xmax, ymin, ymax,
//   bc-left, bc-right, bc-bottom, bc-top
//   source
// ============================================================================
namespace InputParser {
    // Read from file only
    SimConfig from_file(const std::string& filename);

    // Read from CLI args only (backwards-compatible with old main)
    SimConfig from_cli(int argc, char** argv);

    // Load file, then apply CLI overrides (recommended for problems/)
    SimConfig load(const std::string& filename, int argc, char** argv);

    void print(const SimConfig& cfg, int rank = 0);
}
