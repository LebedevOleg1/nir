#pragma once
#include "Base/PhysicsType.hpp"

// ============================================================================
// Sod shock tube — Riemann problem for verification.
//
// Initial conditions (x in [0, 1]):
//   Left  (x < 0.5): rho=1.0, u=0, v=0, p=1.0
//   Right (x > 0.5): rho=0.125, u=0, v=0, p=0.1
//
// Analytical solution contains:
//   - Left rarefaction fan
//   - Contact discontinuity
//   - Right shock wave
// ============================================================================
struct SodPhysics {
    static constexpr PhysicsType type = PhysicsType::Euler;
    static constexpr float gamma      = 1.4f;
    static constexpr float rho_L = 1.0f,   u_L = 0.0f, p_L = 1.0f;
    static constexpr float rho_R = 0.125f, u_R = 0.0f, p_R = 0.1f;
    static constexpr float x_disc = 0.5f;  // discontinuity location
    static constexpr const char* ic_name = "sod";
    static constexpr const char* name    = "sod_shock";
};
