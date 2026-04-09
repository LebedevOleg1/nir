#pragma once
#include "Base/PhysicsType.hpp"

// ============================================================================
// Kelvin-Helmholtz instability — physical parameters.
//
// Two layers of compressible fluid moving in opposite directions:
//   Middle strip (y in [0.25L, 0.75L]): rho=2, u=+0.5
//   Outer region:                        rho=1, u=-0.5
//
// A small sinusoidal perturbation in v seeds the instability.
// Rolls develop and merge over time.
// ============================================================================
struct KHPhysics {
    static constexpr PhysicsType type = PhysicsType::Euler;
    static constexpr float gamma      = 1.4f;
    static constexpr float p0         = 2.5f;    // background pressure
    static constexpr float delta_frac = 0.05f;   // interface width / Ly
    static constexpr const char* ic_name = "kh";
    static constexpr const char* name    = "kelvin_helmholtz";
};
