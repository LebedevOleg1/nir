#pragma once
#include "Base/PhysicsType.hpp"

// ============================================================================
// Rayleigh-Taylor instability — heavy fluid above light fluid.
//
// rho(y) = 1.5 + 0.5*tanh((y - y_mid)/delta)
//   ~1 below y_mid, ~2 above y_mid
//
// Gravity g points in -y direction.
// Multi-mode perturbation seeds multiple wavelengths simultaneously.
// ============================================================================
struct RTPhysics {
    static constexpr PhysicsType type = PhysicsType::Euler;
    static constexpr float gamma      = 1.4f;
    static constexpr float gravity    = 1.0f;
    static constexpr const char* ic_name = "rt";
    static constexpr const char* name    = "rayleigh_taylor";
};
