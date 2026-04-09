#pragma once
#include "Base/PhysicsType.hpp"

// ============================================================================
// Heat equation with Gaussian source — used for convergence verification.
//
// Equation: dT/dt = kappa * laplacian(T) + Q(x, y, t)
// Q is a Gaussian bell centered at (sx, sy) with radius r.
//
// Purpose: run at several resolutions (nx = 50, 100, 200, 400),
// compare against a reference solution to measure convergence order.
// Expected order: 2nd order in space (central differences), 1st in time.
// ============================================================================
struct HeatPhysics {
    static constexpr PhysicsType type  = PhysicsType::Heat;
    static constexpr float kappa       = 1.0f;
    static constexpr float src_x       = 5.0f;
    static constexpr float src_y       = 5.0f;
    static constexpr float src_radius  = 0.5f;
    static constexpr float src_power   = 1000.0f;
    static constexpr const char* name  = "heat_gaussian";
};
