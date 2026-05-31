#pragma once
#include "Base/PhysicsType.hpp"
#include <cmath>

// ============================================================================
// Isentropic vortex — smooth exact solution for Euler equations.
//
// Standard benchmark from Yee, Sandham & Djomehri (1999).
// The isentropic vortex is an EXACT STATIONARY solution of the Euler
// equations. We keep the background flow at rest (u_inf=0) so the exact
// solution equals the initial condition at all times; the L2 error then
// measures purely the scheme's numerical dissipation and converges as h^p.
//
// Domain:  [-5, 5] x [-5, 5], periodic in both directions.
// Background: rho=1, u=0, v=0, T=1, gamma=1.4.
// Vortex centered at (x0, y0) with strength epsilon.
// ============================================================================
struct VortexPhysics {
    static constexpr PhysicsType type = PhysicsType::Euler;
    static constexpr float gamma      = 1.4f;
    static constexpr float u_inf      = 0.0f;   // stationary vortex (convergence test)
    static constexpr float v_inf      = 0.0f;
    static constexpr float T_inf      = 1.0f;   // background temperature (p/rho)
    static constexpr float rho_inf    = 1.0f;
    static constexpr float epsilon    = 5.0f;   // vortex strength
    static constexpr float x0         = 0.0f;   // initial vortex center
    static constexpr float y0         = 0.0f;
    static constexpr float Lx         = 10.0f;  // domain size
    static constexpr const char* ic_name = "vortex";
    static constexpr const char* name    = "isentropic_vortex";
};
