#pragma once
#include "Base/PhysicsType.hpp"

// ============================================================================
// Rayleigh-Taylor instability — Liska & Wendroff (2003) benchmark.
//
// Reference: Liska & Wendroff (2003),
//   "Comparison of Several Difference Schemes on 1D and 2D Test Problems",
//   SIAM J. Sci. Comput. 25(3), 995-1017.  DOI: 10.1137/S1064827502402120
//
// Setup: heavy fluid (rho=2) above light fluid (rho=1), gravity g=0.1 downward.
// Domain [0, 1/6] x [0, 1].  gamma=5/3.
// Initial pressure: hydrostatic equilibrium.
// Perturbation: v = 0.01*(1 + cos(8*pi*x))*(1 + cos(3*pi*y))/4 near interface.
//
// Compare density finger pattern at t=8.9 with Liska & Wendroff (2003) Fig.4.4.
// ============================================================================
struct RTPhysics {
    static constexpr PhysicsType type    = PhysicsType::Euler;
    static constexpr float gamma         = 5.0f / 3.0f;  // Liska 2003
    static constexpr float gravity       = 0.1f;          // |g|, pointing in -y direction
    static constexpr float rho_heavy     = 2.0f;
    static constexpr float rho_light     = 1.0f;
    static constexpr float p_top         = 2.5f;          // pressure at y=1 (top boundary)
    static constexpr const char* ic_name = "rt";
    static constexpr const char* name    = "rayleigh_taylor";
};
