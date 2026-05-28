#pragma once
#include "Base/PhysicsType.hpp"

// ============================================================================
// Kelvin-Helmholtz instability — McNally et al. (2012) benchmark.
//
// Reference: McNally, Lyra & Tassoul (2012),
//   "A Well-Posed Kelvin-Helmholtz Instability Test and Comparison",
//   ApJS 201, 18.  DOI: 10.1088/0067-0049/201/2/18
//
// Smooth density and velocity profiles with tanh transitions.
// Domain [0,1]x[0,1], periodic. gamma=5/3.
//
// Inner strip (0.25 < y < 0.75):  rho=rho2=2, u=+0.5
// Outer region:                   rho=rho1=1, u=-0.5
// Transition width sigma = 0.05/sqrt(2).
// Single-mode perturbation: v = w0*sin(2*pi*x)*(exp[...]+exp[...])
// w0 = 0.1.
//
// Compare density field at t=2 on 512x512 with McNally et al. Fig.1.
// ============================================================================
struct KHPhysics {
    static constexpr PhysicsType type = PhysicsType::Euler;
    static constexpr float gamma      = 5.0f / 3.0f;  // McNally 2012
    static constexpr float rho1       = 1.0f;          // outer density
    static constexpr float rho2       = 2.0f;          // inner density
    static constexpr float u1         = -0.5f;         // outer x-velocity
    static constexpr float u2         =  0.5f;         // inner x-velocity
    static constexpr float p0         = 2.5f;          // uniform background pressure
    static constexpr float sigma      = 0.05f / 1.41421356f;  // 0.05/sqrt(2)
    static constexpr float w0         = 0.1f;          // perturbation amplitude
    static constexpr const char* ic_name = "kh";
    static constexpr const char* name    = "kelvin_helmholtz";
};
