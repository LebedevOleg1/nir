#pragma once
#include "KernelCommon.hpp"
#include <cmath>

// ============================================================================
// Euler equation utilities (all HD for CPU/GPU).
//
// Conservative variables: [rho, rho*u, rho*v, E]
// Primitive variables: [rho, u, v, p]
// Equation of state: p = (gamma - 1) * (E - 0.5 * rho * (u^2 + v^2))
// ============================================================================

HD FORCE_INLINE float euler_pressure(float rho, float rhou, float rhov, float E, float gamma) {
    float u = rhou / rho;
    float v = rhov / rho;
    return (gamma - 1.0f) * (E - 0.5f * rho * (u*u + v*v));
}

HD FORCE_INLINE float euler_sound_speed(float rho, float p, float gamma) {
    float c2 = gamma * p / rho;
    return c2 > 0.0f ? sqrtf(c2) : 0.0f;
}

// Rusanov (Local Lax-Friedrichs) flux.
// Returns numerical flux dotted with face normal (nx, ny).
// Flux = 0.5 * (F_L + F_R) - 0.5 * S_max * (U_R - U_L)
// where S_max = max(|vn_L| + c_L, |vn_R| + c_R)
HD FORCE_INLINE void euler_rusanov_flux(
    float rho_L, float rhou_L, float rhov_L, float E_L,
    float rho_R, float rhou_R, float rhov_R, float E_R,
    float nx, float ny, float gamma,
    float& f_rho, float& f_rhou, float& f_rhov, float& f_E
) {
    // Left primitive
    float uL = rhou_L / rho_L;
    float vL = rhov_L / rho_L;
    float pL = (gamma - 1.0f) * (E_L - 0.5f * rho_L * (uL*uL + vL*vL));
    float cL = euler_sound_speed(rho_L, pL, gamma);
    float vnL = uL * nx + vL * ny;

    // Right primitive
    float uR = rhou_R / rho_R;
    float vR = rhov_R / rho_R;
    float pR = (gamma - 1.0f) * (E_R - 0.5f * rho_R * (uR*uR + vR*vR));
    float cR = euler_sound_speed(rho_R, pR, gamma);
    float vnR = uR * nx + vR * ny;

    // Maximum wave speed
    float absVnL = vnL > 0 ? vnL : -vnL;
    float absVnR = vnR > 0 ? vnR : -vnR;
    float sL = absVnL + cL;
    float sR = absVnR + cR;
    float S_max = sL > sR ? sL : sR;

    // Physical fluxes F(U) dot n
    // F_rho  = rho * vn
    // F_rhou = rho*u * vn + p*nx
    // F_rhov = rho*v * vn + p*ny
    // F_E    = (E + p) * vn
    float FL_rho  = rho_L * vnL;
    float FL_rhou = rhou_L * vnL + pL * nx;
    float FL_rhov = rhov_L * vnL + pL * ny;
    float FL_E    = (E_L + pL) * vnL;

    float FR_rho  = rho_R * vnR;
    float FR_rhou = rhou_R * vnR + pR * nx;
    float FR_rhov = rhov_R * vnR + pR * ny;
    float FR_E    = (E_R + pR) * vnR;

    // Rusanov flux
    f_rho  = 0.5f * (FL_rho  + FR_rho)  - 0.5f * S_max * (rho_R  - rho_L);
    f_rhou = 0.5f * (FL_rhou + FR_rhou) - 0.5f * S_max * (rhou_R - rhou_L);
    f_rhov = 0.5f * (FL_rhov + FR_rhov) - 0.5f * S_max * (rhov_R - rhov_L);
    f_E    = 0.5f * (FL_E    + FR_E)    - 0.5f * S_max * (E_R    - E_L);
}

// Maximum wave speed in a cell (for CFL condition)
HD FORCE_INLINE float euler_max_wavespeed(float rho, float rhou, float rhov, float E, float gamma) {
    float u = rhou / rho;
    float v = rhov / rho;
    float p = (gamma - 1.0f) * (E - 0.5f * rho * (u*u + v*v));
    float c = euler_sound_speed(rho, p, gamma);
    float speed = sqrtf(u*u + v*v) + c;
    return speed;
}
