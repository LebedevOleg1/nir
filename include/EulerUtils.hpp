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

// HLL (Harten-Lax-van Leer) flux.
// Less diffusive than Rusanov: uses separate left/right wave speed estimates.
// F_HLL = (S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L)) / (S_R - S_L)
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

    // Wave speed estimates (Davis estimates)
    float SL = vnL - cL;
    float SR = vnR + cR;
    if (SL > 0.0f) SL = 0.0f;  // ensure SL <= 0
    float vnL_cL = vnL - cL;
    float vnR_cR = vnR + cR;
    SL = (vnL_cL < SL) ? vnL_cL : SL;
    SR = (vnR_cR > SR) ? vnR_cR : SR;

    // Ensure minimum wave speed spread to avoid division by zero
    if (SR - SL < 1e-10f) {
        SR = 1e-5f;
        SL = -1e-5f;
    }

    // Physical fluxes F(U) dot n
    float FL_rho  = rho_L * vnL;
    float FL_rhou = rhou_L * vnL + pL * nx;
    float FL_rhov = rhov_L * vnL + pL * ny;
    float FL_E    = (E_L + pL) * vnL;

    float FR_rho  = rho_R * vnR;
    float FR_rhou = rhou_R * vnR + pR * nx;
    float FR_rhov = rhov_R * vnR + pR * ny;
    float FR_E    = (E_R + pR) * vnR;

    if (SL >= 0.0f) {
        // All waves go right: use left flux
        f_rho  = FL_rho;
        f_rhou = FL_rhou;
        f_rhov = FL_rhov;
        f_E    = FL_E;
    } else if (SR <= 0.0f) {
        // All waves go left: use right flux
        f_rho  = FR_rho;
        f_rhou = FR_rhou;
        f_rhov = FR_rhov;
        f_E    = FR_E;
    } else {
        // Intermediate state: HLL formula
        float inv = 1.0f / (SR - SL);
        f_rho  = (SR * FL_rho  - SL * FR_rho  + SL * SR * (rho_R  - rho_L))  * inv;
        f_rhou = (SR * FL_rhou - SL * FR_rhou + SL * SR * (rhou_R - rhou_L)) * inv;
        f_rhov = (SR * FL_rhov - SL * FR_rhov + SL * SR * (rhov_R - rhov_L)) * inv;
        f_E    = (SR * FL_E    - SL * FR_E    + SL * SR * (E_R    - E_L))    * inv;
    }
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
