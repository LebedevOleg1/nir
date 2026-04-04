#pragma once
#include "KernelCommon.hpp"
#include "PhysicsType.hpp"
#include "EulerUtils.hpp"

// ============================================================================
// FVM cell update kernels for all physics types.
// All functions are HD (host + device) for CPU/GPU compilation.
//
// SoA state layout: U[var * ncells_total + cell]
// ============================================================================

// ---------- Heat Conduction ----------
// dT/dt = kappa * laplacian(T) + source
// Flux = kappa * area * (T_neighbor - T_cell) / distance
HD FORCE_INLINE void compute_cell_update_heat(
    int i,
    const float* RESTRICT U_curr,
    float* RESTRICT U_next,
    const float* RESTRICT volumes,
    const int*   RESTRICT face_owner,
    const int*   RESTRICT face_neighbor,
    const float* RESTRICT face_area,
    const float* RESTRICT face_distance,
    const int*   RESTRICT cell_faces,
    const float* RESTRICT source,
    const int ncells,
    const int ncells_total,
    const float kappa,
    const float dt
) {
    if (i >= ncells) return;

    float T_c = U_curr[i];  // var 0, cell i
    float flux_sum = 0.0f;

    #ifdef __CUDACC__
    #pragma unroll
    #endif
    for (int k = 0; k < 4; ++k) {
        int fi = cell_faces[i * 4 + k];
        int nb = face_neighbor[fi];
        float T_nb = U_curr[nb];
        float grad = (T_nb - T_c) / face_distance[fi];
        flux_sum += kappa * face_area[fi] * grad;
    }

    U_next[i] = T_c + dt * (flux_sum / volumes[i] + source[i]);
}

// ---------- Scalar Diffusion ----------
// dC/dt = D * laplacian(C) + source
// Identical structure to heat, different variable name in VTK
HD FORCE_INLINE void compute_cell_update_diffusion(
    int i,
    const float* RESTRICT U_curr,
    float* RESTRICT U_next,
    const float* RESTRICT volumes,
    const int*   RESTRICT face_owner,
    const int*   RESTRICT face_neighbor,
    const float* RESTRICT face_area,
    const float* RESTRICT face_distance,
    const int*   RESTRICT cell_faces,
    const float* RESTRICT source,
    const int ncells,
    const int ncells_total,
    const float D_coeff,
    const float dt
) {
    if (i >= ncells) return;

    float C_c = U_curr[i];
    float flux_sum = 0.0f;

    #ifdef __CUDACC__
    #pragma unroll
    #endif
    for (int k = 0; k < 4; ++k) {
        int fi = cell_faces[i * 4 + k];
        int nb = face_neighbor[fi];
        float C_nb = U_curr[nb];
        float grad = (C_nb - C_c) / face_distance[fi];
        flux_sum += D_coeff * face_area[fi] * grad;
    }

    U_next[i] = C_c + dt * (flux_sum / volumes[i] + source[i]);
}

// ---------- Euler Equations ----------
// Conservative variables: U = [rho, rho*u, rho*v, E]
// Flux: Rusanov (Local Lax-Friedrichs) approximate Riemann solver
HD FORCE_INLINE void compute_cell_update_euler(
    int i,
    const float* RESTRICT U_curr,
    float* RESTRICT U_next,
    const float* RESTRICT volumes,
    const int*   RESTRICT face_owner,
    const int*   RESTRICT face_neighbor,
    const float* RESTRICT face_area,
    const float* RESTRICT face_distance,
    const float* RESTRICT face_nx,
    const float* RESTRICT face_ny_arr,
    const int*   RESTRICT cell_faces,
    const int ncells,
    const int ncells_total,
    const float gamma,
    const float dt
) {
    if (i >= ncells) return;

    // Load cell state
    float rho_c  = U_curr[0 * ncells_total + i];
    float rhou_c = U_curr[1 * ncells_total + i];
    float rhov_c = U_curr[2 * ncells_total + i];
    float E_c    = U_curr[3 * ncells_total + i];

    float flux_rho  = 0.0f;
    float flux_rhou = 0.0f;
    float flux_rhov = 0.0f;
    float flux_E    = 0.0f;

    #ifdef __CUDACC__
    #pragma unroll
    #endif
    for (int k = 0; k < 4; ++k) {
        int fi = cell_faces[i * 4 + k];
        int nb = face_neighbor[fi];

        // Neighbor state
        float rho_nb  = U_curr[0 * ncells_total + nb];
        float rhou_nb = U_curr[1 * ncells_total + nb];
        float rhov_nb = U_curr[2 * ncells_total + nb];
        float E_nb    = U_curr[3 * ncells_total + nb];

        // Face normal (outward from owner)
        float nx = face_nx[fi];
        float ny = face_ny_arr[fi];

        // Rusanov flux
        float f_rho, f_rhou, f_rhov, f_E;
        euler_rusanov_flux(
            rho_c, rhou_c, rhov_c, E_c,
            rho_nb, rhou_nb, rhov_nb, E_nb,
            nx, ny, gamma,
            f_rho, f_rhou, f_rhov, f_E
        );

        float area = face_area[fi];
        flux_rho  += f_rho  * area;
        flux_rhou += f_rhou * area;
        flux_rhov += f_rhov * area;
        flux_E    += f_E    * area;
    }

    float inv_vol = 1.0f / volumes[i];
    U_next[0 * ncells_total + i] = rho_c  - dt * flux_rho  * inv_vol;
    U_next[1 * ncells_total + i] = rhou_c - dt * flux_rhou * inv_vol;
    U_next[2 * ncells_total + i] = rhov_c - dt * flux_rhov * inv_vol;
    U_next[3 * ncells_total + i] = E_c    - dt * flux_E    * inv_vol;
}
