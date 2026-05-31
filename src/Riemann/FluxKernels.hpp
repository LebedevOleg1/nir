#pragma once
#include "Base/FvmMacros.hpp"
#include "Base/PhysicsType.hpp"
#include "Riemann/EulerUtils.hpp"

// ============================================================================
// FVM cell-update kernels for all physics types.
// All functions are FVM_HOST_DEVICE for CPU/GPU compilation.
//
// SoA state layout: U[var * ncells_total + cell]
// ============================================================================

// ---------- Heat Conduction ----------
// dT/dt = kappa * laplacian(T) + source
FVM_HOST_DEVICE FVM_INLINE void compute_cell_update_heat(
    int i,
    const float* FVM_RESTRICT U_curr,
    float*       FVM_RESTRICT U_next,
    const float* FVM_RESTRICT volumes,
    const int*   FVM_RESTRICT face_owner,
    const int*   FVM_RESTRICT face_neighbor,
    const float* FVM_RESTRICT face_area,
    const float* FVM_RESTRICT face_distance,
    const int*   FVM_RESTRICT cell_faces,
    const float* FVM_RESTRICT source,
    const int ncells,
    const int ncells_total,
    const float kappa,
    const float dt
) {
    if (i >= ncells) return;

    float T_c = U_curr[i];
    float flux_sum = 0.0f;

    #ifdef __CUDACC__
    #pragma unroll
    #endif
    for (int k = 0; k < 4; ++k) {
        int fi = cell_faces[i * 4 + k];
        int nb = face_neighbor[fi];
        float grad = (U_curr[nb] - T_c) / face_distance[fi];
        flux_sum += kappa * face_area[fi] * grad;
    }

    U_next[i] = T_c + dt * (flux_sum / volumes[i] + source[i]);
}

// ---------- Scalar Diffusion ----------
// dC/dt = D * laplacian(C) + source
FVM_HOST_DEVICE FVM_INLINE void compute_cell_update_diffusion(
    int i,
    const float* FVM_RESTRICT U_curr,
    float*       FVM_RESTRICT U_next,
    const float* FVM_RESTRICT volumes,
    const int*   FVM_RESTRICT face_owner,
    const int*   FVM_RESTRICT face_neighbor,
    const float* FVM_RESTRICT face_area,
    const float* FVM_RESTRICT face_distance,
    const int*   FVM_RESTRICT cell_faces,
    const float* FVM_RESTRICT source,
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
        float grad = (U_curr[nb] - C_c) / face_distance[fi];
        flux_sum += D_coeff * face_area[fi] * grad;
    }

    U_next[i] = C_c + dt * (flux_sum / volumes[i] + source[i]);
}

// ---------- Euler Equations ----------
// Conservative variables: U = [rho, rho*u, rho*v, E]
// Numerical flux: HLL (do_muscl=0) or MUSCL-reconstructed HLL (do_muscl=1).
//
// MUSCL (do_muscl=1): per-primitive minmod reconstruction using a 4-cell stencil
//   [stencil_owner, owner, neighbor, stencil_neighbor] per face.
//   Gives 2nd-order spatial accuracy on smooth flows.
FVM_HOST_DEVICE FVM_INLINE void compute_cell_update_euler(
    int i,
    const float* FVM_RESTRICT U_curr,
    float*       FVM_RESTRICT U_next,
    const float* FVM_RESTRICT volumes,
    const int*   FVM_RESTRICT face_owner,
    const int*   FVM_RESTRICT face_neighbor,
    const float* FVM_RESTRICT face_area,
    const float* FVM_RESTRICT face_distance,
    const float* FVM_RESTRICT face_nx,
    const float* FVM_RESTRICT face_ny_arr,
    const int*   FVM_RESTRICT cell_faces,
    const int*   FVM_RESTRICT face_stencil_o,
    const int*   FVM_RESTRICT face_stencil_n,
    const int ncells,
    const int ncells_total,
    const float gamma,
    const float dt,
    const float gravity = 0.0f,
    const int do_muscl = 0,
    const int use_hllc = 1
) {
    if (i >= ncells) return;

    float rho_c  = U_curr[0 * ncells_total + i];
    float rhou_c = U_curr[1 * ncells_total + i];
    float rhov_c = U_curr[2 * ncells_total + i];
    float E_c    = U_curr[3 * ncells_total + i];

    // Owner primitives
    float uc  = rhou_c / rho_c;
    float vc  = rhov_c / rho_c;
    float pc  = euler_pressure(rho_c, rhou_c, rhov_c, E_c, gamma);

    float flux_rho = 0.0f, flux_rhou = 0.0f;
    float flux_rhov = 0.0f, flux_E = 0.0f;

    #ifdef __CUDACC__
    #pragma unroll
    #endif
    for (int k = 0; k < 4; ++k) {
        int fi = cell_faces[i * 4 + k];
        int nb = face_neighbor[fi];

        float rho_nb  = U_curr[0 * ncells_total + nb];
        float rhou_nb = U_curr[1 * ncells_total + nb];
        float rhov_nb = U_curr[2 * ncells_total + nb];
        float E_nb    = U_curr[3 * ncells_total + nb];

        float nx = face_nx[fi];
        float ny = face_ny_arr[fi];

        float rhoL, rhouL, rhovL, EL;
        float rhoR, rhouR, rhovR, ER;

        if (do_muscl) {
            // Load stencil states
            int so = face_stencil_o[fi];
            int sn = face_stencil_n[fi];

            float rho_so  = U_curr[0 * ncells_total + so];
            float rhou_so = U_curr[1 * ncells_total + so];
            float rhov_so = U_curr[2 * ncells_total + so];
            float E_so    = U_curr[3 * ncells_total + so];

            float rho_sn  = U_curr[0 * ncells_total + sn];
            float rhou_sn = U_curr[1 * ncells_total + sn];
            float rhov_sn = U_curr[2 * ncells_total + sn];
            float E_sn    = U_curr[3 * ncells_total + sn];

            // Primitive variables for all 4 stencil cells
            float u_nb = rhou_nb / rho_nb, v_nb = rhov_nb / rho_nb;
            float p_nb = euler_pressure(rho_nb, rhou_nb, rhov_nb, E_nb, gamma);

            float u_so = rhou_so / rho_so, v_so = rhov_so / rho_so;
            float p_so = euler_pressure(rho_so, rhou_so, rhov_so, E_so, gamma);

            float u_sn = rhou_sn / rho_sn, v_sn = rhov_sn / rho_sn;
            float p_sn = euler_pressure(rho_sn, rhou_sn, rhov_sn, E_sn, gamma);

            // Reconstruct each primitive variable at the face
            float rhoL_p, rhoR_p, uL, uR, vL, vR, pL, pR;
            muscl_reconstruct_prim(rho_c,  rho_nb,  rho_so,  rho_sn,  rhoL_p, rhoR_p);
            muscl_reconstruct_prim(uc,     u_nb,    u_so,    u_sn,    uL,     uR);
            muscl_reconstruct_prim(vc,     v_nb,    v_so,    v_sn,    vL,     vR);
            muscl_reconstruct_prim(pc,     p_nb,    p_so,    p_sn,    pL,     pR);

            // Clamp density and pressure to stay physical
            rhoL_p = fmaxf(rhoL_p, 1e-10f);
            rhoR_p = fmaxf(rhoR_p, 1e-10f);
            pL     = fmaxf(pL,     1e-10f);
            pR     = fmaxf(pR,     1e-10f);

            // Convert reconstructed primitives back to conservative
            rhoL  = rhoL_p;
            rhouL = rhoL_p * uL;
            rhovL = rhoL_p * vL;
            EL    = pL / (gamma - 1.0f) + 0.5f * rhoL_p * (uL*uL + vL*vL);

            rhoR  = rhoR_p;
            rhouR = rhoR_p * uR;
            rhovR = rhoR_p * vR;
            ER    = pR / (gamma - 1.0f) + 0.5f * rhoR_p * (uR*uR + vR*vR);
        } else {
            rhoL = rho_c;  rhouL = rhou_c; rhovL = rhov_c; EL = E_c;
            rhoR = rho_nb; rhouR = rhou_nb; rhovR = rhov_nb; ER = E_nb;
        }

        float f_rho, f_rhou, f_rhov, f_E;
        if (use_hllc) {
            euler_hllc_flux(
                rhoL, rhouL, rhovL, EL,
                rhoR, rhouR, rhovR, ER,
                nx, ny, gamma,
                f_rho, f_rhou, f_rhov, f_E);
        } else {
            euler_hll_flux(
                rhoL, rhouL, rhovL, EL,
                rhoR, rhouR, rhovR, ER,
                nx, ny, gamma,
                f_rho, f_rhou, f_rhov, f_E);
        }

        float area  = face_area[fi];
        flux_rho  += f_rho  * area;
        flux_rhou += f_rhou * area;
        flux_rhov += f_rhov * area;
        flux_E    += f_E    * area;
    }

    float inv_vol = 1.0f / volumes[i];

    U_next[0 * ncells_total + i] = rho_c  - dt * flux_rho  * inv_vol;
    U_next[1 * ncells_total + i] = rhou_c - dt * flux_rhou * inv_vol;
    // Gravity source: d(rho*v)/dt += -rho*g
    U_next[2 * ncells_total + i] = rhov_c - dt * flux_rhov * inv_vol - dt * rho_c * gravity;
    // Energy source: d(E)/dt += -rho*g*v
    U_next[3 * ncells_total + i] = E_c    - dt * flux_E    * inv_vol - dt * rho_c * gravity * vc;
}
