#pragma once
#include "Base/FvmMacros.hpp"
#include "Base/PhysicsType.hpp"
#include "Mesh/BC.hpp"

// ============================================================================
// Boundary condition kernels: fill ghost cell values.
//
// Ghost cells sit outside the real domain; their values are computed from
// adjacent interior cells to enforce the desired BC at the face midpoint.
// ============================================================================

// --- Scalar BCs (Heat / Diffusion, NVAR=1) ---
FVM_HOST_DEVICE FVM_INLINE void apply_bc_scalar(
    int ghost_idx, int interior_idx,
    float* FVM_RESTRICT U,
    int ncells_total,
    BCType bc_type, float bc_value
) {
    float U_int = U[interior_idx];

    switch (bc_type) {
        case BCType::Dirichlet:
            // Interpolate: (U_int + U_ghost)/2 = bc_value → ghost = 2*bc - interior
            U[ghost_idx] = 2.0f * bc_value - U_int;
            break;
        case BCType::Neumann:
        case BCType::Wall:
            // Zero flux: ghost mirrors interior
            U[ghost_idx] = U_int;
            break;
        default:
            U[ghost_idx] = U_int;
            break;
    }
}

// --- Euler BCs (NVAR=4) ---
FVM_HOST_DEVICE FVM_INLINE void apply_bc_euler(
    int ghost_idx, int interior_idx,
    float* FVM_RESTRICT U,
    int ncells_total,
    BCType bc_type,
    float nx, float ny,
    float inlet_rho, float inlet_u, float inlet_v, float inlet_p,
    float gamma
) {
    float rho_int  = U[0 * ncells_total + interior_idx];
    float rhou_int = U[1 * ncells_total + interior_idx];
    float rhov_int = U[2 * ncells_total + interior_idx];
    float E_int    = U[3 * ncells_total + interior_idx];

    float u_int = rhou_int / rho_int;
    float v_int = rhov_int / rho_int;

    switch (bc_type) {
        case BCType::Wall: {
            // Reflective: reflect normal velocity, keep tangential
            float vn = u_int * nx + v_int * ny;
            float u_ghost = u_int - 2.0f * vn * nx;
            float v_ghost = v_int - 2.0f * vn * ny;
            U[0 * ncells_total + ghost_idx] = rho_int;
            U[1 * ncells_total + ghost_idx] = rho_int * u_ghost;
            U[2 * ncells_total + ghost_idx] = rho_int * v_ghost;
            U[3 * ncells_total + ghost_idx] = E_int;
            break;
        }
        case BCType::Inlet: {
            float E_in = inlet_p / (gamma - 1.0f) +
                         0.5f * inlet_rho * (inlet_u*inlet_u + inlet_v*inlet_v);
            U[0 * ncells_total + ghost_idx] = inlet_rho;
            U[1 * ncells_total + ghost_idx] = inlet_rho * inlet_u;
            U[2 * ncells_total + ghost_idx] = inlet_rho * inlet_v;
            U[3 * ncells_total + ghost_idx] = E_in;
            break;
        }
        case BCType::Outlet:
        default: {
            // Zero-order extrapolation (outflow / Neumann)
            U[0 * ncells_total + ghost_idx] = rho_int;
            U[1 * ncells_total + ghost_idx] = rhou_int;
            U[2 * ncells_total + ghost_idx] = rhov_int;
            U[3 * ncells_total + ghost_idx] = E_int;
            break;
        }
    }
}
