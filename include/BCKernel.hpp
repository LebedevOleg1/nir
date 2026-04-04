#pragma once
#include "KernelCommon.hpp"
#include "PhysicsType.hpp"
#include "BoundaryCondition.hpp"

// ============================================================================
// Boundary condition application: fills ghost cell values.
//
// For each boundary ghost cell, we know:
//   - ghost_idx: index of the ghost cell
//   - interior_idx: index of the adjacent interior cell
//   - bc_type: type of boundary condition
//   - bc_value: prescribed value (for Dirichlet/Neumann)
//
// SoA layout: U[var * ncells_total + cell]
// ============================================================================

// --- Heat / Diffusion BCs (NVAR=1) ---
HD FORCE_INLINE void apply_bc_scalar(
    int ghost_idx, int interior_idx,
    float* RESTRICT U,
    int ncells_total,
    BCType bc_type, float bc_value
) {
    float U_int = U[ghost_idx];  // will be overwritten
    U_int = U[interior_idx];     // interior cell value

    switch (bc_type) {
        case BCType::Dirichlet:
            // Ghost set so that face-center value = bc_value
            // face_value = (U_int + U_ghost) / 2 = bc_value
            U[ghost_idx] = 2.0f * bc_value - U_int;
            break;
        case BCType::Neumann:
            // dU/dn = bc_value → ghost = interior + gradient*distance
            // For zero Neumann (insulated): ghost = interior
            U[ghost_idx] = U_int;  // simplified: zero gradient
            break;
        case BCType::Wall:
            // Adiabatic wall: zero flux → zero gradient
            U[ghost_idx] = U_int;
            break;
        default:
            // Periodic / Inlet / Outlet should not reach here for scalar
            U[ghost_idx] = U_int;
            break;
    }
}

// --- Euler BCs (NVAR=4) ---
HD FORCE_INLINE void apply_bc_euler(
    int ghost_idx, int interior_idx,
    float* RESTRICT U,
    int ncells_total,
    BCType bc_type,
    float nx, float ny,  // outward face normal direction
    float inlet_rho, float inlet_u, float inlet_v, float inlet_p,
    float gamma
) {
    float rho_int  = U[0 * ncells_total + interior_idx];
    float rhou_int = U[1 * ncells_total + interior_idx];
    float rhov_int = U[2 * ncells_total + interior_idx];
    float E_int    = U[3 * ncells_total + interior_idx];

    float u_int = rhou_int / rho_int;
    float v_int = rhov_int / rho_int;
    float p_int = (gamma - 1.0f) * (E_int - 0.5f * rho_int * (u_int*u_int + v_int*v_int));

    switch (bc_type) {
        case BCType::Wall: {
            // Reflective wall: reflect normal velocity, keep tangential
            // vn = u*nx + v*ny → reflected: u' = u - 2*vn*nx, v' = v - 2*vn*ny
            float vn = u_int * nx + v_int * ny;
            float u_ghost = u_int - 2.0f * vn * nx;
            float v_ghost = v_int - 2.0f * vn * ny;
            U[0 * ncells_total + ghost_idx] = rho_int;
            U[1 * ncells_total + ghost_idx] = rho_int * u_ghost;
            U[2 * ncells_total + ghost_idx] = rho_int * v_ghost;
            U[3 * ncells_total + ghost_idx] = E_int;  // energy unchanged
            break;
        }
        case BCType::Inlet: {
            // Prescribed state
            float E_in = inlet_p / (gamma - 1.0f) + 0.5f * inlet_rho * (inlet_u*inlet_u + inlet_v*inlet_v);
            U[0 * ncells_total + ghost_idx] = inlet_rho;
            U[1 * ncells_total + ghost_idx] = inlet_rho * inlet_u;
            U[2 * ncells_total + ghost_idx] = inlet_rho * inlet_v;
            U[3 * ncells_total + ghost_idx] = E_in;
            break;
        }
        case BCType::Outlet: {
            // Zero-order extrapolation
            U[0 * ncells_total + ghost_idx] = rho_int;
            U[1 * ncells_total + ghost_idx] = rhou_int;
            U[2 * ncells_total + ghost_idx] = rhov_int;
            U[3 * ncells_total + ghost_idx] = E_int;
            break;
        }
        case BCType::Dirichlet: {
            // For Euler: Dirichlet sets density to bc_value, extrapolates rest
            U[0 * ncells_total + ghost_idx] = 2.0f * 1.0f - rho_int;
            U[1 * ncells_total + ghost_idx] = rhou_int;
            U[2 * ncells_total + ghost_idx] = rhov_int;
            U[3 * ncells_total + ghost_idx] = E_int;
            break;
        }
        default: {
            // Neumann / fallback: extrapolate
            U[0 * ncells_total + ghost_idx] = rho_int;
            U[1 * ncells_total + ghost_idx] = rhou_int;
            U[2 * ncells_total + ghost_idx] = rhov_int;
            U[3 * ncells_total + ghost_idx] = E_int;
            break;
        }
    }
}
