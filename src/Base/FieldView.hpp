#pragma once
#include "FvmMacros.hpp"
#include "FvmTypes.hpp"

// ============================================================================
// FieldView — thin SoA accessor (inspired by AMReX Array4).
//
// Wraps a raw pointer to SoA state: data[var * ncells_total + cell].
// Zero overhead: operator() compiles to a single multiply + add + load.
//
// Multi-variable:  view(var, cell)  →  data[var * N + cell]
// Single-variable: view(cell)       →  data[cell]
//
// Typical use in a kernel:
//   FieldView curr{d_curr, ncells_total};
//   FieldView next{d_next, ncells_total};
//   Real rho = curr(0, i);
//   next(0, i) = rho + dt * ...;
// ============================================================================
struct FieldView {
    Real* FVM_RESTRICT data;
    int   ncells_total;

    // Multi-variable access
    FVM_HOST_DEVICE FVM_INLINE
    Real& operator()(int var, int cell) noexcept {
        return data[var * ncells_total + cell];
    }
    FVM_HOST_DEVICE FVM_INLINE
    Real operator()(int var, int cell) const noexcept {
        return data[var * ncells_total + cell];
    }

    // Single-variable shorthand (NVAR=1 physics)
    FVM_HOST_DEVICE FVM_INLINE
    Real& operator()(int cell) noexcept { return data[cell]; }
    FVM_HOST_DEVICE FVM_INLINE
    Real operator()(int cell) const noexcept { return data[cell]; }
};
