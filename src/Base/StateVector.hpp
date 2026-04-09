#pragma once
#include "FvmTypes.hpp"
#include <vector>
#include <algorithm>

// ============================================================================
// StateVector<NVAR> — double-buffered SoA state on CPU.
//
// Variable v of cell c is at data[v * ncells + c].
// SoA layout gives coalesced GPU reads when threads iterate over c.
// ============================================================================
template<int NVAR>
struct StateVector {
    int ncells = 0;
    std::vector<Real> curr;   // NVAR * ncells
    std::vector<Real> next;   // NVAR * ncells

    StateVector() = default;

    void resize(int n) {
        ncells = n;
        curr.assign(NVAR * n, Real(0));
        next.assign(NVAR * n, Real(0));
    }

    // Access variable v of cell c in current buffer
    Real& operator()(int v, int c)       { return curr[v * ncells + c]; }
    Real  operator()(int v, int c) const { return curr[v * ncells + c]; }

    // Raw pointer to variable v in current buffer (for kernels)
    Real*       var_ptr(int v)       { return curr.data() + v * ncells; }
    const Real* var_ptr(int v) const { return curr.data() + v * ncells; }

    // Raw pointer to variable v in next buffer
    Real*       next_var_ptr(int v)       { return next.data() + v * ncells; }
    const Real* next_var_ptr(int v) const { return next.data() + v * ncells; }

    void swap_buffers() { std::swap(curr, next); }
};
