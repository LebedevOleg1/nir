#pragma once
#include <vector>
#include <algorithm>

// SoA double-buffered state vector.
// Variable v of cell c is at data[v * ncells + c].
// This layout gives coalesced GPU access when threads iterate over cells.
template<int NVAR>
struct StateVector {
    int ncells = 0;
    std::vector<float> curr;  // NVAR * ncells
    std::vector<float> next;  // NVAR * ncells

    StateVector() = default;

    void resize(int n) {
        ncells = n;
        curr.assign(NVAR * n, 0.0f);
        next.assign(NVAR * n, 0.0f);
    }

    // Access variable v of cell c in current buffer
    float& operator()(int v, int c)       { return curr[v * ncells + c]; }
    float  operator()(int v, int c) const { return curr[v * ncells + c]; }

    // Raw pointer to variable v in current buffer (for kernels)
    float* var_ptr(int v)             { return curr.data() + v * ncells; }
    const float* var_ptr(int v) const { return curr.data() + v * ncells; }

    // Raw pointer to variable v in next buffer
    float* next_var_ptr(int v)             { return next.data() + v * ncells; }
    const float* next_var_ptr(int v) const { return next.data() + v * ncells; }

    void swap_buffers() { std::swap(curr, next); }
};
