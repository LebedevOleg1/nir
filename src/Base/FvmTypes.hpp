#pragma once
#include <cstdint>

// ============================================================================
// Core scalar types.
// Swap Real = double for double-precision runs (one-line change).
// ============================================================================
using Real = float;
using Int  = int;

struct Vec3 {
    Real x = 0, y = 0, z = 0;
    constexpr Vec3() = default;
    constexpr Vec3(Real x_, Real y_, Real z_ = 0) : x(x_), y(y_), z(z_) {}
};
