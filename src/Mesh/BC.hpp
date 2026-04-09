#pragma once

// ============================================================================
// Boundary condition types and specs.
// ============================================================================

enum class BCType {
    Periodic,
    Dirichlet,
    Neumann,
    Wall,
    Inlet,
    Outlet
};

enum class Boundary { Left = 0, Right = 1, Bottom = 2, Top = 3 };

struct BCSpec {
    BCType type  = BCType::Periodic;
    float  value = 0.0f;   // Dirichlet: fixed value; Neumann: gradient

    // Euler inlet state (used when type == Inlet)
    float inlet_rho = 1.0f;
    float inlet_u   = 0.0f;
    float inlet_v   = 0.0f;
    float inlet_p   = 1.0f;
};
