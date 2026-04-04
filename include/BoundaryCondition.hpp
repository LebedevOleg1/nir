#pragma once

enum class BCType {
    Periodic,
    Dirichlet,
    Neumann,
    Wall,
    Inlet,
    Outlet
};

enum class Boundary { Left, Right, Bottom, Top };

struct BCSpec {
    BCType type = BCType::Periodic;
    float value = 0.0f;   // for Dirichlet: fixed value, for Neumann: gradient

    // Euler inlet state
    float inlet_rho = 1.0f;
    float inlet_u   = 0.0f;
    float inlet_v   = 0.0f;
    float inlet_p   = 1.0f;
};
