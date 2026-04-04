#pragma once
#include "PhysicsType.hpp"
#include "BoundaryCondition.hpp"
#include <string>

struct SourceSpec {
    std::string type = "none";  // "none", "gaussian"
    float x = 5.0f, y = 5.0f;
    float radius = 0.5f;
    float power = 1000.0f;
};

struct SimConfig {
    // Physics
    PhysicsType physics = PhysicsType::Heat;
    bool use_gpu = false;

    // Grid
    int nx = 1000;
    int ny = 1000;

    // Time
    int steps = 400;
    int save_every = 10;
    float cfl = 0.5f;

    // Boundary conditions (left, right, bottom, top)
    BCSpec bc[4];  // indexed by Boundary enum

    // Heat/Diffusion
    float kappa = 1.0f;   // thermal conductivity / diffusion coefficient

    // Euler
    float gamma = 1.4f;

    // Initial conditions
    std::string ic = "default";  // "default", "sod", "blast", "kh", "rt"

    // Gravity (for Euler, in -y direction)
    float gravity = 0.0f;

    // Source
    SourceSpec source;
};

SimConfig parse_cli(int argc, char** argv);
void print_config(const SimConfig& cfg, int rank = 0);
