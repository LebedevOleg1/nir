#include "Config.hpp"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <algorithm>

static BCType parse_bc_type(const std::string& s) {
    // Extract type before ':'
    std::string type_str = s.substr(0, s.find(':'));
    if (type_str == "periodic")  return BCType::Periodic;
    if (type_str == "dirichlet") return BCType::Dirichlet;
    if (type_str == "neumann")   return BCType::Neumann;
    if (type_str == "wall")      return BCType::Wall;
    if (type_str == "inlet")     return BCType::Inlet;
    if (type_str == "outlet")    return BCType::Outlet;
    std::cerr << "Unknown BC type: " << type_str << "\n";
    std::exit(1);
}

static float parse_bc_value(const std::string& s) {
    auto pos = s.find(':');
    if (pos == std::string::npos) return 0.0f;
    return std::stof(s.substr(pos + 1));
}

static BCSpec parse_bc(const std::string& s) {
    BCSpec bc;
    bc.type = parse_bc_type(s);
    bc.value = parse_bc_value(s);
    return bc;
}

static SourceSpec parse_source(const std::string& s) {
    SourceSpec src;
    auto colon = s.find(':');
    if (colon == std::string::npos) {
        src.type = s;
        return src;
    }
    src.type = s.substr(0, colon);
    // Parse "x,y,radius,power"
    std::string params = s.substr(colon + 1);
    std::replace(params.begin(), params.end(), ',', ' ');
    std::istringstream iss(params);
    iss >> src.x >> src.y >> src.radius >> src.power;
    return src;
}

static std::string get_arg_value(const std::string& arg) {
    auto pos = arg.find('=');
    if (pos == std::string::npos) return "";
    return arg.substr(pos + 1);
}

SimConfig parse_cli(int argc, char** argv) {
    SimConfig cfg;

    // Default: all periodic
    for (int i = 0; i < 4; ++i)
        cfg.bc[i].type = BCType::Periodic;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Positional: cpu/gpu
        if (arg == "gpu") { cfg.use_gpu = true; continue; }
        if (arg == "cpu") { cfg.use_gpu = false; continue; }

        std::string val = get_arg_value(arg);

        if (arg.rfind("--physics=", 0) == 0) {
            if (val == "heat")      cfg.physics = PhysicsType::Heat;
            else if (val == "diffusion") cfg.physics = PhysicsType::Diffusion;
            else if (val == "euler") cfg.physics = PhysicsType::Euler;
            else { std::cerr << "Unknown physics: " << val << "\n"; std::exit(1); }
        }
        else if (arg.rfind("--nx=", 0) == 0)         cfg.nx = std::stoi(val);
        else if (arg.rfind("--ny=", 0) == 0)         cfg.ny = std::stoi(val);
        else if (arg.rfind("--steps=", 0) == 0)      cfg.steps = std::stoi(val);
        else if (arg.rfind("--save-every=", 0) == 0)  cfg.save_every = std::stoi(val);
        else if (arg.rfind("--cfl=", 0) == 0)        cfg.cfl = std::stof(val);
        else if (arg.rfind("--kappa=", 0) == 0)      cfg.kappa = std::stof(val);
        else if (arg.rfind("--gamma=", 0) == 0)      cfg.gamma = std::stof(val);
        else if (arg.rfind("--bc-left=", 0) == 0)    cfg.bc[(int)Boundary::Left]   = parse_bc(val);
        else if (arg.rfind("--bc-right=", 0) == 0)   cfg.bc[(int)Boundary::Right]  = parse_bc(val);
        else if (arg.rfind("--bc-bottom=", 0) == 0)  cfg.bc[(int)Boundary::Bottom] = parse_bc(val);
        else if (arg.rfind("--bc-top=", 0) == 0)     cfg.bc[(int)Boundary::Top]    = parse_bc(val);
        else if (arg.rfind("--source=", 0) == 0)     cfg.source = parse_source(val);
        else if (arg.rfind("--ic=", 0) == 0)         cfg.ic = val;
        else if (arg.rfind("--gravity=", 0) == 0)    cfg.gravity = std::stof(val);
        else if (arg.rfind("--inlet-rho=", 0) == 0) {
            cfg.bc[(int)Boundary::Left].inlet_rho = std::stof(val);
            cfg.bc[(int)Boundary::Right].inlet_rho = std::stof(val);
            cfg.bc[(int)Boundary::Bottom].inlet_rho = std::stof(val);
            cfg.bc[(int)Boundary::Top].inlet_rho = std::stof(val);
        }
        else if (arg.rfind("--inlet-u=", 0) == 0) {
            for (auto& b : cfg.bc) b.inlet_u = std::stof(val);
        }
        else if (arg.rfind("--inlet-v=", 0) == 0) {
            for (auto& b : cfg.bc) b.inlet_v = std::stof(val);
        }
        else if (arg.rfind("--inlet-p=", 0) == 0) {
            for (auto& b : cfg.bc) b.inlet_p = std::stof(val);
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::exit(1);
        }
    }

    return cfg;
}

static const char* bc_name(BCType t) {
    switch (t) {
        case BCType::Periodic:  return "periodic";
        case BCType::Dirichlet: return "dirichlet";
        case BCType::Neumann:   return "neumann";
        case BCType::Wall:      return "wall";
        case BCType::Inlet:     return "inlet";
        case BCType::Outlet:    return "outlet";
    }
    return "unknown";
}

static const char* physics_name(PhysicsType p) {
    switch (p) {
        case PhysicsType::Heat:      return "heat";
        case PhysicsType::Diffusion: return "diffusion";
        case PhysicsType::Euler:     return "euler";
    }
    return "unknown";
}

void print_config(const SimConfig& cfg, int rank) {
    if (rank != 0) return;
    std::cout << "=== Configuration ===\n"
              << "Physics: " << physics_name(cfg.physics)
              << " | Device: " << (cfg.use_gpu ? "GPU" : "CPU") << "\n"
              << "Grid: " << cfg.nx << "x" << cfg.ny
              << " | Steps: " << cfg.steps
              << " | Save every: " << cfg.save_every << "\n"
              << "BC: left=" << bc_name(cfg.bc[0].type)
              << " right=" << bc_name(cfg.bc[1].type)
              << " bottom=" << bc_name(cfg.bc[2].type)
              << " top=" << bc_name(cfg.bc[3].type) << "\n";
    if (cfg.physics == PhysicsType::Heat || cfg.physics == PhysicsType::Diffusion)
        std::cout << "kappa=" << cfg.kappa << "\n";
    if (cfg.physics == PhysicsType::Euler) {
        std::cout << "gamma=" << cfg.gamma << " | IC=" << cfg.ic;
        if (cfg.gravity != 0.0f) std::cout << " | g=" << cfg.gravity;
        std::cout << "\n";
    }
    if (cfg.source.type != "none")
        std::cout << "Source: " << cfg.source.type
                  << " at (" << cfg.source.x << "," << cfg.source.y
                  << ") r=" << cfg.source.radius
                  << " P=" << cfg.source.power << "\n";
    std::cout << "=====================\n";
}
