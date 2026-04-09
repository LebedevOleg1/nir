#include "IO/InputParser.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdlib>

// ============================================================================
// Helpers
// ============================================================================
static BCType parse_bc_type(const std::string& s) {
    std::string t = s.substr(0, s.find(':'));
    if (t == "periodic")  return BCType::Periodic;
    if (t == "dirichlet") return BCType::Dirichlet;
    if (t == "neumann")   return BCType::Neumann;
    if (t == "wall")      return BCType::Wall;
    if (t == "inlet")     return BCType::Inlet;
    if (t == "outlet")    return BCType::Outlet;
    std::cerr << "[InputParser] Unknown BC type: " << t << "\n";
    std::exit(1);
}

static float parse_bc_value(const std::string& s) {
    auto pos = s.find(':');
    return (pos == std::string::npos) ? 0.0f : std::stof(s.substr(pos + 1));
}

static BCSpec parse_bc(const std::string& s) {
    BCSpec bc;
    bc.type  = parse_bc_type(s);
    bc.value = parse_bc_value(s);
    return bc;
}

static SourceSpec parse_source(const std::string& s) {
    SourceSpec src;
    auto colon = s.find(':');
    if (colon == std::string::npos) { src.type = s; return src; }
    src.type = s.substr(0, colon);
    std::string params = s.substr(colon + 1);
    std::replace(params.begin(), params.end(), ',', ' ');
    std::istringstream iss(params);
    iss >> src.x >> src.y >> src.radius >> src.power;
    return src;
}

static void apply_kv(SimConfig& cfg, const std::string& key, const std::string& val) {
    if (key == "physics") {
        if (val == "heat")      cfg.physics = PhysicsType::Heat;
        else if (val == "diffusion") cfg.physics = PhysicsType::Diffusion;
        else if (val == "euler")     cfg.physics = PhysicsType::Euler;
        else { std::cerr << "[InputParser] Unknown physics: " << val << "\n"; std::exit(1); }
    }
    else if (key == "device") {
        cfg.use_gpu = (val == "gpu" || val == "1" || val == "true");
    }
    else if (key == "nx")          cfg.nx         = std::stoi(val);
    else if (key == "ny")          cfg.ny         = std::stoi(val);
    else if (key == "steps")       cfg.steps      = std::stoi(val);
    else if (key == "save-every" || key == "save_every") cfg.save_every = std::stoi(val);
    else if (key == "cfl")         cfg.cfl        = std::stof(val);
    else if (key == "kappa")       cfg.kappa      = std::stof(val);
    else if (key == "gamma")       cfg.gamma      = std::stof(val);
    else if (key == "gravity")     cfg.gravity    = std::stof(val);
    else if (key == "ic")          cfg.ic         = val;
    else if (key == "xmin")        cfg.xmin       = std::stof(val);
    else if (key == "xmax")        cfg.xmax       = std::stof(val);
    else if (key == "ymin")        cfg.ymin       = std::stof(val);
    else if (key == "ymax")        cfg.ymax       = std::stof(val);
    else if (key == "bc-left"   || key == "bc_left")   cfg.bc[(int)Boundary::Left]   = parse_bc(val);
    else if (key == "bc-right"  || key == "bc_right")  cfg.bc[(int)Boundary::Right]  = parse_bc(val);
    else if (key == "bc-bottom" || key == "bc_bottom") cfg.bc[(int)Boundary::Bottom] = parse_bc(val);
    else if (key == "bc-top"    || key == "bc_top")    cfg.bc[(int)Boundary::Top]    = parse_bc(val);
    else if (key == "source")      cfg.source     = parse_source(val);
    // silently ignore unknown keys (future compatibility)
}

// ============================================================================
// InputParser::from_file — read key = value file
// ============================================================================
SimConfig InputParser::from_file(const std::string& filename) {
    SimConfig cfg;
    // Default: all periodic
    for (int i = 0; i < 4; ++i) cfg.bc[i].type = BCType::Periodic;

    std::ifstream f(filename);
    if (!f.is_open()) {
        // Not an error: file may not exist (all defaults)
        return cfg;
    }

    std::string line;
    while (std::getline(f, line)) {
        // Strip comment
        auto cpos = line.find('#');
        if (cpos != std::string::npos) line = line.substr(0, cpos);

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);

        // Trim whitespace
        auto trim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t\r\n"));
            s.erase(s.find_last_not_of(" \t\r\n") + 1);
        };
        trim(key); trim(val);
        if (key.empty() || val.empty()) continue;

        apply_kv(cfg, key, val);
    }
    return cfg;
}

// ============================================================================
// InputParser::from_cli — read --key=value command-line arguments
// ============================================================================
SimConfig InputParser::from_cli(int argc, char** argv) {
    SimConfig cfg;
    for (int i = 0; i < 4; ++i) cfg.bc[i].type = BCType::Periodic;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Positional: cpu / gpu
        if (arg == "gpu") { cfg.use_gpu = true; continue; }
        if (arg == "cpu") { cfg.use_gpu = false; continue; }

        // Strip leading --
        if (arg.rfind("--", 0) == 0) arg = arg.substr(2);

        auto eq = arg.find('=');
        if (eq == std::string::npos) continue;
        std::string key = arg.substr(0, eq);
        std::string val = arg.substr(eq + 1);
        apply_kv(cfg, key, val);
    }
    return cfg;
}

// ============================================================================
// InputParser::load — file + CLI overrides
// ============================================================================
SimConfig InputParser::load(const std::string& filename, int argc, char** argv) {
    SimConfig cfg = from_file(filename);

    // Apply CLI on top
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "gpu") { cfg.use_gpu = true; continue; }
        if (arg == "cpu") { cfg.use_gpu = false; continue; }
        if (arg.rfind("--", 0) == 0) arg = arg.substr(2);
        auto eq = arg.find('=');
        if (eq == std::string::npos) continue;
        apply_kv(cfg, arg.substr(0, eq), arg.substr(eq + 1));
    }
    return cfg;
}

// ============================================================================
// InputParser::print
// ============================================================================
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

void InputParser::print(const SimConfig& cfg, int rank) {
    if (rank != 0) return;
    const char* phys = (cfg.physics == PhysicsType::Heat) ? "heat" :
                       (cfg.physics == PhysicsType::Diffusion) ? "diffusion" : "euler";
    std::cout << "=== Config ===\n"
              << "physics=" << phys
              << " device=" << (cfg.use_gpu ? "gpu" : "cpu") << "\n"
              << "domain=[" << cfg.xmin << "," << cfg.xmax << "]x["
                            << cfg.ymin << "," << cfg.ymax << "]\n"
              << "grid=" << cfg.nx << "x" << cfg.ny
              << " steps=" << cfg.steps << " save_every=" << cfg.save_every << "\n"
              << "cfl=" << cfg.cfl;
    if (cfg.physics == PhysicsType::Euler)
        std::cout << " gamma=" << cfg.gamma << " gravity=" << cfg.gravity
                  << " ic=" << cfg.ic;
    else
        std::cout << " kappa=" << cfg.kappa;
    std::cout << "\nbc: left=" << bc_name(cfg.bc[0].type)
              << " right=" << bc_name(cfg.bc[1].type)
              << " bottom=" << bc_name(cfg.bc[2].type)
              << " top=" << bc_name(cfg.bc[3].type) << "\n"
              << "==============\n";
}
