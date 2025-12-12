#pragma once
#include "Mesh.hpp"
#include "KernelCommon.hpp"
#include <vector>

class CpuSolver {
private:
    std::vector<GPUFace> flat_faces;
    bool initialized = false;

public:
    bool init(const Mesh& mesh);
    void step(Mesh& mesh, float dt);
    
    void update_source(const Mesh& mesh) {}
    void read_back(Mesh& mesh) {}
};