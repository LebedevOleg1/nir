#pragma once
#include "Mesh.hpp"
#include "KernelCommon.hpp"
#include <cuda_runtime.h>
#include <iostream>


class CudaSolver {
private:
    float* d_T_curr = nullptr;
    float* d_T_next = nullptr;
    float* d_volumes = nullptr;
    
    GPUFace* d_faces = nullptr;
    int* d_cell_faces = nullptr;
    
    float* d_kappa_face = nullptr;
    float* d_source = nullptr;

    int n_cells = 0;
    int n_faces = 0;
    bool initialized = false;

public:
    CudaSolver();
    ~CudaSolver();

    bool is_available() const { return true; }
    bool init_buffers(Mesh* mesh);
    void update_source(const std::vector<float>& source_data);
    void read_current_to_host(Mesh* mesh);
    void step(float alpha, float dt);
};