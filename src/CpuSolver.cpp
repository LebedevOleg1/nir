#include "CpuSolver.hpp"
#include "HeatKernel.hpp"
#include <iostream>

bool CpuSolver::init(const Mesh& mesh) {
    flat_faces.resize(mesh.faces.size());
    for(size_t i=0; i < mesh.faces.size(); ++i) {
        flat_faces[i].owner = mesh.faces[i].owner;
        flat_faces[i].neighbor = mesh.faces[i].neighbor;
        flat_faces[i].area = mesh.faces[i].area;
        flat_faces[i].distance = mesh.faces[i].distance;
    }
    initialized = true;
    return true;
}

void CpuSolver::step(Mesh& mesh, float dt) {
    if (!initialized) return;

    int ncells = mesh.get_ncells();
    
    const float* T_curr = mesh.data.curr.T.data();
    float* T_next = mesh.data.next.T.data();
    
    const float* volumes = mesh.volumes.data();
    const int* cell_faces = mesh.cell_faces.data();
    const float* kappa_face = mesh.kappa_face.data();
    const float* source = mesh.source.data();
    const GPUFace* faces_ptr = flat_faces.data();

    #pragma omp parallel for
    for (int i = 0; i < ncells; ++i) {
        calculate_heat_flux_core(
            i, 
            T_curr, T_next, volumes, 
            faces_ptr, cell_faces, kappa_face, source, 
            ncells, dt
        );
    }

    mesh.data.swap_buffers();
}