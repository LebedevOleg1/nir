#include "CudaSolver.hpp"
#include "HeatKernel.hpp" 
#include <vector>


__global__ void heat_fvm_kernel_wrapper(
    const float* RESTRICT T_curr,
    float* RESTRICT T_next,
    const float* RESTRICT volumes,
    const GPUFace* RESTRICT faces,
    const int* RESTRICT cell_faces,
    const float* RESTRICT kappa_face,
    const float* RESTRICT source,
    const int ncells,
    const float dt) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    calculate_heat_flux_core(i, T_curr, T_next, volumes, faces, 
                             cell_faces, kappa_face, source, ncells, dt);
}


CudaSolver::CudaSolver() {}

CudaSolver::~CudaSolver() {
    if (d_T_curr) cudaFree(d_T_curr);
    if (d_T_next) cudaFree(d_T_next);
    if (d_volumes) cudaFree(d_volumes);
    if (d_faces) cudaFree(d_faces);
    if (d_cell_faces) cudaFree(d_cell_faces);
    if (d_kappa_face) cudaFree(d_kappa_face);
    if (d_source) cudaFree(d_source);
}

bool CudaSolver::init_buffers(Mesh* mesh) {
    n_cells = mesh->get_ncells();
    int n_all_faces = mesh->faces.size();
    
    cudaMalloc(&d_T_curr, n_cells * sizeof(float));
    cudaMalloc(&d_T_next, n_cells * sizeof(float));
    cudaMalloc(&d_volumes, n_cells * sizeof(float));
    cudaMalloc(&d_faces, n_all_faces * sizeof(GPUFace));
    cudaMalloc(&d_cell_faces, mesh->cell_faces.size() * sizeof(int));
    cudaMalloc(&d_kappa_face, mesh->kappa_face.size() * sizeof(float));
    cudaMalloc(&d_source, n_cells * sizeof(float));

    std::vector<GPUFace> host_gpu_faces(n_all_faces);
    for(int i=0; i<n_all_faces; ++i) {
        host_gpu_faces[i].owner = mesh->faces[i].owner;
        host_gpu_faces[i].neighbor = mesh->faces[i].neighbor;
        host_gpu_faces[i].area = mesh->faces[i].area;
        host_gpu_faces[i].distance = mesh->faces[i].distance;
    }

    cudaMemcpy(d_volumes, mesh->volumes.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, host_gpu_faces.data(), n_all_faces * sizeof(GPUFace), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cell_faces, mesh->cell_faces.data(), mesh->cell_faces.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kappa_face, mesh->kappa_face.data(), mesh->kappa_face.size() * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_T_curr, mesh->data.curr.T.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source, mesh->source.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);

    initialized = true;
    return true;
}

void CudaSolver::update_source(const std::vector<float>& source_data) {
    if (!initialized) return;
    cudaMemcpy(d_source, source_data.data(), n_cells * sizeof(float), cudaMemcpyHostToDevice);
}

void CudaSolver::read_current_to_host(Mesh* mesh) {
    if (!initialized) return;
    cudaMemcpy(mesh->data.curr.T.data(), d_T_curr, n_cells * sizeof(float), cudaMemcpyDeviceToHost);
}

void CudaSolver::step(float alpha, float dt) {
    if (!initialized) return;

    int threads = 256;
    int blocks = (n_cells + threads - 1) / threads;

    heat_fvm_kernel_wrapper<<<blocks, threads>>>(
        d_T_curr, 
        d_T_next, 
        d_volumes, 
        d_faces, 
        d_cell_faces, 
        d_kappa_face, 
        d_source, 
        n_cells, 
        dt
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaDeviceSynchronize();

    float* temp = d_T_curr;
    d_T_curr = d_T_next;
    d_T_next = temp;
}