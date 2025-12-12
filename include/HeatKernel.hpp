#pragma once
#include "KernelCommon.hpp"

HD FORCE_INLINE void calculate_heat_flux_core(
    int i,
    const float* RESTRICT T_curr,
    float* RESTRICT T_next,
    const float* RESTRICT volumes,
    const GPUFace* RESTRICT faces,
    const int* RESTRICT cell_faces,
    const float* RESTRICT kappa_face,
    const float* RESTRICT source,
    const int ncells,
    const float dt
) {
    if (i >= ncells) return;

    float T_c = T_curr[i];
    float flux_sum = 0.0f;

    #ifdef __CUDACC__
    #pragma unroll
    #endif
    for (int k = 0; k < 4; ++k) {
        int face_idx = cell_faces[i * 4 + k];
        
        GPUFace face = faces[face_idx]; 
        
        int neighbor_idx = face.neighbor;
        float T_neighbor = T_curr[neighbor_idx];
        
        float grad_T = (T_neighbor - T_c) / face.distance;
        float flux = kappa_face[face_idx] * face.area * grad_T;
        
        flux_sum += flux;
    }

    T_next[i] = T_c + dt * (flux_sum / volumes[i] + source[i]);
}