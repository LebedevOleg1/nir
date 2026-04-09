#pragma once
#include "FvmMacros.hpp"

// ============================================================================
// ParallelFor — unified CPU/GPU loop dispatch.
//
// Usage (same syntax on CPU and GPU):
//
//   ParallelFor(use_gpu, n_cells, [=] FVM_HOST_DEVICE (int i) {
//       U_next(0, i) = U_curr(0, i) + dt * ...;
//   });
//
// CPU path (compiled by g++):
//   OpenMP parallel for loop.
//
// GPU path (compiled by nvcc, requires --expt-extended-lambda):
//   CUDA kernel with grid-stride loop.
//   The lambda must be annotated with FVM_HOST_DEVICE.
//
// Switching backend: set FVM_BACKEND_THREADPOOL and provide g_thread_pool.
// The interface (ParallelFor signature) never changes.
// ============================================================================

#ifdef __CUDACC__

// Grid-stride kernel: each thread handles multiple elements if grid is small.
template<typename Func>
__global__ void fvm_parallel_for_kernel(int n, Func f) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        f(i);
}

template<typename Func>
FVM_INLINE void ParallelFor(bool use_gpu, int n, Func f) {
    if (use_gpu) {
        constexpr int kThreads = 256;
        int blocks = (n + kThreads - 1) / kThreads;
        fvm_parallel_for_kernel<<<blocks, kThreads>>>(n, f);
        return;
    }
    // CPU fallback inside .cu compilation unit
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) f(i);
}

#else  // CPU-only (g++ path)

template<typename Func>
FVM_INLINE void ParallelFor(bool /*use_gpu*/, int n, Func f) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) f(i);
}

#endif  // __CUDACC__
