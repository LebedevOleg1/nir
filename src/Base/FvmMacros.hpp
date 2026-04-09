#pragma once

// ============================================================================
// Cross-platform CPU/GPU annotation macros.
//
// FVM_HOST_DEVICE  — compiled for both host and device (like AMReX AMREX_GPU_HOST_DEVICE)
// FVM_DEVICE       — device only
// FVM_INLINE       — forced inline (__forceinline__ on GPU, inline on CPU)
// FVM_RESTRICT     — no-alias pointer hint (enables vectorization / reordering)
// ============================================================================

#ifdef __CUDACC__
  #define FVM_HOST_DEVICE __host__ __device__
  #define FVM_DEVICE      __device__
  #define FVM_INLINE      __forceinline__
#else
  #define FVM_HOST_DEVICE
  #define FVM_DEVICE
  #define FVM_INLINE      inline
#endif

#if defined(__CUDACC__) || defined(__GNUC__) || defined(__clang__)
  #define FVM_RESTRICT __restrict__
#elif defined(_MSC_VER)
  #define FVM_RESTRICT __restrict
#else
  #define FVM_RESTRICT
#endif
